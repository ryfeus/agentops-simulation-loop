"""Explicit staged execution with fresh processes for each GPU trainer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from rl.phase11.selection import adapter_files
from rl.phase12 import config
from rl.phase12.artifacts import eligible_methods, verify_lock, verify_run
from rl.phase12.common import fingerprint, read, rows, write
from rl.phase12.corpus import representatives
from rl.phase12.diagnostics import select_budget
from rl.phase12.metrics import choose, final_result, pilot_gate, summarize


def launch(run: Path, kind: str, arguments: dict[str, Any]) -> None:
    job = run / "jobs" / f"{kind}-{fingerprint(arguments)[:16]}.json"
    write(job, arguments)
    log = job.with_suffix(".log")
    with log.open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rl.phase12.runner",
                "worker",
                "--kind",
                kind,
                "--job",
                str(job),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate(
    run: Path,
    role: str,
    output: Path,
    selected: list[str],
    budget: int,
    *,
    adapter: Path | None = None,
    thinking: bool = False,
    attempts: int = 8,
) -> dict[str, Any]:
    exp = config.load()
    if attempts not in (4, 8):
        raise ValueError("evaluation requires four or eight attempts")
    metadata = read(run / f"datasets/{role}/metadata.json")
    subset = {key: metadata[key] for key in selected}
    cached = read(output / "summary.json") if (output / "summary.json").exists() else None
    if (output / "request.json").exists():
        proof = read(output / "request.json")
        if proof != {
            "selected": selected,
            "budget": budget,
            "thinking": thinking,
            "attempts": attempts,
            "adapter": str(adapter.relative_to(run)) if adapter else None,
        }:
            raise ValueError("completed evaluation has a different request")
    output.mkdir(parents=True, exist_ok=True)
    write(
        output / "request.json",
        {
            "selected": selected,
            "budget": budget,
            "thinking": thinking,
            "attempts": attempts,
            "adapter": str(adapter.relative_to(run)) if adapter else None,
        },
    )
    combined = []
    for pass_index in range(attempts // 4):
        target = output / f"pass-{pass_index}"
        if not (target / "result.json").exists():
            if target.exists():
                raise ValueError(
                    f"incomplete worker evidence at {target}; preserve and use a fresh run"
                )
            launch(
                run,
                "evaluate",
                {
                    "root": str(run / f"datasets/{role}"),
                    "selected": selected,
                    "output": str(target),
                    "budget": budget,
                    "seed": exp["evaluation_seed"] + pass_index,
                    "adapter": str(adapter) if adapter else None,
                    "thinking": thinking,
                },
            )
        proof = read(target / "result.json")
        if (
            proof.get("execution_valid") is not True
            or proof.get("training_performed") is not False
            or proof.get("global_step") != 0
            or proof.get("optimizer_created") is not False
            or proof.get("seed") != exp["evaluation_seed"] + pass_index
            or proof.get("inference_sha256") != config.inference_hash(exp, budget, thinking)
            or proof.get("adapter_files") != (adapter_files(adapter) if adapter else None)
        ):
            raise ValueError("invalid evaluation worker")
        for row in rows(target / "rollouts.jsonl"):
            row["attempt"] += pass_index * 4
            combined.append(row)
    result = summarize(combined, subset, attempts)
    result["inference_sha256"] = config.inference_hash(exp, budget, thinking)
    result["seed_schedule"] = [exp["evaluation_seed"] + i for i in range(attempts // 4)]
    if cached is not None and cached != result:
        raise ValueError("cached summary differs from rollout evidence")
    if cached is None:
        write(output / "summary.json", result)
    return result


def diagnose(run: Path) -> None:
    exp = config.load()
    meta = read(run / "datasets/train/metadata.json")
    selected = representatives(meta)
    arms = {}
    for name, budget, thinking in (
        ("thinking-256", 256, True),
        ("nonthinking-256", 256, False),
        ("nonthinking-1024", 1024, False),
    ):
        arms[name] = evaluate(
            run,
            "train",
            run / "diagnostics" / name,
            selected,
            budget,
            thinking=thinking,
            attempts=4,
        )
    if (
        arms["nonthinking-1024"]["diagnostics"]["clipped_fraction"]
        > exp["rollout"]["max_clipped_fraction"]
    ):
        arms["nonthinking-2048"] = evaluate(
            run, "train", run / "diagnostics/nonthinking-2048", selected, 2048, attempts=4
        )
    corpus = read(run / "corpus-manifest.json")
    max_calls = max(
        t["required_tool_calls"] for role in corpus["roles"].values() for t in role.values()
    )
    write(run / "diagnostics/selection.json", select_budget(arms, selected, max_calls, exp))


def budget_for(run: Path) -> int:
    selection = read(run / "diagnostics/selection.json")
    exp = config.load()
    arms = {p.parent.name: read(p) for p in (run / "diagnostics").glob("*/summary.json")}
    corpus = read(run / "corpus-manifest.json")
    max_calls = max(
        t["required_tool_calls"] for role in corpus["roles"].values() for t in role.values()
    )
    expected = select_budget(arms, representatives(corpus["roles"]["train"]), max_calls, exp)
    if selection != expected:
        raise ValueError("rollout selection changed or is unsupported by evidence")
    return selection["budget"]


def train_and_select(run: Path, method: str, seed: int, budget: int) -> None:
    prefix = run / method / str(seed)
    output = prefix / "training"
    if not (output / "training-result.json").exists():
        if output.exists():
            raise ValueError("incomplete training exists; preserve evidence and use a fresh run")
        launch(
            run,
            "train",
            {
                "root": str(run / "datasets/train"),
                "demos": str(run / "demonstrations"),
                "output": str(output),
                "budget": budget,
                "method": method,
                "seed": seed,
            },
        )
    trained = read(output / "training-result.json")
    expected_count = 3 if method == "sft" else 2
    if (
        trained.get("execution_valid") is not True
        or trained.get("method") != method
        or trained.get("seed") != seed
        or len(trained.get("adapters", {})) != expected_count
    ):
        raise ValueError("training evidence does not match the requested method and seed")
    for checkpoint, recorded in trained["adapters"].items():
        if adapter_files(output / "adapters" / checkpoint) != recorded:
            raise ValueError("training checkpoint adapter changed")
    if trained["inference_sha256"] != config.inference_hash(config.load(), budget):
        raise ValueError("training and evaluation inference configurations differ")
    selected = sorted(read(run / "datasets/dev/metadata.json"))
    candidates = {
        int(step): evaluate(
            run, "dev", prefix / "dev" / step, selected, budget, adapter=output / "adapters" / step
        )
        for step in trained["adapters"]
    }
    step = choose(candidates)
    write(
        prefix / "selection.json",
        {
            "selected_step": step,
            "method": method,
            "seed": seed,
            "rule": "action_macro_then_prototype_macro_then_earlier",
            "pilot_eligible": pilot_gate(
                read(run / "dev/baseline/summary.json"), candidates[step], config.load()
            ),
        },
    )


def execute(stage: str, run: Path) -> None:
    verify_run(run)
    if stage != "final" and (
        (run / "selection-lock.json").exists() or (run / "test-opened.json").exists()
    ):
        raise ValueError("locked experiments cannot perform additional fitting or selection")
    if stage == "diagnose":
        diagnose(run)
        return
    budget = budget_for(run)
    if stage == "demos":
        launch(
            run,
            "demos",
            {
                "dataset": str(run / "datasets/train/execution"),
                "output": str(run / "demonstrations"),
                "metadata": read(run / "datasets/train/metadata.json"),
            },
        )
    elif stage in ("pilot", "replicate"):
        methods = ["sft", "grpo"] if stage == "pilot" else eligible_methods(run)
        if not methods:
            raise ValueError("no method qualifies for replication")
        if "sft" in methods:
            demos = read(run / "demonstrations/manifest.json")
            captured = rows(run / "demonstrations/demonstrations.jsonl")
            expected_ids = set(read(run / "datasets/train/metadata.json"))
            if (
                demos.get("execution_valid") is not True
                or len(captured) != 160
                or {r["task_id"] for r in captured} != expected_ids
                or demos.get("records_sha256") != fingerprint(captured)
            ):
                raise ValueError("verified training demonstrations are required before SFT")
        evaluate(
            run,
            "dev",
            run / "dev/baseline",
            sorted(read(run / "datasets/dev/metadata.json")),
            budget,
        )
        for method in methods:
            for seed in [42] if stage == "pilot" else [43, 44]:
                train_and_select(run, method, seed, budget)
    elif stage == "final":
        lock = verify_lock(run)
        if not (run / "final-inputs.json").is_file():
            raise ValueError("prepare-final must verify and package the final evaluation inputs")
        from rl.phase12.artifacts import verify_files

        final_inputs = read(run / "final-inputs.json")
        from rl.phase12.common import digest

        if final_inputs["selection_lock_sha256"] != digest(run / "selection-lock.json"):
            raise ValueError("final payload refers to another selection lock")
        verify_files(run, final_inputs["files"])
        write(run / "test-opened.json", {"selection_lock_sha256": fingerprint(lock)})
        test_ids = sorted(read(run / "datasets/test/metadata.json"))
        base = evaluate(run, "test", run / "final/test/baseline", test_ids, budget)
        anchor_ids = sorted(read(run / "datasets/anchors/metadata.json"))
        anchors = evaluate(run, "anchors", run / "final/anchors/baseline", anchor_ids, budget)
        results = {}
        for method in lock["methods"]:
            candidates = {}
            for seed in config.load()["seeds"]:
                selection = lock["selections"][f"{method}/{seed}"]
                adapter = run / selection["adapter"]
                candidates[seed] = evaluate(
                    run,
                    "test",
                    run / f"final/test/{method}/{seed}",
                    test_ids,
                    budget,
                    adapter=adapter,
                )
                anchor = evaluate(
                    run,
                    "anchors",
                    run / f"final/anchors/{method}/{seed}",
                    anchor_ids,
                    budget,
                    adapter=adapter,
                )
                write(
                    run / f"final/anchors/{method}/{seed}/comparison.json",
                    {
                        "prototype_macro_delta": anchor["prototype_macro_pass_rate"]
                        - anchors["prototype_macro_pass_rate"],
                        "regression_observed": anchor["prototype_macro_pass_rate"]
                        < anchors["prototype_macro_pass_rate"],
                    },
                )
            results[method] = final_result(base, candidates, config.load())
        verify_lock(run)
        write(run / "final/result.json", results)
    else:
        raise ValueError("unknown execution stage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["worker", "execute"])
    parser.add_argument("--kind", choices=["evaluate", "train", "demos"])
    parser.add_argument("--job", type=Path)
    parser.add_argument("--stage", choices=["diagnose", "demos", "pilot", "replicate", "final"])
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.command == "execute":
        execute(args.stage, args.run.resolve())
    else:
        values = read(args.job)
        for key in ("root", "dataset", "demos", "output", "adapter"):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        if args.kind == "demos":
            from rl.phase12.demonstrations import collect

            collect(**values)
        else:
            from rl.phase12 import runtime

            getattr(runtime, args.kind)(**values, exp=config.load())


if __name__ == "__main__":
    main()
