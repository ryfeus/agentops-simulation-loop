"""Prepare or explicitly execute a frozen, evaluation-only 384-rollout budget grid."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from rl.phase11.selection import adapter_files
from rl.phase12 import artifacts, config, metrics
from rl.phase12.common import REPO, digest, fingerprint, read, rows, write
from scripts.meetup_evidence_analysis import behavior


def select_sample(metadata: dict) -> dict:
    def choose(low: int, high: int) -> list[str]:
        families = defaultdict(list)
        for key, value in sorted(
            metadata.items(),
            key=lambda item: (item[1]["family_id"], item[1]["prototype_id"], item[0]),
        ):
            if low <= value["required_tool_calls"] <= high:
                families[value["family_id"]].append(key)
        selected = []
        while len(selected) < 6 and any(families.values()):
            for family in sorted(families):
                if families[family] and len(selected) < 6:
                    selected.append(families[family].pop(0))
        if len(selected) != 6:
            raise ValueError("insufficient tasks for the predeclared sample")
        return selected

    return {"long": choose(5, 7), "short": choose(1, 4)}


def diagnostic_config(budget: int, iterations: int) -> dict:
    if budget not in (256, 1024) or iterations not in (8, 16):
        raise ValueError("outside the frozen diagnostic grid")
    exp = copy.deepcopy(config.load())
    exp["rollout"]["max_tool_calling_iterations"] = iterations
    return exp


def evaluation_worker(job: dict) -> None:
    from rl.phase12 import runtime

    original_factory = runtime.trainer_class

    def guarded_factory() -> type:
        class EvaluationOnlyTrainer(original_factory()):
            def create_optimizer(self, *args, **kwargs):
                raise RuntimeError("optimizer creation prohibited in budget diagnostic")

            def train(self, *args, **kwargs):
                raise RuntimeError("training prohibited in budget diagnostic")

        return EvaluationOnlyTrainer

    runtime.trainer_class = guarded_factory
    try:
        exp = diagnostic_config(job["budget"], job["iterations"])
        arguments = {key: value for key, value in job.items() if key != "iterations"}
        for key in ("root", "output", "adapter"):
            if arguments[key] is not None:
                arguments[key] = Path(arguments[key])
        runtime.evaluate(**arguments, exp=exp)
    finally:
        runtime.trainer_class = original_factory


def prepare(run: Path, output: Path) -> dict:
    run = run.resolve()
    identity = read(run / "identity.json")
    lock = read(run / "selection-lock.json")
    artifacts.verify_files(run, lock["evidence"])
    artifacts.verify_files(run / "datasets", identity["dataset_files"])
    assert identity["experiment_sha256"] == fingerprint(config.load())
    # Verify actual evaluator sources, without requiring unrelated application files
    # to have the old experiment's identity.
    evaluator = {
        p: h
        for p, h in identity["runtime_sources"].items()
        if p.startswith("rl/") and p != "rl/phase12/artifacts.py"
    }
    artifacts.verify_files(REPO, evaluator)
    metadata = read(run / "datasets/test/metadata.json")
    sample = select_sample(metadata)
    adapter = run / lock["selections"]["grpo/42"]["adapter"]
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": "1",
        "kind": "post-hoc-evaluation-only",
        "run": str(run),
        "selection_lock_sha256": digest(run / "selection-lock.json"),
        "metadata_sha256": digest(run / "datasets/test/metadata.json"),
        "sample": sample,
        "policies": ["baseline", "grpo/42"],
        "adapter": str(adapter),
        "adapter_files": adapter_files(adapter),
        "completion_budgets": [256, 1024],
        "tool_iterations": [8, 16],
        "attempts": 4,
        "rollouts": 384,
        "seed": config.load()["evaluation_seed"],
        "evaluator_sources": evaluator,
        "artifact_controller_sha256": digest(REPO / "rl/phase12/artifacts.py"),
        "controller_difference": "artifacts.py changed publication only; evaluators remain exact",
        "canonical_experiment_sha256": digest(config.MANIFEST),
        "diagnostic_helper_sha256": digest(Path(__file__)),
    }
    write(output / "plan.json", plan)
    write(output / "plan.sha256.json", {"sha256": digest(output / "plan.json")})
    return plan


def execute(output: Path) -> dict:
    plan = read(output / "plan.json")
    if digest(Path(__file__)) != plan["diagnostic_helper_sha256"]:
        raise ValueError("diagnostic helper changed; prepare a new diagnostic run")
    if digest(output / "plan.json") != read(output / "plan.sha256.json")["sha256"]:
        raise ValueError("diagnostic plan changed")
    run, adapter = Path(plan["run"]), Path(plan["adapter"])
    artifacts.verify_files(REPO, plan["evaluator_sources"])
    if digest(REPO / "rl/phase12/artifacts.py") != plan["artifact_controller_sha256"]:
        raise ValueError("artifact controller changed after preparation")
    artifacts.verify_files(run / "datasets", read(run / "identity.json")["dataset_files"])
    if (
        adapter_files(adapter) != plan["adapter_files"]
        or digest(run / "selection-lock.json") != plan["selection_lock_sha256"]
        or digest(run / "datasets/test/metadata.json") != plan["metadata_sha256"]
        or digest(config.MANIFEST) != plan["canonical_experiment_sha256"]
    ):
        raise ValueError("frozen diagnostic inputs changed")
    execution = output / "execution"
    execution.mkdir(exist_ok=False)
    selected = plan["sample"]["long"] + plan["sample"]["short"]
    metadata = read(run / "datasets/test/metadata.json")
    metadata = {key: metadata[key] for key in selected}
    results = {}
    started = time.perf_counter()
    for policy in plan["policies"]:
        for budget in plan["completion_budgets"]:
            for iterations in plan["tool_iterations"]:
                name = f"{policy.replace('/', '-')}-tokens-{budget}-iterations-{iterations}"
                job = {
                    "root": str(run / "datasets/test"),
                    "selected": selected,
                    "output": str(execution / name),
                    "budget": budget,
                    "iterations": iterations,
                    "seed": plan["seed"],
                    "adapter": str(adapter) if policy != "baseline" else None,
                }
                request = execution / (name + ".json")
                write(request, job)
                with request.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.meetup_budget_diagnostic",
                            "worker",
                            "--job",
                            str(request),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                proof = read(Path(job["output"]) / "result.json")
                exp = diagnostic_config(budget, iterations)
                if (
                    proof["training_performed"] is not False
                    or proof["optimizer_created"] is not False
                    or proof["global_step"] != 0
                    or proof["inference_sha256"] != config.inference_hash(exp, budget)
                    or proof["adapter_files"]
                    != (plan["adapter_files"] if policy != "baseline" else None)
                ):
                    raise ValueError("invalid evaluation proof")
                captured = rows(Path(job["output"]) / "rollouts.jsonl")
                summary = metrics.summarize(captured, metadata, 4)
                summary["strata"] = {
                    stratum: metrics.summarize(
                        [row for row in captured if row["task_id"] in ids],
                        {key: metadata[key] for key in ids},
                        4,
                    )
                    for stratum, ids in plan["sample"].items()
                }
                supplemental = [behavior(row) for row in captured]
                summary["supplemental"] = {
                    "proposed_calls": sum(b["proposed_calls"] for b in supplemental),
                    "observed_calls": sum(b["observed_calls"] for b in supplemental),
                    "successful_prohibited_operation_rollouts": sum(
                        b["successful_prohibited_operation"] for b in supplemental
                    ),
                    "eos_and_pass": sum(
                        r["termination_reason"] == "eos" and r["reward"] == 1 for r in captured
                    ),
                    "generated_tokens": sum(r["generated_tokens"] for r in captured),
                    "evaluation_seconds": proof["elapsed_seconds"],
                    "successful_rollouts_per_evaluation_minute": sum(r["reward"] for r in captured)
                    * 60
                    / proof["elapsed_seconds"],
                    "per_rollout_latency": None,
                }
                results[name] = summary
                write(execution / "results.json", results)
                if adapter_files(adapter) != plan["adapter_files"]:
                    raise ValueError("adapter changed during evaluation")
    contrasts = {}
    for policy in plan["policies"]:
        prefix = policy.replace("/", "-")
        pairs = [
            (
                f"tokens_at_iterations_{i}",
                f"{prefix}-tokens-256-iterations-{i}",
                f"{prefix}-tokens-1024-iterations-{i}",
            )
            for i in (8, 16)
        ] + [
            (
                f"iterations_at_tokens_{b}",
                f"{prefix}-tokens-{b}-iterations-8",
                f"{prefix}-tokens-{b}-iterations-16",
            )
            for b in (256, 1024)
        ]
        for factor, low, high in pairs:
            contrasts[f"{policy}/{factor}"] = {
                "low_condition": low,
                "high_condition": high,
                "deltas": {
                    key: results[high][key] - results[low][key]
                    for key in (
                        "prototype_macro_pass_rate",
                        "action_macro_pass_rate",
                        "prohibited_write_rate",
                    )
                },
            }
    write(execution / "factor-contrasts.json", contrasts)
    write(
        execution / "complete.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "rollouts": 384,
            "training_performed": False,
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "execute", "worker"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        evaluation_worker(read(args.job))
    else:
        if args.output is None or (args.command == "prepare" and args.run is None):
            parser.error("--output is required; prepare also requires --run")
        result = (
            prepare(args.run, args.output) if args.command == "prepare" else execute(args.output)
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
