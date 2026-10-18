"""Frozen focused fine-tuning comparison: released, trained and LLM-only, 72 trials."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.decision_model.next_evaluate import (
    build_final_freeze,
    check_final_freeze,
    digest,
    evaluate_tasks,
    load_tasks,
    model_predictor,
    summarize,
    validate_task,
    write,
)

ARMS = ("released", "tuned", "llm")
EXPERIMENT = "focused-finetuning-v1"


def planned_order(task_ids: list[str]) -> list[tuple[str, str, int]]:
    result = []
    for attempt in range(2):
        for index, task in enumerate(task_ids):
            shift = (index + attempt) % 3
            result.extend((task, arm, attempt) for arm in ARMS[shift:] + ARMS[:shift])
    return result


def access_path(manifest: Path) -> Path:
    return manifest.resolve().parent / (".evaluation-access-" + digest(manifest) + ".json")


def validate_cohort(manifest: Path) -> list[dict[str, Any]]:
    cohort = json.loads(manifest.read_text())
    if cohort.get("cohort_previously_exposed") is not False or any(
        cohort.get(key) != 0
        for key in (
            "pre_freeze_policy_rollouts",
            "pre_freeze_model_inferences",
            "pre_freeze_authorization_parser_checks",
            "pre_freeze_teacher_collections",
        )
    ):
        raise ValueError("Untouched test provenance required")
    tasks = load_tasks(manifest, "test", 13)
    roots = Counter(task["composition_root"] for task in tasks)
    if len(tasks) != 12 or len(roots) != 6 or set(roots.values()) != {2}:
        raise ValueError("Require 12 untouched tasks, 6 families, 2 variants")
    novelty = cohort.get("novelty_audit", {})
    for key in ("root_collisions", "structural_collisions", "normalized_instruction_collisions"):
        if novelty.get(key) != 0:
            raise ValueError("Held-out novelty validation missing or failed: " + key)
    for task in tasks:
        validate_task(task)
    return tasks


def fixture_evidence(manifest: Path, tasks: list[dict[str, Any]]) -> dict[str, str]:
    fixtures_path = manifest.parent / "fixture-validation.json"
    prototypes_path = manifest.parent / "prototype-specs.json"
    fixtures = json.loads(fixtures_path.read_text())
    prototypes = json.loads(prototypes_path.read_text())
    prototype_hash = hashlib.sha256(
        json.dumps(prototypes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        len(fixtures) != 12
        or {f.get("task_id") for f in fixtures} != {t["task_id"] for t in tasks}
        or any(
            f.get("fixture_only") is not True
            or f.get("agent_or_policy_executed") is not False
            or f.get("expected_endpoint_accepted") is not True
            or f.get("missing_writes_rejected") is not True
            for f in fixtures
        )
        or len(prototypes) != 6
        or {p.get("id") for p in prototypes} != {t["composition_root"] for t in tasks}
        or prototype_hash != json.loads(manifest.read_text()).get("prototype_specs_sha256")
    ):
        raise ValueError("Authored endpoint fixture/prototype evidence missing or invalid")
    return {str(path.resolve()): digest(path) for path in (fixtures_path, prototypes_path)}


def tensor_changes(released: Path, tuned: Path) -> dict[str, Any]:
    """Compare numeric state dicts across upstream export/save_pretrained formats."""
    import torch
    from safetensors.torch import load_file

    def head(path: Path) -> dict[str, Any]:
        # Match pinned upstream load_head_state: both formats contain the same
        # pointer-head state dict; two files would be ambiguous and are refused.
        exported, saved = path / "head.safetensors", path / "slot_head.pt"
        if exported.exists() and saved.exists():
            raise ValueError("Ambiguous head: both head.safetensors and slot_head.pt")
        if exported.is_file():
            state = load_file(str(exported))
        elif saved.is_file():
            state = torch.load(saved, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Missing pointer-head weights: " + str(path))
        if (
            not isinstance(state, dict)
            or not state
            or not all(
                isinstance(name, str) and isinstance(tensor, torch.Tensor)
                for name, tensor in state.items()
            )
        ):
            raise ValueError("Pointer head must contain a nonempty tensor state dict")
        return state

    counts = {"adapter": 0, "head": 0}
    l1 = {"adapter": 0.0, "head": 0.0}

    def compare(before: dict, after: dict, group: str) -> None:
        if before.keys() != after.keys():
            raise ValueError("Changed tensor names: " + group)
        for name, old in before.items():
            new = after[name]
            if old.shape != new.shape or not torch.isfinite(new).all():
                raise ValueError("Invalid trained tensor: " + name)
            delta = (new.to(torch.float64) - old.to(torch.float64)).abs()
            counts[group] += int(torch.count_nonzero(delta))
            l1[group] += float(delta.sum())

    compare(head(released), head(tuned), "head")
    adapter_files = {
        path.relative_to(released) for path in (released / "lora").rglob("*.safetensors")
    }
    tuned_files = {path.relative_to(tuned) for path in (tuned / "lora").rglob("*.safetensors")}
    if not adapter_files or adapter_files != tuned_files:
        raise ValueError("Adapter weight files missing or changed")
    for relative in sorted(adapter_files):
        compare(load_file(str(released / relative)), load_file(str(tuned / relative)), "adapter")
    if any(counts[group] == 0 for group in counts):
        raise ValueError("Tuned arm requires nonzero adapter AND pointer-head updates")
    return {"changed_elements": counts, "absolute_delta_sum": l1}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.freeze.exists() or access_path(args.manifest).exists():
        raise ValueError("Freeze or test access already exists")
    if not 65 <= args.max_job_seconds <= 3600:
        raise ValueError("Evaluation budget must be 65-3600 seconds")
    tasks = validate_cohort(args.manifest)
    evidence = fixture_evidence(args.manifest, tasks)
    selection = json.loads(args.selection_manifest.read_text())
    if selection.get("selected_checkpoint_role") != "trained" or not (
        isinstance(selection.get("optimizer_steps"), int) and selection["optimizer_steps"] > 0
    ):
        raise ValueError("Selection must name a genuinely trained checkpoint and positive steps")
    changes = tensor_changes(args.released, args.tuned)
    freeze = build_final_freeze(
        args.manifest,
        released=args.released,
        tuned=args.tuned,
        released_calibration=args.released_calibration,
        tuned_calibration=args.tuned_calibration,
        fallback_config=args.fallback_config,
        wheel=args.wheel,
        selection_manifest=args.selection_manifest,
        invoice_local_fallback=True,
        cohort_previously_exposed=False,
        selected_checkpoint_role="trained",
    )
    freeze["artifact_sha256"].update(evidence)
    cohort = json.loads(args.manifest.read_text())
    for path, expected in cohort.get("source_sha256", {}).items():
        if digest(Path(path)) != expected:
            raise ValueError("Cohort source changed: " + path)
        freeze["artifact_sha256"][str(Path(path).resolve())] = expected
    for source in (Path(__file__), Path(__file__).with_name("focused_report.py")):
        freeze["source_sha256"][str(source.resolve())] = digest(source)
    thresholds = {"llm": 0.0}
    for arm in ("released", "tuned"):
        threshold = float(json.loads(getattr(args, arm + "_calibration").read_text())["threshold"])
        if not 0 <= threshold <= 1.000001:
            raise ValueError("Invalid frozen confidence threshold")
        thresholds[arm] = threshold
    freeze.update(
        experiment=EXPERIMENT,
        task_ids=[t["task_id"] for t in tasks],
        repeats=2,
        planned_trials=72,
        max_job_seconds=args.max_job_seconds,
        device=args.device,
        thresholds=thresholds,
        weight_changes=changes,
        optimizer_steps=selection["optimizer_steps"],
        evaluation_access_path=str(access_path(args.manifest)),
        artifact_paths={
            name: str(getattr(args, name).resolve())
            for name in (
                "released",
                "tuned",
                "released_calibration",
                "tuned_calibration",
                "fallback_config",
                "wheel",
                "selection_manifest",
            )
        },
        success_target={
            "fallback_reduction_fraction": 0.2,
            "completion_loss": 0,
            "successful_unwanted_writes": 0,
        },
    )
    args.freeze.parent.mkdir(parents=True, exist_ok=True)
    with args.freeze.open("x") as handle:
        handle.write(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return freeze


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.decision_model import harbor
    from experiments.decision_model.model import DecisionModel, checkpoint_identity

    started = time.perf_counter()
    freeze = json.loads(args.freeze.read_text())
    check_final_freeze(freeze, args.manifest)
    if (
        freeze.get("experiment") != EXPERIMENT
        or freeze.get("cohort_previously_exposed") is not False
    ):
        raise ValueError("Wrong experiment/freshness freeze")
    if args.device != freeze["device"] or args.max_job_seconds != freeze["max_job_seconds"]:
        raise ValueError("Execution bounds differ from freeze")
    for name, path in freeze["artifact_paths"].items():
        if str(getattr(args, name).resolve()) != path:
            raise ValueError("Command artifact differs from freeze: " + name)
    for arm in ("released", "tuned"):
        if checkpoint_identity(getattr(args, arm)) != freeze["checkpoints"][arm]:
            raise ValueError("Checkpoint differs from freeze: " + arm)
    tasks = validate_cohort(args.manifest)
    if [t["task_id"] for t in tasks] != freeze["task_ids"]:
        raise ValueError("Task selection differs from freeze")
    claim = access_path(args.manifest)
    if str(claim) != freeze["evaluation_access_path"] or claim.exists():
        raise ValueError("Test access already claimed; no outcome-based retry")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "freeze.json", freeze)
    (args.output / "cohort-manifest.json").write_bytes(args.manifest.read_bytes())
    for path in map(Path, freeze["source_sha256"]):
        (args.output / ("source-" + path.name)).write_bytes(path.read_bytes())
    os.environ.update(
        AWS_REGION="us-west-2",
        AWS_DEFAULT_REGION="us-west-2",
        BILLING_MODEL_TEMPERATURE="0",
        BILLING_MODEL_MAX_TOKENS="512",
        BILLING_MODEL_TOTAL_ATTEMPTS="1",
    )
    manifest = dict(
        schema_version=EXPERIMENT,
        complete=False,
        error=None,
        task_ids=freeze["task_ids"],
        repeats=2,
        planned_trials=72,
        max_job_seconds=args.max_job_seconds,
        device=args.device,
        backend="native-harbor",
        cohort_previously_exposed=False,
        invoice_local_fallback=True,
        max_tool_calls=16,
        max_llm_calls=16,
        fallback_timeout=120,
        task_timeout=180,
        thresholds=freeze["thresholds"],
        freeze_sha256=digest(args.freeze),
        cohort_manifest_sha256=digest(args.manifest),
        execution_order=[],
        model_load_ms={},
        limitations=[
            "Focused authored billing tasks; not a broad external benchmark.",
            "Variants and attempts are correlated within six families.",
            "Native verifier grades database/trajectory, not every prose claim.",
            "Tuned checkpoint is reported even when it regresses.",
        ],
    )
    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    write(args.output / "manifest.json", manifest)
    try:
        with claim.open("x") as handle:
            json.dump(
                dict(
                    cohort_manifest_sha256=digest(args.manifest),
                    freeze_sha256=digest(args.freeze),
                    output=str(args.output.resolve()),
                    claimed_before_model_initialization=True,
                ),
                handle,
                sort_keys=True,
            )
            handle.write("\n")
        (args.output / "evaluation-access.json").write_bytes(claim.read_bytes())
        manifest["evaluation_access_sha256"] = digest(claim)
        predictors = {"llm": lambda state: {"action": "defer", "confidence": 1.0}}
        for arm in ("released", "tuned"):
            loaded = time.perf_counter()
            model = DecisionModel(
                getattr(args, arm),
                device=args.device,
                calibration=getattr(args, arm + "_calibration"),
            )
            predictors[arm] = model_predictor(model)
            manifest["model_load_ms"][arm] = (time.perf_counter() - loaded) * 1000
        fallback = harbor.llm_fallback(args.fallback_config, invoice_local_fallback=True)
        by_id = {task["task_id"]: task for task in tasks}
        for position, (task_id, arm, attempt) in enumerate(planned_order(freeze["task_ids"])):
            remaining = args.max_job_seconds - (time.perf_counter() - started)
            if remaining < 65:
                raise TimeoutError("Insufficient frozen budget for another isolated trial")
            check_final_freeze(freeze, args.manifest)
            manifest["execution_order"].append(
                dict(task_id=task_id, arm=arm, attempt=attempt, position=position)
            )
            write(args.output / "manifest.json", manifest)
            metrics = await evaluate_tasks(
                predictors[arm],
                [by_id[task_id]],
                args.output / "trials" / f"{position:03d}-{arm}-{attempt}-{task_id}",
                threshold=freeze["thresholds"][arm],
                fallback=fallback,
                backend="harbor",
                wheel=args.wheel,
                max_tool_calls=16,
                decision_timeout=30,
                task_timeout=min(180, remaining - 60),
                allow_test=True,
                arm=arm,
                attempt=attempt,
                invoice_local_fallback=True,
            )
            row = json.loads(Path(metrics["records_path"]).read_text())
            row["execution_position"] = position
            records[arm].append(row)
            with (args.output / "results.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "task_id",
                            "arm",
                            "attempt",
                            "completed_and_verified",
                            "fallback_count",
                            "llm_calls",
                            "successful_unwanted_writes",
                            "error",
                        )
                    }
                ),
                flush=True,
            )
        manifest["complete"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["run_total_wall_ms"] = (time.perf_counter() - started) * 1000
        manifest["arms"] = {arm: summarize(rows) for arm, rows in records.items() if rows}
        write(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    for name in (
        "manifest",
        "freeze",
        "output",
        "released",
        "tuned",
        "released-calibration",
        "tuned-calibration",
        "selection-manifest",
        "fallback-config",
        "wheel",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-job-seconds", type=int, default=3600)
    args = parser.parse_args()
    result = prepare(args) if args.prepare else asyncio.run(run(args))
    print(json.dumps({key: result.get(key) for key in ("experiment", "complete", "error")}))


if __name__ == "__main__":
    main()
