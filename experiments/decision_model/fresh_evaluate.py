"""Frozen old/new fallback comparison on fresh families; no fitting or quality retries.

The old episode/fallback is loaded from the retained invoice-local run, not
reimplemented as an approximation. All arms use the same environment, guards,
checkpoint and calibration. Only the episode prompt and fallback tool schema vary.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.decision_model.next_evaluate import (
    build_final_freeze,
    check_final_freeze,
    digest,
    evaluate_tasks,
    final_source_paths,
    load_tasks,
    model_predictor,
    summarize,
    validate_task,
    write,
)

ARMS = ("released_old", "released_new", "llm_old", "llm_new")
OLD_RUNTIME_SHA256 = "89feffa95775b5189b8200658d2821bec534c7ed422bc4905e94397a39641f37"
RETAINED_RUN = Path(
    ".artifacts/decision-model-next-large/download/.artifacts/decision-model-next/run"
)


def contracts(threshold: float, old_source: Path) -> dict[str, Any]:
    current = Path(__file__).with_name("harbor.py")
    return {
        arm: {
            "predictor": arm.split("_")[0],
            "runtime": arm.split("_")[1],
            "runtime_sha256": digest(old_source if arm.endswith("old") else current),
            "threshold": threshold if arm.startswith("released") else 0.0,
            "fallback_reason_defaults": arm.endswith("new"),
            "low_confidence_authority_clarified": arm.endswith("new"),
        }
        for arm in ARMS
    }


def load_old_runtime(path: Path) -> Any:
    if digest(path) != OLD_RUNTIME_SHA256:
        raise ValueError("Old runtime must be the exact retained invoice-local-v2 source")
    spec = importlib.util.spec_from_file_location("retained_invoice_local_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.freeze.exists():
        raise FileExistsError(args.freeze)
    if digest(args.old_runtime) != OLD_RUNTIME_SHA256:
        raise ValueError("Old runtime differs from retained evidence")
    cohort = json.loads(args.manifest.read_text())
    novelty = cohort.get("novelty_audit", {})
    if (
        cohort.get("cohort_previously_exposed") is not False
        or cohort.get("pre_freeze_policy_rollouts") != 0
        or cohort.get("pre_freeze_model_inferences") != 0
        or len(cohort.get("tasks", {})) != 12
        or novelty.get("unique_new_structures") != 6
        or any(
            novelty.get(k) != 0
            for k in (
                "root_collisions",
                "structural_collisions",
                "normalized_instruction_collisions",
            )
        )
    ):
        raise ValueError("Fresh-family provenance audit missing or invalid")
    tasks = load_tasks(args.manifest, "test", 12)
    if len(tasks) != 12 or len({t["composition_root"] for t in tasks}) != 6:
        raise ValueError("Predeclared design needs12 tasks in6 fresh families")
    for task in tasks:
        validate_task(task)
    calibration = json.loads(args.calibration.read_text())
    threshold = float(calibration["threshold"])
    if not 0 <= threshold <= 1.000001:
        raise ValueError("Invalid frozen calibration")
    freeze = build_final_freeze(
        args.manifest,
        released=args.checkpoint,
        tuned=args.checkpoint,
        released_calibration=args.calibration,
        tuned_calibration=args.calibration,
        fallback_config=args.fallback_config,
        wheel=args.wheel,
        selection_manifest=args.selection_manifest,
        invoice_local_fallback=True,
        cohort_previously_exposed=False,
        selected_checkpoint_role="released_baseline",
    )
    freeze["source_sha256"].update(
        {
            str(p.resolve()): digest(p)
            for p in (
                Path(__file__),
                args.old_runtime,
                Path(__file__).with_name("fresh_report.py"),
                Path(__file__).with_name("fresh_holdout.py"),
            )
        }
    )
    freeze.update(
        experiment="fresh-invoice-local-fallback-repair-v1",
        arm_contracts=contracts(threshold, args.old_runtime),
        task_ids=[t["task_id"] for t in tasks],
        repeats=2,
        planned_trials=96,
        max_job_seconds=3600,
        device="mps",
        training_steps=0,
        calibration_fits=0,
        cohort_id=cohort.get("cohort_id"),
        checkpoint_path=str(args.checkpoint.resolve()),
        calibration_path=str(args.calibration.resolve()),
        fallback_config_path=str(args.fallback_config.resolve()),
        wheel_path=str(args.wheel.resolve()),
        old_runtime_path=str(args.old_runtime.resolve()),
    )
    args.freeze.parent.mkdir(parents=True, exist_ok=True)
    with args.freeze.open("x") as handle:
        json.dump(freeze, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return freeze


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.decision_model import harbor
    from experiments.decision_model.model import DecisionModel, checkpoint_identity

    started = time.perf_counter()
    freeze = json.loads(args.freeze.read_text())
    check_final_freeze(freeze, args.manifest)
    if freeze.get("experiment") != "fresh-invoice-local-fallback-repair-v1":
        raise ValueError("Wrong experiment freeze")
    for name in ("checkpoint", "calibration", "fallback_config", "wheel", "old_runtime"):
        if str(getattr(args, name).resolve()) != freeze[name + "_path"]:
            raise ValueError("Command differs from frozen artifact: " + name)
    if checkpoint_identity(args.checkpoint) != freeze["checkpoints"]["released"]:
        raise ValueError("Checkpoint changed")
    tasks = load_tasks(args.manifest, "test", 12)
    if [t["task_id"] for t in tasks] != freeze["task_ids"]:
        raise ValueError("Task identities changed")
    threshold = float(json.loads(args.calibration.read_text())["threshold"])
    if contracts(threshold, args.old_runtime) != freeze["arm_contracts"]:
        raise ValueError("Frozen arm contracts changed")
    for task in tasks:
        validate_task(task)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "freeze.json", freeze)
    (args.output / "cohort-manifest.json").write_bytes(args.manifest.read_bytes())
    for path in [
        *final_source_paths(),
        Path(__file__),
        Path(__file__).with_name("fresh_report.py"),
        Path(__file__).with_name("fresh_holdout.py"),
    ]:
        (args.output / ("source-" + path.name)).write_bytes(path.read_bytes())
    (args.output / "source-harbor-old.py").write_bytes(args.old_runtime.read_bytes())
    os.environ.update(
        AWS_REGION="us-west-2",
        AWS_DEFAULT_REGION="us-west-2",
        BILLING_MODEL_TEMPERATURE="0",
        BILLING_MODEL_MAX_TOKENS="512",
        BILLING_MODEL_TOTAL_ATTEMPTS="1",
    )
    manifest = {
        "schema_version": "fresh-paired-fallback-v1",
        "complete": False,
        "error": None,
        "task_ids": freeze["task_ids"],
        "repeats": 2,
        "planned_trials": 96,
        "max_job_seconds": 3600,
        "device": "mps",
        "backend": "native-harbor",
        "cohort_previously_exposed": False,
        "invoice_local_fallback": True,
        "arm_contracts": freeze["arm_contracts"],
        "training_steps": 0,
        "calibration_fits": 0,
        "max_tool_calls": 16,
        "max_llm_calls": 16,
        "fallback_timeout": 120,
        "task_timeout": 180,
        "freeze_sha256": digest(args.freeze),
        "cohort_manifest_sha256": digest(args.manifest),
        "execution_order": [],
        "limitations": [
            "Authored billing families, not an external benchmark.",
            "Two variants and two attempts are correlated within each of six families.",
            "Both fallback fixes change together; no individual causal attribution.",
            "No tuning or calibration on this cohort and no quality retries.",
        ],
    }
    write(args.output / "manifest.json", manifest)
    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    try:
        old = load_old_runtime(args.old_runtime)
        assert old.PROMPT_SUFFIX == harbor.PROMPT_SUFFIX
        loaded = time.perf_counter()
        model = DecisionModel(args.checkpoint, device="mps", calibration=args.calibration)
        manifest["model_load_ms"] = (time.perf_counter() - loaded) * 1000
        predict = model_predictor(model)
        fallbacks = {
            "old": old.llm_fallback(args.fallback_config),
            "new": harbor.llm_fallback(args.fallback_config, invoice_local_fallback=True),
        }
        runners = {"old": old.run_episode, "new": harbor.run_episode}
        for attempt in range(2):
            for index, task in enumerate(tasks):
                shift = (index + attempt) % len(ARMS)
                for arm in ARMS[shift:] + ARMS[:shift]:
                    remaining = 3600 - (time.perf_counter() - started)
                    if remaining < 65:
                        raise TimeoutError("Insufficient frozen budget for another trial")
                    check_final_freeze(freeze, args.manifest)
                    position = len(manifest["execution_order"])
                    manifest["execution_order"].append(
                        dict(task_id=task["task_id"], arm=arm, attempt=attempt, position=position)
                    )
                    write(args.output / "manifest.json", manifest)
                    runtime = arm.split("_")[1]
                    metrics = await evaluate_tasks(
                        predict
                        if arm.startswith("released")
                        else lambda state: {"action": "defer", "confidence": 1.0},
                        [task],
                        args.output
                        / "trials"
                        / f"{position:03d}-{arm}-{attempt}-{task['task_id']}",
                        threshold=freeze["arm_contracts"][arm]["threshold"],
                        fallback=fallbacks[runtime],
                        episode_runner=runners[runtime],
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
                                k: row.get(k)
                                for k in (
                                    "task_id",
                                    "arm",
                                    "attempt",
                                    "completed_and_verified",
                                    "llm_calls",
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
        manifest["arms"] = {a: summarize(rows) for a, rows in records.items() if rows}
        write(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--old-runtime",
        type=Path,
        default=Path(".artifacts/decision-model-invoice-local/final-harbor-v2/source-harbor.py"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(".artifacts/decision-model/source-checkpoint")
    )
    parser.add_argument(
        "--calibration", type=Path, default=RETAINED_RUN / "calibration-released/calibration.json"
    )
    parser.add_argument(
        "--selection-manifest", type=Path, default=RETAINED_RUN / "selection-manifest.json"
    )
    parser.add_argument(
        "--fallback-config",
        type=Path,
        default=Path(".artifacts/decision-model/fallback-config.json"),
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=Path(
            ".artifacts/decision-model/wheel/agentops_simulation_loop_aws-0.1.0-py3-none-any.whl"
        ),
    )
    args = parser.parse_args()
    result = prepare(args) if args.prepare else asyncio.run(run(args))
    print(
        json.dumps(
            {k: result.get(k) for k in ("experiment", "complete", "planned_trials", "error")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
