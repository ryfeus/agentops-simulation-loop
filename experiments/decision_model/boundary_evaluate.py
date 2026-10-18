"""Frozen paired boundary repairs on exposed cases; no fitting or quality retries.

The retained episode and authorization dependency are loaded without modifying
their source or global imports. Current and retained policies share the released
checkpoint, calibration, native billing service, and bounded sequential scheduler.
"""

from __future__ import annotations

import argparse
import asyncio
import builtins
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
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
OLD_RUNTIME_SHA256 = "189eda8f73ac386874089b1d39168fd5c15805abc0aa8516bd0a488b8597373a"
OLD_AUTHORIZATION_SHA256 = "54a6c59e90b93c41bc6464cc4ca22016f4b441751f53d47c1dced6bdccd57beb"
RETAINED_COHORT_SHA256 = "232f8b6d322adf5a060b2e07dbe79916937d11eb5ce3608dc2da063f1b0b0460"
AUTHORIZATION_MODULE = "experiments.decision_model.authorization"
RETAINED_RUN = Path(
    ".artifacts/decision-model-next-large/download/.artifacts/decision-model-next/run"
)


def contracts(threshold: float, old_source: Path, old_authorization: Path) -> dict[str, Any]:
    current = Path(__file__).with_name("harbor.py")
    return {
        arm: {
            "predictor": arm.split("_")[0],
            "runtime": arm.split("_")[1],
            "runtime_sha256": digest(old_source if arm.endswith("old") else current),
            "threshold": threshold if arm.startswith("released") else 0.0,
            "authorization_sha256": digest(
                old_authorization if arm.endswith("old") else current.with_name("authorization.py")
            ),
            "fallback_reason_defaults": True,
            "low_confidence_authority_clarified": True,
            "authorization_boundary_repaired": arm.endswith("new"),
            "authorization_import_binding": "private_retained_dependency"
            if arm.endswith("old")
            else "current_module",
        }
        for arm in ARMS
    }


def load_old_runtime(path: Path, authorization_path: Path) -> Any:
    if digest(path) != OLD_RUNTIME_SHA256:
        raise ValueError("Old runtime must be the exact retained fresh-fallback source")
    if digest(authorization_path) != OLD_AUTHORIZATION_SHA256:
        raise ValueError("Old authorization must be the exact retained source")
    auth_spec = importlib.util.spec_from_file_location(
        "retained_boundary_authorization", authorization_path
    )
    auth = importlib.util.module_from_spec(auth_spec)
    assert auth_spec.loader is not None
    auth_spec.loader.exec_module(auth)
    spec = importlib.util.spec_from_file_location("retained_boundary_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    # Function-local imports use this module-private builtins dictionary. Redirect
    # only the known dependency, leaving sys.modules and current runtime untouched.
    # This also remains bound across awaits and exceptions, with no restoration gap.
    def frozen_import(
        name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0
    ) -> ModuleType:
        if name == AUTHORIZATION_MODULE and level == 0:
            if not fromlist:
                raise ImportError("Retained authorization imports must use explicit from-import")
            return auth
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": frozen_import}
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    module.__dict__["_retained_authorization"] = auth
    return module


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.freeze.exists():
        raise FileExistsError(args.freeze)
    if digest(args.old_runtime) != OLD_RUNTIME_SHA256:
        raise ValueError("Old runtime differs from retained evidence")
    if digest(args.old_authorization) != OLD_AUTHORIZATION_SHA256:
        raise ValueError("Old authorization differs from retained evidence")
    cohort = json.loads(args.manifest.read_text())
    if digest(args.manifest) != RETAINED_COHORT_SHA256:
        raise ValueError("Cohort must be the exact retained previous evaluation cohort")
    if args.cohort_previously_exposed is not True:
        raise ValueError("Must explicitly acknowledge this cohort was previously exposed")
    tasks = load_tasks(args.manifest, "test", 12)
    roots = Counter(task["composition_root"] for task in tasks)
    if len(tasks) != 12 or len(roots) != 6 or set(roots.values()) != {2}:
        raise ValueError("Predeclared design needs12 tasks in6 exposed families")
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
        cohort_previously_exposed=True,
        selected_checkpoint_role="released_baseline",
    )
    freeze["source_sha256"].update(
        {
            str(p.resolve()): digest(p)
            for p in (
                Path(__file__),
                args.old_runtime,
                args.old_authorization,
                Path(__file__).with_name("boundary_report.py"),
            )
        }
    )
    freeze.update(
        experiment="exposed-authorization-boundary-repair-v1",
        arm_contracts=contracts(threshold, args.old_runtime, args.old_authorization),
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
        old_authorization_path=str(args.old_authorization.resolve()),
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
    if freeze.get("experiment") != "exposed-authorization-boundary-repair-v1":
        raise ValueError("Wrong experiment freeze")
    if (
        args.cohort_previously_exposed is not True
        or freeze.get("cohort_previously_exposed") is not True
    ):
        raise ValueError("Exposed cohort acknowledgment must match freeze")
    for name in (
        "checkpoint",
        "calibration",
        "fallback_config",
        "wheel",
        "old_runtime",
        "old_authorization",
    ):
        if str(getattr(args, name).resolve()) != freeze[name + "_path"]:
            raise ValueError("Command differs from frozen artifact: " + name)
    if checkpoint_identity(args.checkpoint) != freeze["checkpoints"]["released"]:
        raise ValueError("Checkpoint changed")
    tasks = load_tasks(args.manifest, "test", 12)
    if [t["task_id"] for t in tasks] != freeze["task_ids"]:
        raise ValueError("Task identities changed")
    threshold = float(json.loads(args.calibration.read_text())["threshold"])
    if contracts(threshold, args.old_runtime, args.old_authorization) != freeze["arm_contracts"]:
        raise ValueError("Frozen arm contracts changed")
    for task in tasks:
        validate_task(task)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "freeze.json", freeze)
    (args.output / "cohort-manifest.json").write_bytes(args.manifest.read_bytes())
    for path in [
        *final_source_paths(),
        Path(__file__),
        Path(__file__).with_name("boundary_report.py"),
    ]:
        (args.output / ("source-" + path.name)).write_bytes(path.read_bytes())
    (args.output / "source-harbor-old.py").write_bytes(args.old_runtime.read_bytes())
    (args.output / "source-authorization-old.py").write_bytes(args.old_authorization.read_bytes())
    os.environ.update(
        AWS_REGION="us-west-2",
        AWS_DEFAULT_REGION="us-west-2",
        BILLING_MODEL_TEMPERATURE="0",
        BILLING_MODEL_MAX_TOKENS="512",
        BILLING_MODEL_TOTAL_ATTEMPTS="1",
    )
    manifest = {
        "schema_version": "exposed-paired-boundary-v1",
        "complete": False,
        "error": None,
        "task_ids": freeze["task_ids"],
        "repeats": 2,
        "planned_trials": 96,
        "max_job_seconds": 3600,
        "device": "mps",
        "backend": "native-harbor",
        "cohort_previously_exposed": True,
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
            "Previously exposed authored cases informed these repairs; not fresh holdout evidence.",
            "Two variants and two attempts are correlated within each of six families.",
            "Authorization, no-write completion, and tool descriptions change together.",
            "No model tuning or calibration fits in this run; no quality retries.",
        ],
    }
    write(args.output / "manifest.json", manifest)
    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    try:
        old = load_old_runtime(args.old_runtime, args.old_authorization)
        assert old.PROMPT_SUFFIX == harbor.PROMPT_SUFFIX
        loaded = time.perf_counter()
        model = DecisionModel(args.checkpoint, device="mps", calibration=args.calibration)
        manifest["model_load_ms"] = (time.perf_counter() - loaded) * 1000
        predict = model_predictor(model)
        fallbacks = {
            "old": old.llm_fallback(args.fallback_config, invoice_local_fallback=True),
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
    parser.add_argument("--cohort-previously-exposed", action="store_true")
    parser.add_argument(
        "--old-authorization",
        type=Path,
        default=Path(
            ".artifacts/decision-model-fresh-fallback/final-harbor/source-authorization.py"
        ),
    )
    parser.add_argument(
        "--old-runtime",
        type=Path,
        default=Path(".artifacts/decision-model-fresh-fallback/final-harbor/source-harbor.py"),
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
