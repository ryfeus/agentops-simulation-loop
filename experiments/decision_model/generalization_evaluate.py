"""Prospective released-decider/LLM comparison with one immutable repaired runtime.

48 interleaved native trials; no policy preflight, training, calibration or quality
retries. A cohort access claim is written before model construction so rerunning
the same cohort cannot silently present it as an untouched fresh evaluation.
"""

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
    final_source_paths,
    load_tasks,
    model_predictor,
    summarize,
    validate_task,
    write,
)

ARMS = ("released", "llm")
EXPERIMENT = "prospective-generalization-v1"
COHORT_ID = "billing-generalization-v1"
REFERENCE_FREEZE = Path(".artifacts/decision-model-boundary-repair/final-freeze.json")
REFERENCE_FREEZE_SHA256 = "6d4041ddb36aef945a65bee891b2bbff99379b7f9252d81cd34e60e909f2ce16"
RETAINED_RUN = Path(
    ".artifacts/decision-model-next-large/download/.artifacts/decision-model-next/run"
)


def reference(args: argparse.Namespace) -> dict[str, Any]:
    if digest(args.reference_freeze) != REFERENCE_FREEZE_SHA256:
        raise ValueError("Reference must be the exact completed boundary-repair freeze")
    baseline = json.loads(args.reference_freeze.read_text())
    for source in final_source_paths():
        expected = baseline["source_sha256"].get(str(source.resolve()))
        snapshot = args.reference_freeze.parent / "final-harbor" / ("source-" + source.name)
        if expected is None or digest(source) != expected or digest(snapshot) != expected:
            raise ValueError("Repaired runtime or retained snapshot changed: " + str(source))
    return baseline


def validate_fresh_cohort(manifest: Path) -> list[dict[str, Any]]:
    cohort = json.loads(manifest.read_text())
    novelty = cohort.get("novelty_audit", {})
    if (
        cohort.get("cohort_id") != COHORT_ID
        or cohort.get("cohort_previously_exposed") is not False
        or cohort.get("pre_freeze_policy_rollouts") != 0
        or cohort.get("pre_freeze_model_inferences") != 0
        or len(cohort.get("tasks", {})) != 12
        or novelty.get("unique_new_structures") != 6
        or novelty.get("source_task_counts", {}).get("fallback-fresh") != 12
        or any(
            novelty.get(key) != 0
            for key in (
                "root_collisions",
                "structural_collisions",
                "normalized_instruction_collisions",
            )
        )
    ):
        raise ValueError("Prospective cohort freshness provenance missing or invalid")
    tasks = load_tasks(manifest, "test", 12)
    roots = Counter(task["composition_root"] for task in tasks)
    if len(tasks) != 12 or len(roots) != 6 or set(roots.values()) != {2}:
        raise ValueError("Design requires12 tasks,6 roots and2 variants per root")
    for task in tasks:
        validate_task(task)
    return tasks


def cohort_sources(manifest: Path) -> list[Path]:
    sources = json.loads(manifest.read_text()).get("source_sha256", {})
    if not sources:
        raise ValueError("Cohort source provenance is missing")
    for source, expected in sources.items():
        if digest(Path(source)) != expected:
            raise ValueError("Authored cohort source changed: " + source)
    return [Path(source).resolve() for source in sources]


def access_path(manifest: Path) -> Path:
    return manifest.resolve().parent / (".evaluation-access-" + digest(manifest) + ".json")


def cohort_evidence(manifest: Path, tasks: list[dict[str, Any]]) -> dict[str, str]:
    fixtures_path = manifest.parent / "fixture-validation.json"
    prototypes_path = manifest.parent / "prototype-specs.json"
    fixtures = json.loads(fixtures_path.read_text())
    prototypes = json.loads(prototypes_path.read_text())
    cohort = json.loads(manifest.read_text())
    prototype_hash = hashlib.sha256(
        json.dumps(prototypes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        len(fixtures) != 12
        or {row.get("task_id") for row in fixtures} != {t["task_id"] for t in tasks}
        or any(
            row.get("fixture_only") is not True
            or row.get("agent_or_policy_executed") is not False
            or row.get("expected_endpoint_accepted") is not True
            or row.get("missing_writes_rejected") is not True
            for row in fixtures
        )
        or len(prototypes) != 6
        or {row.get("id") for row in prototypes} != {t["composition_root"] for t in tasks}
        or prototype_hash != cohort.get("prototype_specs_sha256")
    ):
        raise ValueError(
            "Authored fixture/prototype evidence is inconsistent or contains policy exposure"
        )
    return {str(path.resolve()): digest(path) for path in (fixtures_path, prototypes_path)}


def contracts() -> dict[str, Any]:
    directory = Path(__file__).parent
    return {
        arm: {
            "predictor": arm,
            "runtime_sha256": digest(directory / "harbor.py"),
            "authorization_sha256": digest(directory / "authorization.py"),
            "threshold": 0.7 if arm == "released" else 0.0,
            "invoice_local_fallback": True,
            "fallback_reason_defaults": True,
            "authorization_boundary_repaired": True,
        }
        for arm in ARMS
    }


def additional_sources() -> list[Path]:
    return [
        Path(__file__),
        Path(__file__).with_name("generalization_report.py"),
        Path(__file__).with_name("generalization_holdout.py"),
        Path(__file__).with_name("next_report.py"),
    ]


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.freeze.exists():
        raise FileExistsError(args.freeze)
    baseline = reference(args)
    tasks = validate_fresh_cohort(args.manifest)
    evidence = cohort_evidence(args.manifest, tasks)
    if digest(args.manifest) == baseline["cohort_manifest_sha256"]:
        raise ValueError("Previous evaluation cohort cannot be labeled fresh")
    if access_path(args.manifest).exists():
        raise ValueError("Cohort evaluation access already claimed; no longer pristine")
    if float(json.loads(args.calibration.read_text())["threshold"]) != 0.7:
        raise ValueError("Fixed released routing threshold must remain0.7")
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
    if freeze["checkpoints"]["released"] != baseline["checkpoints"]["released"]:
        raise ValueError("Released checkpoint must remain the prior frozen checkpoint")
    if freeze["artifact_sha256"] != baseline["artifact_sha256"]:
        raise ValueError("Calibration, checkpoint, fallback configuration or wheel changed")
    freeze["source_sha256"].update(
        {
            str(path.resolve()): digest(path)
            for path in [*additional_sources(), *cohort_sources(args.manifest)]
        }
    )
    names = [Path(path).name for path in freeze["source_sha256"]]
    if len(names) != len(set(names)):
        raise ValueError("Frozen source snapshots require unique basenames")
    freeze["artifact_sha256"][str(args.reference_freeze.resolve())] = digest(args.reference_freeze)
    freeze.update(
        experiment=EXPERIMENT,
        arm_contracts=contracts(),
        task_ids=[t["task_id"] for t in tasks],
        repeats=2,
        planned_trials=48,
        max_job_seconds=3600,
        device="mps",
        training_steps=0,
        calibration_fits=0,
        cohort_id=json.loads(args.manifest.read_text()).get("cohort_id"),
        cohort_evidence_sha256=evidence,
        baseline_reference_freeze_path=str(args.reference_freeze.resolve()),
        baseline_reference_freeze_sha256=digest(args.reference_freeze),
        evaluation_access_path=str(access_path(args.manifest)),
        checkpoint_path=str(args.checkpoint.resolve()),
        calibration_path=str(args.calibration.resolve()),
        fallback_config_path=str(args.fallback_config.resolve()),
        wheel_path=str(args.wheel.resolve()),
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
    reference(args)
    if (
        freeze.get("experiment") != EXPERIMENT
        or freeze.get("cohort_previously_exposed") is not False
    ):
        raise ValueError("Wrong experiment or nonprospective freeze")
    if str(args.reference_freeze.resolve()) != freeze["baseline_reference_freeze_path"]:
        raise ValueError("Command reference differs from freeze")
    for name in ("checkpoint", "calibration", "fallback_config", "wheel"):
        if str(getattr(args, name).resolve()) != freeze[name + "_path"]:
            raise ValueError("Command differs from frozen artifact: " + name)
    if checkpoint_identity(args.checkpoint) != freeze["checkpoints"]["released"]:
        raise ValueError("Checkpoint changed")
    tasks = validate_fresh_cohort(args.manifest)
    if cohort_evidence(args.manifest, tasks) != freeze.get("cohort_evidence_sha256"):
        raise ValueError("Frozen authored cohort evidence changed")
    if [task["task_id"] for task in tasks] != freeze["task_ids"] or contracts() != freeze[
        "arm_contracts"
    ]:
        raise ValueError("Frozen task identities or arm contracts changed")
    claim = access_path(args.manifest)
    if str(claim) != freeze["evaluation_access_path"] or claim.exists():
        raise ValueError("Cohort evaluation access already claimed or path changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "freeze.json", freeze)
    (args.output / "cohort-manifest.json").write_bytes(args.manifest.read_bytes())
    for source in freeze["cohort_evidence_sha256"]:
        path = Path(source)
        (args.output / path.name).write_bytes(path.read_bytes())
    for path in map(Path, freeze["source_sha256"]):
        (args.output / ("source-" + path.name)).write_bytes(path.read_bytes())
    os.environ.update(
        AWS_REGION="us-west-2",
        AWS_DEFAULT_REGION="us-west-2",
        BILLING_MODEL_TEMPERATURE="0",
        BILLING_MODEL_MAX_TOKENS="512",
        BILLING_MODEL_TOTAL_ATTEMPTS="1",
    )
    manifest = {
        "schema_version": EXPERIMENT,
        "complete": False,
        "error": None,
        "task_ids": freeze["task_ids"],
        "repeats": 2,
        "planned_trials": 48,
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
        "evaluation_access_path": str(claim),
        "execution_order": [],
        "limitations": [
            "Fresh authored compositions share known billing primitives; "
            "not an external benchmark.",
            "Two variants and two attempts are correlated within each of six families.",
            "One repaired runtime is shared. Comparison measures routing, not learned improvement.",
            "Fixed prior calibration is not refitted or validated by the nominal threshold alone.",
            "No outcome-based retries or selection; every planned completed trial is retained.",
        ],
    }
    write(args.output / "manifest.json", manifest)
    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    try:
        # Exclusive creation precedes any model/fallback construction. Failed
        # starts retain the claim conservatively; they cannot be hidden by retries.
        with claim.open("x") as handle:
            json.dump(
                {
                    "schema_version": "prospective-cohort-access-v1",
                    "cohort_manifest_sha256": digest(args.manifest),
                    "freeze_sha256": digest(args.freeze),
                    "output": str(args.output.resolve()),
                    "claimed_before_model_initialization": True,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
        (args.output / "evaluation-access.json").write_bytes(claim.read_bytes())
        manifest["evaluation_access_sha256"] = digest(claim)
        write(args.output / "manifest.json", manifest)
        loaded = time.perf_counter()
        model = DecisionModel(args.checkpoint, device="mps", calibration=args.calibration)
        manifest["model_load_ms"] = (time.perf_counter() - loaded) * 1000
        predict = model_predictor(model)
        loaded = time.perf_counter()
        fallback = harbor.llm_fallback(args.fallback_config, invoice_local_fallback=True)
        manifest["fallback_load_ms"] = (time.perf_counter() - loaded) * 1000
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
                    metrics = await evaluate_tasks(
                        predict
                        if arm == "released"
                        else lambda state: {"action": "defer", "confidence": 1.0},
                        [task],
                        args.output
                        / "trials"
                        / f"{position:03d}-{arm}-{attempt}-{task['task_id']}",
                        threshold=freeze["arm_contracts"][arm]["threshold"],
                        fallback=fallback,
                        episode_runner=harbor.run_episode,
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
        manifest["arms"] = {arm: summarize(rows) for arm, rows in records.items() if rows}
        write(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    for name in ("manifest", "freeze", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-freeze", type=Path, default=REFERENCE_FREEZE)
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
            {key: result.get(key) for key in ("experiment", "complete", "planned_trials", "error")}
        )
    )


if __name__ == "__main__":
    main()
