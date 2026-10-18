"""Repair contradictory legacy risk flags on calibration only, preserving raw artifacts.

Gold actions come from authored tasks and verified BillingService demonstrations.
This does not alter gold, predictions, model parameters, runtime, or test data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments.decision_model import data, model, next_data

ROOT = Path(__file__).resolve().parents[2] / ".artifacts/decision-model-focused"
WORKER = ROOT / "download/.artifacts/decision-model-focused/run"


def reconcile(predictions: list[dict], gold: list[dict], tasks: dict, demonstrations: dict):
    """Remove only a gold action's contradictory legacy flag, with strict evidence."""
    if not gold or any(row.get("split") != "calibration" for row in gold + predictions):
        raise ValueError("Only calibration examples are permitted")
    lookup = {row["example_id"]: row for row in gold}
    if len(lookup) != len(gold) or len({p["example_id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate example identity")
    if set(lookup) != {p["example_id"] for p in predictions}:
        raise ValueError("Prediction and gold coverage differ")
    corrected, changes = [], []
    for prediction in predictions:
        row = lookup[prediction["example_id"]]
        metadata = tasks[row["task_id"]]
        demonstration = demonstrations[row["task_id"]]
        if metadata["split"] != "calibration" or demonstration["split"] != "calibration":
            raise ValueError("Task or demonstration split mismatch")
        if not (
            demonstration.get("complete") is True
            and demonstration.get("reward") == 1
            and demonstration.get("verifier_components", {}).get("reward") == 1
            and demonstration["scenario_sha256"] == metadata["scenario_sha256"]
        ):
            raise ValueError("Unverified independent demonstration")
        if (
            row["state"] != data.serialize_state(row["input_state"])
            or next_data.label_state(row["input_state"], metadata) != row["label_action"]
        ):
            raise ValueError("Gold disagrees with observed state and authored intent")
        for key in ("group_id", "label_action", "prohibited_actions"):
            if prediction[key] != row[key]:
                raise ValueError("Prediction differs from frozen gold metadata: " + key)
        probabilities = prediction.get("probabilities") or {}
        if (
            prediction.get("error")
            or set(probabilities) != set(data.ACTIONS)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5)
            or prediction["action"] != max(probabilities, key=probabilities.get)
        ):
            raise ValueError("Invalid or incomplete model prediction")
        result = copy.deepcopy(prediction)
        action = row["label_action"]
        if action in row["prohibited_actions"]:
            state, target = row["input_state"], row["input_state"]["target_invoice"]
            operation = {"refund": "refund_invoice", "escalate": "escalate_dispute"}.get(action)
            if not operation or metadata["expected_operations"].get(target) != operation:
                raise ValueError("Contradiction lacks an independently required operation")
            evidence = [
                o
                for o in demonstration["tool_observations"]
                if o["name"] == operation
                and o["arguments"].get("invoice_id") == target
                and o["result"].get("ok") is True
            ]
            if not evidence:
                raise ValueError("Contradiction lacks successful actual tool evidence")
            # Diagnose this exact known legacy bug; arbitrary contradictions fail closed.
            if not (
                action == "escalate"
                and data._excluded(state["request"], target)
                and f"do not refund {target}" in state["request"].lower()
                and state["observation"]["escalation_history"]["records"] == []
            ):
                raise ValueError("Contradiction is not the diagnosed operation-scope defect")
            result["prohibited_actions"] = [a for a in row["prohibited_actions"] if a != action]
            changes.append(
                {
                    "example_id": row["example_id"],
                    "task_id": row["task_id"],
                    "gold_action": action,
                    "target_invoice": target,
                    "before": row["prohibited_actions"],
                    "after": result["prohibited_actions"],
                    "reason": (
                        "Legacy _excluded treats do-not-refund as prohibiting escalation; "
                        "authored review and actual tool evidence establish escalation."
                    ),
                }
            )
        corrected.append(result)
    return corrected, changes


def run(root: Path = ROOT, worker: Path = WORKER, output: Path | None = None):
    output = output or root / "calibration-corrected"
    if output.exists():
        raise FileExistsError(output)
    dataset_manifest_path = root / "data/dataset-manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    paths = [
        dataset_manifest_path,
        root / "data/calibration.jsonl",
        root / "data/demonstrations.jsonl",
        root / "corpus/cohort-manifest.json",
        worker / "selection-manifest.json",
        worker / "training/training.json",
        Path(__file__),
    ]
    paths.extend(
        Path(__file__).with_name(name)
        for name in ("model.py", "data.py", "history.py", "next_data.py", "authorization.py")
    )
    for name in ("calibration.jsonl", "demonstrations.jsonl"):
        if next_data.digest(root / "data" / name) != dataset_manifest["file_sha256"][name]:
            raise ValueError("Frozen dataset changed")
    if (
        next_data.digest(root / "corpus/cohort-manifest.json")
        != dataset_manifest["corpus_manifest_sha256"]
    ):
        raise ValueError("Frozen corpus changed")
    gold = model.read_rows(root / "data/calibration.jsonl")
    tasks = json.loads((root / "corpus/cohort-manifest.json").read_text())["tasks"]
    demonstrations = {r["task_id"]: r for r in model.read_rows(root / "data/demonstrations.jsonl")}
    selection = json.loads((worker / "selection-manifest.json").read_text())
    if (
        selection.get("test_accessed") is not False
        or selection.get("selected_checkpoint_role") != "trained"
    ):
        raise ValueError("Expected pre-test trained selection")
    for relative, expected in selection["evidence_sha256"].items():
        if next_data.digest(worker / relative) != expected:
            raise ValueError("Original worker selection evidence changed")
    fitted, audited = {}, {}
    for arm in ("released", "tuned"):
        original_path = worker / f"calibration-{arm}/calibration.json"
        prediction_path = worker / f"calibration-{arm}/predictions.jsonl"
        paths.extend((original_path, prediction_path))
        original = json.loads(original_path.read_text())
        predictions = model.read_rows(prediction_path)
        corrected, changes = reconcile(predictions, gold, tasks, demonstrations)
        if any(p["checkpoint"] != original["checkpoint"] for p in predictions):
            raise ValueError("Checkpoint identity changed")
        calibration = model.fit_calibration(corrected, original["checkpoint"])
        fitted[arm] = (calibration, corrected)
        audited[arm] = {
            "corrections": changes,
            "corrected_count": len(changes),
            "before": original,
            "after": calibration,
            "predictions_and_gold_unchanged": True,
        }
    if audited["released"]["corrections"] != audited["tuned"]["corrections"]:
        raise ValueError("Correction must be arm-independent")
    output.mkdir(parents=True)
    for arm, (calibration, corrected) in fitted.items():
        model.write_json(output / arm / "calibration.json", calibration)
        model.write_jsonl(output / arm / "corrected-predictions.jsonl", corrected)
    audit = {
        "scope": "Calibration-only correction before test access; no new inference or training",
        "arms": audited,
        "input_sha256": {str(p.resolve()): next_data.digest(p) for p in paths},
        "original_optimizer_steps": selection["optimizer_steps"],
        "original_selected_epoch": selection["selected_epoch"],
        "test_accessed": False,
    }
    model.write_json(output / "correction-audit.json", audit)
    corrected_selection = {
        **selection,
        "original_selection_sha256": next_data.digest(worker / "selection-manifest.json"),
        "original_selection_path": str((worker / "selection-manifest.json").resolve()),
        "original_evidence_sha256": selection["evidence_sha256"],
        "evidence_sha256": {
            str(p.relative_to(output)): next_data.digest(p)
            for p in (
                output / "released/calibration.json",
                output / "tuned/calibration.json",
                output / "correction-audit.json",
            )
        },
        "calibration_correction": (
            "Remove only independently disproved legacy gold-action "
            "prohibitions; original weights/epoch and predictions retained"
        ),
        "checkpoint_identities": {arm: pair[0]["checkpoint"] for arm, pair in fitted.items()},
    }
    model.write_json(output / "selection-manifest.json", corrected_selection)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--worker", type=Path, default=WORKER)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = run(args.root, args.worker, args.output)
    print(
        json.dumps(
            {
                arm: {"corrected_count": a["corrected_count"], "calibration": a["after"]}
                for arm, a in audit["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
