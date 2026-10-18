"""Bind the passing regression audit and remaining execution budget before fresh access."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from experiments.decision_model.next_evaluate import digest, final_source_paths

ROOT = Path(".artifacts/decision-model-generalization-v4")
REPORT = Path("reports/decision-model-generalization-v4")


def main():
    run = ROOT / "regression-paid"
    audit_path = REPORT / "regression-audit.json"
    audit = json.loads(audit_path.read_text())
    spec = importlib.util.spec_from_file_location("v4_gate_audit", REPORT / "audit.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    current_audit = auditor.audit_run(run, expected_tasks=36, exposed=True, expected_repeats=1)
    if current_audit != audit:
        raise ValueError("Regression evidence changed after saved audit")
    manifest = json.loads((run / "manifest.json").read_text())
    freeze = json.loads((run / "freeze.json").read_text())
    if not (
        audit["passed"]
        and audit["strict_gate_passed"]
        and manifest["complete"]
        and manifest["strict_regression_gate_passed"]
        and audit["retained_trials"] == 72
    ):
        raise ValueError("Full strict regression gate must pass before any fresh model access")
    evidence = [
        run / name
        for name in ("manifest.json", "freeze.json", "cohort-manifest.json", "results.jsonl")
    ]
    evidence += [audit_path, REPORT / "semantic-review.json", REPORT / "cohort-freeze-proof.json"]
    evidence += [Path(p) for p in audit["native_evidence_sha256"]]
    sources = {str(p.resolve()): digest(p) for p in final_source_paths()}
    if any(freeze["source_sha256"].get(p) != h for p, h in sources.items()):
        raise ValueError("Runtime drifted after regression")
    elapsed = manifest["run_total_wall_ms"] / 1000
    if not 0 <= elapsed < 3535:
        raise ValueError("Combined paid execution allowance exhausted")
    lock = {
        "regression_passed": True,
        "gate": (
            "all72 strict completions, native passes, all required writes, "
            "zero unwanted writes, complete evidence and owned cleanup"
        ),
        "source_sha256": sources,
        "regression_evidence_sha256": {str(p.resolve()): digest(p) for p in evidence},
        "regression_elapsed_seconds": elapsed,
        "max_combined_paid_seconds": 3600,
        "remaining_paid_seconds": int(3600 - elapsed),
        "training_steps": 0,
        "calibration_fits": 0,
    }
    with (REPORT / "runtime-lock.json").open("x") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"strict_gate_passed": True, "remaining_paid_seconds": int(3600 - elapsed)}))


if __name__ == "__main__":
    main()
