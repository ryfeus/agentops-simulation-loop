"""Remote Phase 11 training/selection and locked final-evaluation orchestration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rl.phase11.compare import compare
from rl.phase11.evaluate import evaluate_policy
from rl.phase11.selection import adapter_files, choose_checkpoint, dev_gate, digest, verify_lock
from rl.phase11.train import write_json

SEED = 20260925


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(payload: Path, role: str) -> Path:
    return payload / "manifests" / f"{role}-ids.json"


def _evaluate(
    payload: Path,
    root: Path,
    *,
    role: str,
    policy: str,
    destination: Path,
    adapter: Path | None = None,
) -> dict[str, Any]:
    passes = 2 if role in {"dev", "holdout"} else 1
    dataset_root = payload / ("anchors" if role == "anchors" else "dataset")
    metadata = (
        payload / "manifests" / ("anchor-metadata.json" if role == "anchors" else "metadata.json")
    )
    return evaluate_policy(
        dataset_root=dataset_root,
        metadata_path=metadata,
        task_ids_path=_ids(payload, role),
        adapter=adapter,
        passes=passes,
        seed=SEED,
        output=destination,
        run_id=root.name,
    )


def _copy_selected(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    adapter_files(destination)


def train(payload: Path, run: Path) -> dict[str, Any]:
    identity = _load(payload / "manifests" / "identity.json")
    if identity.get("worktree_clean") is not True:
        raise ValueError("canonical Phase 11 source revision must be clean")
    acceptance = _load(payload / "manifests" / "phase10-acceptance.json")
    if (
        acceptance.get("status") != "PASS"
        or acceptance.get("execution_suite_sha256") != identity["phase10"]["execution_suite_sha256"]
        or acceptance.get("oracle", {}).get("passes") != 200
    ):
        raise ValueError("Phase 10 verifier acceptance is not proven")
    for name in ("train", "dev", "holdout", "train-representatives"):
        ids = json.loads(_ids(payload, name).read_text())
        if len(ids) != {"train": 160, "dev": 20, "holdout": 20, "train-representatives": 32}[name]:
            raise ValueError(f"Phase 11 {name} count changed")
    baseline_reps = _evaluate(
        payload,
        run,
        role="train-representatives",
        policy="baseline",
        destination=run / "preflight" / "train-representatives",
    )
    baseline_dev = _evaluate(
        payload, run, role="dev", policy="baseline", destination=run / "dev" / "baseline"
    )
    mixed = sum(0 < row["passes"] < row["attempts"] for row in baseline_reps["tasks"].values())
    signal = {
        "mixed_groups": mixed,
        "all_zero_groups": sum(row["passes"] == 0 for row in baseline_reps["tasks"].values()),
        "all_one_groups": sum(row["passes"] == 4 for row in baseline_reps["tasks"].values()),
    }
    write_json(run / "preflight" / "grpo-signal.json", signal)
    if mixed == 0:
        summary = {
            "status": "NO_OBSERVED_GRPO_SIGNAL",
            "holdout_opened": False,
            "preflight": signal,
            "training_execution_valid": False,
        }
        write_json(run / "summary.json", summary)
        return summary
    command = [
        sys.executable,
        "-m",
        "rl.phase11.train",
        "--dataset-root",
        str(payload / "dataset"),
        "--metadata",
        str(payload / "manifests" / "metadata.json"),
        "--split",
        str(payload / "manifests" / "split.json"),
        "--output",
        str(run / "training"),
    ]
    subprocess.run(command, check=True)
    result = _load(run / "training" / "training-result.json")
    for step in (160, 320):
        _copy_selected(
            run / "training" / "adapters" / f"step-{step}", run / "adapters" / f"step-{step}"
        )
    dev = {
        160: _evaluate(
            payload,
            run,
            role="dev",
            policy="trained",
            destination=run / "dev" / "step-160",
            adapter=run / "adapters" / "step-160",
        ),
        320: _evaluate(
            payload,
            run,
            role="dev",
            policy="trained",
            destination=run / "dev" / "step-320",
            adapter=run / "adapters" / "step-320",
        ),
    }
    decision = choose_checkpoint(dev)
    write_json(run / "selection-decision.json", decision)
    selected_step = decision["selected_step"]
    _copy_selected(run / "adapters" / f"step-{selected_step}", run / "selected-adapter")
    selected_train = _evaluate(
        payload,
        run,
        role="train-representatives",
        policy="trained",
        destination=run / "train-representatives-trained",
        adapter=run / "selected-adapter",
    )
    train_comparison = compare(baseline_reps, selected_train)
    write_json(run / "train-representatives-comparison.json", train_comparison)
    dev_comparison = compare(baseline_dev, dev[selected_step])
    write_json(run / "dev" / "comparison.json", dev_comparison)
    lock = {
        "schema_version": "1",
        "experiment_id": identity["experiment_id"],
        "experiment_sha256": identity["experiment_sha256"],
        "source_git_revision": identity["source_git_revision"],
        "worktree_clean": identity["worktree_clean"],
        **identity["phase10"],
        "training_config_sha256": digest(run / "training" / "training-config.json"),
        "seed": SEED,
        "selected_step": selected_step,
        "adapter_files": adapter_files(run / "selected-adapter"),
        "adapter_manifest_sha256": digest(run / "training" / "adapter-manifest.json"),
        "dev_baseline_sha256": digest(run / "dev" / "baseline" / "rollouts.jsonl"),
        "dev_step_160_sha256": digest(run / "dev" / "step-160" / "rollouts.jsonl"),
        "dev_step_320_sha256": digest(run / "dev" / "step-320" / "rollouts.jsonl"),
        "selection_decision_sha256": digest(run / "selection-decision.json"),
        "holdout_opened": False,
    }
    write_json(run / "selection-lock.json", lock)
    verify_lock(
        lock,
        run,
        identity["experiment_sha256"],
        identity["phase10"]["split_manifest_sha256"],
        identity["source_git_revision"],
    )
    gate = dev_gate(baseline_dev, dev[selected_step])
    summary = {
        "status": "PASS",
        "training_execution_valid": True,
        "checkpoint_selection_valid": True,
        "global_step": result["global_step"],
        "selected_step": selected_step,
        "dev_gate_passed": gate,
        "holdout_opened": False,
        "preflight": signal,
        "train_rep_improvement_observed": train_comparison["micro_delta"] > 0,
        "dev_improvement_observed": gate,
        "human_review_required": True,
    }
    write_json(run / "summary.json", summary)
    return summary


def final(payload: Path, run: Path) -> dict[str, Any]:
    identity = _load(payload / "manifests" / "identity.json")
    lock = _load(payload / "selection-lock.json")
    verify_lock(
        lock,
        payload,
        identity["experiment_sha256"],
        identity["phase10"]["split_manifest_sha256"],
        identity["source_git_revision"],
    )
    baseline_dev = _load(payload / "dev" / "baseline" / "task-summary.json")
    selected_dev = _load(payload / "dev" / f"step-{lock['selected_step']}" / "task-summary.json")
    if not dev_gate(baseline_dev, selected_dev):
        raise ValueError("canonical dev gate failed; holdout must stay unopened")
    adapter = payload / "selected-adapter"
    anchors_base = _evaluate(
        payload, run, role="anchors", policy="baseline", destination=run / "anchors" / "baseline"
    )
    anchors_trained = _evaluate(
        payload,
        run,
        role="anchors",
        policy="trained",
        destination=run / "anchors" / "trained",
        adapter=adapter,
    )
    anchor_comparison = compare(anchors_base, anchors_trained)
    write_json(run / "anchors" / "comparison.json", anchor_comparison)
    holdout_base = _evaluate(
        payload, run, role="holdout", policy="baseline", destination=run / "holdout" / "baseline"
    )
    holdout_trained = _evaluate(
        payload,
        run,
        role="holdout",
        policy="trained",
        destination=run / "holdout" / "trained",
        adapter=adapter,
    )
    holdout_comparison = compare(holdout_base, holdout_trained)
    write_json(run / "holdout" / "comparison.json", holdout_comparison)
    trained_rows = [
        json.loads(line)
        for line in (run / "holdout" / "trained" / "rollouts.jsonl").read_text().splitlines()
        if line
    ]
    shortcut_candidates = [
        row["rollout_id"]
        for row in trained_rows
        if row["reward"] == 1.0
        and (
            int(row.get("tool_call_count", 0)) == 0
            or bool(row.get("verifier_diagnostics", {}).get("trajectory_error"))
        )
    ]
    summary = {
        "status": "PASS",
        "holdout_evaluation_valid": True,
        "training_execution_valid": True,
        "checkpoint_selection_valid": True,
        "holdout_micro_improvement_observed": holdout_comparison["micro_delta"] > 0,
        "holdout_prototype_macro_improvement_observed": holdout_comparison["prototype_macro_delta"]
        > 0,
        "holdout_prototypes_improved": holdout_comparison["prototypes_improved"],
        "holdout_prototypes_unchanged": holdout_comparison["prototypes_unchanged"],
        "holdout_prototypes_worsened": holdout_comparison["prototypes_worsened"],
        "anchor_regression_observed": anchor_comparison["micro_delta"] < 0,
        "obvious_shortcut_candidates": shortcut_candidates,
        "generalization_signal_supported": holdout_comparison["micro_delta"] > 0
        and holdout_comparison["prototype_macro_delta"] > 0
        and holdout_comparison["prototypes_improved"] >= 2
        and not shortcut_candidates,
        "human_review_required": True,
    }
    write_json(run / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "final"))
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    try:
        value = (
            train(args.payload, args.run) if args.mode == "train" else final(args.payload, args.run)
        )
    except Exception as exc:
        write_json(
            args.run / "summary.json",
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "mode": args.mode,
            },
        )
        raise
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
