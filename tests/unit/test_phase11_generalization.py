"""Phase 11 canonical protocol invariants that run without GPU or AWS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.phase8c.execution_suite import EXECUTION_PROFILE
from rl.phase11.compare import compare, summarize
from rl.phase11.corpus import subset_suite, validate_subset
from rl.phase11.experiment import validate_lineage
from rl.phase11.selection import adapter_files, choose_checkpoint, dev_gate, digest, verify_lock
from rl.phase11.train import validate_exposure


def test_frozen_phase10_lineage_and_representatives() -> None:
    value = validate_lineage(
        suite_sha256="4596d59cec6df9832584bd3529589f78170c8a84b67260aeed96d61b3cadaa67"
    )
    assert len(value["representatives"]) == 32
    assert all(task_id.endswith("-v00") for task_id in value["representatives"])
    with pytest.raises(ValueError, match="lineage"):
        validate_lineage(suite_sha256="0" * 64)


def test_subset_payload_contains_only_allowed_executable_tasks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "tasks").mkdir(parents=True)
    for task_id in ("train-v00", "dev-v00", "holdout-v00"):
        (source / "tasks" / task_id).mkdir()
        (source / "tasks" / task_id / "scenario.yaml").write_text(task_id)
    manifest = {
        "schema_version": "1",
        "task_count": 3,
        "task_ids": ["dev-v00", "holdout-v00", "train-v00"],
        "tasks": {key: {} for key in ("dev-v00", "holdout-v00", "train-v00")},
        "base_suite_sha256": "a" * 64,
        "execution_suite_sha256": "b" * 64,
        "execution_profile": EXECUTION_PROFILE,
    }
    (source / "execution-suite.json").write_text(json.dumps(manifest))
    destination = tmp_path / "train-payload"
    subset_suite(source, destination, {"train-v00", "dev-v00"})
    validate_subset(destination, {"train-v00", "dev-v00"})
    assert not (destination / "tasks" / "holdout-v00").exists()
    assert not list(destination.rglob("holdout-v00/scenario.yaml"))
    with pytest.raises(ValueError, match="differ"):
        validate_subset(destination, {"train-v00"})


def _training_evidence() -> tuple[list[dict], list[dict], set[str], dict[str, dict]]:
    task_ids = {f"family-{index:03d}-v00" for index in range(160)}
    groups, rollouts = [], []
    for exposure in range(2):
        for task_id in sorted(task_ids):
            index = len(groups)
            ids = [f"train:{task_id}:{exposure}:{attempt}" for attempt in range(4)]
            groups.append(
                {
                    "task_id": task_id,
                    "group_index": index,
                    "rollout_ids": ids,
                    "rewards": [0.0, 1.0, 0.0, 1.0],
                    "has_reward_variance": True,
                }
            )
            rollouts.extend(
                {"task_id": task_id, "rollout_id": rollout_id, "reward": 0.0} for rollout_id in ids
            )
    metadata = {
        task_id: {
            "family_id": task_id[:-4],
            "prototype_id": task_id[:-4],
            "archetype": "normal-refund",
            "difficulty": "easy",
        }
        for task_id in task_ids
    }
    return groups, rollouts, task_ids, metadata


def test_exact_training_exposure_and_leakage_guard() -> None:
    groups, rollouts, train_ids, metadata = _training_evidence()
    result = validate_exposure(groups, rollouts, train_ids, {"dev-v00"}, {"holdout-v00"}, metadata)
    assert result["group_count"] == 320
    assert all(item == {"groups": 2, "rollouts": 8} for item in result["tasks"].values())
    groups[0]["task_id"] = "dev-v00"
    with pytest.raises(ValueError, match="group"):
        validate_exposure(groups, rollouts, train_ids, {"dev-v00"}, {"holdout-v00"}, metadata)


@pytest.mark.parametrize(
    ("early", "late", "selected"),
    [((0.5, 0.8), (0.6, 0.1), 320), ((0.5, 0.8), (0.5, 0.9), 320), ((0.5, 0.8), (0.5, 0.8), 160)],
)
def test_checkpoint_selection_order(
    early: tuple[float, float], late: tuple[float, float], selected: int
) -> None:
    candidates = {
        step: {"prototype_macro_pass_rate": values[0], "micro_pass_rate": values[1]}
        for step, values in ((160, early), (320, late))
    }
    assert choose_checkpoint(candidates)["selected_step"] == selected


def test_dev_gate_requires_both_improvements() -> None:
    baseline = {"prototype_macro_pass_rate": 0.5, "micro_pass_rate": 0.5}
    assert not dev_gate(baseline, {"prototype_macro_pass_rate": 0.6, "micro_pass_rate": 0.5})
    assert dev_gate(baseline, {"prototype_macro_pass_rate": 0.6, "micro_pass_rate": 0.6})


def test_holdout_micro_and_prototype_macro_differ() -> None:
    metadata = {
        key: {"family_id": key, "prototype_id": proto, "archetype": "normal-refund"}
        for key, proto in (("a", "shared"), ("b", "shared"), ("c", "one"), ("d", "two"))
    }
    baseline = summarize(
        [
            {"task_id": key, "attempt": attempt, "reward": 1.0 if key in {"a", "b"} else 0.0}
            for key in metadata
            for attempt in range(4)
        ],
        metadata,
        4,
    )
    trained = summarize(
        [
            {"task_id": key, "attempt": attempt, "reward": 1.0 if key in {"c", "d"} else 0.0}
            for key in metadata
            for attempt in range(4)
        ],
        metadata,
        4,
    )
    result = compare(baseline, trained)
    assert result["micro_delta"] == 0
    assert result["prototype_macro_delta"] == pytest.approx(1 / 3)
    assert result["prototypes_improved"] == 2
    assert result["prototypes_worsened"] == 1


def test_selection_lock_detects_adapter_and_evidence_tampering(tmp_path: Path) -> None:
    adapter = tmp_path / "selected-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    evidence = [
        "training/training-config.json",
        "training/adapter-manifest.json",
        "selection-decision.json",
        "dev/baseline/rollouts.jsonl",
        "dev/step-160/rollouts.jsonl",
        "dev/step-320/rollouts.jsonl",
    ]
    for name in evidence:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"selected_step": 160}) if name == "selection-decision.json" else "evidence"
        )
    lock = {
        "holdout_opened": False,
        "experiment_sha256": "exp",
        "split_manifest_sha256": "split",
        "source_git_revision": "rev",
        "worktree_clean": True,
        "selected_step": 160,
        "adapter_files": adapter_files(adapter),
    }
    for key, name in (
        ("training_config_sha256", evidence[0]),
        ("adapter_manifest_sha256", evidence[1]),
        ("selection_decision_sha256", evidence[2]),
        ("dev_baseline_sha256", evidence[3]),
        ("dev_step_160_sha256", evidence[4]),
        ("dev_step_320_sha256", evidence[5]),
    ):
        lock[key] = digest(tmp_path / name)
    verify_lock(lock, tmp_path, "exp", "split", "rev")
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="adapter"):
        verify_lock(lock, tmp_path, "exp", "split", "rev")
