from __future__ import annotations

import ast
import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from rl.phase8c.execution_suite import derive_execution_suite
from rl.phase9.report import build_reports
from rl.phase9.task_set import load_task_set
from rl.phase9.training_config import (
    EVAL_BATCH_SIZE,
    MAX_COMPLETION_LENGTH,
    MAX_TOOL_CALLING_ITERATIONS,
    MODEL_ID,
    NUM_GENERATIONS,
    TRAIN_BATCH_SIZE,
    config_evidence,
)
from rl.phase9.training_reward import reset_stage, set_stage, training_reward
from scripts import rl_smoke
from scripts.rl_harbor_overfit import (
    OverfitControllerError,
    _retain_exact_task_set,
    validate_rollout_evidence,
    write_summary,
)


def _manifest(suite: dict[str, object]) -> dict[str, object]:
    tasks = suite["tasks"]
    assert isinstance(tasks, dict)
    train = tasks["readonly-before-action"]
    control = tasks["missing-status"]
    assert isinstance(train, dict) and isinstance(control, dict)
    return {
        "schema_version": "1",
        "name": "test",
        "execution_suite_sha256": suite["execution_suite_sha256"],
        "source_phase8c_runs": ["phase8c-test"],
        "training_tasks": [
            {
                "task_id": "readonly-before-action",
                "archetype": train["archetype"],
                "difficulty": train["difficulty"],
                "baseline_attempts": 4,
                "baseline_passes": 3,
                "baseline_pass_rate": 0.75,
                "selection_reason": "mixed",
            }
        ],
        "regression_tasks": [
            {
                "task_id": "missing-status",
                "archetype": control["archetype"],
                "difficulty": control["difficulty"],
                "selection_reason": "control",
            }
        ],
    }


def test_task_set_validates_full_suite_lineage_and_disjoint_sets(tmp_path: Path) -> None:
    suite = derive_execution_suite(tmp_path / "suite").manifest
    path = tmp_path / "task-set.json"
    path.write_text(json.dumps(_manifest(suite)))
    selected = load_task_set(path, suite)
    assert selected.training_ids == {"readonly-before-action"}
    assert selected.regression_ids == {"missing-status"}


def test_task_set_rejects_overlap(tmp_path: Path) -> None:
    suite = derive_execution_suite(tmp_path / "suite").manifest
    value = _manifest(suite)
    value["regression_tasks"] = [dict(value["training_tasks"][0])]
    path = tmp_path / "task-set.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlap"):
        load_task_set(path, suite)


def test_exact_task_set_recovery_preserves_bytes_and_rejects_mismatch(tmp_path: Path) -> None:
    raw = b'{ "name": "operator selection", "training_tasks": [] }\n'
    (tmp_path / "identity.json").write_text(
        json.dumps({"training_task_set_sha256": hashlib.sha256(raw).hexdigest()})
    )
    target = tmp_path / "training-task-set.json"
    target.write_text(json.dumps(json.loads(raw), sort_keys=True))
    _retain_exact_task_set(tmp_path, raw)
    assert target.read_bytes() == raw
    assert list(tmp_path.glob("training-task-set.invalid-*.json"))
    with pytest.raises(OverfitControllerError, match="SHA"):
        _retain_exact_task_set(tmp_path, b'{"name":"different"}')
    assert target.read_bytes() == raw


def test_summary_uses_run_source_provenance_after_worktree_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "identity.json").write_text(
        json.dumps({"source_revision": "clean-run-revision", "worktree_clean": True})
    )
    (tmp_path / "training-task-set.json").write_text(
        json.dumps({"training_tasks": [], "regression_tasks": []})
    )
    monkeypatch.setattr(rl_smoke, "git_provenance", lambda: ("later-revision", False))
    outputs = {
        name: {"value": value}
        for name, value in {
            "instance_type": "g6.2xlarge",
            "ami_id": "ami-123",
            "availability_zone": "us-west-2c",
        }.items()
    }
    summary = write_summary(tmp_path, outputs, None)
    assert summary["git_revision"] == "clean-run-revision"
    assert summary["worktree_clean"] is True


def test_pinned_phase9_config_contract_is_documented_without_model_load() -> None:
    evidence = config_evidence(max_steps=20, learning_rate=1e-5, seed=20260922)
    assert MODEL_ID == "Qwen/Qwen3-0.6B"
    assert (NUM_GENERATIONS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE) == (4, 4, 4)
    assert (MAX_COMPLETION_LENGTH, MAX_TOOL_CALLING_ITERATIONS) == (256, 4)
    assert evidence["lora"]["r"] == 8
    assert evidence["lora"]["lora_alpha"] == 16


def test_phase9_declares_the_fixed_phase8b_tool_contract_without_evaluator_import() -> None:
    source = Path("rl/phase9/train_overfit.py").read_text(encoding="utf-8")
    assert 'EXPECTED_TOOLS = {"get_invoice", "refund_invoice", "escalate_dispute"}' in source
    assert "evaluate_baseline import EXPECTED_TOOLS" not in source


def test_native_runner_has_evaluate_train_evaluate_lifecycle() -> None:
    tree = ast.parse(Path("rl/phase9/train_overfit.py").read_text())
    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_stage"
            and len(node.args) >= 3
            and isinstance(node.args[1], ast.Constant)
        ),
        key=lambda node: node.lineno,
    )
    stages = [node.args[1].value for node in calls]
    assert stages == ["before", "train", "after"]


def test_before_after_report_separates_training_and_control(tmp_path: Path) -> None:
    def rows(stage: str, rewards: list[float], task_id: str) -> None:
        with (tmp_path / f"rollouts-{stage}.jsonl").open("a") as stream:
            for reward in rewards:
                stream.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "archetype": "read-only",
                            "difficulty": "easy",
                            "reward": reward,
                            "tool_call_count": 1,
                            "tool_calls": [],
                            "verifier_components": {"reward": reward},
                        }
                    )
                    + "\n"
                )

    rows("before", [0.0] * 4, "train")
    rows("after", [1.0] * 4, "train")
    rows("before", [1.0] * 4, "control")
    rows("after", [1.0] * 4, "control")
    (tmp_path / "reward-groups.jsonl").write_text(
        json.dumps(
            {
                "task_id": "train",
                "rewards": [0.0, 1.0, 0.0, 1.0],
                "has_reward_variance": True,
            }
        )
        + "\n"
    )
    result = build_reports(output=tmp_path, training_ids={"train"}, regression_ids={"control"})
    assert result["training_tasks_improved"] == 1
    report = json.loads((tmp_path / "before-after-summary.json").read_text())
    assert {row["task_id"] for row in report["tasks"] if row["is_regression_control"]} == {
        "control"
    }


def test_adapter_delta_counts_tensors_and_scalars() -> None:
    source = Path("rl/phase9/train_overfit.py").read_text(encoding="utf-8")
    assert '"trainable_tensor_count": len(before)' in source
    assert '"trainable_parameter_count": scalar_parameters' in source
    assert '"changed_tensor_count": changed_tensors' in source


def test_safe_archive_extraction_requires_exact_regular_manifest_files(tmp_path: Path) -> None:
    payload = BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        content = b"adapter"
        info = tarfile.TarInfo("adapter_config.json")
        info.size = len(content)
        archive.addfile(info, BytesIO(content))
    expected = {
        "adapter_config.json": {"size": 7, "sha256": hashlib.sha256(b"adapter").hexdigest()}
    }
    destination = tmp_path / "adapter"
    rl_smoke._extract_verified_archive(payload.getvalue(), destination, expected)
    assert (destination / "adapter_config.json").read_bytes() == b"adapter"


def test_rollout_evidence_requires_stable_group_references(tmp_path: Path) -> None:
    task_set = {
        "training_tasks": [{"task_id": "train"}],
        "regression_tasks": [{"task_id": "control"}],
    }
    (tmp_path / "training-task-set.json").write_text(json.dumps(task_set))
    rows = {
        "before": [("train", 0.0), ("control", 1.0)],
        "train": [("train", float(index % 2)) for index in range(4)],
        "after": [("train", 1.0), ("control", 1.0)],
    }
    for stage, values in rows.items():
        (tmp_path / f"rollouts-{stage}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "stage": stage,
                        "task_id": task,
                        "reward": reward,
                        "rollout_id": f"{stage}:{task}:{index:06d}",
                    }
                )
                + "\n"
                for index, (task, reward) in enumerate(values)
            )
        )
    (tmp_path / "reward-groups.jsonl").write_text(
        json.dumps({"rollout_ids": [f"train:train:{index:06d}" for index in range(4)]}) + "\n"
    )
    (tmp_path / "training-metrics.jsonl").write_text(json.dumps({"loss": 1.0}) + "\n")
    assert validate_rollout_evidence(tmp_path)["valid"] is True


def test_training_reward_emits_stable_ids_and_group_references(tmp_path: Path) -> None:
    class Environment:
        reward = 1.0

        @staticmethod
        def _baseline_evidence() -> dict[str, object]:
            return {}

    tokens = set_stage("train", tmp_path)
    try:
        training_reward(
            environments=[Environment() for _ in range(4)],
            completions=["x"] * 4,
            task_id=["task"] * 4,
            archetype=["a"] * 4,
            difficulty=["d"] * 4,
        )
    finally:
        reset_stage(tokens)
    rows = [
        json.loads(line) for line in (tmp_path / "rollouts-train.jsonl").read_text().splitlines()
    ]
    group = json.loads((tmp_path / "reward-groups.jsonl").read_text())
    assert [row["rollout_id"] for row in rows] == [f"train:task:{index:06d}" for index in range(4)]
    assert group["rollout_ids"] == [row["rollout_id"] for row in rows]


def test_catastrophic_regression_requires_severe_control_drop(tmp_path: Path) -> None:
    def rows(stage: str, reward: float) -> None:
        with (tmp_path / f"rollouts-{stage}.jsonl").open("a") as stream:
            for task_id in ("train", "control"):
                for _ in range(4):
                    stream.write(json.dumps({"task_id": task_id, "reward": reward}) + "\n")

    rows("before", 1.0)
    rows("after", 0.0)
    (tmp_path / "reward-groups.jsonl").write_text("")
    result = build_reports(output=tmp_path, training_ids={"train"}, regression_ids={"control"})
    assert result["catastrophic_regression_detected"] is True
