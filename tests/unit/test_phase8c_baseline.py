from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from rl.phase8c.evaluate_baseline import (
    ATTEMPTS_PER_PASS,
    EVALUATION_MAX_STEPS,
    EXPECTED_TOOLS,
    FROZEN_EVALUATION_SETTINGS,
    classify_infrastructure_failure,
)
from rl.phase8c.execution_suite import derive_execution_suite, load_execution_suite
from rl.phase8c.report import build_reports, wilson_interval
from rl.phase8c.suite_check import load_oracle_cache
from scripts import rl_harbor_baseline


def _rollout(
    task_id: str, attempt: int, reward: float, *, archetype: str = "normal"
) -> dict[str, object]:
    return {
        "run_id": "test",
        "task_id": task_id,
        "archetype": archetype,
        "difficulty": "easy",
        "attempt": attempt,
        "reward": reward,
        "retry_count": 0,
        "tool_call_count": 1,
        "tool_calls": [{"name": "get_invoice", "arguments": {"invoice_id": "inv-1"}}],
        "verifier_components": {"reward": reward, "trajectory": 1.0},
        "environment_reset_ms": 1.0,
        "tool_execution_ms": 2.0,
        "verifier_ms": 3.0,
    }


def test_execution_suite_derives_full_shared_verifier_copy_and_hashes(tmp_path: Path) -> None:
    suite = derive_execution_suite(tmp_path / "suite")
    manifest = load_execution_suite(tmp_path / "suite" / "execution-suite.json")
    assert manifest == suite.manifest
    assert manifest["task_count"] == 25
    assert manifest["task_ids"] == sorted(manifest["tasks"])
    assert manifest["base_suite_sha256"] != manifest["execution_suite_sha256"]
    task = tmp_path / "suite" / "tasks" / "paid-refund-direct"
    source = Path("benchmarks/billing/tasks/paid-refund-direct")
    assert task.joinpath("task.toml").read_text() != source.joinpath("task.toml").read_text()
    for path in source.rglob("*"):
        if path.is_file() and path.name != "task.toml":
            assert path.read_bytes() == task.joinpath(path.relative_to(source)).read_bytes()


def test_oracle_cache_requires_exact_passing_suite(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    payload = {
        "schema_version": "1",
        "status": "PASS",
        "execution_suite_sha256": "a" * 64,
        "task_count": 25,
        "passed_tasks": 25,
        "failed_tasks": [],
    }
    path.write_text(json.dumps(payload))
    assert load_oracle_cache(path, "a" * 64, 25)
    assert not load_oracle_cache(path, "b" * 64, 25)
    payload["failed_tasks"] = [{"task_id": "x"}]
    path.write_text(json.dumps(payload))
    assert not load_oracle_cache(path, "a" * 64, 25)


def test_frozen_config_preserves_native_eval_contract() -> None:
    assert ATTEMPTS_PER_PASS == 4
    assert EVALUATION_MAX_STEPS == 25
    assert FROZEN_EVALUATION_SETTINGS["model_init_kwargs"] == {"dtype": "bfloat16"}
    assert FROZEN_EVALUATION_SETTINGS["use_vllm"]
    assert FROZEN_EVALUATION_SETTINGS["vllm_mode"] == "colocate"
    assert FROZEN_EVALUATION_SETTINGS["vllm_tensor_parallel_size"] == 1
    assert FROZEN_EVALUATION_SETTINGS["max_completion_length"] == 256
    assert FROZEN_EVALUATION_SETTINGS["max_tool_calling_iterations"] == 4
    assert {"get_invoice", "refund_invoice", "escalate_dispute"} == EXPECTED_TOOLS


def test_frozen_runner_uses_native_evaluate_and_never_train() -> None:
    source = Path("rl/phase8c/evaluate_baseline.py").read_text()
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "trainer"
    ]
    assert "evaluate" in calls
    assert "train" not in calls


def test_reports_bucket_wilson_and_candidate_aggregation(tmp_path: Path) -> None:
    rows = [
        *[_rollout("always-fail", index, 0.0, archetype="read") for index in range(4)],
        *[_rollout("mixed", index, float(index % 2), archetype="refund") for index in range(4)],
        *[_rollout("always-pass", index, 1.0, archetype="refund") for index in range(4)],
    ]
    result = build_reports(rows, attempts_per_task=4, destination=tmp_path)
    by_id = {task["task_id"]: task for task in result["tasks"]}
    assert by_id["always-fail"]["observed_bucket"] == "observed_always_fail"
    assert by_id["mixed"]["observed_bucket"] == "observed_mixed"
    assert by_id["always-pass"]["observed_bucket"] == "observed_always_pass"
    assert wilson_interval(0, 4)[0] == 0.0
    assert result["candidates"]["mixed_task_count"] == 1
    assert (tmp_path / "task-summary.json").is_file()


def test_reports_fail_closed_on_duplicate_or_missing_attempts(tmp_path: Path) -> None:
    rows = [_rollout("task", 0, 0.0), _rollout("task", 0, 1.0)]
    with pytest.raises(ValueError, match="identities"):
        build_reports(rows, attempts_per_task=2, destination=tmp_path)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("cannot connect to Docker daemon", "DOCKER_UNAVAILABLE"),
        ("RewardFileNotFoundError", "HARBOR_VERIFIER"),
        ("CUDA out of memory", "VLLM_OOM"),
        ("invoice is already refunded", None),
    ],
)
def test_infrastructure_failure_classification_does_not_retry_policy_errors(
    message: str, expected: str | None
) -> None:
    assert classify_infrastructure_failure(message) == expected


def test_summary_requires_no_training_complete_coverage_and_matching_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rl_harbor_baseline.phase8a, "git_provenance", lambda: ("a" * 40, True))
    (tmp_path / "run-status.json").write_text(json.dumps({"evaluation": "PASS"}))
    (tmp_path / "versions.json").write_text(json.dumps({"gpu": "NVIDIA L4"}))
    suite = {"base_suite_sha256": "b" * 64, "execution_suite_sha256": "c" * 64}
    (tmp_path / "execution-suite.json").write_text(json.dumps(suite))
    (tmp_path / "oracle-suite-check.json").write_text(
        json.dumps({"status": "PASS", "execution_suite_sha256": "c" * 64})
    )
    (tmp_path / "phase9-candidates.json").write_text(
        json.dumps({"phase9_candidate_threshold_met": False})
    )
    (tmp_path / "evaluation-result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_type": "canonical",
                "training_performed": False,
                "global_step": 0,
                "optimizer_created": False,
                "checkpoint_emitted": False,
                "task_count": 25,
                "attempts_per_task": 4,
                "expected_rollouts": 100,
                "valid_rollouts": 100,
                "infrastructure_invalid_rollouts": 0,
            }
        )
    )
    outputs = {
        "instance_type": {"value": "g6.2xlarge"},
        "ami_id": {"value": "ami-test"},
        "availability_zone": {"value": "us-west-2a"},
    }
    summary = rl_harbor_baseline.write_summary(tmp_path, outputs, None)
    assert summary["status"] == "PASS"
    assert summary["canonical_acceptance"] is True
    failed = json.loads((tmp_path / "evaluation-result.json").read_text())
    failed["optimizer_created"] = True
    (tmp_path / "evaluation-result.json").write_text(json.dumps(failed))
    assert rl_harbor_baseline.write_summary(tmp_path, outputs, None)["status"] == "FAIL"
