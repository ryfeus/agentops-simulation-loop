from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl.phase9b.adapter import _adapter_manifest_files, validate_persisted_adapter
from rl.phase9b.compare import build_comparison
from rl.phase9b.evaluate import ATTEMPTS_PER_PASS, PASSES, _pass_command
from scripts import rl_harbor_overfit_eval
from scripts.rl_harbor_overfit_eval import _valid_policy


def _rows(policy: str, task_id: str, rewards: list[float]) -> list[dict[str, object]]:
    return [
        {
            "policy": policy,
            "task_id": task_id,
            "reward": value,
            "tool_call_count": 0,
            "tool_calls": [],
            "verifier_components": {},
        }
        for value in rewards
    ]


def test_phase9b_has_fixed_four_native_passes() -> None:
    assert (PASSES, ATTEMPTS_PER_PASS, PASSES * ATTEMPTS_PER_PASS) == (4, 4, 16)


def test_each_native_pass_is_a_separate_worker_process(tmp_path: Path) -> None:
    commands = [
        _pass_command(
            policy="baseline",
            evaluation_pass=index,
            dataset_root=tmp_path / "suite",
            output=tmp_path / "baseline",
            run_id="phase9b-test",
            seed=20260923,
            source_run="phase9-source",
            source_dir=tmp_path / "source",
            task_set_path=tmp_path / "source" / "training-task-set.json",
        )
        for index in range(PASSES)
    ]
    assert [command[-1] for command in commands] == ["0", "1", "2", "3"]
    assert all(command[1:3] == ["-m", "rl.phase9b.evaluate"] for command in commands)
    assert all("--worker-policy" in command for command in commands)
    with pytest.raises(ValueError, match="invalid Phase 9b policy or pass"):
        _pass_command(
            policy="baseline",
            evaluation_pass=PASSES,
            dataset_root=tmp_path,
            output=tmp_path,
            run_id="bad",
            seed=0,
            source_run="bad",
            source_dir=tmp_path,
            task_set_path=tmp_path,
        )


def test_remote_adapter_validator_has_no_local_controller_dependency() -> None:
    source = Path("rl/phase9b/adapter.py").read_text(encoding="utf-8")
    assert "from scripts" not in source
    digest = hashlib.sha256(b"weights").hexdigest()
    assert _adapter_manifest_files(
        {
            "artifacts": [
                {"path": "adapter/final/adapter_model.safetensors", "size": 7, "sha256": digest}
            ]
        }
    ) == {"adapter_model.safetensors": {"size": 7, "sha256": digest}}
    with pytest.raises(ValueError, match="unsafe path"):
        _adapter_manifest_files(
            {"artifacts": [{"path": "adapter/final/../escape", "size": 7, "sha256": digest}]}
        )


def test_persisted_adapter_requires_manifest_and_phase9_lineage(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter" / "final"
    adapter.mkdir(parents=True)
    config = {
        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "bias": "none",
        "target_modules": ["q_proj"],
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "training_args.bin").write_bytes(b"trainer arguments, not adapter weights")
    artifacts = []
    for path in sorted(adapter.iterdir()):
        artifacts.append(
            {
                "path": f"adapter/final/{path.name}",
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (tmp_path / "artifacts-manifest.json").write_text(json.dumps({"artifacts": artifacts}))
    task_set = b'{"schema_version":"1","name":"source"}\n'
    (tmp_path / "training-task-set.json").write_bytes(task_set)
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "adapter_persisted_locally": True,
                "adapter_checksums_verified": True,
                "execution_suite_sha256": "suite",
                "source_revision": "a" * 40,
                "training_task_set_sha256": hashlib.sha256(task_set).hexdigest(),
            }
        )
    )
    identity = validate_persisted_adapter(
        tmp_path, source_run="phase9-test", execution_suite_sha256="suite"
    )
    assert identity.weights_sha256 == hashlib.sha256(b"weights").hexdigest()
    assert identity.task_set_sha256 == hashlib.sha256(task_set).hexdigest()


def test_payload_places_adapter_at_validator_expected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"wheel")
    suite = tmp_path / "suite"
    (suite / "tasks").mkdir(parents=True)
    (suite / "execution-suite.json").write_text("{}")
    oracle = tmp_path / "oracle.json"
    oracle.write_text("{}")
    source = tmp_path / "source"
    adapter_dir = source / "adapter" / "final"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "artifacts-manifest.json").write_text("{}")
    (source / "summary.json").write_text("{}")
    task_set = source / "training-task-set.json"
    task_set.write_text("{}")
    adapter = SimpleNamespace(
        source_dir=source,
        adapter_dir=adapter_dir,
        task_set_path=task_set,
        files={"adapter_model.safetensors": {}},
        value=lambda: {"source_run": "phase9-test"},
    )
    monkeypatch.setattr(rl_harbor_overfit_eval, "_payload_files", lambda _wheel: ())
    payload, _ = rl_harbor_overfit_eval.package_payload(
        wheel=wheel, suite_root=suite, oracle=oracle, task_set=task_set, adapter=adapter
    )
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert "source-adapter/adapter/final/adapter_model.safetensors" in names
    assert "source-adapter/final/adapter_model.safetensors" not in names


def _policy_evidence(tmp_path: Path, *, bad: str | None = None) -> None:
    expected = {"a", "b", "c", "d", "control"}
    policy = "baseline"
    directory = tmp_path / policy
    directory.mkdir()
    (directory / "evaluation-result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "training_performed": False,
                "optimizer_created": False,
                "global_step": 0,
                "checkpoint_emitted": False,
            }
        )
    )
    rows = []
    for task_id in sorted(expected):
        for attempt in range(16):
            rows.append(
                {
                    "policy": policy,
                    "task_id": task_id,
                    "attempt": attempt,
                    "evaluation_pass": attempt // 4,
                    "pass_local_attempt": attempt % 4,
                    "reward": 1.0,
                }
            )
    if bad == "duplicate":
        rows[-1]["attempt"] = 14
    if bad == "uneven":
        rows[-1]["task_id"] = "a"
    (directory / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert _valid_policy(tmp_path, policy, expected) is (bad is None)


def test_phase9b_policy_requires_exact_per_task_attempt_coverage(tmp_path: Path) -> None:
    _policy_evidence(tmp_path)


@pytest.mark.parametrize("bad", ["duplicate", "uneven"])
def test_phase9b_policy_rejects_duplicate_or_uneven_attempts(tmp_path: Path, bad: str) -> None:
    _policy_evidence(tmp_path, bad=bad)


def test_comparison_requires_two_improvements_and_preserves_control_flag(tmp_path: Path) -> None:
    baseline, trained = [], []
    for task in ("a", "b", "c", "d"):
        baseline += _rows("baseline", task, [0.0] * 16)
        trained += _rows("trained", task, [1.0] * 16)
    baseline += _rows("baseline", "control", [1.0] * 16)
    trained += _rows("trained", "control", [0.75] * 16 + [])
    result = build_comparison(
        baseline=baseline,
        trained=trained,
        training_ids={"a", "b", "c", "d"},
        control_id="control",
        destination=tmp_path,
    )
    assert result["behavioral_improvement_supported"] is True
    assert result["catastrophic_regression_detected"] is False
    assert json.loads((tmp_path / "comparison.json").read_text())["human_review_required"] is True
