from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib
import json
import sqlite3
import sys
import tarfile
import tomllib
from io import BytesIO
from pathlib import Path

import pytest
from harbor.models.trial.paths import TrialPaths

from agentops_demo.harbor.provenance import PackageProvenance
from agentops_demo.taskify.integrity import harbor_task_sha256
from scripts import rl_harbor_smoke

PHASE8B = Path("rl/phase8b").resolve()
if str(PHASE8B) not in sys.path:
    sys.path.insert(0, str(PHASE8B))
bridge = importlib.import_module("sandbox_billing_bridge")
execution_task = importlib.import_module("execution_task")
harbor_compat = importlib.import_module("harbor_compat")
SEED = Path("benchmarks/billing/tasks/paid-refund-direct/environment/seed.sql")
TASK = Path("benchmarks/billing/tasks/paid-refund-direct")


def prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    database, trajectory = tmp_path / "billing.db", tmp_path / "agent-run.json"
    with sqlite3.connect(database) as connection:
        connection.executescript(SEED.read_text())
    monkeypatch.setenv("BILLING_DATABASE_PATH", str(database))
    monkeypatch.setenv("AGENT_RUN_PATH", str(trajectory))
    return database, trajectory


def calls(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text())["tool_calls"]


def test_bridge_reads_and_records_invoice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, trajectory = prepare(monkeypatch, tmp_path)
    result = asyncio.run(bridge.invoke("get_invoice", {"invoice_id": "inv-201"}))
    assert result["ok"] is True
    assert result["invoice"]["status"] == "paid"
    assert calls(trajectory) == [{"name": "get_invoice", "arguments": {"invoice_id": "inv-201"}}]


def test_bridge_refund_and_ordered_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database, trajectory = prepare(monkeypatch, tmp_path)
    asyncio.run(bridge.invoke("get_invoice", {"invoice_id": "inv-201"}))
    result = asyncio.run(
        bridge.invoke("refund_invoice", {"invoice_id": "inv-201", "reason": "test"})
    )
    assert result["ok"] is True
    with sqlite3.connect(database) as connection:
        result = connection.execute("SELECT status FROM invoices WHERE id = 'inv-201'").fetchone()
        assert result == ("refunded",)
    assert [call["name"] for call in calls(trajectory)] == ["get_invoice", "refund_invoice"]


def test_bridge_records_missing_and_duplicate_domain_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, trajectory = prepare(monkeypatch, tmp_path)
    missing = asyncio.run(bridge.invoke("get_invoice", {"invoice_id": "inv-999"}))
    assert missing == {
        "ok": False,
        "error_type": "InvoiceNotFoundError",
        "error": "invoice 'inv-999' does not exist",
    }
    asyncio.run(bridge.invoke("refund_invoice", {"invoice_id": "inv-201", "reason": "first"}))
    duplicate = asyncio.run(
        bridge.invoke("refund_invoice", {"invoice_id": "inv-201", "reason": "duplicate"})
    )
    assert duplicate["ok"] is False
    assert duplicate["error_type"] == "RefundAlreadyExistsError"
    assert [call["name"] for call in calls(trajectory)] == [
        "get_invoice",
        "refund_invoice",
        "refund_invoice",
    ]


def test_harness_declares_only_billing_tools() -> None:
    module = ast.parse((PHASE8B / "billing_harbor_env.py").read_text())
    definition = next(node for node in module.body if isinstance(node, ast.ClassDef))
    public = {
        node.name
        for node in definition.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public == {"get_invoice", "refund_invoice", "escalate_dispute"}


def test_harness_preserves_the_valid_wheel_filename_for_pip() -> None:
    source = (PHASE8B / "billing_harbor_env.py").read_text()
    assert 'wheel_destination = f"/tmp/{wheel.name}"' in source
    assert "upload_file(wheel, wheel_destination)" in source


def test_phase8b_failure_classes_honor_structured_runtime_phases() -> None:
    assert rl_harbor_smoke.classify_failure("", "HARBOR_SANDBOX_START") == "HARBOR_SANDBOX_START"
    assert rl_harbor_smoke.classify_failure("", "HARBOR_VERIFIER") == "HARBOR_VERIFIER"
    assert rl_harbor_smoke.classify_failure("CUDA out of memory") == "VLLM_OOM"
    assert rl_harbor_smoke.classify_failure("RewardFileNotFoundError") == "HARBOR_VERIFIER"


def test_trial_log_mounts_restore_the_harbor_trial_contract(tmp_path: Path) -> None:
    paths = TrialPaths(trial_dir=tmp_path / "trial")
    mounts = harbor_compat.trial_log_mounts(paths)
    assert mounts == [
        {"type": "bind", "source": str(paths.verifier_dir.resolve()), "target": "/logs/verifier"},
        {"type": "bind", "source": str(paths.agent_dir.resolve()), "target": "/logs/agent"},
        {"type": "bind", "source": str(paths.artifacts_dir.resolve()), "target": "/logs/artifacts"},
    ]


def test_execution_task_derivation_preserves_canonical_semantics(tmp_path: Path) -> None:
    execution = execution_task.derive_execution_task(TASK, tmp_path / "execution")
    with (execution.path / "task.toml").open("rb") as source:
        derived_toml = tomllib.load(source)
    assert execution.base_task_sha256 == harbor_task_sha256(TASK)
    assert execution.base_task_sha256 != execution.execution_task_sha256
    assert execution.execution_profile == "trl-harbor-shared-verifier"
    assert derived_toml["verifier"]["environment_mode"] == "shared"
    assert "environment" not in derived_toml["verifier"]
    for source in sorted(path for path in TASK.rglob("*") if path.is_file()):
        relative = source.relative_to(TASK)
        if relative != Path("task.toml"):
            assert source.read_bytes() == (execution.path / relative).read_bytes()


def test_payload_contains_only_the_locked_runtime_one_task_and_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "agentops_demo-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"clean wheel fixture")
    execution = execution_task.derive_execution_task(TASK, tmp_path / "execution")
    payload, digest = rl_harbor_smoke.package_payload(wheel, execution.path)
    assert hashlib.sha256(payload).hexdigest() == digest
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert {
        "rl/pyproject.toml",
        "rl/uv.lock",
        "rl/phase8b/billing_harbor_env.py",
        "rl/phase8b/harbor_compat.py",
        "rl/phase8b/execution_task.py",
        "rl/phase8b/sandbox_billing_bridge.py",
        "rl/phase8b/env_smoke.py",
        "rl/phase8b/train_harbor_smoke.py",
        "rl/phase8b/remote_run.sh",
        f"wheel/{wheel.name}",
    } <= names
    assert "dataset/tasks/paid-refund-direct/task.toml" in names
    assert all(
        not name.startswith("dataset/tasks/")
        or name == "dataset/tasks/paid-refund-direct"
        or name.startswith("dataset/tasks/paid-refund-direct/")
        for name in names
    )
    assert not any(name == ".git" or name.startswith(".git/") for name in names)
    assert not any("credential" in name.lower() or "token" in name.lower() for name in names)


def test_task_identity_uses_the_shared_harbor_task_digest(tmp_path: Path) -> None:
    package = PackageProvenance(
        source_revision="a" * 40,
        filename="agentops_demo-0.0.0-py3-none-any.whl",
        sha256="b" * 64,
        worktree_clean=True,
    )
    execution = execution_task.derive_execution_task(TASK, tmp_path / "execution")
    identity = rl_harbor_smoke._write_identity(
        tmp_path,
        package=package,
        payload_sha256="c" * 64,
        execution_task=execution,
    )
    assert identity["harbor_task_sha256"] == harbor_task_sha256(rl_harbor_smoke.TASK)
    assert identity["base_task_sha256"] == execution.base_task_sha256
    assert identity["execution_task_sha256"] == execution.execution_task_sha256
    assert identity["execution_profile"] == "trl-harbor-shared-verifier"


def outputs() -> dict[str, dict[str, str]]:
    return {
        "instance_id": {"value": "i-123"},
        "instance_type": {"value": "g6.2xlarge"},
        "ami_id": {"value": "ami-123"},
        "availability_zone": {"value": "us-west-2a"},
    }


def write_pass_evidence(path: Path) -> None:
    (path / "run-status.json").write_text(
        json.dumps({"harbor_env_preflight": "PASS", "training": "PASS"})
    )
    (path / "versions.json").write_text(
        json.dumps({"gpu": "NVIDIA L4", "trl": "1.13.0", "vllm": "0.28.0", "harbor": "0.22.0"})
    )
    (path / "task-identity.json").write_text(
        json.dumps(
            {
                "task_id": "paid-refund-direct",
                "harbor_task_sha256": "a" * 64,
                "base_task_sha256": "a" * 64,
                "execution_task_sha256": "d" * 64,
                "execution_profile": "trl-harbor-shared-verifier",
                "wheel_sha256": "b" * 64,
                "payload_sha256": "c" * 64,
            }
        )
    )
    (path / "harbor-env-preflight.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "noop_reward": 0.0,
                "correct_reward": 1.0,
                "noop_reward_json_exists": True,
                "correct_reward_json_exists": True,
            }
        )
    )
    (path / "training-harbor.json").write_text(
        json.dumps({"status": "PASS", "global_step": 1, "tool_call_frequency": 0.5})
    )


def test_phase8b_summary_fails_closed_on_required_runtime_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rl_harbor_smoke.phase8a, "git_provenance", lambda: ("d" * 40, True))
    write_pass_evidence(tmp_path)
    assert rl_harbor_smoke.write_summary(tmp_path, outputs(), None)["status"] == "PASS"
    for filename, key, bad in (
        ("harbor-env-preflight.json", "noop_reward", 1.0),
        ("harbor-env-preflight.json", "correct_reward", 0.0),
        ("harbor-env-preflight.json", "noop_reward_json_exists", False),
        ("training-harbor.json", "global_step", 0),
        ("training-harbor.json", "tool_call_frequency", 0.0),
    ):
        write_pass_evidence(tmp_path)
        payload = json.loads((tmp_path / filename).read_text())
        payload[key] = bad
        (tmp_path / filename).write_text(json.dumps(payload))
        assert rl_harbor_smoke.write_summary(tmp_path, outputs(), None)["status"] == "FAIL"


def test_phase8b_summary_fails_closed_when_training_record_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rl_harbor_smoke.phase8a, "git_provenance", lambda: ("d" * 40, True))
    write_pass_evidence(tmp_path)
    (tmp_path / "training-harbor.json").unlink()
    summary = rl_harbor_smoke.write_summary(tmp_path, outputs(), None)
    assert summary["status"] == "FAIL"


def test_phase8b_summary_fails_closed_when_manual_preflight_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rl_harbor_smoke.phase8a, "git_provenance", lambda: ("d" * 40, True))
    write_pass_evidence(tmp_path)
    (tmp_path / "harbor-env-preflight.json").unlink()
    summary = rl_harbor_smoke.write_summary(tmp_path, outputs(), None)
    assert summary["status"] == "FAIL"
    assert summary["canonical_acceptance"] is False


def test_phase8b_summary_prefers_structured_training_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rl_harbor_smoke.phase8a, "git_provenance", lambda: ("d" * 40, True))
    write_pass_evidence(tmp_path)
    (tmp_path / "training-harbor.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "failure_phase": "MODEL_NO_TOOL_CALL",
                "error": "tool call frequency was zero",
            }
        )
    )
    summary = rl_harbor_smoke.write_summary(tmp_path, outputs(), None)
    assert summary["failure_phase"] == "MODEL_NO_TOOL_CALL"
    assert summary["failure_class"] == "MODEL_NO_TOOL_CALL"


def test_phase8b_summary_preserves_structured_verifier_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rl_harbor_smoke.phase8a, "git_provenance", lambda: ("d" * 40, True))
    write_pass_evidence(tmp_path)
    (tmp_path / "harbor-env-preflight.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "failure_phase": "HARBOR_VERIFIER",
                "error_type": "RewardFileNotFoundError",
                "error": "No reward file found",
                "traceback": "verifier traceback",
            }
        )
    )
    summary = rl_harbor_smoke.write_summary(tmp_path, outputs(), None)
    assert summary["failure_phase"] == "HARBOR_VERIFIER"
    assert summary["failure_class"] == "HARBOR_VERIFIER"
