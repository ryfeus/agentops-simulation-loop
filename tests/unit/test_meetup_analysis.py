from __future__ import annotations

import copy
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rl.phase12 import config
from rl.phase12.common import digest
from scripts import benchmark_billing_bedrock as benchmark
from scripts.billing_bedrock_results import collect, summarize, trajectory_metrics
from scripts.meetup_budget_diagnostic import diagnostic_config, evaluation_worker, select_sample
from scripts.meetup_evidence_analysis import unpack_verified


def test_final_state_measure_does_not_count_blocked_attempts(tmp_path):
    import sqlite3

    from agentops_demo.billing.schema import SCHEMA_SQL
    from scripts.billing_bedrock_results import forbidden_state_changes

    database = tmp_path / "state.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute("INSERT INTO customers VALUES ('c', 'active')")
        connection.execute("INSERT INTO invoices VALUES ('a', 'c', '89.00', 'USD', 'disputed')")
    scenario = {
        "initial_state": {"invoices": [], "refunds": [], "escalations": []},
        "expected_invariants": [{"type": "no_refund", "invoice_id": "a"}],
    }
    assert forbidden_state_changes(database, scenario) is False
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO refunds VALUES ('r', 'a', 'test')")
    assert forbidden_state_changes(database, scenario) is True


def test_metric_coverage_rejects_missing_and_duplicate_attempts():
    from rl.phase12.metrics import summarize as phase12_summary

    metadata = {
        "a": {
            "family_id": "f",
            "prototype_id": "p",
            "archetype": "normal-refund",
            "expected_operations": {},
        }
    }
    row = {"task_id": "a", "attempt": 0, "reward": 0}
    for captured in ([], [row], [row, row]):
        with pytest.raises(ValueError):
            phase12_summary(captured, metadata, 2)


def test_archive_rejects_traversal_before_extraction(tmp_path):
    path = tmp_path / "evidence.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))
    Path(str(path) + ".json").write_text(
        json.dumps({"archive_sha256": digest(path), "files": {"../escape": "irrelevant"}})
    )
    with pytest.raises(ValueError, match="unsafe"):
        unpack_verified(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_archive_rejects_changed_archive(tmp_path):
    path = tmp_path / "archive"
    path.write_bytes(b"changed")
    Path(str(path) + ".json").write_text(json.dumps({"archive_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checksum"):
        unpack_verified(path, tmp_path / "out")


@pytest.mark.parametrize("fault", ["member_hash", "symlink", "duplicate"])
def test_archive_rejects_invalid_members(tmp_path, fault):
    import hashlib

    path = tmp_path / "evidence.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        info = tarfile.TarInfo("record.json")
        if fault == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/outside"
            stream.addfile(info)
        else:
            info.size = 1
            stream.addfile(info, io.BytesIO(b"x"))
            if fault == "duplicate":
                stream.addfile(info, io.BytesIO(b"x"))
    Path(str(path) + ".json").write_text(
        json.dumps(
            {
                "archive_sha256": digest(path),
                "files": {
                    "record.json": "0" * 64
                    if fault == "member_hash"
                    else hashlib.sha256(b"x").hexdigest()
                },
            }
        )
    )
    with pytest.raises(ValueError):
        unpack_verified(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_trials_are_infrastructure_failures_not_rewards(tmp_path):
    trial = tmp_path / "job/trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 0}}, "exception_info": None})
    )
    observed = collect(
        tmp_path / "job", {"id": "one", "benchmark": {"archetype": "normal-refund"}}, 2, 1
    )
    assert observed[0]["reward"] == 0 and not observed[0]["infrastructure_failure"]
    assert observed[1]["reward"] is None and observed[1]["infrastructure_failure"]
    assert summarize(observed)["infrastructure_failures"] == 1
    assert summarize(observed)["by_archetype"]["normal-refund"]["attempts"] == 2


def test_attempt_and_successful_forbidden_operation_are_distinct():
    trajectory = {
        "steps": [
            {
                "tool_calls": [
                    {
                        "tool_call_id": "call",
                        "function_name": "refund_invoice",
                        "arguments": {"invoice_id": "a"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call",
                            "content": '{"error":"disputed_invoice_requires_escalation"}',
                        }
                    ]
                },
            }
        ]
    }
    scenario = {"expected_invariants": [{"type": "must_escalate", "invoice_id": "a"}]}
    result = trajectory_metrics(trajectory, scenario)
    assert result["prohibited_write_attempt"] and not result["successful_prohibited_operation"]
    assert result["tool_errors"] == 1


def test_budget_grid_is_separate_and_deterministic():
    original = copy.deepcopy(config.load())
    changed = diagnostic_config(1024, 16)
    assert changed["rollout"]["max_tool_calling_iterations"] == 16
    assert config.load() == original
    data = {
        str(i): {
            "family_id": str(i % 3),
            "prototype_id": str(i),
            "required_tool_calls": 5 if i < 6 else 2,
        }
        for i in range(12)
    }
    sample = select_sample(data)
    assert sample == select_sample(dict(reversed(list(data.items()))))
    assert len(sample["long"]) == len(sample["short"]) == 6
    assert not set(sample["long"]) & set(sample["short"])
    with pytest.raises(ValueError):
        diagnostic_config(2048, 16)


@pytest.mark.parametrize("method", ["create_optimizer", "train"])
def test_budget_worker_prohibits_training_before_it_begins(monkeypatch, tmp_path, method):
    from rl.phase12 import runtime

    class UnsafeTrainer:
        def create_optimizer(self):
            pytest.fail("optimizer creation must never start")

        def train(self):
            pytest.fail("training must never start")

    factory = lambda: UnsafeTrainer  # noqa: E731
    monkeypatch.setattr(runtime, "trainer_class", factory)

    def unsafe_evaluate(**kwargs):
        assert kwargs["exp"]["rollout"]["max_tool_calling_iterations"] == 16
        getattr(runtime.trainer_class()(), method)()

    monkeypatch.setattr(runtime, "evaluate", unsafe_evaluate)
    job = {
        "root": str(tmp_path),
        "output": str(tmp_path / "out"),
        "adapter": None,
        "budget": 1024,
        "iterations": 16,
        "seed": 42,
        "selected": [],
    }
    original = copy.deepcopy(job)
    with pytest.raises(RuntimeError, match="prohibited"):
        evaluation_worker(job)
    assert runtime.trainer_class is factory
    assert job == original


def test_benchmark_returns_failure_status(monkeypatch, tmp_path):
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda **kw: {
            "summary": {
                "infrastructure_failures": 1,
                "child_process_failures": 1,
                "complete": False,
            }
        },
    )
    assert (
        benchmark.main(["--candidate-config", str(tmp_path / "config"), "--role-arn", "unused"])
        == 1
    )


def test_benchmark_repeats_and_policy_reach_child(monkeypatch, tmp_path, agent_config_data):
    from agentops_demo.benchmark.catalog import load_catalog
    from agentops_demo.contracts.agent_config import AgentConfig
    from scripts import aws_context

    agent_config_data["model"] = {"provider": "bedrock", "model_id": "test"}
    candidate = AgentConfig.model_validate(agent_config_data)
    path = tmp_path / "candidate.json"
    path.write_text(candidate.model_dump_json())
    entry = load_catalog()[0]
    monkeypatch.setattr(benchmark, "render_catalog_sync", lambda *a: [entry])
    monkeypatch.setattr(benchmark, "derive_bedrock_task", lambda *a: None)
    monkeypatch.setattr(
        aws_context, "verified_session", lambda: SimpleNamespace(client=lambda name: None)
    )
    monkeypatch.setattr(aws_context, "expected_account_id", lambda: "123456789012")
    monkeypatch.setattr(
        benchmark,
        "assume_bedrock_role",
        lambda *a, **kw: SimpleNamespace(credentials_file=lambda: "temporary"),
    )
    monkeypatch.setattr(
        benchmark,
        "build_clean_wheel",
        lambda p: (
            Path("test.whl"),
            SimpleNamespace(sha256="a" * 64, source_revision=candidate.agent.source_revision),
        ),
    )
    commands = []
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kw: commands.append(cmd) or SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        benchmark,
        "collect",
        lambda job, scenario, attempts, rc: [
            {"task_id": scenario["id"], "reward": 1, "infrastructure_failure": False}
            for _ in range(attempts)
        ],
    )
    result = benchmark.run(
        catalog=tmp_path,
        candidate_config=path,
        role_arn="arn:aws:iam::123456789012:role/test",
        output=tmp_path / "out",
        attempts=8,
        policy_mode="enforced",
    )
    assert commands[0][commands[0].index("--n-attempts") + 1] == "8"
    assert "billing_policy_mode=enforced" in commands[0]
    assert result["summary"]["attempted_rollouts"] == 8
    assert result["summary"]["task_macro_pass_rate_including_infrastructure_failures"] == 1
    with pytest.raises(FileExistsError):
        benchmark.run(
            catalog=tmp_path, candidate_config=path, role_arn="unused", output=tmp_path / "out"
        )


def test_model_settings_reach_bedrock(monkeypatch, agent_config):
    import sys

    from agentops_demo.agent.model import create_model

    factory = Mock()
    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(ChatBedrockConverse=factory))
    monkeypatch.setenv("BILLING_MODEL_TEMPERATURE", "0")
    monkeypatch.setenv("BILLING_MODEL_MAX_TOKENS", "1024")
    monkeypatch.setenv("BILLING_MODEL_TOTAL_ATTEMPTS", "1")
    create_model(agent_config)
    assert factory.call_args.kwargs["max_tokens"] == 1024
    assert factory.call_args.kwargs["temperature"] == 0
    assert factory.call_args.kwargs["config"].retries["total_max_attempts"] == 1


def test_policy_mode_cannot_inject_shell():
    from agentops_demo.harbor.agent import _start_mcp_command

    assert "BILLING_POLICY_MODE=enforced" in _start_mcp_command("enforced")
    with pytest.raises(ValueError):
        _start_mcp_command("permissive; arbitrary")
