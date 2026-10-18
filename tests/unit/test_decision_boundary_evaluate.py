"""Offline orchestration contracts; no fresh-family contents, Docker, or provider calls."""

from __future__ import annotations

import argparse
import asyncio
import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.decision_model import boundary_evaluate, harbor, model


@pytest.fixture
def boundary_args(tmp_path, monkeypatch):
    paths = {}
    for name in ("calibration", "fallback_config", "selection_manifest"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"threshold": 0.91, "selection_complete": True}))
        paths[name] = path
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"offline checkpoint fixture")
    paths["checkpoint"] = checkpoint
    paths["wheel"] = tmp_path / "fixture.whl"
    paths["wheel"].write_bytes(b"offline wheel fixture")
    old = tmp_path / "old_runtime.py"
    old.write_text(
        f"PROMPT_SUFFIX = {harbor.PROMPT_SUFFIX!r}\n"
        "def llm_fallback(config, *, invoice_local_fallback=False):\n"
        "    assert invoice_local_fallback\n    return 'old-fallback'\n"
        "async def run_episode(**kwargs):\n    raise AssertionError('must remain mocked')\n"
    )
    paths["old_runtime"] = old
    auth = tmp_path / "old_authorization.py"
    auth.write_text("VARIANT = 'retained-test-authority'\n")
    paths["old_authorization"] = auth
    monkeypatch.setattr(
        boundary_evaluate, "OLD_AUTHORIZATION_SHA256", boundary_evaluate.digest(auth)
    )
    monkeypatch.setattr(boundary_evaluate, "OLD_RUNTIME_SHA256", boundary_evaluate.digest(old))
    tasks = {
        f"offline-{family}-{variant}": {
            "task_id": f"offline-{family}-{variant}",
            "split": "test",
            "composition_root": f"offline-family-{family}",
            "task_path": "unused-offline-fixture",
        }
        for family in range(6)
        for variant in range(2)
    }
    manifest = tmp_path / "cohort.json"
    manifest.write_text(
        json.dumps(
            {
                "cohort_id": "offline-fixture",
                "tasks": tasks,
                "cohort_previously_exposed": False,
                "pre_freeze_policy_rollouts": 0,
                "pre_freeze_model_inferences": 0,
                "novelty_audit": {
                    "unique_new_structures": 6,
                    "root_collisions": 0,
                    "structural_collisions": 0,
                    "normalized_instruction_collisions": 0,
                },
            }
        )
    )
    monkeypatch.setattr(boundary_evaluate, "validate_task", lambda task: None)
    monkeypatch.setattr(
        boundary_evaluate, "RETAINED_COHORT_SHA256", boundary_evaluate.digest(manifest)
    )
    for name in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "BILLING_MODEL_TEMPERATURE",
        "BILLING_MODEL_MAX_TOKENS",
        "BILLING_MODEL_TOTAL_ATTEMPTS",
    ):
        monkeypatch.setenv(name, "offline")
    args = argparse.Namespace(
        manifest=manifest,
        freeze=tmp_path / "freeze.json",
        output=tmp_path / "result",
        cohort_previously_exposed=True,
        **paths,
    )
    boundary_evaluate.prepare(args)
    return args


def install_offline_execution(monkeypatch, args, *, after_trial=None):
    loaded, calls, predictors, new_fallback_calls = [], [], [], []
    new_fallback = object()

    class Model:
        def __init__(self, checkpoint, **kwargs):
            loaded.append((checkpoint, kwargs))

        def predict(self, row):
            assert "expected_operations" not in json.loads(row["state"])
            return {"action": "no_action", "confidence": 0.8}

    def make_new_fallback(config, *, invoice_local_fallback=False):
        new_fallback_calls.append((config, invoice_local_fallback))
        return new_fallback

    async def evaluate(predict, tasks, output, **kwargs):
        arm, attempt = kwargs["arm"], kwargs["attempt"]
        assert len(tasks) == 1
        assert kwargs["threshold"] == (0.91 if arm.startswith("released") else 0)
        assert kwargs["max_tool_calls"] == 16
        assert kwargs["decision_timeout"] == 30
        assert 0 < kwargs["task_timeout"] <= 180
        assert kwargs["allow_test"] is True
        assert kwargs["invoice_local_fallback"] is True
        assert kwargs["backend"] == "harbor"
        assert kwargs["wheel"] == args.wheel
        if arm.endswith("new"):
            assert kwargs["episode_runner"] is harbor.run_episode
            assert kwargs["fallback"] is new_fallback
        else:
            assert kwargs["episode_runner"].__module__ == "retained_boundary_runtime"
            assert kwargs["fallback"] == "old-fallback"
        if arm.startswith("released"):
            predictors.append(predict)
            assert predict({"target_invoice": "inv-1"}) == {
                "action": "no_action",
                "confidence": 0.8,
            }
        else:
            assert predict({}) == {"action": "defer", "confidence": 1}
        calls.append((tasks[0]["task_id"], arm, attempt))
        output.mkdir(parents=True, exist_ok=False)
        # A policy failure must remain in evidence and never trigger a quality retry.
        success = len(calls) % 13 != 0
        row = {
            "task_id": tasks[0]["task_id"],
            "arm": arm,
            "attempt": attempt,
            "completed_and_verified": success,
            "verifier_pass": success,
            "successful_unwanted_writes": 0,
            "missing_required_operations": int(not success),
            "successful_required_operations": int(success),
            "required_operations": 1,
            "llm_calls": 1,
            "llm_input_tokens": 10,
            "llm_output_tokens": 2,
        }
        path = output / "results.jsonl"
        path.write_text(json.dumps(row) + "\n")
        if after_trial:
            after_trial()
        return {"records_path": str(path)}

    monkeypatch.setattr(model, "DecisionModel", Model)
    monkeypatch.setattr(harbor, "llm_fallback", make_new_fallback)
    monkeypatch.setattr(boundary_evaluate, "evaluate_tasks", evaluate)
    return loaded, calls, predictors, new_fallback_calls


async def test_frozen_96_trial_interleaving_uses_exact_old_new_runtimes_and_one_model(
    boundary_args, monkeypatch
):
    loaded, calls, predictors, fallback_calls = install_offline_execution(
        monkeypatch, boundary_args
    )
    result = await boundary_evaluate.run(boundary_args)
    freeze = json.loads(boundary_args.freeze.read_text())
    expected = []
    for attempt in range(2):
        for index, task_id in enumerate(freeze["task_ids"]):
            shift = (index + attempt) % 4
            for arm in boundary_evaluate.ARMS[shift:] + boundary_evaluate.ARMS[:shift]:
                expected.append((task_id, arm, attempt))
    assert calls == expected and len(set(calls)) == 96
    assert loaded == [
        (boundary_args.checkpoint, {"device": "mps", "calibration": boundary_args.calibration})
    ]
    assert len(predictors) == 48 and all(p is predictors[0] for p in predictors)
    assert fallback_calls == [(boundary_args.fallback_config, True)]
    assert result["complete"] and result["error"] is None
    assert result["training_steps"] == result["calibration_fits"] == 0
    assert result["cohort_previously_exposed"] is True
    assert result["max_job_seconds"] == 3600
    assert result["max_llm_calls"] == result["max_tool_calls"] == 16
    assert result["fallback_timeout"] == 120 and result["task_timeout"] == 180
    assert all(metrics["tasks"] == 24 for metrics in result["arms"].values())
    rows = [
        json.loads(line)
        for line in (boundary_args.output / "results.jsonl").read_text().splitlines()
    ]
    assert [r["execution_position"] for r in rows] == list(range(96))
    assert sum(not r["completed_and_verified"] for r in rows) == 7
    assert (
        boundary_args.output / "source-harbor-old.py"
    ).read_bytes() == boundary_args.old_runtime.read_bytes()
    assert (boundary_args.output / "source-harbor.py").read_bytes() == Path(
        harbor.__file__
    ).read_bytes()


async def test_freeze_and_output_cannot_be_overwritten(boundary_args, monkeypatch):
    original = boundary_args.freeze.read_bytes()
    with pytest.raises(FileExistsError):
        boundary_evaluate.prepare(boundary_args)
    assert boundary_args.freeze.read_bytes() == original
    loaded, calls, _, _ = install_offline_execution(monkeypatch, boundary_args)
    boundary_args.output.mkdir()
    sentinel = boundary_args.output / "keep.txt"
    sentinel.write_text("existing run")
    with pytest.raises(FileExistsError):
        await boundary_evaluate.run(boundary_args)
    assert sentinel.read_text() == "existing run" and not loaded and not calls


def test_prepare_requires_explicit_exposure_acknowledgment(boundary_args):
    boundary_args.freeze.unlink()
    boundary_args.cohort_previously_exposed = False
    with pytest.raises(ValueError, match="previously exposed"):
        boundary_evaluate.prepare(boundary_args)


@pytest.mark.parametrize(
    "artifact", ["old_runtime", "old_authorization", "calibration", "checkpoint", "manifest"]
)
async def test_frozen_artifact_drift_stops_before_model_or_execution(
    boundary_args, monkeypatch, artifact
):
    loaded, calls, _, _ = install_offline_execution(monkeypatch, boundary_args)
    path = getattr(boundary_args, artifact)
    if artifact == "checkpoint":
        path /= "adapter.safetensors"
    path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises(ValueError, match="changed"):
        await boundary_evaluate.run(boundary_args)
    assert not boundary_args.output.exists() and not loaded and not calls


@pytest.mark.parametrize("stop", ["budget", "source_drift"])
async def test_midrun_budget_or_source_drift_preserves_partial_evidence(
    boundary_args, monkeypatch, tmp_path, stop
):
    clock = [0.0]
    monkeypatch.setattr(boundary_evaluate, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    source = tmp_path / "frozen-runtime-extra.py"
    source.write_text("unchanged source")
    freeze = json.loads(boundary_args.freeze.read_text())
    freeze["source_sha256"][str(source.resolve())] = boundary_evaluate.digest(source)
    boundary_args.freeze.write_text(json.dumps(freeze))

    def after_trial():
        if stop == "budget":
            clock[0] = 3540  # Cleanup reserve cannot become a new trial's execution budget.
        else:
            source.write_text("changed source")

    _, calls, _, _ = install_offline_execution(monkeypatch, boundary_args, after_trial=after_trial)
    with pytest.raises((TimeoutError, ValueError)):
        await boundary_evaluate.run(boundary_args)
    result = json.loads((boundary_args.output / "manifest.json").read_text())
    assert result["complete"] is False and result["error"]
    assert len(calls) == len(result["execution_order"]) == 1
    assert len((boundary_args.output / "results.jsonl").read_text().splitlines()) == 1


async def test_private_import_binding_survives_await_exception_and_cancellation(
    tmp_path, monkeypatch
):
    from experiments.decision_model import authorization

    auth_path = tmp_path / "old_authorization.py"
    auth_path.write_text("VALUE = 'retained'\n")
    runtime_path = tmp_path / "old_harbor.py"
    runtime_path.write_text(
        "from experiments.decision_model.authorization import VALUE as initial\n"
        "async def probe(pause, fail=False):\n"
        "    from experiments.decision_model.authorization import VALUE as before\n"
        "    await pause()\n"
        "    from experiments.decision_model.authorization import VALUE as after\n"
        "    if fail: raise RuntimeError(before + ':' + after)\n"
        "    return initial, before, after\n"
    )
    monkeypatch.setattr(
        boundary_evaluate, "OLD_RUNTIME_SHA256", boundary_evaluate.digest(runtime_path)
    )
    monkeypatch.setattr(
        boundary_evaluate, "OLD_AUTHORIZATION_SHA256", boundary_evaluate.digest(auth_path)
    )
    monkeypatch.setattr(authorization, "VALUE", "current", raising=False)
    modules = dict(sys.modules)
    original_import = builtins.__import__
    old = boundary_evaluate.load_old_runtime(runtime_path, auth_path)

    async def pause():
        await asyncio.sleep(0)
        assert sys.modules[boundary_evaluate.AUTHORIZATION_MODULE] is authorization
        assert authorization.VALUE == "current"

    assert await old.probe(pause) == ("retained",) * 3
    with pytest.raises(RuntimeError, match="retained:retained"):
        await old.probe(pause, fail=True)

    async def cancel():
        await pause()
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await old.probe(cancel)
    assert sys.modules == modules
    assert builtins.__import__ is original_import
    assert old._retained_authorization is not authorization


async def test_actual_retained_episode_uses_old_state_guard_and_completion(monkeypatch):
    if not Path(".artifacts/decision-model-fresh-fallback/final-harbor/source-harbor.py").is_file():
        pytest.skip("Optional historical runtime snapshots are not distributed")
    from experiments.decision_model import authorization, data, history

    # Import dependencies before the identity snapshot; neither private module is
    # registered in sys.modules, and the current authorization binding is untouched.
    assert data and history
    modules = dict(sys.modules)
    old = boundary_evaluate.load_old_runtime(
        Path(".artifacts/decision-model-fresh-fallback/final-harbor/source-harbor.py"),
        Path(".artifacts/decision-model-fresh-fallback/final-harbor/source-authorization.py"),
    )
    used = []
    for name in ("make_state", "authorize_action", "scoped_rule_action"):
        original = getattr(old._retained_authorization, name)

        def observe(*args, _name=name, _original=original, **kwargs):
            used.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(old._retained_authorization, name, observe)

        def current_must_not_run(*args, **kwargs):
            raise AssertionError("Retained episode imported current authorization")

        monkeypatch.setattr(authorization, name, current_must_not_run)
    invocations = []

    async def invoke(name, args):
        await asyncio.sleep(0)
        invocations.append(name)
        if name == "get_invoice":
            return {
                "ok": True,
                "invoice": {"id": "inv-123", "status": "paid"},
                "escalation_history": {"invoice_id": "inv-123", "complete": True, "records": []},
            }
        return {"ok": True, "refund": {"invoice_id": "inv-123", "success": True}}

    async def fallback(messages, call, counts):
        await asyncio.sleep(0)
        counts["llm_calls"] += 1
        result = await call("refund_invoice", {"invoice_id": "inv-123"})
        assert result["ok"]
        return {"final_response": "Refund completed"}

    result = await old.run_episode(
        request="Refund inv-123.",
        targets=["inv-123"],
        invoke=invoke,
        predict=lambda state: {"action": "defer", "confidence": 1.0},
        fallback=fallback,
        invoice_authorization=True,
        escalation_history=True,
        invoice_local_fallback=True,
    )
    assert result["policy_completed"] is True
    assert invocations == ["get_invoice", "refund_invoice"]
    assert set(used) == {"make_state", "authorize_action", "scoped_rule_action"}
    assert sys.modules == modules


async def test_old_authorization_identity_is_saved_separately(boundary_args, monkeypatch):
    install_offline_execution(monkeypatch, boundary_args)
    result = await boundary_evaluate.run(boundary_args)
    assert (boundary_args.output / "source-authorization-old.py").read_bytes() == (
        boundary_args.old_authorization.read_bytes()
    )
    for arm, contract in result["arm_contracts"].items():
        path = (
            boundary_args.old_authorization
            if arm.endswith("old")
            else Path("experiments/decision_model/authorization.py")
        )
        assert contract["authorization_sha256"] == boundary_evaluate.digest(path)
        assert contract["fallback_reason_defaults"] is True
        assert contract["low_confidence_authority_clarified"] is True
