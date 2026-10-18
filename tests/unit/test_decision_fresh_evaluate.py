"""Offline orchestration contracts; no fresh-family contents, Docker, or provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.decision_model import fresh_evaluate, harbor, model


@pytest.fixture
def fresh_args(tmp_path, monkeypatch):
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
        "def llm_fallback(config):\n    return 'old-fallback'\n"
        "async def run_episode(**kwargs):\n    raise AssertionError('must remain mocked')\n"
    )
    paths["old_runtime"] = old
    monkeypatch.setattr(fresh_evaluate, "OLD_RUNTIME_SHA256", fresh_evaluate.digest(old))
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
    monkeypatch.setattr(fresh_evaluate, "validate_task", lambda task: None)
    for name in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "BILLING_MODEL_TEMPERATURE",
        "BILLING_MODEL_MAX_TOKENS",
        "BILLING_MODEL_TOTAL_ATTEMPTS",
    ):
        monkeypatch.setenv(name, "offline")
    args = argparse.Namespace(
        manifest=manifest, freeze=tmp_path / "freeze.json", output=tmp_path / "result", **paths
    )
    fresh_evaluate.prepare(args)
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
            assert kwargs["episode_runner"].__module__ == "retained_invoice_local_runtime"
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
    monkeypatch.setattr(fresh_evaluate, "evaluate_tasks", evaluate)
    return loaded, calls, predictors, new_fallback_calls


async def test_frozen_96_trial_interleaving_uses_exact_old_new_runtimes_and_one_model(
    fresh_args, monkeypatch
):
    loaded, calls, predictors, fallback_calls = install_offline_execution(monkeypatch, fresh_args)
    result = await fresh_evaluate.run(fresh_args)
    freeze = json.loads(fresh_args.freeze.read_text())
    expected = []
    for attempt in range(2):
        for index, task_id in enumerate(freeze["task_ids"]):
            shift = (index + attempt) % 4
            for arm in fresh_evaluate.ARMS[shift:] + fresh_evaluate.ARMS[:shift]:
                expected.append((task_id, arm, attempt))
    assert calls == expected and len(set(calls)) == 96
    assert loaded == [
        (fresh_args.checkpoint, {"device": "mps", "calibration": fresh_args.calibration})
    ]
    assert len(predictors) == 48 and all(p is predictors[0] for p in predictors)
    assert fallback_calls == [(fresh_args.fallback_config, True)]
    assert result["complete"] and result["error"] is None
    assert result["training_steps"] == result["calibration_fits"] == 0
    assert result["cohort_previously_exposed"] is False
    assert result["max_job_seconds"] == 3600
    assert result["max_llm_calls"] == result["max_tool_calls"] == 16
    assert result["fallback_timeout"] == 120 and result["task_timeout"] == 180
    assert all(metrics["tasks"] == 24 for metrics in result["arms"].values())
    rows = [
        json.loads(line) for line in (fresh_args.output / "results.jsonl").read_text().splitlines()
    ]
    assert [r["execution_position"] for r in rows] == list(range(96))
    assert sum(not r["completed_and_verified"] for r in rows) == 7
    assert (
        fresh_args.output / "source-harbor-old.py"
    ).read_bytes() == fresh_args.old_runtime.read_bytes()
    assert (fresh_args.output / "source-harbor.py").read_bytes() == Path(
        harbor.__file__
    ).read_bytes()


async def test_freeze_and_output_cannot_be_overwritten(fresh_args, monkeypatch):
    original = fresh_args.freeze.read_bytes()
    with pytest.raises(FileExistsError):
        fresh_evaluate.prepare(fresh_args)
    assert fresh_args.freeze.read_bytes() == original
    loaded, calls, _, _ = install_offline_execution(monkeypatch, fresh_args)
    fresh_args.output.mkdir()
    sentinel = fresh_args.output / "keep.txt"
    sentinel.write_text("existing run")
    with pytest.raises(FileExistsError):
        await fresh_evaluate.run(fresh_args)
    assert sentinel.read_text() == "existing run" and not loaded and not calls


@pytest.mark.parametrize(
    "key,value",
    [
        ("cohort_previously_exposed", True),
        ("pre_freeze_policy_rollouts", 1),
        ("pre_freeze_model_inferences", 1),
        ("unique_new_structures", 5),
        ("root_collisions", 1),
        ("structural_collisions", 1),
        ("normalized_instruction_collisions", 1),
    ],
)
def test_prepare_rejects_exposure_or_colliding_family_provenance(fresh_args, key, value):
    cohort = json.loads(fresh_args.manifest.read_text())
    (cohort["novelty_audit"] if key in cohort["novelty_audit"] else cohort)[key] = value
    fresh_args.manifest.write_text(json.dumps(cohort))
    fresh_args.freeze.unlink()
    with pytest.raises(ValueError, match="provenance audit"):
        fresh_evaluate.prepare(fresh_args)
    assert not fresh_args.freeze.exists() and not fresh_args.output.exists()


@pytest.mark.parametrize("artifact", ["old_runtime", "calibration", "checkpoint", "manifest"])
async def test_frozen_artifact_drift_stops_before_model_or_execution(
    fresh_args, monkeypatch, artifact
):
    loaded, calls, _, _ = install_offline_execution(monkeypatch, fresh_args)
    path = getattr(fresh_args, artifact)
    if artifact == "checkpoint":
        path /= "adapter.safetensors"
    path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises(ValueError, match="changed"):
        await fresh_evaluate.run(fresh_args)
    assert not fresh_args.output.exists() and not loaded and not calls


@pytest.mark.parametrize("stop", ["budget", "source_drift"])
async def test_midrun_budget_or_source_drift_preserves_partial_evidence(
    fresh_args, monkeypatch, tmp_path, stop
):
    clock = [0.0]
    monkeypatch.setattr(fresh_evaluate, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    source = tmp_path / "frozen-runtime-extra.py"
    source.write_text("unchanged source")
    freeze = json.loads(fresh_args.freeze.read_text())
    freeze["source_sha256"][str(source.resolve())] = fresh_evaluate.digest(source)
    fresh_args.freeze.write_text(json.dumps(freeze))

    def after_trial():
        if stop == "budget":
            clock[0] = 3540  # Cleanup reserve cannot become a new trial's execution budget.
        else:
            source.write_text("changed source")

    _, calls, _, _ = install_offline_execution(monkeypatch, fresh_args, after_trial=after_trial)
    with pytest.raises((TimeoutError, ValueError)):
        await fresh_evaluate.run(fresh_args)
    result = json.loads((fresh_args.output / "manifest.json").read_text())
    assert result["complete"] is False and result["error"]
    assert len(calls) == len(result["execution_order"]) == 1
    assert len((fresh_args.output / "results.jsonl").read_text().splitlines()) == 1
