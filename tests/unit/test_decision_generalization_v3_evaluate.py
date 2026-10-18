"""Offline orchestration contracts; no fresh-family contents, Docker, or provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.decision_model import generalization_v3_evaluate, harbor, model


@pytest.fixture
def generalization_args(tmp_path, monkeypatch):
    paths = {}
    for name in ("calibration", "fallback_config", "selection_manifest"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"threshold": 0.7, "selection_complete": True}))
        paths[name] = path
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"offline checkpoint fixture")
    paths["checkpoint"] = checkpoint
    paths["wheel"] = tmp_path / "fixture.whl"
    paths["wheel"].write_bytes(b"offline wheel fixture")
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
    prototypes = [{"id": f"offline-family-{index}"} for index in range(6)]
    (tmp_path / "prototype-specs.json").write_text(json.dumps(prototypes))
    (tmp_path / "fixture-validation.json").write_text(
        json.dumps(
            [
                {
                    "task_id": task_id,
                    "fixture_only": True,
                    "agent_or_policy_executed": False,
                    "expected_endpoint_accepted": True,
                    "missing_writes_rejected": True,
                }
                for task_id in tasks
            ]
        )
    )
    provenance = tmp_path / "synthetic_provenance.py"
    provenance.write_text("# Synthetic fixture, no held-out data\n")
    manifest.write_text(
        json.dumps(
            {
                "cohort_id": "billing-generalization-v3",
                "prototype_specs_sha256": hashlib.sha256(
                    json.dumps(prototypes, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "tasks": tasks,
                "cohort_previously_exposed": False,
                "pre_freeze_policy_rollouts": 0,
                "pre_freeze_model_inferences": 0,
                "pre_freeze_authorization_parser_checks": 0,
                "pre_freeze_teacher_collections": 0,
                "source_sha256": {str(provenance): generalization_v3_evaluate.digest(provenance)},
                "novelty_audit": {
                    "unique_new_structures": 6,
                    "source_task_counts": {
                        "fallback-fresh": 12,
                        "generalization-v1": 12,
                        "generalization-v2": 12,
                    },
                    "root_collisions": 0,
                    "structural_collisions": 0,
                    "normalized_instruction_collisions": 0,
                },
            }
        )
    )
    monkeypatch.setattr(generalization_v3_evaluate, "validate_task", lambda task: None)
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
    reference_path = tmp_path / "reference/final-freeze.json"
    reference_path.parent.mkdir()
    snapshots = reference_path.parent / "final-harbor"
    snapshots.mkdir()
    for source in generalization_v3_evaluate.final_source_paths():
        (snapshots / ("source-" + source.name)).write_bytes(source.read_bytes())
    baseline = generalization_v3_evaluate.build_final_freeze(
        manifest,
        released=checkpoint,
        tuned=checkpoint,
        released_calibration=paths["calibration"],
        tuned_calibration=paths["calibration"],
        fallback_config=paths["fallback_config"],
        wheel=paths["wheel"],
        selection_manifest=paths["selection_manifest"],
    )
    baseline["cohort_manifest_sha256"] = "older-cohort-fixture"
    reference_path.write_text(json.dumps(baseline))
    args.reference_freeze = reference_path
    args.runtime_lock = tmp_path / "runtime-lock.json"
    args.runtime_lock.write_text(
        json.dumps(
            {
                "regression_passed": True,
                "max_combined_paid_seconds": 3600,
                "regression_elapsed_seconds": 900,
                "source_sha256": {
                    str(p.resolve()): generalization_v3_evaluate.digest(p)
                    for p in generalization_v3_evaluate.final_source_paths()
                },
                "regression_evidence_sha256": {
                    str(provenance): generalization_v3_evaluate.digest(provenance)
                },
            }
        )
    )
    monkeypatch.setattr(
        generalization_v3_evaluate,
        "REFERENCE_FREEZE_SHA256",
        generalization_v3_evaluate.digest(reference_path),
    )
    monkeypatch.setattr(
        generalization_v3_evaluate,
        "additional_sources",
        lambda: [Path(generalization_v3_evaluate.__file__)],
    )
    generalization_v3_evaluate.prepare(args)
    return args


def install_offline_execution(monkeypatch, args, *, after_trial=None):
    loaded, calls, predictors, new_fallback_calls = [], [], [], []
    new_fallback = object()

    class Model:
        def __init__(self, checkpoint, **kwargs):
            assert generalization_v3_evaluate.access_path(args.manifest).exists()
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
        assert kwargs["threshold"] == (0.7 if arm == "released" else 0)
        assert kwargs["max_tool_calls"] == 16
        assert kwargs["decision_timeout"] == 30
        assert 0 < kwargs["task_timeout"] <= 180
        assert kwargs["allow_test"] is True
        assert kwargs["invoice_local_fallback"] is True
        assert kwargs["backend"] == "harbor"
        assert kwargs["wheel"] == args.wheel
        assert kwargs["episode_runner"] is harbor.run_episode
        assert kwargs["fallback"] is new_fallback
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
    monkeypatch.setattr(generalization_v3_evaluate, "evaluate_tasks", evaluate)
    return loaded, calls, predictors, new_fallback_calls


async def test_frozen_48_trial_interleaving_uses_common_runtime_fallback_and_one_model(
    generalization_args, monkeypatch
):
    loaded, calls, predictors, fallback_calls = install_offline_execution(
        monkeypatch, generalization_args
    )
    result = await generalization_v3_evaluate.run(generalization_args)
    freeze = json.loads(generalization_args.freeze.read_text())
    expected = []
    for attempt in range(2):
        for index, task_id in enumerate(freeze["task_ids"]):
            shift = (index + attempt) % 2
            for arm in (
                generalization_v3_evaluate.ARMS[shift:] + generalization_v3_evaluate.ARMS[:shift]
            ):
                expected.append((task_id, arm, attempt))
    assert calls == expected and len(set(calls)) == 48
    assert loaded == [
        (
            generalization_args.checkpoint,
            {"device": "mps", "calibration": generalization_args.calibration},
        )
    ]
    assert len(predictors) == 24 and all(p is predictors[0] for p in predictors)
    assert fallback_calls == [(generalization_args.fallback_config, True)]
    assert result["complete"] and result["error"] is None
    assert result["training_steps"] == result["calibration_fits"] == 0
    assert result["cohort_previously_exposed"] is False
    assert result["max_job_seconds"] == 2700
    assert result["max_llm_calls"] == result["max_tool_calls"] == 16
    assert result["fallback_timeout"] == 120 and result["task_timeout"] == 180
    assert all(metrics["tasks"] == 24 for metrics in result["arms"].values())
    rows = [
        json.loads(line)
        for line in (generalization_args.output / "results.jsonl").read_text().splitlines()
    ]
    assert [r["execution_position"] for r in rows] == list(range(48))
    assert sum(not r["completed_and_verified"] for r in rows) == 3
    assert (generalization_args.output / "source-harbor.py").read_bytes() == Path(
        harbor.__file__
    ).read_bytes()
    claim = generalization_v3_evaluate.access_path(generalization_args.manifest)
    assert claim.exists()
    assert (
        generalization_args.output / "evaluation-access.json"
    ).read_bytes() == claim.read_bytes()


async def test_freeze_and_output_cannot_be_overwritten(generalization_args, monkeypatch):
    original = generalization_args.freeze.read_bytes()
    with pytest.raises(FileExistsError):
        generalization_v3_evaluate.prepare(generalization_args)
    assert generalization_args.freeze.read_bytes() == original
    loaded, calls, _, _ = install_offline_execution(monkeypatch, generalization_args)
    generalization_args.output.mkdir()
    sentinel = generalization_args.output / "keep.txt"
    sentinel.write_text("existing run")
    with pytest.raises(FileExistsError):
        await generalization_v3_evaluate.run(generalization_args)
    assert sentinel.read_text() == "existing run" and not loaded and not calls


@pytest.mark.parametrize(
    "key,value",
    [
        ("cohort_previously_exposed", True),
        ("pre_freeze_policy_rollouts", 1),
        ("pre_freeze_model_inferences", 1),
        ("pre_freeze_authorization_parser_checks", 1),
        ("pre_freeze_teacher_collections", 1),
        ("unique_new_structures", 5),
        ("root_collisions", 1),
        ("structural_collisions", 1),
        ("normalized_instruction_collisions", 1),
    ],
)
def test_prepare_rejects_exposure_or_colliding_family_provenance(generalization_args, key, value):
    cohort = json.loads(generalization_args.manifest.read_text())
    (cohort["novelty_audit"] if key in cohort["novelty_audit"] else cohort)[key] = value
    generalization_args.manifest.write_text(json.dumps(cohort))
    generalization_args.freeze.unlink()
    with pytest.raises(ValueError, match="provenance"):
        generalization_v3_evaluate.prepare(generalization_args)
    assert not generalization_args.freeze.exists() and not generalization_args.output.exists()


@pytest.mark.parametrize(
    "field", ["pre_freeze_authorization_parser_checks", "pre_freeze_teacher_collections"]
)
def test_missing_parser_or_teacher_exposure_declaration_is_not_zero(generalization_args, field):
    cohort = json.loads(generalization_args.manifest.read_text())
    del cohort[field]
    generalization_args.manifest.write_text(json.dumps(cohort))
    generalization_args.freeze.unlink()
    with pytest.raises(ValueError, match="provenance"):
        generalization_v3_evaluate.prepare(generalization_args)
    assert not generalization_args.freeze.exists()


@pytest.mark.parametrize("mutation", ["empty_lock", "empty_evidence", "missing_evidence", "failed"])
def test_prepare_requires_positive_regression_with_nonempty_evidence(generalization_args, mutation):
    lock = json.loads(generalization_args.runtime_lock.read_text())
    if mutation == "empty_lock":
        lock = {}
    elif mutation == "empty_evidence":
        lock["regression_evidence_sha256"] = {}
    elif mutation == "missing_evidence":
        del lock["regression_evidence_sha256"]
    else:
        lock["regression_passed"] = False
    generalization_args.runtime_lock.write_text(json.dumps(lock))
    generalization_args.freeze.unlink()
    with pytest.raises(ValueError, match="completed regression evidence"):
        generalization_v3_evaluate.prepare(generalization_args)
    assert not generalization_args.freeze.exists() and not generalization_args.output.exists()


def test_prepare_rejects_changed_regression_evidence_even_with_pass_flag(generalization_args):
    lock = json.loads(generalization_args.runtime_lock.read_text())
    evidence = Path(next(iter(lock["regression_evidence_sha256"])))
    evidence.write_bytes(evidence.read_bytes() + b"\n# changed regression evidence\n")
    generalization_args.freeze.unlink()
    with pytest.raises(ValueError, match="Regression evidence changed"):
        generalization_v3_evaluate.prepare(generalization_args)
    assert not generalization_args.freeze.exists()


async def test_run_rejects_runtime_lock_tamper_before_access_or_model(
    generalization_args, monkeypatch
):
    loaded, calls, _, _ = install_offline_execution(monkeypatch, generalization_args)
    generalization_args.runtime_lock.write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed"):
        await generalization_v3_evaluate.run(generalization_args)
    assert not loaded and not calls and not generalization_args.output.exists()
    assert not generalization_v3_evaluate.access_path(generalization_args.manifest).exists()


async def test_run_rechecks_external_regression_evidence_before_access_or_model(
    generalization_args, monkeypatch, tmp_path
):
    # Separate external regression evidence from frozen source provenance so the
    # reference gate itself, not the generic source hash check, must reject drift.
    external = tmp_path / "external-regression.json"
    external.write_text('{"passed": true}')
    lock = json.loads(generalization_args.runtime_lock.read_text())
    lock["regression_evidence_sha256"] = {
        str(external): generalization_v3_evaluate.digest(external)
    }
    generalization_args.runtime_lock.write_text(json.dumps(lock))
    generalization_args.freeze.unlink()
    generalization_v3_evaluate.prepare(generalization_args)
    loaded, calls, _, _ = install_offline_execution(monkeypatch, generalization_args)
    external.write_text('{"passed": false}')
    with pytest.raises(ValueError, match="Regression evidence changed"):
        await generalization_v3_evaluate.run(generalization_args)
    assert not loaded and not calls and not generalization_args.output.exists()
    assert not generalization_v3_evaluate.access_path(generalization_args.manifest).exists()


@pytest.mark.parametrize("artifact", ["calibration", "checkpoint", "manifest"])
async def test_frozen_artifact_drift_stops_before_model_or_execution(
    generalization_args, monkeypatch, artifact
):
    loaded, calls, _, _ = install_offline_execution(monkeypatch, generalization_args)
    path = getattr(generalization_args, artifact)
    if artifact == "checkpoint":
        path /= "adapter.safetensors"
    path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises(ValueError, match="changed"):
        await generalization_v3_evaluate.run(generalization_args)
    assert not generalization_args.output.exists() and not loaded and not calls


@pytest.mark.parametrize("stop", ["budget", "source_drift"])
async def test_midrun_budget_or_source_drift_preserves_partial_evidence(
    generalization_args, monkeypatch, tmp_path, stop
):
    clock = [0.0]
    monkeypatch.setattr(
        generalization_v3_evaluate, "time", SimpleNamespace(perf_counter=lambda: clock[0])
    )
    source = tmp_path / "frozen-runtime-extra.py"
    source.write_text("unchanged source")
    freeze = json.loads(generalization_args.freeze.read_text())
    freeze["source_sha256"][str(source.resolve())] = generalization_v3_evaluate.digest(source)
    generalization_args.freeze.write_text(json.dumps(freeze))

    def after_trial():
        if stop == "budget":
            clock[0] = 3540  # Cleanup reserve cannot become a new trial's execution budget.
        else:
            source.write_text("changed source")

    _, calls, _, _ = install_offline_execution(
        monkeypatch, generalization_args, after_trial=after_trial
    )
    with pytest.raises((TimeoutError, ValueError)):
        await generalization_v3_evaluate.run(generalization_args)
    result = json.loads((generalization_args.output / "manifest.json").read_text())
    assert result["complete"] is False and result["error"]
    assert len(calls) == len(result["execution_order"]) == 1
    assert len((generalization_args.output / "results.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("mutation", ["historical_id", "missing_prior_cohort", "provenance_drift"])
def test_cannot_relabel_historical_manifest_or_omit_prior_exposure(generalization_args, mutation):
    payload = json.loads(generalization_args.manifest.read_text())
    if mutation == "historical_id":
        payload["cohort_id"] = "billing-invoice-local-fresh-v1"
    elif mutation == "missing_prior_cohort":
        del payload["novelty_audit"]["source_task_counts"]["fallback-fresh"]
    else:
        Path(next(iter(payload["source_sha256"]))).write_text("source drift")
    generalization_args.manifest.write_text(json.dumps(payload))
    generalization_args.freeze.unlink()
    with pytest.raises(ValueError):
        generalization_v3_evaluate.prepare(generalization_args)
    assert not generalization_args.freeze.exists()


async def test_access_claim_prevents_repeated_fresh_runs_or_refreezing(
    generalization_args, monkeypatch
):
    loaded, calls, _, _ = install_offline_execution(monkeypatch, generalization_args)
    await generalization_v3_evaluate.run(generalization_args)
    original_output = generalization_args.output
    generalization_args.output = original_output.with_name("another-fresh-run")
    with pytest.raises(ValueError, match="access already claimed"):
        await generalization_v3_evaluate.run(generalization_args)
    assert len(loaded) == 1 and len(calls) == 48 and not generalization_args.output.exists()
    generalization_args.freeze = generalization_args.freeze.with_name("another-freeze.json")
    with pytest.raises(ValueError, match="access already claimed"):
        generalization_v3_evaluate.prepare(generalization_args)


async def test_failed_model_start_retains_access_claim_and_incomplete_manifest(
    generalization_args, monkeypatch
):
    def fail(*args, **kwargs):
        assert generalization_v3_evaluate.access_path(generalization_args.manifest).exists()
        raise RuntimeError("synthetic model initialization failure")

    monkeypatch.setattr(model, "DecisionModel", fail)
    with pytest.raises(RuntimeError, match="initialization failure"):
        await generalization_v3_evaluate.run(generalization_args)
    manifest = json.loads((generalization_args.output / "manifest.json").read_text())
    assert not manifest["complete"] and manifest["execution_order"] == []
    assert "initialization failure" in manifest["error"]
    assert generalization_v3_evaluate.access_path(generalization_args.manifest).exists()


@pytest.mark.parametrize("mutation", ["retained_source", "reference", "threshold"])
def test_prepare_rejects_drift_from_prior_repaired_experiment(generalization_args, mutation):
    generalization_args.freeze.unlink()
    if mutation == "retained_source":
        snapshot = generalization_args.reference_freeze.parent / "final-harbor/source-harbor.py"
        snapshot.write_bytes(snapshot.read_bytes() + b"\n# changed\n")
    elif mutation == "reference":
        generalization_args.reference_freeze.write_bytes(b"{}")
    else:
        payload = json.loads(generalization_args.calibration.read_text())
        payload["threshold"] = 0.8
        generalization_args.calibration.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        generalization_v3_evaluate.prepare(generalization_args)
    assert not generalization_args.freeze.exists()


@pytest.mark.parametrize("evidence_file", ["fixture-validation.json", "prototype-specs.json"])
async def test_authored_evidence_hashes_are_frozen_copied_and_checked_before_model(
    generalization_args, monkeypatch, evidence_file
):
    freeze = json.loads(generalization_args.freeze.read_text())
    path = generalization_args.manifest.parent / evidence_file
    assert freeze["cohort_evidence_sha256"][
        str(path.resolve())
    ] == generalization_v3_evaluate.digest(path)
    loaded, calls, _, _ = install_offline_execution(monkeypatch, generalization_args)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="cohort evidence changed"):
        await generalization_v3_evaluate.run(generalization_args)
    assert not loaded and not calls and not generalization_args.output.exists()
    assert not generalization_v3_evaluate.access_path(generalization_args.manifest).exists()


def test_fixture_policy_exposure_cannot_be_frozen_as_fresh(generalization_args):
    generalization_args.freeze.unlink()
    path = generalization_args.manifest.parent / "fixture-validation.json"
    fixtures = json.loads(path.read_text())
    fixtures[0]["agent_or_policy_executed"] = True
    path.write_text(json.dumps(fixtures))
    with pytest.raises(ValueError, match="policy exposure"):
        generalization_v3_evaluate.prepare(generalization_args)


async def test_authored_evidence_is_copied_verbatim(generalization_args, monkeypatch):
    install_offline_execution(monkeypatch, generalization_args)
    await generalization_v3_evaluate.run(generalization_args)
    for filename in ("fixture-validation.json", "prototype-specs.json"):
        assert (generalization_args.output / filename).read_bytes() == (
            generalization_args.manifest.parent / filename
        ).read_bytes()


@pytest.mark.parametrize("elapsed", [-1, 3535, 3600, "900", None])
def test_combined_execution_allowance_cannot_reset_or_overrun(tmp_path, elapsed):
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"max_combined_paid_seconds": 3600, "regression_elapsed_seconds": elapsed})
    )
    with pytest.raises(ValueError, match="execution budget"):
        generalization_v3_evaluate.remaining_paid_budget(lock)


def test_remaining_allowance_subtracts_regression_and_rounds_down(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"max_combined_paid_seconds": 3600, "regression_elapsed_seconds": 1234.6})
    )
    assert generalization_v3_evaluate.remaining_paid_budget(lock) == 2365
