"""Synthetic orchestration fixtures only: never inspect actual held-out task behavior."""

from __future__ import annotations

import argparse
import hashlib
import json

import pytest

from experiments.decision_model import focused_evaluate as evaluate
from experiments.decision_model import focused_report as report
from experiments.decision_model import harbor, model


@pytest.fixture
def args(tmp_path, monkeypatch):
    paths = {}
    for arm in ("released", "tuned"):
        checkpoint = tmp_path / arm
        checkpoint.mkdir()
        (checkpoint / "head.safetensors").write_bytes(arm.encode())
        paths[arm] = checkpoint
        calibration = tmp_path / (arm + "-calibration.json")
        calibration.write_text(json.dumps({"threshold": 0.7 if arm == "released" else 0.6}))
        paths[arm + "_calibration"] = calibration
    for name, value in (
        (
            "selection_manifest",
            {
                "selection_complete": True,
                "selected_checkpoint_role": "trained",
                "optimizer_steps": 20,
            },
        ),
        ("fallback_config", {}),
        ("wheel", {}),
    ):
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_text(json.dumps(value))
    tasks = {
        f"fixture-{family}-{variant}": {
            "task_id": f"fixture-{family}-{variant}",
            "split": "test",
            "composition_root": f"fixture-{family}",
            "task_path": "synthetic-no-access",
            "expected_operations": {},
        }
        for family in range(6)
        for variant in range(2)
    }
    prototypes = [{"id": f"fixture-{i}"} for i in range(6)]
    (tmp_path / "prototype-specs.json").write_text(json.dumps(prototypes))
    (tmp_path / "fixture-validation.json").write_text(
        json.dumps(
            [
                dict(
                    task_id=task_id,
                    fixture_only=True,
                    agent_or_policy_executed=False,
                    expected_endpoint_accepted=True,
                    missing_writes_rejected=True,
                )
                for task_id in tasks
            ]
        )
    )
    manifest = tmp_path / "cohort.json"
    manifest.write_text(
        json.dumps(
            dict(
                tasks=tasks,
                prototype_specs_sha256=hashlib.sha256(
                    json.dumps(prototypes, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                cohort_previously_exposed=False,
                pre_freeze_policy_rollouts=0,
                pre_freeze_model_inferences=0,
                pre_freeze_authorization_parser_checks=0,
                pre_freeze_teacher_collections=0,
                novelty_audit=dict(
                    root_collisions=0, structural_collisions=0, normalized_instruction_collisions=0
                ),
            )
        )
    )
    monkeypatch.setattr(evaluate, "validate_task", lambda task: None)
    monkeypatch.setattr(report, "validate_task", lambda task: None)
    monkeypatch.setattr(
        evaluate,
        "tensor_changes",
        lambda *args: {
            "changed_elements": {"adapter": 1, "head": 1},
            "absolute_delta_sum": {"adapter": 0.1, "head": 0.2},
        },
    )
    return argparse.Namespace(
        manifest=manifest,
        freeze=tmp_path / "freeze.json",
        output=tmp_path / "run",
        device="mps",
        max_job_seconds=3600,
        **paths,
    )


def install_execution(args, monkeypatch, *, fail_load=False):
    loaded, calls = [], []

    class Model:
        def __init__(self, checkpoint, **kwargs):
            assert evaluate.access_path(args.manifest).exists()
            loaded.append((checkpoint, kwargs))
            if fail_load:
                raise RuntimeError("synthetic initialization failure")

        def predict(self, row):
            assert "expected_operations" not in json.loads(row["state"])
            return {"action": "no_action", "confidence": 0.9}

    async def trial(predict, tasks, output, **kwargs):
        arm, attempt = kwargs["arm"], kwargs["attempt"]
        assert kwargs["threshold"] == {"released": 0.7, "tuned": 0.6, "llm": 0}[arm]
        assert kwargs["backend"] == "harbor" and kwargs["invoice_local_fallback"] is True
        assert kwargs["max_tool_calls"] == 16 and kwargs["allow_test"] is True
        assert 0 < kwargs["task_timeout"] <= 180
        assert kwargs["decision_timeout"] == 30
        assert predict({"target_invoice": "inv-1"})["action"] == (
            "defer" if arm == "llm" else "no_action"
        )
        task_id = tasks[0]["task_id"]
        calls.append((task_id, arm, attempt))
        row = dict(
            task_id=task_id,
            arm=arm,
            attempt=attempt,
            completed_and_verified=True,
            verifier_pass=True,
            policy_completed=True,
            successful_unwanted_writes=0,
            missing_required_operations=0,
            successful_required_operations=0,
            missing_required_operation_details=[],
            required_operations=0,
            llm_calls=1,
            llm_input_tokens=10,
            llm_output_tokens=2,
            tool_observations=[],
            decisions=[],
            tool_call_count=0,
            invoice_local_fallback=True,
            cleanup="deleted-owned-environment",
            tool_evidence_complete=True,
            total_wall_ms=100,
            verifier={"rewards": {"reward": 1.0}, "diagnostics": {}},
        )
        output.mkdir(parents=True)
        native = output / task_id / "verifier"
        native.mkdir(parents=True)
        (native / "reward.json").write_text(json.dumps(row["verifier"]["rewards"]))
        (native / "diagnostics.json").write_text("{}")
        path = output / "results.jsonl"
        path.write_text(json.dumps(row) + "\n")
        return {"records_path": str(path)}

    monkeypatch.setattr(model, "DecisionModel", Model)
    monkeypatch.setattr(harbor, "llm_fallback", lambda *a, **k: object())
    monkeypatch.setattr(evaluate, "evaluate_tasks", trial)
    return loaded, calls


async def test_three_arm_run_and_complete_native_audit(args, monkeypatch):
    freeze = evaluate.prepare(args)
    loaded, calls = install_execution(args, monkeypatch)
    result = await evaluate.run(args)
    assert result["complete"] is True
    assert len(loaded) == 2 and len(calls) == 72
    assert calls == evaluate.planned_order(freeze["task_ids"])
    assert all(value["tasks"] == 24 for value in result["arms"].values())
    audit = report.audit_run(args.output, expected_tasks=12, exposed=False)
    assert audit["passed"], audit["failures"]
    assert audit["native_evidence_files_verified"] == 216
    metrics = report.report(args.output, args.output.parent / "report")
    assert metrics["paired"]["tuned_vs_released"]["pairs"] == 24
    assert metrics["target_met"] is False  # zero baseline fallbacks cannot claim a reduction


async def test_failed_initialization_claim_cannot_be_retried(args, monkeypatch):
    evaluate.prepare(args)
    loaded, calls = install_execution(args, monkeypatch, fail_load=True)
    with pytest.raises(RuntimeError, match="initialization"):
        await evaluate.run(args)
    assert loaded and not calls and evaluate.access_path(args.manifest).exists()
    args.output = args.output.parent / "retry"
    with pytest.raises(ValueError, match="already claimed"):
        await evaluate.run(args)


@pytest.mark.parametrize(
    "key,value",
    [
        ("selected_checkpoint_role", "released_baseline"),
        ("optimizer_steps", 0),
        ("optimizer_steps", None),
        ("selection_complete", False),
    ],
)
def test_epoch_zero_and_incomplete_selection_rejected(args, key, value):
    selection = json.loads(args.selection_manifest.read_text())
    selection[key] = value
    args.selection_manifest.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        evaluate.prepare(args)
    assert not args.freeze.exists()


@pytest.mark.parametrize(
    "key",
    [
        "pre_freeze_model_inferences",
        "pre_freeze_policy_rollouts",
        "pre_freeze_authorization_parser_checks",
        "pre_freeze_teacher_collections",
    ],
)
def test_test_exposure_refused(args, key):
    cohort = json.loads(args.manifest.read_text())
    cohort[key] = 1
    args.manifest.write_text(json.dumps(cohort))
    with pytest.raises(ValueError, match="provenance"):
        evaluate.prepare(args)


async def test_frozen_artifact_change_stops_before_model_access(args, monkeypatch):
    evaluate.prepare(args)
    loaded, calls = install_execution(args, monkeypatch)
    args.tuned_calibration.write_text('{"threshold": 0.1}')
    with pytest.raises(ValueError, match="changed"):
        await evaluate.run(args)
    assert not loaded and not calls and not evaluate.access_path(args.manifest).exists()


async def test_native_tampering_is_integrity_failure(args, monkeypatch):
    evaluate.prepare(args)
    install_execution(args, monkeypatch)
    await evaluate.run(args)
    reward = next((args.output / "trials").glob("*/*/verifier/reward.json"))
    reward.write_text('{"reward":0}')
    audit = report.audit_run(args.output, expected_tasks=12, exposed=False)
    assert not audit["passed"] and any("native evidence agrees" in f for f in audit["failures"])


def test_pairing_never_claims_lift_from_incomplete_coverage():
    rows = [
        dict(
            task_id="one",
            attempt=0,
            arm=arm,
            completed_and_verified=True,
            fallback_count=1 if arm == "released" else 0,
            llm_calls=1,
            successful_unwanted_writes=0,
        )
        for arm in ("released", "tuned")
    ]
    result = report.compare(rows, "tuned", "released")
    assert result["fallback_reduction_fraction"] == 1 and not result["target_met"]


def test_actual_tensor_updates_required(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    paths = [tmp_path / name for name in ("released", "tuned")]
    for path in paths:
        (path / "lora").mkdir(parents=True)
        safetensors.save_file({"weight": torch.zeros(2)}, str(path / "head.safetensors"))
        safetensors.save_file({"weight": torch.zeros(2)}, str(path / "lora/adapter.safetensors"))
    with pytest.raises(ValueError, match="nonzero"):
        evaluate.tensor_changes(*paths)
    safetensors.save_file({"weight": torch.ones(2)}, str(paths[1] / "head.safetensors"))
    with pytest.raises(ValueError, match="nonzero"):
        evaluate.tensor_changes(*paths)
    safetensors.save_file({"weight": torch.ones(2)}, str(paths[1] / "lora/adapter.safetensors"))
    assert evaluate.tensor_changes(*paths)["changed_elements"] == {"adapter": 2, "head": 2}


def test_exported_head_and_save_pretrained_head_formats(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    released, tuned = tmp_path / "released", tmp_path / "trained"
    for path in (released, tuned):
        (path / "lora").mkdir(parents=True)
    safetensors.save_file({"weight": torch.zeros(2)}, str(released / "head.safetensors"))
    torch.save({"weight": torch.zeros(2)}, tuned / "slot_head.pt")
    for path, value in ((released, 0), (tuned, 1)):
        safetensors.save_file(
            {"lora_weight": torch.full((2,), float(value))},
            str(path / "lora/adapter_model.safetensors"),
        )
    with pytest.raises(ValueError, match="nonzero"):
        evaluate.tensor_changes(released, tuned)
    torch.save({"weight": torch.ones(2)}, tuned / "slot_head.pt")
    assert evaluate.tensor_changes(released, tuned)["changed_elements"] == {"head": 2, "adapter": 2}
    torch.save({"wrong_key": torch.ones(2)}, tuned / "slot_head.pt")
    with pytest.raises(ValueError, match="tensor names"):
        evaluate.tensor_changes(released, tuned)
    safetensors.save_file({"weight": torch.ones(2)}, str(tuned / "head.safetensors"))
    with pytest.raises(ValueError, match="Ambiguous"):
        evaluate.tensor_changes(released, tuned)
