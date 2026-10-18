"""Whole-task quality, abstention and option permutations retain distinct meanings."""

import json

from experiments.decision_model.next_train import permutation_check, task_score


def metrics(**kwargs):
    return {
        "selection_eligible": True,
        "successful_unwanted_writes": 0,
        "accepted_prohibited_proposals": 0,
        "completed_and_verified": 2,
        "missing_required_operations": 0,
        "raw_prohibited_proposals": 0,
        "proposal_count": 8,
        **kwargs,
    }


def test_task_selection_prioritizes_safety_then_real_completion():
    assert task_score(metrics()) > task_score(metrics(successful_unwanted_writes=1), 1)
    assert task_score(metrics()) > task_score(metrics(accepted_prohibited_proposals=1), 1)
    assert task_score(metrics(), 0.5) > task_score(metrics(completed_and_verified=0), 1)
    assert task_score(metrics()) > task_score(metrics(missing_required_operations=2))
    assert task_score(metrics()) > task_score(metrics(selection_eligible=False), 1)


def test_permutation_check_remaps_gold_without_leaking_order(tmp_path):
    class Predictor:
        def predict(self, row):
            assert row["options"][row["label"]][0] == row["label_action"]
            return {"action": row["options"][0][0], "error": None}

    row = {
        "example_id": "one",
        "group_id": "family",
        "label_action": "refund",
        "options": [[name, name] for name in ("refund", "escalate", "no_action", "defer")],
    }
    path = tmp_path / "permutations.json"
    permutation_check(Predictor(), [row], path)
    report = json.loads(path.read_text())
    assert report["examples"] == 1 and report["consistent_examples"] == 0
    assert len({tuple(p["option_order"]) for p in report["results"][0]["predictions"]}) == 24


def test_candidate_predictor_records_in_memory_identity():
    from types import SimpleNamespace

    from experiments.decision_model.next_train import state_predictor

    source = {"path": "released", "files_sha256": {"head": "source-hash"}}
    candidate = {"kind": "in_memory_training_candidate", "epoch": 2, "global_step": 10}
    runner = SimpleNamespace(
        checkpoint=source, predict=lambda row: {"action": "refund", "checkpoint": source}
    )
    prediction = state_predictor(runner, candidate)({"request": "refund inv-1"})
    assert prediction["checkpoint"] == candidate
    assert prediction["source_checkpoint"] == source
    assert runner.checkpoint == source


def test_selection_manifest_requires_exact_completion(tmp_path):
    import hashlib

    import pytest

    from experiments.decision_model.next_train import require_selection_manifest

    with pytest.raises(ValueError, match="selection-manifest"):
        require_selection_manifest(None)
    path = tmp_path / "selection.json"
    for value in (False, "true", 1, None):
        path.write_text(json.dumps({"selection_complete": value}))
        with pytest.raises(ValueError, match="selection_complete"):
            require_selection_manifest(path)
    path.write_text(json.dumps({"selection_complete": True}))
    assert require_selection_manifest(path) == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection_complete": True,
    }


def test_replay_gate_runs_before_test_data_or_model_access(monkeypatch, tmp_path):
    import sys

    import pytest

    from experiments.decision_model import next_train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "next_train",
            "replay",
            "--checkpoint",
            "unused",
            "--data",
            "reserved-test",
            "--manifest",
            "unused",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    accessed = []
    monkeypatch.setattr(next_train.model, "read_rows", lambda *a: accessed.append("data"))
    monkeypatch.setattr(next_train.model, "DecisionModel", lambda *a: accessed.append("model"))
    with pytest.raises(ValueError, match="selection-manifest"):
        next_train.main()
    assert accessed == []


def test_rollout_collection_uses_train_only_independent_labels(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from experiments.decision_model import next_data, next_train

    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"tasks": {"task": {"task_id": "task", "split": "train"}}}))
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "task_id": "task",
                "decisions": [{"state": {"request": "observed"}, "proposal": {"action": "refund"}}],
            }
        )
        + "\n"
    )
    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps({"state": "dev", "split": "dev", "label_action": "defer"}) + "\n")
    calls = []
    warmup_rows = []

    def predict(row):
        warmup_rows.append(row)
        return {"error": None, "latency_ms": 123}

    def evaluate(runner, args, output, **kwargs):
        assert len(warmup_rows) == 1 and warmup_rows[0]["split"] == "train"
        calls.append(kwargs)
        return {"selection_eligible": True, "records_path": str(records)}

    monkeypatch.setattr(next_train, "evaluate_tasks", evaluate)
    monkeypatch.setattr(
        next_data,
        "example",
        lambda state, metadata, index, snapshot: {
            "state": json.dumps(state),
            "label_action": "no_action",
            "split": metadata["split"],
        },
    )
    output = tmp_path / "collection"
    args = SimpleNamespace(output=output, dev_data=dev, manifest=manifest)
    next_train.collect(
        SimpleNamespace(checkpoint={"path": "released"}, predict=predict),
        [
            {"state": "initial", "split": "train", "label_action": "defer"},
        ],
        args,
    )
    collected = next_train.model.read_rows(output / "on-policy.jsonl")
    assert calls == [{"split": "train", "limit": 48}]
    assert collected[0]["label_action"] == "no_action"  # Never copy model's refund proposal.
    assert (
        collected[0]["teacher_source"] == "actual-released-policy-state+independent-offline-label"
    )
    assert {row["split"] for row in next_train.model.read_rows(output / "train-dev.jsonl")} == {
        "train",
        "dev",
    }

    report = json.loads((output / "collection.json").read_text())
    assert report["warmup_source_split"] == "train"
    assert report["warmup_wall_ms"] >= 0
    assert report["warmup_inference_ms"] == 123


def test_collection_warmup_error_stops_before_task_execution(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest

    from experiments.decision_model import next_train

    evaluated = []
    monkeypatch.setattr(next_train, "evaluate_tasks", lambda *a, **kw: evaluated.append(True))
    runner = SimpleNamespace(predict=lambda row: {"error": "device compilation failed"})
    args = SimpleNamespace(output=tmp_path / "collection", dev_data=tmp_path / "dev.jsonl")
    with pytest.raises(RuntimeError, match="warmup failed"):
        next_train.collect(runner, [{"split": "train"}], args)
    assert evaluated == []
