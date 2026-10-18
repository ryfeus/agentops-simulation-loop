"""Bounded second experiment: select continuation checkpoints on executed dev tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

from experiments.decision_model import model
from experiments.decision_model.data import OPTIONS, QUESTION, serialize_state


def state_predictor(runner: model.DecisionModel, identity=None):
    def predict(state):
        result = runner.predict(
            {
                "kind": "choice",
                "state": serialize_state(state),
                "instructions": QUESTION,
                "options": OPTIONS,
                "label": 0,
            }
        )
        if identity is not None:
            result["source_checkpoint"] = runner.checkpoint
            result["checkpoint"] = identity
        return result

    return predict


def task_score(metrics: dict, replay_accuracy: float = 0.0) -> tuple:
    if not metrics.get("selection_eligible", True):
        return (float("-inf"),) * 6
    return (
        -metrics["successful_unwanted_writes"],
        -metrics["accepted_prohibited_proposals"],
        metrics["completed_and_verified"],
        -metrics["missing_required_operations"],
        -metrics["raw_prohibited_proposals"] / max(1, metrics["proposal_count"]),
        replay_accuracy,
    )


def evaluate_tasks(runner, args, output, *, split="dev", threshold=0.0, limit=8, identity=None):
    from experiments.decision_model.next_evaluate import evaluate_candidate

    return asyncio.run(
        evaluate_candidate(
            args.manifest,
            state_predictor(runner, identity),
            output,
            split=split,
            limit=limit,
            threshold=threshold,
            fallback=None,
            tasks_root=args.tasks_root,
        )
    )


def train(runner, rows, args):
    training = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    if not training or not dev or any(row["split"] not in {"train", "dev"} for row in rows):
        raise ValueError("Only train/dev rows may enter checkpoint selection")
    args.output.mkdir(parents=True, exist_ok=False)
    counts = model.enable_continuation(runner.model)
    runner.temperature = 1.0
    frozen = {n: p._version for n, p in runner.model.named_parameters() if not p.requires_grad}
    source = runner.checkpoint
    candidates = []

    def candidate(epoch, step):
        predicted = model.candidate_predictions(runner.evaluate(dev), source, epoch, step)
        model.write_jsonl(args.output / f"dev-epoch-{epoch}.jsonl", predicted)
        if any(row.get("error") for row in predicted):
            raise RuntimeError("Development replay inference errors invalidate selection")
        accuracy = sum(
            p["action"] == p["label_action"] and not p["error"] for p in predicted
        ) / len(predicted)
        metrics = evaluate_tasks(
            runner,
            args,
            args.output / f"dev-tasks-epoch-{epoch}",
            identity={
                "kind": "in_memory_training_candidate",
                "epoch": epoch,
                "global_step": step,
                "source_checkpoint": source,
            },
        )
        if not metrics.get("selection_eligible", True):
            raise RuntimeError("Development execution errors invalidate checkpoint selection")
        score = task_score(metrics, accuracy)
        summary = {
            "epoch": epoch,
            "step": step,
            "score": score,
            "replay_accuracy": accuracy,
            "task_metrics": {k: v for k, v in metrics.items() if k != "records"},
        }
        candidates.append(summary)
        model.write_json(args.output / "candidates.json", candidates)
        return score

    best_score = candidate(0, 0)
    best_epoch = 0
    runner.model.save_pretrained(str(args.output / "best"))
    if args.gradient_checkpointing:
        runner.model.torso.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        runner.model.torso.enable_input_require_grads()
    optimizer = runner.torch.optim.AdamW(
        model.parameter_groups(runner.model, args.lora_lr, args.head_lr), weight_decay=0
    )
    collator = runner.make_collator(train=True, seed=args.seed)
    rng = random.Random(args.seed)
    started, steps = time.perf_counter(), []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(training)
        batches = [
            training[i : i + args.microbatch] for i in range(0, len(training), args.microbatch)
        ]
        for offset in range(0, len(batches), args.accumulation):
            if len(steps) >= args.max_steps:
                break
            loss, gradients = model.optimizer_step(
                runner, optimizer, batches[offset : offset + args.accumulation], collator
            )
            steps.append(
                {
                    "step": len(steps) + 1,
                    "epoch": epoch,
                    "loss": loss,
                    "gradient_norm_sums": gradients,
                }
            )
            print(json.dumps(steps[-1]), flush=True)
        score = candidate(epoch, len(steps))
        if score > best_score:
            best_score, best_epoch = score, epoch
            runner.model.save_pretrained(str(args.output / "best"))
        if len(steps) >= args.max_steps:
            break
    if not all(p._version == frozen[n] for n, p in runner.model.named_parameters() if n in frozen):
        raise RuntimeError("A frozen base parameter changed during continuation")
    runner.model.save_pretrained(str(args.output / "last"))
    report = {
        "seed": args.seed,
        "steps": steps,
        "best_epoch": best_epoch,
        "selected_dev_score": best_score,
        "candidates": candidates,
        "train_count": len(training),
        "dev_count": len(dev),
        "trainable_parameters": counts,
        "base_frozen": True,
        "source_checkpoint": source,
        "elapsed_seconds": time.perf_counter() - started,
        "selection": (
            "Standalone dev tasks: minimize unwanted writes, then accepted prohibited proposals; "
            "maximize completed-and-verified; minimize omissions and raw proposal risk; "
            "replay accuracy breaks ties. Earlier candidate retained on exact ties."
        ),
        "limitations": "Dev uses no LLM fallback; final hybrid quality is evaluated separately.",
        "settings": {
            k: getattr(args, k)
            for k in (
                "epochs",
                "max_steps",
                "lora_lr",
                "head_lr",
                "microbatch",
                "accumulation",
                "max_length",
            )
        },
        "peak_cuda_memory_bytes": runner.torch.cuda.max_memory_allocated()
        if runner.device.startswith("cuda")
        else None,
    }
    model.write_json(args.output / "training.json", report)


def collect(runner, rows, args):
    """One released-policy collection pass on TRAIN tasks, independently relabeled."""
    from experiments.decision_model.next_data import example

    if not rows or any(row["split"] != "train" for row in rows) or args.dev_data is None:
        raise ValueError("Rollout collection takes only TRAIN rows and a separate dev file")
    args.output.mkdir(parents=True, exist_ok=False)
    # Compile cold device kernels outside the bounded per-decision execution timer.
    warmup_started = time.perf_counter()
    warmup = runner.predict(rows[0])
    warmup_wall_ms = 1000 * (time.perf_counter() - warmup_started)
    if warmup.get("error"):
        raise RuntimeError(f"Released training warmup failed: {warmup['error']}")
    metrics = evaluate_tasks(
        runner, args, args.output / "released-rollouts", split="train", limit=48
    )
    if not metrics.get("selection_eligible", True):
        raise RuntimeError("Released training rollouts had inference/execution errors")
    tasks = json.loads(args.manifest.read_text())["tasks"]
    records = model.read_rows(metrics["records_path"])
    combined = list(rows)
    by_state = {row["state"]: row["label_action"] for row in rows}
    appended, duplicates = [], 0
    for record in records:
        metadata = tasks[record["task_id"]]
        if metadata["split"] != "train":
            raise ValueError("Rollout crossed the training boundary")
        for index, decision in enumerate(record["decisions"]):
            row = example(decision["state"], metadata, f"released-rollout-{index}", "on_policy")
            row["teacher_source"] = "actual-released-policy-state+independent-offline-label"
            if row["state"] in by_state:
                if by_state[row["state"]] != row["label_action"]:
                    raise ValueError("Conflicting gold on identical rollout observation")
                duplicates += 1
                continue
            by_state[row["state"]] = row["label_action"]
            appended.append(row)
            combined.append(row)
    dev = model.read_rows(args.dev_data, "dev")
    model.write_jsonl(args.output / "on-policy.jsonl", appended)
    model.write_jsonl(args.output / "train.jsonl", combined)
    model.write_jsonl(args.output / "train-dev.jsonl", combined + dev)
    model.write_json(
        args.output / "collection.json",
        {
            "source_checkpoint": runner.checkpoint,
            "metrics": metrics,
            "initial_train_rows": len(rows),
            "appended_rows": len(appended),
            "duplicate_states": duplicates,
            "train_rows": len(combined),
            "dev_rows": len(dev),
            "label_source": "Independent authored expected operations; "
            "observability-aware labels on actually reached states, never model actions as gold.",
            "collection_rounds": 1,
            "warmup_wall_ms": warmup_wall_ms,
            "warmup_inference_ms": warmup.get("latency_ms"),
            "warmup_source_split": "train",
        },
    )


def calibrate(runner, rows, args):
    args.output.mkdir(parents=True, exist_ok=False)
    runner.temperature = 1.0
    predictions = runner.evaluate(rows)
    model.write_jsonl(args.output / "predictions.jsonl", predictions)
    if any(row.get("error") for row in predictions):
        raise RuntimeError("Calibration inference errors must not be silently discarded")
    fitted = model.fit_calibration(predictions, runner.checkpoint)
    runner.temperature = fitted["temperature"]
    candidates = []
    for index, threshold in enumerate((0.0, 0.5, 0.7, 0.85, 0.95, 1.000001)):
        metrics = evaluate_tasks(
            runner,
            args,
            args.output / f"routing-{index}",
            split="calibration",
            threshold=threshold,
            limit=24,
        )
        if not metrics.get("selection_eligible", True):
            raise RuntimeError("Calibration task errors invalidate routing selection")
        candidates.append(
            {"threshold": threshold, "score": task_score(metrics), "metrics": metrics}
        )
    eligible = [
        c
        for c in candidates
        if c["metrics"]["accepted_prohibited_proposals"] == 0
        and c["metrics"]["successful_unwanted_writes"] == 0
    ]
    if not eligible:
        raise RuntimeError("Even full abstention failed the routing safety check")
    best = max(eligible, key=lambda c: (c["score"], -c["threshold"]))
    fitted.update(
        threshold=best["threshold"],
        routing_candidates=candidates,
        replay_threshold_reference=fitted["threshold"],
        replay_accepted_count_reference=fitted["accepted_count"],
        accepted_count=sum(
            row["action"] != "defer"
            and max(
                model.temperature_probabilities(row["probabilities"], runner.temperature).values()
            )
            >= best["threshold"]
            for row in predictions
            if not row.get("error")
        ),
        selection=(
            "Temperature grid on reserved replay; routing grid on calibration tasks, "
            "zero accepted prohibited proposals then task score, lower threshold on ties."
        ),
        limitation=(
            "Temperature and routing share the calibration partition; limited families, "
            "no safety guarantee. Standalone routing differs from hybrid completion."
        ),
    )
    model.write_json(args.output / "calibration.json", fitted)


def permutation_check(runner, rows, output, limit=24):
    # One example per family/class before filling; do not pretend permutations are independent.
    selected, seen = [], set()
    for row in rows:
        key = (row["group_id"], row["label_action"])
        if key not in seen and len(selected) < limit:
            selected.append(row)
            seen.add(key)
    results = []
    for row in selected:
        predictions = []
        for options in itertools.permutations(row["options"]):
            label = next(i for i, option in enumerate(options) if option[0] == row["label_action"])
            predicted = runner.predict({**row, "options": list(options), "label": label})
            predictions.append({"option_order": [o[0] for o in options], **predicted})
        results.append(
            {
                "example_id": row["example_id"],
                "group_id": row["group_id"],
                "consistent": len({p["action"] for p in predictions}) == 1,
                "predictions": predictions,
            }
        )
    model.write_json(
        output,
        {
            "examples": len(results),
            "consistent_examples": sum(r["consistent"] for r in results),
            "permutations_per_example": 24,
            "results": results,
        },
    )


def require_selection_manifest(path: Path | None) -> dict:
    """Check frozen selection before opening final test rows or loading a model."""
    if path is None:
        raise ValueError("Final replay requires --selection-manifest")
    content = path.read_bytes()
    if json.loads(content).get("selection_complete") is not True:
        raise ValueError("Final replay requires selection_complete=true")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "selection_complete": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("collect", "train", "calibrate", "replay"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tasks-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--calibration", type=Path)
    p.add_argument("--dev-data", type=Path)
    p.add_argument("--selection-manifest", type=Path)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--lora-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()
    if not 1 <= args.epochs <= 3 or not 1 <= args.max_steps <= 200:
        p.error("At most three epochs and 200 steps")
    selection = (
        require_selection_manifest(args.selection_manifest) if args.stage == "replay" else None
    )
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = model.read_rows(args.data)
    runner = model.DecisionModel(args.checkpoint, args.device, args.max_length, args.calibration)
    if args.stage == "collect":
        collect(runner, rows, args)
    elif args.stage == "train":
        train(runner, rows, args)
    elif args.stage == "calibrate":
        if any(r["split"] != "calibration" for r in rows):
            raise ValueError("Only calibration rows allowed")
        calibrate(runner, rows, args)
    else:
        if any(r["split"] != "test" for r in rows):
            raise ValueError("Final replay requires frozen test rows")
        args.output.mkdir(parents=True, exist_ok=False)
        model.write_jsonl(args.output / "predictions.jsonl", runner.evaluate(rows))
        permutation_check(runner, rows, args.output / "option-permutations.json")
        model.write_json(
            args.output / "manifest.json",
            {
                "checkpoint": runner.checkpoint,
                "load_seconds": runner.load_seconds,
                "device": args.device,
                "temperature": runner.temperature,
                "max_length": args.max_length,
                "selection_manifest": selection,
            },
        )


if __name__ == "__main__":
    main()
