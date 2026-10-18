"""One bounded continuation run selected for safe direct-decision coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from experiments.decision_model import model


def coverage_score(predictions: list[dict], threshold: float = 0.7) -> tuple:
    """DEV-only ordering; an incorrect confident no-op also counts as an error."""
    if not predictions or any(p.get("error") for p in predictions):
        raise ValueError("Development inference must be complete and error-free")
    accepted = [p for p in predictions if p["action"] != "defer" and p["confidence"] >= threshold]
    incorrect = sum(p["action"] != p["label_action"] for p in accepted)
    correct = sum(p["action"] == p["label_action"] for p in accepted)
    accuracy = sum(p["action"] == p["label_action"] for p in predictions) / len(predictions)
    return -incorrect, correct, accuracy


def validate_partitions(rows: list[dict]) -> dict[str, list[dict]]:
    partitions = {
        split: [r for r in rows if r["split"] == split] for split in ("train", "dev", "calibration")
    }
    if not all(partitions.values()) or len(rows) != sum(map(len, partitions.values())):
        raise ValueError("Only nonempty TRAIN/DEV/CALIBRATION partitions may enter training")
    ownership = {}
    for row in rows:
        group = row["group_id"]
        if group in ownership and ownership[group] != row["split"]:
            raise ValueError("Family appears in multiple partitions")
        ownership[group] = row["split"]
        if row["options"][row["label"]][0] != row["label_action"]:
            raise ValueError("Option label mismatch")
    return partitions


def train(runner, partitions, args):
    output = args.output / "training"
    output.mkdir(exist_ok=False)
    training, dev = partitions["train"], partitions["dev"]
    runner.temperature = 1.0
    counts = model.enable_continuation(runner.model)
    source = runner.checkpoint
    frozen = {n: p._version for n, p in runner.model.named_parameters() if not p.requires_grad}
    before = model.trainable_snapshot(runner.model)
    released = runner.evaluate(dev)
    baseline_score = coverage_score(released)
    model.write_jsonl(output / "dev-released.jsonl", released)
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
    steps, candidates = [], []
    best_score, best_epoch = None, None
    started = time.perf_counter()
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
        predictions = model.candidate_predictions(runner.evaluate(dev), source, epoch, len(steps))
        score = coverage_score(predictions)
        model.write_jsonl(output / f"dev-epoch-{epoch}.jsonl", predictions)
        candidates.append({"epoch": epoch, "optimizer_steps": len(steps), "score": score})
        if best_score is None or score > best_score:
            best_score, best_epoch = score, epoch
            runner.model.save_pretrained(str(output / "best-trained"))
        model.write_json(output / "candidates.json", candidates)
        if len(steps) >= args.max_steps:
            break
    base_frozen = all(
        p._version == frozen[n] for n, p in runner.model.named_parameters() if n in frozen
    )
    changed = model.changed_groups(runner.model, before)
    if not base_frozen or not all(changed.values()):
        raise RuntimeError("Continuation failed frozen-base or trainable-update checks")
    runner.model.save_pretrained(str(output / "last"))
    report = {
        "seed": args.seed,
        "steps": steps,
        "candidates": candidates,
        "best_epoch": best_epoch,
        "baseline_dev_score": baseline_score,
        "selected_dev_score": best_score,
        "selected_checkpoint_role": "trained",
        "selected_beats_release_on_dev": best_score > baseline_score,
        "trainable_parameters": counts,
        "base_frozen": base_frozen,
        "changed": changed,
        "source_checkpoint": source,
        "elapsed_seconds": time.perf_counter() - started,
        "train_count": len(training),
        "dev_count": len(dev),
        "selection": "Among trained epochs: fewest incorrect accepted DEV decisions at 0.7, "
        "then most correct accepted, then accuracy; earlier wins ties. "
        "Released baseline is reported separately even if it wins.",
        "settings": {
            k: getattr(args, k)
            for k in (
                "epochs",
                "max_steps",
                "microbatch",
                "accumulation",
                "lora_lr",
                "head_lr",
                "max_length",
                "gradient_checkpointing",
            )
        },
        "peak_cuda_memory_bytes": runner.torch.cuda.max_memory_allocated()
        if runner.device.startswith("cuda")
        else None,
    }
    model.write_json(output / "training.json", report)
    del optimizer, before
    runner.reload(str(output / "best-trained"))
    selected = runner.evaluate(dev)
    saved = model.read_rows(output / f"dev-epoch-{best_epoch}.jsonl")
    delta = model.prediction_delta(saved, selected)
    if delta > 1e-5:
        raise RuntimeError(f"Selected checkpoint reload changed probabilities by {delta}")
    report["selected_reload_max_probability_delta"] = delta
    model.write_json(output / "training.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 3 or not 1 <= args.max_steps <= 200:
        raise ValueError("Training bounds exceeded")
    rows = model.read_rows(args.data)
    partitions = validate_partitions(rows)
    args.output.mkdir(exist_ok=False, parents=True)
    model.write_json(
        args.output / "input.json",
        {
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "counts": {k: len(v) for k, v in partitions.items()},
            "seed": args.seed,
        },
    )
    random.seed(args.seed)
    runner = model.DecisionModel(args.checkpoint, device=args.device, max_length=args.max_length)
    runner.torch.manual_seed(args.seed)
    smoke_args = SimpleNamespace(**{**vars(args), "output": args.output / "smoke"})
    smoke = model.run_smoke(runner, partitions["train"], smoke_args)
    estimate = smoke["elapsed_seconds"] / args.steps * args.max_steps * 2 + 1800
    model.write_json(args.output / "runtime-estimate.json", {"remaining_seconds": estimate})
    if estimate > 10800:
        raise RuntimeError("Measured smoke throughput exceeds bounded worker budget")
    training = train(runner, partitions, args)
    for arm, checkpoint in (
        ("released", args.checkpoint),
        ("tuned", str(args.output / "training/best-trained")),
    ):
        runner.reload(checkpoint)
        runner.temperature = 1.0
        predicted = runner.evaluate(partitions["calibration"])
        if any(p.get("error") for p in predicted):
            raise RuntimeError("Calibration inference errors")
        dest = args.output / f"calibration-{arm}"
        model.write_jsonl(dest / "predictions.jsonl", predicted)
        model.write_json(
            dest / "calibration.json", model.fit_calibration(predicted, runner.checkpoint)
        )
    files = [
        args.output / "training/training.json",
        args.output / "calibration-released/calibration.json",
        args.output / "calibration-tuned/calibration.json",
    ]
    model.write_json(
        args.output / "selection-manifest.json",
        {
            "selection_complete": True,
            "selected_checkpoint_role": "trained",
            "optimizer_steps": next(
                c["optimizer_steps"]
                for c in training["candidates"]
                if c["epoch"] == training["best_epoch"]
            ),
            "total_main_optimizer_steps": len(training["steps"]),
            "smoke_steps": args.steps,
            "selected_epoch": training["best_epoch"],
            "selected_beats_release_on_dev": training["selected_beats_release_on_dev"],
            "completed_at": datetime.now(UTC).isoformat(),
            "evidence_sha256": {
                str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            "test_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
