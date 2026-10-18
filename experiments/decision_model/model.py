"""Small continuation experiment; model dependencies belong in its separate environment.

The pinned upstream loader preserves the released pointer head and adapter. Never
use training.train's init_from ablation, which replaces that head. Both replay and
the hybrid runner use this module's identical, non-truncating inference path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "6039cf8b64c6074e75804000c07ac3d4cce795f7"
ACTIONS = ("refund", "escalate", "no_action", "defer")


def upstream() -> None:
    source = Path(
        os.environ.get("STRANDS_DECIDER_SOURCE", ".artifacts/decision-model/vendor/strands-decider")
    ).resolve()
    if not (source / "training/data/__init__.py").exists():
        raise RuntimeError(f"Missing pinned upstream checkout: {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f"Upstream revision {revision} differs from pinned {SOURCE_COMMIT}")
    for path in (source, source / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def read_rows(path: str | Path, split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows in {path} for split {split!r}")
    return rows


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def checkpoint_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError("Use a prepared local snapshot with pinned base_revision")
    hashes = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and (
            file.suffix in {".pt", ".safetensors"}
            or file.name in {"strands_decider_config.json", "adapter_config.json"}
        ):
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[str(file.relative_to(path))] = digest.hexdigest()
    return {"path": str(path), "files_sha256": hashes, "upstream_commit": SOURCE_COMMIT}


class StrictTokenizer:
    """Reject the exact rendered batch before upstream can silently truncate it."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, texts: Any, **kwargs: Any) -> Any:
        lengths = self.tokenizer(texts, truncation=False, padding=False)["input_ids"]
        if isinstance(texts, str):
            lengths = [lengths]
        maximum = kwargs.get("max_length")
        if maximum and any(len(tokens) > maximum for tokens in lengths):
            raise ValueError(f"Input exceeds {maximum} tokens; truncation is prohibited")
        return self.tokenizer(texts, **{**kwargs, "truncation": False})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def enable_continuation(model: Any) -> dict:
    """Unfreeze only the existing adapter and readout, retaining their dimensions."""
    counts = {"lora": 0, "head": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        kind = (
            "head"
            if name.startswith("head.")
            else ("lora" if ".lora_A." in name or ".lora_B." in name else "frozen")
        )
        parameter.requires_grad_(kind != "frozen")
        counts[kind] += parameter.numel()
    if not counts["lora"] or not counts["head"] or not counts["frozen"]:
        raise ValueError(f"Expected existing adapter, head and frozen base: {counts}")
    model.config.temperature = 1.0
    model.config.temperature_by_kind = {}
    if hasattr(model.torso.config, "use_cache"):
        model.torso.config.use_cache = False
    return counts


def parameter_groups(model: Any, lora_lr: float, head_lr: float) -> list[dict]:
    return [
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")
            ],
            "lr": head_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("head.")
            ],
            "lr": lora_lr,
        },
    ]


class DecisionModel:
    """One model per process; all predictions use upstream renderer/collator/loss."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        max_length: int = 1024,
        calibration: str | None = None,
    ):
        upstream()
        import torch
        from strands_decider.modeling import StrandsDeciderModel

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.checkpoint = checkpoint_identity(checkpoint)
        started = time.perf_counter()
        if device == "cpu":
            from strands_decider.cpu_kernels import install

            install()
        elif device == "mps":
            from strands_decider.mps_kernels import install

            install()
        self.model = StrandsDeciderModel.load(checkpoint).to(device)
        if not self.model.config.base_revision:
            raise ValueError("Prepared checkpoint must pin actual base_revision")
        if self.model.config.head_type != "pointer":
            raise ValueError("This experiment requires the released pointer head")
        if device == "cpu":
            self.model.torso.float()
        self.model.eval()
        self.temperature = float(
            self.model.config.temperature_by_kind.get("choice", self.model.config.temperature)
        )
        if calibration:
            fitted = json.loads(Path(calibration).read_text())
            if fitted["checkpoint"]["files_sha256"] != self.checkpoint["files_sha256"]:
                raise ValueError("Calibration belongs to a different checkpoint")
            self.temperature = float(fitted["temperature"])
        self.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.collator = self.make_collator(train=False)

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        elif self.device.startswith("mps"):
            self.torch.mps.synchronize()

    def make_collator(self, train: bool, seed: int = 42) -> Any:
        from training.data import CollatorConfig, SystemOneCollator

        return SystemOneCollator(
            StrictTokenizer(self.model.tokenizer),
            CollatorConfig(
                max_length=self.max_length,
                num_slots=self.model.config.num_slots,
                head_type=self.model.config.head_type,
                shuffle_options=train,
                vary_instructions=False,
                ordinal_smoothing=0.0,
                seed=seed,
            ),
            train=train,
        )

    def batch(self, rows: list[dict], collator: Any = None) -> dict:
        from training.data import Example

        examples = [Example.from_dict({"label": 0, **row}) for row in rows]
        result = (collator or self.collator)(examples)
        return {key: value.to(self.device) for key, value in result.items()}

    def predict(self, row: dict) -> dict:
        result = {
            key: row[key]
            for key in ("example_id", "group_id", "split", "label_action", "prohibited_actions")
            if key in row
        }
        result.update(
            checkpoint=self.checkpoint,
            action="defer",
            probabilities={},
            confidence=0.0,
            input_tokens=None,
            latency_ms=None,
            error=None,
        )
        started = time.perf_counter()
        try:
            batch = self.batch([row])
            result["input_tokens"] = int(batch["attention_mask"].sum())
            self.model.eval()
            self.synchronize()
            started = time.perf_counter()
            with self.torch.inference_mode():
                output = self.model(**batch, temperature=self.temperature)
            self.synchronize()
            result["latency_ms"] = 1000 * (time.perf_counter() - started)
            probabilities = output["log_probs"][0, : len(row["options"])].exp().tolist()
            if not all(math.isfinite(p) for p in probabilities):
                raise ValueError("Non-finite predicted probabilities")
            result["probabilities"] = {
                option[0]: p for option, p in zip(row["options"], probabilities, strict=True)
            }
            result["action"] = max(result["probabilities"], key=result["probabilities"].get)
            result["confidence"] = max(probabilities)
        except (ValueError, RuntimeError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result["latency_ms"] = 1000 * (time.perf_counter() - started)
        return result

    def evaluate(self, rows: list[dict], warmup: bool = True) -> list[dict]:
        if warmup:
            self.predict(rows[0])
        return [self.predict(row) for row in rows]

    def reload(self, checkpoint: str) -> None:
        from strands_decider.modeling import StrandsDeciderModel

        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
        self.model = StrandsDeciderModel.load(checkpoint).to(self.device).eval()
        if self.device == "cpu":
            self.model.torso.float()
        self.checkpoint = checkpoint_identity(checkpoint)
        self.temperature = float(
            self.model.config.temperature_by_kind.get("choice", self.model.config.temperature)
        )
        self.collator = self.make_collator(train=False)


def selection_score(predictions: list[dict]) -> tuple[int, float]:
    prohibited = sum(row["action"] in row.get("prohibited_actions", []) for row in predictions)
    correct = sum(row["action"] == row["label_action"] and not row["error"] for row in predictions)
    return -prohibited, correct / len(predictions)


def candidate_predictions(
    predictions: list[dict], source_checkpoint: dict, epoch: int, global_step: int
) -> list[dict]:
    """Identify evaluated in-memory weights without claiming a saved checkpoint hash."""
    identity = {
        "kind": "in_memory_training_candidate",
        "epoch": epoch,
        "global_step": global_step,
        "source_checkpoint": source_checkpoint,
    }
    return [
        {**row, "source_checkpoint": source_checkpoint, "checkpoint": identity}
        for row in predictions
    ]


def balanced_fixture(rows: list[dict]) -> list[dict]:
    result = []
    for action in ACTIONS:
        matches = [row for row in rows if row["label_action"] == action]
        if not matches:
            raise ValueError(f"Smoke fixture missing class {action}")
        result.append(matches[0])
    return result


def optimizer_step(
    runner: DecisionModel, optimizer: Any, batches: list[list[dict]], collator: Any
) -> tuple[float, dict]:
    torch = runner.torch
    runner.model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for rows in batches:
        output = runner.model(**runner.batch(rows, collator), temperature=1.0)
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        (loss / len(batches)).backward()
        losses.append(float(loss.detach()))
    gradients = {"head": 0.0, "lora": 0.0}
    for name, parameter in runner.model.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received gradient: {name}")
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Non-finite gradient: {name}")
            kind = "head" if name.startswith("head.") else "lora"
            gradients[kind] += float(parameter.grad.detach().float().norm())
    if not all(value > 0 for value in gradients.values()):
        raise RuntimeError(f"Missing adapter/head gradient: {gradients}")
    torch.nn.utils.clip_grad_norm_([p for p in runner.model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return sum(losses) / len(losses), gradients


def trainable_snapshot(model: Any) -> dict:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def changed_groups(model: Any, before: dict) -> dict:
    import torch

    changed = {"head": False, "lora": False}
    for name, parameter in model.named_parameters():
        if name in before and not torch.equal(parameter.detach().cpu(), before[name]):
            changed["head" if name.startswith("head.") else "lora"] = True
    return changed


def prediction_delta(left: list[dict], right: list[dict]) -> float:
    if any(row["error"] for row in left + right):
        raise RuntimeError("Cannot compare errored checkpoint predictions")
    return max(
        abs(a["probabilities"][action] - b["probabilities"][action])
        for a, b in zip(left, right, strict=True)
        for action in ACTIONS
    )


def run_smoke(runner: DecisionModel, rows: list[dict], args: Any) -> dict:
    if not 10 <= args.steps <= 20:
        raise ValueError("Smoke run must use 10-20 optimizer steps")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "smoke.json").exists():
        raise ValueError("Use a fresh smoke output directory")
    source = runner.checkpoint["path"]
    fixture = balanced_fixture(rows)
    original = runner.evaluate(fixture)
    runner.model.save_pretrained(str(output / "roundtrip"))
    runner.reload(str(output / "roundtrip"))
    roundtrip = prediction_delta(original, runner.evaluate(fixture))
    if roundtrip > 1e-5:
        raise RuntimeError(f"Initial checkpoint round-trip delta {roundtrip} > 1e-5")
    counts = enable_continuation(runner.model)
    runner.temperature = 1.0
    if args.gradient_checkpointing:
        runner.model.torso.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        runner.model.torso.enable_input_require_grads()
    before_predictions = runner.evaluate(fixture)
    before = trainable_snapshot(runner.model)
    frozen_versions = {
        name: parameter._version
        for name, parameter in runner.model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = runner.torch.optim.AdamW(
        parameter_groups(runner.model, args.lora_lr, args.head_lr), weight_decay=0.0
    )
    collator = runner.make_collator(train=True, seed=args.seed)
    started = time.perf_counter()
    steps = []
    for step in range(args.steps):
        loss, gradients = optimizer_step(runner, optimizer, [fixture[:2], fixture[2:]], collator)
        steps.append({"step": step + 1, "loss": loss, "gradient_norm_sums": gradients})
        print(json.dumps(steps[-1]), flush=True)
    after = runner.evaluate(fixture)
    changed = changed_groups(runner.model, before)
    frozen = all(
        parameter._version == frozen_versions[name]
        for name, parameter in runner.model.named_parameters()
        if name in frozen_versions
    )
    if not all(changed.values()) or not frozen:
        raise RuntimeError(f"Smoke parameter check failed: {changed}, base_frozen={frozen}")
    runner.model.save_pretrained(str(output / "checkpoint"))
    del optimizer, before
    runner.reload(str(output / "checkpoint"))
    trained_roundtrip = prediction_delta(after, runner.evaluate(fixture))
    if trained_roundtrip > 1e-5:
        raise RuntimeError(f"Trained checkpoint round-trip delta {trained_roundtrip} > 1e-5")
    before_nll = -sum(
        math.log(max(row["probabilities"][row["label_action"]], 1e-12))
        for row in before_predictions
    ) / len(fixture)
    after_nll = -sum(
        math.log(max(row["probabilities"][row["label_action"]], 1e-12)) for row in after
    ) / len(fixture)
    report = {
        "trainable_parameters": counts,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "initial_roundtrip_max_probability_delta": roundtrip,
        "trained_roundtrip_max_probability_delta": trained_roundtrip,
        "changed": changed,
        "base_frozen": frozen,
        "fixture_ids": [row["example_id"] for row in fixture],
        "before_nll": before_nll,
        "after_nll": after_nll,
        "before_accuracy": selection_score(before_predictions)[1],
        "after_accuracy": selection_score(after)[1],
        "overfit_pass": after_nll < before_nll and selection_score(after)[1] >= 0.75,
        "note": "Four training examples; wiring evidence only, not quality evaluation",
    }
    write_json(output / "smoke.json", report)
    runner.reload(source)
    if not report["overfit_pass"]:
        raise RuntimeError(
            "Balanced smoke overfit check failed; inspect smoke.json before training"
        )
    return report


def run_train(runner: DecisionModel, rows: list[dict], args: Any) -> dict:
    if not 1 <= args.epochs <= 3 or not 1 <= args.max_steps <= 200:
        raise ValueError("Training bound is 1-3 epochs and 1-200 optimizer steps")
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    if not train or not dev:
        raise ValueError("Training requires nonempty train and dev splits")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best").exists():
        raise ValueError("Use a fresh training output directory")
    counts = enable_continuation(runner.model)
    runner.temperature = 1.0
    # Retain the starting model if every candidate loses under the fixed dev criterion.
    best_score = selection_score(runner.evaluate(dev))
    baseline_score = best_score
    runner.model.save_pretrained(str(output / "best"))
    best_epoch = 0
    if args.gradient_checkpointing:
        runner.model.torso.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        runner.model.torso.enable_input_require_grads()
    optimizer = runner.torch.optim.AdamW(
        parameter_groups(runner.model, args.lora_lr, args.head_lr), weight_decay=0.0
    )
    collator = runner.make_collator(train=True, seed=args.seed)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    steps, candidates = [], []
    for epoch in range(args.epochs):
        rng.shuffle(train)
        batches = [train[i : i + args.microbatch] for i in range(0, len(train), args.microbatch)]
        for offset in range(0, len(batches), args.accumulation):
            if len(steps) >= args.max_steps:
                break
            loss, _ = optimizer_step(
                runner, optimizer, batches[offset : offset + args.accumulation], collator
            )
            steps.append({"step": len(steps) + 1, "epoch": epoch + 1, "loss": loss})
            print(json.dumps(steps[-1]), flush=True)
        predicted = candidate_predictions(
            runner.evaluate(dev), runner.checkpoint, epoch + 1, len(steps)
        )
        score = selection_score(predicted)
        candidates.append(
            {"epoch": epoch + 1, "dev_prohibited": -score[0], "dev_accuracy": score[1]}
        )
        write_jsonl(output / f"dev-epoch-{epoch + 1}.jsonl", predicted)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            runner.model.save_pretrained(str(output / "best"))
        if len(steps) >= args.max_steps:
            break
    runner.model.save_pretrained(str(output / "last"))
    report = {
        "seed": args.seed,
        "trainable_parameters": counts,
        "steps": steps,
        "candidates": candidates,
        "best_epoch": best_epoch,
        "baseline_dev_score": baseline_score,
        "selected_dev_score": best_score,
        "selection": "Fewest prohibited proposals, then highest dev accuracy; ties retain earlier",
        "elapsed_seconds": time.perf_counter() - started,
        "train_count": len(train),
        "dev_count": len(dev),
        "source_checkpoint": runner.checkpoint,
        "settings": {
            key: getattr(args, key)
            for key in (
                "epochs",
                "max_steps",
                "lora_lr",
                "head_lr",
                "microbatch",
                "accumulation",
                "max_length",
                "gradient_checkpointing",
            )
        },
        "peak_cuda_memory_bytes": runner.torch.cuda.max_memory_allocated()
        if runner.device.startswith("cuda")
        else None,
    }
    write_json(output / "training.json", report)
    return report


def temperature_probabilities(probabilities: dict[str, float], temperature: float) -> dict:
    logits = {action: math.log(max(p, 1e-30)) / temperature for action, p in probabilities.items()}
    offset = max(logits.values())
    values = {action: math.exp(logit - offset) for action, logit in logits.items()}
    total = sum(values.values())
    return {action: value / total for action, value in values.items()}


def fit_calibration(predictions: list[dict], checkpoint: dict) -> dict:
    if any(row.get("split") != "calibration" for row in predictions):
        raise ValueError("Calibration may use only the disjoint calibration partition")
    valid = [row for row in predictions if not row["error"] and row["probabilities"]]
    if not valid:
        raise ValueError("No valid calibration predictions")

    def nll(temperature: float) -> float:
        return -sum(
            math.log(
                max(
                    temperature_probabilities(row["probabilities"], temperature)[
                        row["label_action"]
                    ],
                    1e-30,
                )
            )
            for row in valid
        ) / len(valid)

    temperatures = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    temperature = min(temperatures, key=nll)
    calibrated = [
        {
            **row,
            "confidence": max(
                temperature_probabilities(row["probabilities"], temperature).values()
            ),
        }
        for row in valid
    ]
    candidates = sorted({0.0, 1.000001, *(row["confidence"] for row in calibrated)})
    threshold = 1.000001
    accepted_count = 0
    for candidate in candidates:
        accepted = [
            row for row in calibrated if row["confidence"] >= candidate and row["action"] != "defer"
        ]
        if accepted and all(
            row["action"] == row["label_action"]
            and row["action"] not in row.get("prohibited_actions", [])
            for row in accepted
        ):
            threshold, accepted_count = candidate, len(accepted)
            break
    return {
        "checkpoint": checkpoint,
        "temperature": temperature,
        "threshold": threshold,
        "sample_count": len(valid),
        "family_count": len({row["group_id"] for row in valid}),
        "error_count": len(predictions) - len(valid),
        "accepted_count": accepted_count,
        "nll_before": nll(1.0),
        "nll_after": nll(temperature),
        "selection": "Grid minimum NLL; maximum coverage with zero observed accepted errors",
        "limitation": "Sparse calibration; zero observed errors is no safety guarantee",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "smoke", "train", "calibrate"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--split")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--calibration")
    parser.add_argument("--reverse-options", action="store_true")
    parser.add_argument("--limit", type=int, help="Evaluation-only bounded sample")
    args = parser.parse_args()
    upstream()
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    split = args.split
    if args.command == "calibrate":
        split = "calibration"
    elif args.command == "smoke":
        split = "train"
    elif args.command == "evaluate" and split is None:
        split = "test"
    rows = read_rows(args.data, split)
    if args.limit:
        if args.command != "evaluate" or args.limit < 1:
            raise ValueError("--limit is a positive evaluation-only sample bound")
        rows = rows[: args.limit]
    if args.reverse_options:
        rows = [
            {
                **row,
                "options": list(reversed(row["options"])),
                "label": len(row["options"]) - 1 - row["label"],
            }
            for row in rows
        ]
    runner = DecisionModel(args.checkpoint, args.device, args.max_length, args.calibration)
    if args.command == "evaluate":
        write_jsonl(args.output, runner.evaluate(rows))
        write_json(
            str(args.output) + ".metadata.json",
            {
                "checkpoint": runner.checkpoint,
                "load_seconds": runner.load_seconds,
                "device": args.device,
                "batch_size": 1,
                "max_length": args.max_length,
                "temperature": runner.temperature,
                "reverse_options": args.reverse_options,
            },
        )
    elif args.command == "smoke":
        run_smoke(runner, rows, args)
    elif args.command == "train":
        run_train(runner, rows, args)
    else:
        runner.temperature = 1.0
        predictions = runner.evaluate(rows)
        write_jsonl(str(args.output) + ".predictions.jsonl", predictions)
        write_json(args.output, fit_calibration(predictions, runner.checkpoint))


if __name__ == "__main__":
    main()
