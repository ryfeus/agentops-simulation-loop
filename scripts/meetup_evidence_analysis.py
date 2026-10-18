"""Offline reconciliation of retained Phase 12 evidence; never launches a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath

from rl.phase11.selection import adapter_files
from rl.phase12 import artifacts, calibration, config, corpus, metrics
from rl.phase12.common import digest, fingerprint, read, rows, write
from rl.phase12.demonstrations import validate


def unpack_verified(archive: Path, output: Path) -> dict:
    manifest = read(Path(str(archive) + ".json"))
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("archive checksum mismatch")
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
            raise ValueError("archive inventory mismatch or duplicate members")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError("unsafe archive member")
            payload = stream.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != manifest["files"][member.name]:
                raise ValueError("member checksum mismatch: " + member.name)
        if output.exists():
            artifacts.verify_files(output, manifest["files"])
            return manifest
        output.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.extractfile(member).read())
    return manifest


def behavior(row: dict) -> dict:
    """Supplement the frozen attempt metric with observable successful writes."""
    inspected, writes = set(), Counter()
    prohibited, successful, unknown = 0, 0, 0
    for observation in row["observations"]:
        if observation.get("bridge_rejected"):
            continue
        name = observation["name"]
        target = observation["arguments"].get("invoice_id", "")
        if name == "get_invoice":
            inspected.add(target)
            continue
        writes[target] += 1
        bad = (
            row["expected_operations"].get(target) != name
            or target not in inspected
            or writes[target] > 1
        )
        prohibited += int(bad)
        try:
            response = json.loads(observation["response"])
            payload = response.get(
                "refund", response.get("escalation", response.get("result", response))
            )
            ok = payload.get("success") is True
            successful += int(bad and ok)
        except (ValueError, AttributeError):
            unknown += int(bad)
    conversation = row["completion"]
    proposed = sum(len(m.get("tool_calls", [])) for m in conversation)
    retained_responses = sum(m["role"] == "tool" for m in conversation)
    executed = len(row["observations"])
    return {
        "prohibited_attempt": bool(prohibited),
        "successful_prohibited_operation": bool(successful),
        "unknown_write_responses": unknown,
        "proposed_calls": proposed,
        "observed_calls": executed,
        "retained_tool_responses": retained_responses,
        "executed_responses_not_retained": max(0, executed - retained_responses),
        "assistant_tool_rounds": sum(bool(m.get("tool_calls")) for m in conversation),
        "repeated_identical_calls": executed
        - len(
            {json.dumps([o["name"], o["arguments"]], sort_keys=True) for o in row["observations"]}
        ),
    }


def gpu_trace(path: Path) -> dict:
    samples = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].strip() == "timestamp":
                continue
            samples.append(
                (
                    datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"),
                    int(row[2].strip().split()[0]),
                )
            )
    intervals = [((b[0] - a[0]).total_seconds(), a[1]) for a, b in pairwise(samples)]
    if not intervals or any(seconds <= 0 for seconds, _ in intervals):
        raise ValueError("invalid GPU trace")
    return {
        "sha256": digest(path),
        "samples": len(samples),
        "memory_resident_seconds": sum(s for s, memory in intervals if memory > 0),
        "trace_span_seconds": sum(s for s, _ in intervals),
        "maximum_sample_gap_seconds": max(s for s, _ in intervals),
    }


def analyze(archive: Path, saved_run: Path, output: Path) -> dict:
    evidence = output / "evidence"
    archive_manifest = unpack_verified(archive, evidence)
    identity = read(evidence / "identity.json")
    exp = read(evidence / "experiment.json")
    assert exp == config.load()
    # The portable export intentionally excludes executable datasets.
    artifacts.verify_files(saved_run / "datasets", identity["dataset_files"])
    frozen = corpus.check()
    assert frozen["corpus_sha256"] == identity["corpus_sha256"]
    assert read(evidence / "corpus-manifest.json") == corpus.transport_manifest(frozen)
    calibration.validate(read(evidence / "calibration.json"), frozen)
    metadata = {
        r: read(saved_run / f"datasets/{r}/metadata.json")
        for r in ("train", "dev", "test", "anchors")
    }
    for role in ("train", "dev", "test"):
        assert metadata[role] == frozen["roles"][role]
    artifacts.verify_files(evidence, identity["inputs"])
    source_checks = {}
    for name, expected in identity["runtime_sources"].items():
        payload = subprocess.check_output(["git", "show", f"{identity['source_revision']}:{name}"])
        source_checks[name] = hashlib.sha256(payload).hexdigest() == expected
    assert all(source_checks.values())
    lock = read(evidence / "selection-lock.json")
    artifacts.verify_files(evidence, lock["evidence"])
    assert lock["experiment_sha256"] == identity["experiment_sha256"]
    assert read(evidence / "test-opened.json")["selection_lock_sha256"] == fingerprint(lock)
    assert read(evidence / "final-inputs.json")["selection_lock_sha256"] == digest(
        evidence / "selection-lock.json"
    )
    normalized, summaries, diagnostics, all_rows = [], {}, {}, {}
    targets = [("dev", "baseline", "dev/baseline")]
    for method, seed, steps in (
        ("sft", 42, (29, 58, 87)),
        ("grpo", 42, (160, 320)),
        ("grpo", 43, (160, 320)),
        ("grpo", 44, (160, 320)),
    ):
        targets.extend(
            ("dev", f"{method}/{seed}/{step}", f"{method}/{seed}/dev/{step}") for step in steps
        )
    targets.extend(
        (role, policy, f"final/{role}/{policy}")
        for role in ("test", "anchors")
        for policy in ("baseline", "grpo/42", "grpo/43", "grpo/44")
    )
    for role, policy, relative in targets:
        target = evidence / relative
        captured, elapsed = [], 0.0
        for index in range(2):
            proof = read(target / f"pass-{index}/result.json")
            assert proof["execution_valid"] and proof["training_performed"] is False
            assert proof["optimizer_created"] is False and proof["global_step"] == 0
            assert proof["seed"] == exp["evaluation_seed"] + index
            assert proof["inference_sha256"] == config.inference_hash(exp, 256)
            if policy == "baseline":
                adapter = None
            elif role == "dev":
                method, seed, step = policy.split("/")
                adapter = evidence / f"{method}/{seed}/training/adapters/{step}"
            else:
                adapter = evidence / lock["selections"][policy]["adapter"]
            assert proof["adapter_files"] == (adapter_files(adapter) if adapter else None)
            elapsed += proof["elapsed_seconds"]
            batch = rows(target / f"pass-{index}/rollouts.jsonl")
            assert len(batch) == len(metadata[role]) * 4
            for row in batch:
                assert all(row[k] == v for k, v in metadata[role][row["task_id"]].items())
                row["attempt"] += index * 4
            captured.extend(batch)
        summary = metrics.summarize(captured, metadata[role], 8)
        summary.update(
            inference_sha256=config.inference_hash(exp, 256),
            seed_schedule=[exp["evaluation_seed"], exp["evaluation_seed"] + 1],
        )
        assert summary == read(target / "summary.json"), relative
        key = f"{role}/{policy}"
        summaries[key], all_rows[key] = summary, captured
        grouped = defaultdict(Counter)
        behaviors = []
        for row in captured:
            b = behavior(row)
            behaviors.append(b)
            c = grouped[str(row.get("required_tool_calls", "unrecorded"))]
            c.update(
                rollouts=1,
                passes=int(row["reward"]),
                pending=int(row["pending_tool_call"]),
                **{row["termination_reason"] + "_termination": 1},
            )
        assert (
            sum(b["prohibited_attempt"] for b in behaviors) == summary["prohibited_write_rollouts"]
        )
        diagnostics[key] = {
            "by_required_calls": dict(grouped),
            "pending_with_observed_responses_omitted": sum(
                r["pending_tool_call"] and b["executed_responses_not_retained"] > 0
                for r, b in zip(captured, behaviors, strict=True)
            ),
            "pending_with_eight_rounds": sum(
                r["pending_tool_call"] and b["assistant_tool_rounds"] >= 8
                for r, b in zip(captured, behaviors, strict=True)
            ),
            "pending_and_reward_one": sum(
                r["pending_tool_call"] and r["reward"] == 1 for r in captured
            ),
            "repetition_rollouts": sum(b["repeated_identical_calls"] > 0 for b in behaviors),
            "successful_prohibited_operation_rollouts": sum(
                b["successful_prohibited_operation"] for b in behaviors
            ),
            "unknown_write_responses": sum(b["unknown_write_responses"] for b in behaviors),
        }
        normalized.append(
            {
                "evaluation": key,
                "evidence_status": "recomputed_from_rollouts",
                "tasks": len(metadata[role]),
                "prototypes": len(summary["prototypes"]),
                "attempts_per_task": 8,
                "rollouts": len(captured),
                **{
                    k: summary[k]
                    for k in (
                        "prototype_macro_pass_rate",
                        "action_macro_pass_rate",
                        "prohibited_write_rate",
                    )
                },
                **summary["diagnostics"]["termination_reasons"],
                "tool_errors": summary["diagnostics"]["tool_errors"],
                "successful_prohibited_operation_rollouts": diagnostics[key][
                    "successful_prohibited_operation_rollouts"
                ],
                "evaluation_seconds_excludes_model_load_and_final_teardown": elapsed,
                "verifier_seconds_sum": sum(r["verifier_ms"] for r in captured) / 1000,
                "tool_seconds_sum": sum(r["tool_execution_ms"] for r in captured) / 1000,
                "environment_reset_seconds_sum": sum(r["environment_reset_ms"] for r in captured)
                / 1000,
                "generated_tokens": sum(r["generated_tokens"] for r in captured),
                "conversation_tokens": sum(r["completion_token_count"] for r in captured),
                "source": relative,
                "summary_sha256": digest(target / "summary.json"),
            }
        )
    training, selection = {}, {}
    for method, seed in (("sft", 42), ("grpo", 42), ("grpo", 43), ("grpo", 44)):
        prefix = evidence / f"{method}/{seed}"
        result = read(prefix / "training/training-result.json")
        for step, hashes in result["adapters"].items():
            assert adapter_files(prefix / f"training/adapters/{step}") == hashes
        options = {
            int(step): summaries[f"dev/{method}/{seed}/{step}"] for step in result["adapters"]
        }
        decision = read(prefix / "selection.json")
        assert metrics.choose(options) == decision["selected_step"]
        assert (
            metrics.pilot_gate(summaries["dev/baseline"], options[decision["selected_step"]], exp)
            == decision["pilot_eligible"]
        )
        if method == "grpo":
            assert metrics.reward_groups(rows(prefix / "training/rollouts.jsonl")) == read(
                prefix / "training/reward-groups.json"
            )
        training[f"{method}/{seed}"] = result
        selection[f"{method}/{seed}"] = decision
    demos = rows(evidence / "demonstrations/demonstrations.jsonl")
    assert len(demos) == 160 and {d["task_id"] for d in demos} == set(metadata["train"])
    for demo in demos:
        validate(demo, set(metadata["train"]))
    assert read(evidence / "demonstrations/manifest.json")["records_sha256"] == fingerprint(demos)
    sft = read(evidence / "sft/42/training/sft-data.json")
    assert sft["assistant_turn_examples"] == sum(
        m["role"] == "assistant" for d in demos for m in d["messages"]
    )
    final = metrics.final_result(
        summaries["test/baseline"], {s: summaries[f"test/grpo/{s}"] for s in (42, 43, 44)}, exp
    )
    assert {"grpo": final} == read(evidence / "final/result.json")
    gpu = {
        p.stem: gpu_trace(p)
        for p in sorted((evidence / "jobs").glob("gpu-*.csv"))
        if p.stat().st_size > 200
    }
    examples = []
    candidate = all_rows["test/grpo/42"]
    baseline = {(r["task_id"], r["attempt"]): r for r in all_rows["test/baseline"]}
    criteria = [
        ("success", lambda r: r["reward"] == 1 and r["termination_reason"] == "eos"),
        ("prohibited-write", lambda r: behavior(r)["successful_prohibited_operation"]),
        ("unfinished-sequence", lambda r: r["pending_tool_call"] and r["reward"] == 0),
        (
            "paired-improvement",
            lambda r: r["reward"] == 1 and baseline[r["task_id"], r["attempt"]]["reward"] == 0,
        ),
    ]
    for category, predicate in criteria:
        row = next(
            r for r in sorted(candidate, key=lambda r: (r["task_id"], r["attempt"])) if predicate(r)
        )
        example = {
            "category": category,
            "selection": "first lexicographic task/attempt satisfying criterion",
            "candidate": row,
            "baseline": baseline[row["task_id"], row["attempt"]]
            if category == "paired-improvement"
            else None,
        }
        examples.append(example)
    write(output / "examples-private.json", examples)
    fields = sorted({key for row in normalized for key in row})
    with (output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    proof = {
        "archive": archive_manifest,
        "dataset_files_verified": len(identity["dataset_files"]),
        "runtime_sources_verified_against_historical_commit": source_checks,
        "identity": identity,
        "development_rollouts": sum(
            r["rollouts"] for r in normalized if r["evaluation"].startswith("dev/")
        ),
        "final_and_anchor_rollouts": sum(
            r["rollouts"] for r in normalized if not r["evaluation"].startswith("dev/")
        ),
        "final": final,
        "selection": selection,
        "training": training,
        "demonstrations": len(demos),
        "sft_examples": sft["assistant_turn_examples"],
        "diagnostics": diagnostics,
        "gpu": gpu,
        "results": normalized,
    }
    write(output / "reconciliation.json", proof)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--saved-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    proof = analyze(args.archive, args.saved_run, args.output)
    print(
        json.dumps(
            {k: proof[k] for k in ("development_rollouts", "final_and_anchor_rollouts", "final")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
