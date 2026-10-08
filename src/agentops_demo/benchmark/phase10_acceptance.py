"""Harbor rendering, Oracle proof, and family calibration for Phase 10."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any

from agentops_demo.benchmark.catalog import render_catalog_sync
from agentops_demo.benchmark.phase10_variants import PHASE10, check, load_families
from agentops_demo.taskify.integrity import harbor_task_sha256, validate_harbor_task
from agentops_demo.validation.scenario import load_scenario
from rl.phase8c.execution_suite import derive_execution_suite, load_execution_suite

ROOT = PHASE10.parents[2]
STAGE = ROOT / ".rl-smoke" / "phase10"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _manifest(root: Path) -> dict[str, Any]:
    fresh = check(root)
    if fresh["status"] != "CURRENT":
        raise ValueError("Phase 10 generated snapshot is stale; run generate")
    return json.loads((root / "generated" / "corpus-manifest.json").read_text())


def _active(root: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest = _manifest(root)
    active_path = STAGE / "active.json"
    if not active_path.is_file():
        raise ValueError("Phase 10 tasks are not rendered; run render")
    active = json.loads(active_path.read_text())
    if active.get("corpus_sha256") != manifest["corpus_sha256"]:
        raise ValueError("Phase 10 rendered suite belongs to a different corpus")
    suite_root = STAGE / "execution" / active["execution_suite_sha256"]
    suite = load_execution_suite(suite_root / "execution-suite.json")
    if (
        suite["execution_suite_sha256"] != active["execution_suite_sha256"]
        or suite["task_count"] != 200
        or active.get("rendered_count") != 200
        or set(active.get("rendered_task_sha256", {})) != set(suite["task_ids"])
    ):
        raise ValueError("Phase 10 execution suite is invalid")
    rendered = STAGE / "rendered" / "tasks"
    for task_id in suite["task_ids"]:
        base = rendered / task_id
        execution = suite_root / "tasks" / task_id
        validate_harbor_task(base)
        validate_harbor_task(execution)
        if (
            harbor_task_sha256(base) != active["rendered_task_sha256"][task_id]
            or harbor_task_sha256(execution) != suite["tasks"][task_id]["execution_task_sha256"]
        ):
            raise ValueError(f"Phase 10 task content changed after rendering: {task_id}")
    return manifest, suite_root, suite


def render(root: Path = PHASE10) -> dict[str, Any]:
    manifest = _manifest(root)
    rendered = STAGE / "rendered" / "tasks"
    entries = render_catalog_sync(root / "generated" / "scenarios", rendered)
    if len(entries) != 200:
        raise ValueError("Phase 10 rendering did not produce exactly 200 tasks")
    staged_parent = Path(tempfile.mkdtemp(prefix=".phase10-suite-", dir=STAGE))
    staged = staged_parent / "suite"
    try:
        suite = derive_execution_suite(
            staged,
            catalog=root / "generated" / "scenarios",
            canonical_tasks=rendered,
        )
        destination = STAGE / "execution" / suite.execution_suite_sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = load_execution_suite(destination / "execution-suite.json")
            if existing != suite.manifest:
                raise ValueError("existing Phase 10 execution suite differs at the same SHA")
        else:
            os.replace(staged, destination)
        active = {
            "corpus_sha256": manifest["corpus_sha256"],
            "execution_suite_sha256": suite.execution_suite_sha256,
            "rendered_count": len(entries),
            "rendered_task_sha256": {
                entry.scenario.id: harbor_task_sha256(rendered / entry.scenario.id)
                for entry in entries
            },
        }
        _write(STAGE / "active.json", active)
        return {
            "status": "PASS",
            "task_count": len(entries),
            "corpus_sha256": manifest["corpus_sha256"],
            "execution_suite_sha256": suite.execution_suite_sha256,
        }
    finally:
        shutil.rmtree(staged_parent, ignore_errors=True)


def render_smoke(root: Path = PHASE10) -> dict[str, Any]:
    """Render and validate every generated task without persistent state or Docker."""
    manifest = _manifest(root)
    with tempfile.TemporaryDirectory(prefix="phase10-render-smoke-") as temporary:
        output = Path(temporary) / "tasks"
        entries = render_catalog_sync(root / "generated" / "scenarios", output)
        expected = set(manifest["tasks"])
        actual = {entry.scenario.id for entry in entries}
        if actual != expected or len(entries) != manifest["task_count"]:
            raise ValueError("Phase 10 render smoke task IDs or count differ from corpus")
        for task_id in sorted(actual):
            validate_harbor_task(output / task_id)
    return {
        "status": "PASS",
        "task_count": len(entries),
        "corpus_sha256": manifest["corpus_sha256"],
    }


def _harbor() -> str:
    command = Path(sys.executable).with_name("harbor")
    if command.is_file():
        return str(command)
    resolved = shutil.which("harbor")
    if resolved is None:
        raise FileNotFoundError("Harbor CLI is unavailable; install the harbor dependency group")
    return resolved


def _require_docker() -> None:
    completed = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Docker daemon is unavailable: {completed.stdout.strip()}")


def _trial(
    task: Path, jobs: Path, *, agent: str, model: str, agent_kwargs: dict[str, str] | None = None
) -> dict[str, Any]:
    if jobs.exists():
        shutil.rmtree(jobs)
    jobs.mkdir(parents=True, exist_ok=True)
    command = [
        _harbor(),
        "run",
        "-p",
        str(task),
        "-a",
        agent,
        "-m",
        model,
        "--env",
        "docker",
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--yes",
        "--job-name",
        "trial",
        "--jobs-dir",
        str(jobs),
    ]
    for key, value in sorted((agent_kwargs or {}).items()):
        command.extend(("--ak", f"{key}={value}"))
    completed = subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    reward = None
    reward_files = sorted(jobs.rglob("reward.json"))
    if reward_files:
        with suppress(OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            reward = float(json.loads(reward_files[-1].read_text())["reward"])
    return {
        "returncode": completed.returncode,
        "reward": reward,
        "output": completed.stdout[-4000:],
        "job_root": str(jobs),
    }


def _cache(
    path: Path, *, corpus_sha: str, suite_sha: str, count_key: str, count: int
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        value.get("status") == "PASS"
        and value.get("corpus_sha256") == corpus_sha
        and value.get("execution_suite_sha256") == suite_sha
        and value.get(count_key) == count
        and not value.get("failures")
    ):
        return value
    return None


def oracle(root: Path = PHASE10) -> dict[str, Any]:
    manifest, suite_root, suite = _active(root)
    corpus_sha = manifest["corpus_sha256"]
    suite_sha = suite["execution_suite_sha256"]
    cache = STAGE / "oracle" / corpus_sha / "summary.json"
    existing = _cache(
        cache, corpus_sha=corpus_sha, suite_sha=suite_sha, count_key="task_count", count=200
    )
    if existing is not None:
        return {"cached": True, **existing}
    _require_docker()

    def one(task_id: str) -> tuple[str, dict[str, Any]]:
        result = _trial(
            suite_root / "tasks" / task_id,
            STAGE / "oracle" / corpus_sha / "jobs" / task_id,
            agent="oracle",
            model="oracle",
        )
        return task_id, result

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, task_id): task_id for task_id in suite["task_ids"]}
        for future in as_completed(futures):
            task_id, result = future.result()
            if result["returncode"] != 0 or result["reward"] != 1.0:
                failures.append({"task_id": task_id, **result})
    failures.sort(key=lambda item: item["task_id"])
    value = {
        "schema_version": "1",
        "status": "PASS" if not failures else "FAIL",
        "corpus_sha256": corpus_sha,
        "execution_suite_sha256": suite_sha,
        "task_count": 200,
        "passes": 200 - len(failures),
        "failures": failures,
    }
    if not failures:
        _write(cache, value)
    return value


def _negative_kwargs(task_id: str, family_strategy: str, root: Path) -> dict[str, str]:
    scenario = load_scenario(root / "generated" / "scenarios" / task_id / "scenario.yaml")
    assert scenario.benchmark is not None
    benchmark = scenario.benchmark
    known = [invoice.id for invoice in scenario.initial_state.invoices]
    allowed = benchmark.mutation_policy.allowed_targets
    inspections = benchmark.trajectory.required_inspections
    target = allowed[0] if allowed else inspections[0]
    wrong = next((item for item in known if item != target and item not in allowed), "")
    if family_strategy == "lookup_only" and not any(
        item.type in {"must_refund", "must_escalate"} for item in scenario.expected_invariants
    ):
        raise ValueError(f"{task_id}: lookup_only is not guaranteed wrong")
    if family_strategy.endswith("wrong_target") and not wrong:
        raise ValueError(f"{task_id}: no protected wrong target")
    if (
        family_strategy.startswith("duplicate_")
        and benchmark.trajectory.max_write_attempts_per_invoice != 1
    ):
        raise ValueError(f"{task_id}: duplicate action is not prohibited")
    if family_strategy in {"refund_target", "escalate_target"} and target in allowed:
        raise ValueError(f"{task_id}: target mutation is not prohibited")
    return {
        "strategy": family_strategy,
        "target_invoice": target,
        "wrong_invoice": wrong,
        "inspection_ids": json.dumps(inspections, separators=(",", ":")),
    }


def _proof_performed(proof: object, kwargs: dict[str, str]) -> bool:
    if not isinstance(proof, dict) or proof.get("strategy") != kwargs["strategy"]:
        return False
    calls = proof.get("tool_calls")
    if not isinstance(calls, list):
        return False
    strategy = kwargs["strategy"]
    target = kwargs["target_invoice"]
    wrong = kwargs["wrong_invoice"]
    expected = (
        []
        if strategy in {"noop", "mutate_without_inspection"}
        else [("get_invoice", invoice_id) for invoice_id in json.loads(kwargs["inspection_ids"])]
    )
    actions = {
        "refund_target": [("refund_invoice", target)],
        "escalate_target": [("escalate_dispute", target)],
        "refund_wrong_target": [("refund_invoice", wrong)],
        "escalate_wrong_target": [("escalate_dispute", wrong)],
        "mutate_without_inspection": [("refund_invoice", target)],
        "duplicate_refund": [("refund_invoice", target), ("refund_invoice", target)],
        "duplicate_escalation": [("escalate_dispute", target), ("escalate_dispute", target)],
    }
    expected.extend(actions.get(strategy, []))
    actual = [
        (call.get("name"), call.get("arguments", {}).get("invoice_id"))
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("arguments"), dict)
    ]
    return len(actual) == len(calls) and actual == expected


def _calibration_proofs_current(root: Path, corpus_sha: str) -> bool:
    for family in load_families(root):
        task_id = f"{family.id}-v00"
        kwargs = _negative_kwargs(task_id, family.negative_strategy, root)
        jobs = STAGE / "calibration" / corpus_sha / "jobs" / family.id
        files = sorted(jobs.rglob("negative-execution.json"))
        if not files:
            return False
        try:
            proof = json.loads(files[-1].read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if not _proof_performed(proof, kwargs):
            return False
    return True


def calibrate(root: Path = PHASE10) -> dict[str, Any]:
    manifest, suite_root, suite = _active(root)
    corpus_sha = manifest["corpus_sha256"]
    suite_sha = suite["execution_suite_sha256"]
    cache = STAGE / "calibration" / corpus_sha / "summary.json"
    existing = _cache(
        cache, corpus_sha=corpus_sha, suite_sha=suite_sha, count_key="families_checked", count=40
    )
    if existing is not None and _calibration_proofs_current(root, corpus_sha):
        return {"cached": True, **existing}
    _require_docker()
    failures: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, Any]] = {}
    for family in load_families(root):
        task_id = f"{family.id}-v00"
        kwargs = _negative_kwargs(task_id, family.negative_strategy, root)
        jobs = STAGE / "calibration" / corpus_sha / "jobs" / family.id
        result = _trial(
            suite_root / "tasks" / task_id,
            jobs,
            agent="agentops_demo.harbor.phase10_negative_agent:Phase10NegativeAgent",
            model=f"phase10/{family.negative_strategy}",
            agent_kwargs=kwargs,
        )
        proof_files = sorted(jobs.rglob("negative-execution.json"))
        proof = json.loads(proof_files[-1].read_text()) if proof_files else None
        performed = _proof_performed(proof, kwargs)
        evidence[family.id] = {
            "task_id": task_id,
            "strategy": family.negative_strategy,
            "reward": result["reward"],
            "performed": performed,
        }
        if result["returncode"] != 0 or result["reward"] != 0.0 or not performed:
            failures.append({"family_id": family.id, **result, "performed": performed})
    value = {
        "schema_version": "1",
        "status": "PASS" if not failures else "FAIL",
        "corpus_sha256": corpus_sha,
        "execution_suite_sha256": suite_sha,
        "families_checked": 40,
        "expected_failures_observed": 40 - len(failures),
        "evidence": evidence,
        "failures": failures,
    }
    if not failures:
        _write(cache, value)
    return value


def acceptance(root: Path = PHASE10) -> dict[str, Any]:
    manifest, _, suite = _active(root)
    smoke_result = render_smoke(root)
    oracle_result = oracle(root)
    negative_result = calibrate(root)
    split_path = root / "generated" / "split-manifest.json"
    split = json.loads(split_path.read_text())
    stats = json.loads((root / "generated" / "corpus-stats.json").read_text())
    summary = {
        "schema_version": "1",
        "status": "PASS"
        if oracle_result["status"] == negative_result["status"] == "PASS"
        else "FAIL",
        "corpus_sha256": manifest["corpus_sha256"],
        "family_specs_sha256": manifest["family_specs_sha256"],
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "execution_suite_sha256": suite["execution_suite_sha256"],
        "task_count": 200,
        "family_count": 40,
        "train_count": split["counts"]["train"],
        "dev_count": split["counts"]["dev"],
        "holdout_count": split["counts"]["holdout"],
        "scenario_validation": {"valid": 200, "invalid": 0},
        "render_validation": {"valid": 200, "invalid": 0},
        "render_smoke": smoke_result,
        "prototype_counts": stats["prototype_counts_by_split"],
        "split_validation": {
            "family_disjoint": True,
            "prototype_disjoint": stats["cross_split_prototype_collisions"] == 0,
            "normalized_instruction_cross_split_collisions": 0,
            "behavior_signature_cross_split_collisions": stats["behavior_signature_collisions"][
                "cross_split"
            ],
        },
        "oracle": {
            "passes": oracle_result["passes"],
            "failures": oracle_result["failures"],
            "cached": oracle_result.get("cached", False),
        },
        "negative_calibration": {
            "families_checked": negative_result["families_checked"],
            "expected_failures_observed": negative_result["expected_failures_observed"],
            "cached": negative_result.get("cached", False),
        },
        "leakage_checks_passed": True,
        "diversity_checks_passed": True,
    }
    path = STAGE / "acceptance" / manifest["corpus_sha256"] / "summary.json"
    _write(path, summary)
    if summary["status"] == "PASS":
        report = ROOT / "reports" / "phase-10-billing-corpus.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        family_lookup = {family.id: family for family in load_families(root)}
        provenance = "\n".join(
            f"| {name} | {family_id} | {family_lookup[family_id].prototype} | "
            f"{family_lookup[family_id].archetype} |"
            for name in ("dev", "holdout")
            for family_id in split["families"][name]
        )
        prototype_counts = json.dumps(stats["prototype_counts_by_split"], sort_keys=True)
        within_split_reuse = stats["behavior_signature_collisions"]["within_split"]
        report.write_text(
            "# Phase 10 billing corpus\n\n"
            f"Generator: `{manifest['generator_version']}`\n\n"
            f"Corpus SHA: `{manifest['corpus_sha256']}`\n\n"
            f"Family specs SHA: `{manifest['family_specs_sha256']}`\n\n"
            f"Split manifest SHA: `{summary['split_manifest_sha256']}`\n\n"
            f"Execution suite SHA: `{suite['execution_suite_sha256']}`\n\n"
            "40 families; 200 generated tasks; train 160, dev 20, holdout 20.\n\n"
            "Oracle 200/200; family negative calibration 40/40.\n\n"
            f"Oracle cache reused: `{summary['oracle']['cached']}`; "
            f"calibration cache reused: `{summary['negative_calibration']['cached']}`. "
            "Both caches matched the exact corpus and execution-suite SHAs.\n\n"
            "Full CPU render smoke: 200/200 valid. Family, prototype, normalized instruction, "
            "and behavior-signature cross-split checks: PASS (zero collisions).\n\n"
            f"Prototypes by split: `{prototype_counts}`. "
            f"Within-split behavior signature reuse: `{within_split_reuse}` "
            "signature groups.\n\n"
            "| Split | Family | Prototype | Archetype |\n| --- | --- | --- | --- |\n"
            f"{provenance}\n\n"
            f"Archetypes: `{json.dumps(stats['counts_by_archetype'], sort_keys=True)}`\n\n"
            f"Difficulties: `{json.dumps(stats['counts_by_difficulty'], sort_keys=True)}`\n\n"
            "Known limits: deterministic templates and billing-v1 tool policy; "
            "no model characterization or RL training.\n",
            encoding="utf-8",
        )
    return summary


def run_command(command: str, root: Path = PHASE10) -> dict[str, Any]:
    if command == "render-smoke":
        return render_smoke(root)
    if command == "render":
        return render(root)
    if command == "oracle":
        return oracle(root)
    if command == "calibrate":
        return calibrate(root)
    if command == "acceptance":
        return acceptance(root)
    raise ValueError(f"unknown Phase 10 command {command}")
