"""Phase 12 controller. CPU checks are separate from every execution command."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from agentops_demo.benchmark.catalog import load_catalog
from agentops_demo.harbor.provenance import build_clean_wheel
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from rl.phase8c.execution_suite import derive_execution_suite
from rl.phase12 import artifacts, calibration, config, corpus
from rl.phase12.common import REPO, digest, fingerprint, read, write
from scripts import rl_smoke as smoke

ROOT = REPO / ".rl-smoke/phase12"


def runtime_sources() -> list[Path]:
    sources = [REPO / "rl/pyproject.toml", REPO / "rl/uv.lock"]
    for phase in ("phase8b", "phase8c", "phase9", "phase11", "phase12"):
        sources.extend(sorted((REPO / "rl" / phase).glob("*.py")))
    sources.extend([config.MANIFEST, REPO / "rl/phase12/remote_run.sh"])
    sources.extend(sorted((REPO / "scripts").glob("*.py")))
    return sources


def dispatch(run: Path, stage: str) -> dict[str, Any]:
    """Submit one stage to an already provisioned account-guarded worker; no provisioning."""
    from scripts.rl_harbor_generalization import _aws

    run_id = smoke.validate_run_id(f"{run.name}-{stage}")
    issued = run / "dispatch" / f"{stage}.json"
    if issued.exists():
        raise ValueError("stage already dispatched; fetch its status/evidence instead")
    bundle = ROOT / "bundles" / f"{run_id}.tar.gz"
    proof = pack(run, stage, bundle)
    ssm, instance_id, _ = _aws()
    remote = smoke.upload_payload(
        ssm, instance_id, run_id, bundle.read_bytes(), proof["archive_sha256"], parallel_chunks=True
    )
    command = shlex.join(["bash", f"{remote}/rl/phase12/remote_run.sh", remote, stage])
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command], "executionTimeout": ["172800"]},
        TimeoutSeconds=172800,
    )
    value = {
        "command_id": response["Command"]["CommandId"],
        "instance_id": instance_id,
        "remote": remote,
        "stage": stage,
        "bundle_sha256": proof["archive_sha256"],
    }
    write(issued, value)
    return value


def fetch(run: Path, stage: str) -> dict[str, Any]:
    """Nonblocking status; after completion, checksum and retain every evidence file."""
    from scripts.rl_harbor_generalization import _aws

    issued = read(run / "dispatch" / f"{stage}.json")
    ssm, instance_id, _ = _aws()
    if instance_id != issued["instance_id"]:
        raise ValueError("provisioned worker differs from dispatched worker")
    status = ssm.get_command_invocation(CommandId=issued["command_id"], InstanceId=instance_id)
    if status["Status"] in {"Pending", "InProgress", "Delayed", "Cancelling"}:
        return {"status": status["Status"], "evidence_fetched": False}
    remote = issued["remote"]
    # stdlib-only so diagnostics remain recoverable if runtime imports fail.
    code = """import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); items={}
for path in sorted(root.rglob('*')):
    relative=path.relative_to(root)
    excluded={'datasets','wheel','checkpoints','dispatch','.remote-download'}
    if any(p in excluded for p in relative.parts): continue
    if path.name == '.rl-artifacts.tar.gz': continue
    if path.is_symlink(): raise ValueError('symlink in remote evidence')
    if path.is_file():
        items[relative.as_posix()]={'size':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
Path(sys.argv[2]).write_text(json.dumps(items))
"""
    command = (
        f"python3 - {shlex.quote(remote + '/run')} "
        f"{shlex.quote(remote + '/inventory.json')} <<'PY'\n{code}PY"
    )
    result = smoke.run_shell(ssm, instance_id, [command], timeout=600)
    if not result.succeeded:
        raise ValueError("could not inventory remote evidence")
    inventory = json.loads(smoke._read_remote_file(ssm, instance_id, remote + "/inventory.json"))
    with tempfile.TemporaryDirectory(prefix="phase12-fetch-") as temporary:
        staging = Path(temporary) / "evidence"
        smoke.download_remote_directory(
            ssm,
            instance_id,
            remote + "/run",
            staging,
            inventory,
            journal_path=run / "dispatch" / f"{stage}-transfer.json",
        )
        for relative in inventory:
            source, target = staging / relative, run / relative
            if target.exists() and digest(target) != digest(source):
                raise ValueError(f"local evidence differs from worker: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write(
        run / "dispatch" / f"{stage}-status.json",
        {
            "status": status["Status"],
            "response_code": status.get("ResponseCode"),
            "stderr": smoke.redact_text(status.get("StandardErrorContent", "")),
        },
    )
    return {"status": status["Status"], "evidence_fetched": True, "files": len(inventory)}


def config_check(*, native: bool = False) -> dict[str, Any]:
    exp, frozen = config.load(), corpus.check()
    maximum = max(
        t["required_tool_calls"] for role in frozen["roles"].values() for t in role.values()
    )
    if maximum + 1 > exp["rollout"]["max_tool_calling_iterations"]:
        raise ValueError("tool iteration budget too small")
    if native:
        config.grpo_config(
            exp, Path("/tmp/phase12-config-check"), 1024, 42, training=True, cpu=True
        )
        config.sft_config(exp, Path("/tmp/phase12-config-check"), 42, cpu=True)
        import inspect

        from transformers.utils import get_json_schema

        from rl.phase12.environment import ComparisonEnv
        from rl.phase12.runtime import tool_schemas

        env = ComparisonEnv(environment_type="docker")
        try:
            tools = [
                method
                for name, method in inspect.getmembers(env, inspect.ismethod)
                if not name.startswith("_") and name != "reset"
            ]
            if [get_json_schema(method) for method in tools] != tool_schemas():
                raise ValueError("SFT and native GRPO tool schemas or order differ")
        finally:
            env._run(env._stop())
    return {
        "execution_valid": True,
        "training_performed": False,
        "native_config_checked": native,
        "counts": {r: len(tasks) for r, tasks in frozen["roles"].items()},
        "maximum_required_calls": maximum,
        "corpus_sha256": frozen["corpus_sha256"],
    }


def prepare(run: Path, calibration_path: Path) -> dict[str, Any]:
    exp, frozen = config.load(), corpus.check()
    proof = read(calibration_path)
    calibration.validate(proof, frozen)
    revision, clean = smoke.git_provenance()
    if not clean:
        raise ValueError("execution preparation requires a clean committed source revision")
    run.mkdir(parents=True, exist_ok=False)
    corpus.render(run / "datasets", ("train", "dev"))
    write(run / "experiment.json", exp)
    write(run / "corpus-manifest.json", corpus.transport_manifest(frozen))
    write(run / "calibration.json", proof)
    wheel, _ = build_clean_wheel(ROOT / "wheel")
    destination = run / "wheel" / wheel.name
    destination.parent.mkdir()
    shutil.copy2(wheel, destination)
    identity = {
        "schema_version": "1",
        "source_revision": revision,
        "worktree_clean": True,
        "experiment_sha256": fingerprint(exp),
        "corpus_sha256": frozen["corpus_sha256"],
        "inputs": {
            p: digest(run / p)
            for p in (
                "experiment.json",
                "corpus-manifest.json",
                "calibration.json",
                str(destination.relative_to(run)),
            )
        },
        "dataset_files": artifacts.files(run / "datasets"),
        "runtime_sources": {p.relative_to(REPO).as_posix(): digest(p) for p in runtime_sources()},
    }
    write(run / "identity.json", identity)
    return identity


def prepare_final(run: Path) -> dict[str, Any]:
    lock = artifacts.verify_lock(run)
    if (run / "test-opened.json").exists():
        raise ValueError("cannot change final payload after test evaluation starts")
    if (run / "final-inputs.json").exists():
        existing = read(run / "final-inputs.json")
        artifacts.verify_files(run, existing["files"])
        return existing
    frozen = corpus.check()
    if frozen["corpus_sha256"] != lock["corpus_sha256"]:
        raise ValueError("final corpus differs from training identity")
    with tempfile.TemporaryDirectory(prefix="phase12-final-") as temporary:
        source = Path(temporary) / "datasets"
        corpus.render(source, ("test",))
        shutil.copytree(source / "test", run / "datasets/test")
    catalog = run / "datasets/anchors/scenarios"
    canonical = run / "datasets/anchors/canonical"
    metadata = {}
    for entry in load_catalog():
        scenario = entry.scenario
        asyncio.run(render_harbor_task(scenario, canonical / scenario.id))
        target = catalog / scenario.id / "scenario.yaml"
        target.parent.mkdir(parents=True)
        target.write_bytes((canonical / scenario.id / "scenario.yaml").read_bytes())
        b = scenario.benchmark
        metadata[scenario.id] = {
            "family_id": scenario.id,
            "prototype_id": scenario.id,
            "archetype": b.archetype,
            "difficulty": b.difficulty,
            "expected_write_targets": b.mutation_policy.allowed_targets,
            "expected_operations": {
                i.invoice_id: "refund_invoice" if i.type == "must_refund" else "escalate_dispute"
                for i in scenario.expected_invariants
                if i.type in {"must_refund", "must_escalate"}
            },
        }
    derive_execution_suite(
        run / "datasets/anchors/execution", catalog=catalog, canonical_tasks=canonical
    )
    write(run / "datasets/anchors/metadata.json", metadata)
    result = {
        "selection_lock_sha256": digest(run / "selection-lock.json"),
        "files": {
            f"datasets/{role}/{p}": sha
            for role in ("test", "anchors")
            for p, sha in artifacts.files(run / "datasets" / role).items()
        },
    }
    write(run / "final-inputs.json", result)
    return result


def report(run: Path, output: Path) -> dict[str, Any]:
    identity = artifacts.verify_run(run)
    final = read(run / "final/result.json") if (run / "final/result.json").exists() else None
    value = {
        "source_revision": identity["source_revision"],
        "corpus_sha256": identity["corpus_sha256"],
        "execution_valid": None
        if final is None
        else all(v["execution_valid"] for v in final.values()),
        "behavioral_improvement": None
        if final is None
        else {k: v["behavioral_improvement"] for k, v in final.items()},
        "replication_complete": final is not None,
        "final": final,
        "final_test_opened": (run / "test-opened.json").exists(),
        "gpu_acceptance_pending": final is None,
    }
    lines = [
        "# Phase 12 SFT/GRPO comparison",
        "",
        f"Source: `{identity['source_revision']}`.",
        "",
        "Technical execution, behavioral improvement, and replication are separate results.",
        "",
        "No final learning result is available." if final is None else "Final results:",
        "",
        "```json",
        json.dumps(value, indent=2),
        "```",
        "",
        "## Evidence",
        "",
    ]
    for pattern in (
        "diagnostics/selection.json",
        "*/42/selection.json",
        "*/*/training/training-result.json",
        "*/*/training/sft-data.json",
        "*/*/training/reward-groups.json",
        "final/anchors/*/*/comparison.json",
    ):
        for path in sorted(run.glob(pattern)):
            lines.extend(
                [
                    f"### {path.relative_to(run)}",
                    "",
                    "```json",
                    json.dumps(read(path), indent=2),
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "Per-task outcomes, clipping, parser/tool errors, prohibited writes, and failures",
            "are retained in each evaluation's summary.json and pass-level rollouts.jsonl.",
            "Elapsed trainer seconds measure occupied single-GPU wall time; SFT and GRPO budgets",
            "are declared separately, not presented as equal compute.",
            "",
            "These synthetic-task results do not establish production readiness.",
        ]
    )
    lines.extend(
        [
            "",
            "## Evaluation summary",
            "",
            "| Evaluation | Prototype | Action | Prohibited writes | Clipped | Tool errors |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for path in sorted(run.rglob("summary.json")):
        summary = read(path)
        if "prototype_macro_pass_rate" not in summary:
            continue
        lines.append(
            f"| {path.parent.relative_to(run)} | {summary['prototype_macro_pass_rate']:.3f} "
            f"| {summary['action_macro_pass_rate']:.3f} | {summary['prohibited_write_rate']:.3f} "
            f"| {summary['diagnostics']['clipped_fraction']:.3f} "
            f"| {summary['diagnostics']['tool_errors']} |"
        )
    lines.extend(["", "## Archetype outcomes", ""])
    for path in sorted((run / "final/test").rglob("summary.json")):
        summary = read(path)
        lines.append(
            f"- {path.parent.relative_to(run)}: "
            + "; ".join(f"{key} {v['pass_rate']:.3f}" for key, v in summary["archetypes"].items())
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    write(output.with_suffix(".json"), value)
    return value


def pack(run: Path, stage: str, destination: Path) -> dict[str, Any]:
    """Prepare a portable worker bundle; final bundles exclude train/dev task contents."""
    identity = artifacts.verify_run(run)
    revision, clean = smoke.git_provenance()
    if revision != identity["source_revision"] or not clean:
        raise ValueError("worker source must match the clean run revision")
    final = stage == "final"
    if final:
        artifacts.verify_lock(run)
        if not (run / "final-inputs.json").exists():
            raise ValueError("prepare-final is required")
    elif (run / "selection-lock.json").exists():
        raise ValueError("locked run cannot dispatch training stages")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run)
        if path.is_symlink():
            raise ValueError("run contains symlink")
        if not path.is_file() or any(part in {"checkpoints", "jobs"} for part in relative.parts):
            continue
        if relative.parts[0] == "datasets" and relative.parts[1] not in (
            ("test", "anchors") if final else ("train", "dev")
        ):
            continue
        if final and relative.parts[0] == "demonstrations":
            continue
        entries.append((path, "run/" + relative.as_posix()))
    # Authored prototype instructions and generated test YAMLs never enter training bundles.
    entries.extend((path, path.relative_to(REPO).as_posix()) for path in runtime_sources())
    with tarfile.open(destination, "w:gz") as archive:
        for path, name in entries:
            archive.add(path, arcname=name, recursive=False)
    return {
        "archive_sha256": digest(destination),
        "stage": stage,
        "files": [name for _, name in entries],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "config-check",
            "render-smoke",
            "calibrate",
            "prepare",
            "execute",
            "dispatch",
            "fetch",
            "lock",
            "prepare-final",
            "pack",
            "report",
            "export",
            "publish",
        ],
    )
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--stage", choices=["diagnose", "demos", "pilot", "replicate", "final"])
    parser.add_argument("--destination")
    args = parser.parse_args()
    if (
        args.command not in ("config-check", "render-smoke", "calibrate", "publish")
        and args.run is None
    ):
        parser.error("--run is required")
    if args.command in ("calibrate", "pack", "report", "export", "publish") and args.output is None:
        parser.error("--output is required")
    if args.command in ("pack", "execute", "dispatch", "fetch") and args.stage is None:
        parser.error("--stage is required")
    run = args.run.resolve() if args.run else None
    if args.command == "config-check":
        result = config_check(native=args.native)
    elif args.command == "render-smoke":
        with tempfile.TemporaryDirectory(prefix="phase12-render-") as temporary:
            corpus.render(Path(temporary) / "datasets", ("train", "dev", "test"))
        result = {"execution_valid": True, "rendered": 340, "model_evaluation_performed": False}
    elif args.command == "calibrate":
        result = calibration.calibrate(args.output)
    elif args.command == "prepare":
        if args.calibration is None:
            parser.error("--calibration is required")
        result = prepare(run, args.calibration)
    elif args.command == "execute":
        from rl.phase12.runner import execute

        execute(args.stage, run)
        result = {"stage_completed": args.stage}
    elif args.command == "lock":
        result = artifacts.create_lock(run)
    elif args.command == "dispatch":
        result = dispatch(run, args.stage)
    elif args.command == "fetch":
        result = fetch(run, args.stage)
    elif args.command == "prepare-final":
        result = prepare_final(run)
    elif args.command == "pack":
        result = pack(run, args.stage, args.output)
    elif args.command == "report":
        result = report(run, args.output)
    elif args.command == "export":
        result = artifacts.export(run, args.output)
    else:
        if not args.destination:
            parser.error("--destination is required")
        result = artifacts.publish(args.output, args.destination)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
