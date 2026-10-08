"""SSM-only controller for canonical Phase 11 GRPO generalization."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentops_demo.benchmark.catalog import load_catalog
from agentops_demo.benchmark.phase10_acceptance import _active
from agentops_demo.benchmark.phase10_variants import PHASE10
from agentops_demo.harbor.provenance import build_clean_wheel
from agentops_demo.validation.scenario import load_scenario
from rl.phase8c.execution_suite import derive_execution_suite
from rl.phase11.corpus import subset_suite, validate_subset
from rl.phase11.experiment import EXPERIMENT_PATH, sha256, validate_lineage
from rl.phase11.selection import adapter_files, dev_gate, verify_lock
from rl.phase11.train import write_json
from rl.phase11.training_config import training_config
from scripts import rl_smoke as smoke

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / ".rl-smoke" / "phase11"
PHASE10_ACCEPTANCE = REPO / ".rl-smoke" / "phase10" / "acceptance"
OUTPUTS = REPO / ".rl-smoke" / "terraform-outputs.json"


def _run_id(requested: str | None) -> str:
    return smoke.validate_run_id(
        requested or "phase11-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    )


def _resolve_run(requested: str | None) -> tuple[str, Path]:
    if requested:
        run_id = smoke.validate_run_id(requested)
    else:
        run_id = str(
            smoke.load_json_object(ROOT / "latest-run.json", "Phase 11 latest run")["run_id"]
        )
    return run_id, ROOT / "runs" / run_id


def _suite() -> tuple[Path, dict[str, Any]]:
    _, path, manifest = _active(PHASE10)
    return path, manifest


def _acceptance(lineage: dict[str, str]) -> dict[str, Any]:
    path = PHASE10_ACCEPTANCE / lineage["corpus_sha256"] / "summary.json"
    value = smoke.load_json_object(path, "Phase 10 acceptance")
    if (
        value.get("status") != "PASS"
        or value.get("execution_suite_sha256") != lineage["execution_suite_sha256"]
        or value.get("split_manifest_sha256") != lineage["split_manifest_sha256"]
        or value.get("oracle", {}).get("passes") != 200
        or value.get("negative_calibration", {}).get("expected_failures_observed") != 40
    ):
        raise ValueError("Phase 10 Oracle/calibration acceptance is unavailable")
    return value


def config_check() -> dict[str, Any]:
    source, suite = _suite()
    validated = validate_lineage(suite_sha256=suite["execution_suite_sha256"])
    _acceptance(validated["lineage"])
    config = training_config(Path("/tmp/phase11-config-check"), cpu_check=True)
    if (
        config.max_steps,
        config.num_generations,
        config.save_steps,
        config.per_device_train_batch_size,
    ) != (320, 4, 160, 4):
        raise ValueError("Phase 11 TRL config differs from frozen contract")
    if set(suite["task_ids"]) != set(
        json.loads((PHASE10 / "generated" / "corpus-manifest.json").read_text())["tasks"]
    ):
        raise ValueError("Phase 10 suite task IDs differ from corpus")
    return {
        "status": "PASS",
        "experiment_id": validated["experiment"]["experiment_id"],
        "lineage": validated["lineage"],
        "train_representatives": len(validated["representatives"]),
        "counts": {role: len(data["task_ids"]) for role, data in validated["roles"].items()},
        "suite_root": str(source),
    }


def _copy_code(stage: Path, wheel: Path) -> None:
    required = [
        REPO / "rl" / "pyproject.toml",
        REPO / "rl" / "uv.lock",
        *sorted((REPO / "rl" / "phase8b").glob("*.py")),
        *sorted((REPO / "rl" / "phase8c").glob("*.py")),
        *sorted((REPO / "rl" / "phase9").glob("*.py")),
        *sorted((REPO / "rl" / "phase11").glob("*.py")),
        REPO / "rl" / "phase11" / "experiment.json",
        REPO / "rl" / "phase11" / "remote_train.sh",
        REPO / "rl" / "phase11" / "remote_final_eval.sh",
    ]
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = stage / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    target = stage / "wheel" / wheel.name
    target.parent.mkdir(parents=True)
    shutil.copy2(wheel, target)


def _pack(stage: Path) -> tuple[bytes, str]:
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w:gz") as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(stage).as_posix())
    payload = memory.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def _metadata(ids: set[str]) -> dict[str, dict[str, Any]]:
    corpus = json.loads((PHASE10 / "generated" / "corpus-manifest.json").read_text())
    result = {}
    for key in sorted(ids):
        scenario = load_scenario(PHASE10 / "generated" / "scenarios" / key / "scenario.yaml")
        result[key] = {
            **corpus["tasks"][key],
            "expected_write_targets": list(scenario.benchmark.mutation_policy.allowed_targets),
        }
    return result


def _identity(validated: dict[str, Any], wheel_sha: str, anchor_sha: str) -> dict[str, Any]:
    revision, clean = smoke.git_provenance()
    if not clean:
        raise ValueError("canonical Phase 11 run requires a clean committed worktree")
    return {
        "schema_version": "1",
        "experiment_id": validated["experiment"]["experiment_id"],
        "experiment_sha256": validated["experiment_sha256"],
        "source_git_revision": revision,
        "worktree_clean": clean,
        "phase10": validated["lineage"],
        "model_id": validated["experiment"]["model_id"],
        "wheel_sha256": wheel_sha,
        "anchor_suite_sha256": anchor_sha,
        "task_id_sha256": {
            role: hashlib.sha256(
                json.dumps(data["task_ids"], separators=(",", ":")).encode()
            ).hexdigest()
            for role, data in validated["roles"].items()
        },
        "train_representative_ids_sha256": hashlib.sha256(
            json.dumps(validated["representatives"], separators=(",", ":")).encode()
        ).hexdigest(),
    }


def build_train_payload(run: Path) -> tuple[bytes, str]:
    source, suite = _suite()
    validated = validate_lineage(suite_sha256=suite["execution_suite_sha256"])
    acceptance = _acceptance(validated["lineage"])
    stage = ROOT / "stage" / f"{run.name}-train"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    wheel, package = build_clean_wheel(ROOT / "wheel")
    retained_wheel = run / "source-wheel" / wheel.name
    retained_wheel.parent.mkdir(parents=True)
    shutil.copy2(wheel, retained_wheel)
    anchor_check = derive_execution_suite(ROOT / "stage" / f"{run.name}-anchor-check")
    identity = _identity(validated, package.sha256, anchor_check.execution_suite_sha256)
    _copy_code(stage, wheel)
    roles = validated["roles"]
    allowed = set(roles["train"]["task_ids"]) | set(roles["dev"]["task_ids"])
    subset_suite(source, stage / "dataset", allowed)
    validate_subset(stage / "dataset", allowed)
    if any(task_id in allowed for task_id in roles["holdout"]["task_ids"]):
        raise ValueError("holdout executable task leaked into train payload")
    manifests = stage / "manifests"
    write_json(manifests / "metadata.json", _metadata(allowed))
    write_json(manifests / "split.json", {role: data["task_ids"] for role, data in roles.items()})
    for role in roles:
        write_json(manifests / f"{role}-ids.json", roles[role]["task_ids"])
    write_json(manifests / "train-representatives-ids.json", validated["representatives"])
    write_json(manifests / "identity.json", identity)
    write_json(manifests / "phase10-acceptance.json", acceptance)
    write_json(run / "identity.json", identity)
    shutil.copy2(EXPERIMENT_PATH, run / "experiment.json")
    write_json(run / "train-representatives.json", validated["representatives"])
    payload, digest_value = _pack(stage)
    write_json(
        run / "train-payload.json",
        {
            "sha256": digest_value,
            "executable_task_ids": sorted(allowed),
            "holdout_task_directories": 0,
        },
    )
    return payload, digest_value


def _anchor_metadata() -> dict[str, dict[str, Any]]:
    entries = load_catalog()
    if len(entries) != 25:
        raise ValueError("historical anchor suite must contain 25 tasks")
    return {
        entry.scenario.id: {
            "family_id": entry.scenario.id,
            "prototype_id": entry.scenario.id,
            "archetype": entry.scenario.benchmark.archetype,
            "difficulty": entry.scenario.benchmark.difficulty,
            "expected_write_targets": list(
                entry.scenario.benchmark.mutation_policy.allowed_targets
            ),
        }
        for entry in entries
    }


def build_final_payload(run: Path) -> tuple[bytes, str]:
    source, suite = _suite()
    validated = validate_lineage(suite_sha256=suite["execution_suite_sha256"])
    selection_check(run)
    identity = smoke.load_json_object(run / "identity.json", "Phase 11 identity")
    stage = ROOT / "stage" / f"{run.name}-final"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    revision, clean = smoke.git_provenance()
    if not clean or revision != identity["source_git_revision"]:
        raise ValueError("final evaluation source revision differs from selection lock")
    wheels = list((run / "source-wheel").glob("*.whl"))
    if len(wheels) != 1 or sha256(wheels[0]) != identity["wheel_sha256"]:
        raise ValueError("final evaluation wheel differs from selected source revision")
    _copy_code(stage, wheels[0])
    holdout = set(validated["roles"]["holdout"]["task_ids"])
    subset_suite(source, stage / "dataset", holdout)
    validate_subset(stage / "dataset", holdout)
    anchor = derive_execution_suite(stage / "anchors")
    if anchor.manifest["task_count"] != 25:
        raise ValueError("historical anchor execution suite changed")
    if anchor.execution_suite_sha256 != identity["anchor_suite_sha256"]:
        raise ValueError("historical anchor suite changed after selection")
    manifests = stage / "manifests"
    write_json(manifests / "metadata.json", _metadata(holdout))
    write_json(manifests / "anchor-metadata.json", _anchor_metadata())
    write_json(manifests / "holdout-ids.json", sorted(holdout))
    write_json(manifests / "anchors-ids.json", anchor.manifest["task_ids"])
    write_json(manifests / "identity.json", identity)
    for relative in (
        "selected-adapter/adapter_config.json",
        "selected-adapter/adapter_model.safetensors",
        "selection-lock.json",
        "selection-decision.json",
        "training/training-config.json",
        "training/adapter-manifest.json",
        "dev/baseline/rollouts.jsonl",
        "dev/step-160/rollouts.jsonl",
        "dev/step-320/rollouts.jsonl",
        "dev/baseline/task-summary.json",
        "dev/step-160/task-summary.json",
        "dev/step-320/task-summary.json",
    ):
        source_file = run / relative
        if not source_file.is_file():
            raise ValueError(f"final payload missing locked evidence: {relative}")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
    if any(
        (stage / "dataset" / "tasks" / task_id).exists()
        for task_id in validated["roles"]["train"]["task_ids"]
    ):
        raise ValueError("train executable task leaked into final payload")
    payload, digest_value = _pack(stage)
    write_json(
        run / "final-payload.json",
        {"sha256": digest_value, "holdout_task_directories": 20, "anchor_task_directories": 25},
    )
    return payload, digest_value


def _aws() -> tuple[Any, str, Any]:
    outputs = smoke.load_json_object(OUTPUTS, "RL smoke Terraform outputs")
    instance_id = smoke.terraform_output_value(outputs, "instance_id")
    sts, ec2, ssm = smoke._aws_clients()
    smoke.require_account(sts)
    smoke.wait_for_instance(ec2, instance_id)
    smoke.wait_for_ssm(ssm, instance_id)
    return ssm, instance_id, outputs


def _remote_manifest(ssm: Any, instance_id: str, remote: str) -> dict[str, Any]:
    run = f"{remote}/run"
    command = f"test -d {run} && python3 {remote}/rl/phase11/artifacts.py --run {run} >/dev/null"
    result = smoke.run_shell(ssm, instance_id, [command], timeout=3600)
    if not result.succeeded:
        raise RuntimeError(f"remote evidence manifest failed: {result.stderr}")
    return json.loads(smoke._read_remote_file(ssm, instance_id, f"{run}/evidence-manifest.json"))


def _fetch(ssm: Any, instance_id: str, run_id: str, destination: Path) -> None:
    remote = f"{smoke.REMOTE_ROOT}/{run_id}"
    manifest = _remote_manifest(ssm, instance_id, remote)
    files = {
        item["path"]: {"size": item["size"], "sha256": item["sha256"]}
        for item in manifest["artifacts"]
    }
    staging = destination / ".remote-download"
    if staging.exists():
        shutil.rmtree(staging)
    smoke.download_remote_directory(
        ssm,
        instance_id,
        f"{remote}/run",
        staging,
        files,
        journal_path=destination / "evidence-transfer.journal.json",
    )
    for source in staging.rglob("*"):
        if source.is_file():
            target = destination / source.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.read_bytes() != source.read_bytes():
                raise ValueError(f"retained evidence changed: {target}")
            shutil.copy2(source, target)
    shutil.rmtree(staging)
    write_json(destination / "evidence-manifest.json", manifest)


def _execute(mode: str, run_id: str, destination: Path, payload: bytes, digest_value: str) -> None:
    ssm, instance_id, _ = _aws()
    remote = smoke.upload_payload(
        ssm, instance_id, run_id, payload, digest_value, parallel_chunks=True
    )
    script = "remote_train.sh" if mode == "train" else "remote_final_eval.sh"
    command = f"bash {remote}/rl/phase11/{script} {remote} {remote}/run"
    issued = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command], "executionTimeout": ["172800"]},
        TimeoutSeconds=172800,
    )
    command_id = issued["Command"]["CommandId"]
    write_json(
        destination / "ssm-command.json",
        {"command_id": command_id, "instance_id": instance_id, "mode": mode},
    )
    result = smoke.wait_command(ssm, instance_id, command_id, timeout=172800)
    try:
        _fetch(ssm, instance_id, run_id, destination)
    finally:
        if not result.succeeded:
            try:
                log = smoke._read_remote_file(ssm, instance_id, f"{remote}/run.log")
                (destination / "remote.log").write_bytes(log)
            except smoke.SmokeError:
                pass
    if not result.succeeded:
        raise RuntimeError(f"Phase 11 {mode} failed; evidence: {destination}")


def train(requested: str | None) -> Path:
    run_id = _run_id(requested)
    destination = ROOT / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    write_json(ROOT / "latest-run.json", {"run_id": run_id})
    payload, digest_value = build_train_payload(destination)
    _execute("train", run_id, destination, payload, digest_value)
    return destination


def train_evidence(requested: str | None) -> Path:
    run_id, destination = _resolve_run(requested)
    if (destination / "selection-lock.json").is_file():
        selection_check(destination)
        return destination
    ssm, instance_id, _ = _aws()
    _fetch(ssm, instance_id, run_id, destination)
    return destination


def selection_check(run: Path) -> dict[str, Any]:
    identity = smoke.load_json_object(run / "identity.json", "Phase 11 identity")
    if identity.get("experiment_sha256") != sha256(EXPERIMENT_PATH):
        raise ValueError("Phase 11 experiment changed since source selection")
    _, active_suite = _suite()
    current = validate_lineage(suite_sha256=active_suite["execution_suite_sha256"])
    if identity.get("phase10") != current["lineage"]:
        raise ValueError("Phase 11 source lineage changed since selection")
    lock = smoke.load_json_object(run / "selection-lock.json", "Phase 11 selection lock")
    verify_lock(
        lock,
        run,
        identity["experiment_sha256"],
        identity["phase10"]["split_manifest_sha256"],
        identity["source_git_revision"],
    )
    summary = smoke.load_json_object(run / "summary.json", "Phase 11 training summary")
    if summary.get("status") != "PASS" or summary.get("training_execution_valid") is not True:
        raise ValueError("Phase 11 training technical acceptance failed")
    baseline = smoke.load_json_object(
        run / "dev" / "baseline" / "task-summary.json", "dev baseline"
    )
    selected = smoke.load_json_object(
        run / "dev" / f"step-{lock['selected_step']}" / "task-summary.json", "selected dev"
    )
    return {
        "status": "PASS",
        "selected_step": lock["selected_step"],
        "dev_gate_passed": dev_gate(baseline, selected),
        "adapter_files": len(adapter_files(run / "selected-adapter")),
    }


def final_eval(requested: str | None) -> Path:
    run_id, destination = _resolve_run(requested)
    if not selection_check(destination)["dev_gate_passed"]:
        raise ValueError("canonical dev gate failed; holdout remains unopened")
    payload, digest_value = build_final_payload(destination)
    final_destination = destination / "final"
    final_destination.mkdir(parents=True, exist_ok=False)
    _execute("final", run_id, final_destination, payload, digest_value)
    return final_destination


def final_evidence(requested: str | None) -> Path:
    run_id, destination = _resolve_run(requested)
    final_destination = destination / "final"
    if (final_destination / "summary.json").is_file():
        return final_destination
    ssm, instance_id, _ = _aws()
    final_destination.mkdir(parents=True, exist_ok=True)
    _fetch(ssm, instance_id, run_id, final_destination)
    return final_destination


def report(requested: str | None) -> Path:
    _, run = _resolve_run(requested)
    summary = smoke.load_json_object(run / "summary.json", "Phase 11 training summary")
    final_path = run / "final" / "summary.json"
    final = (
        smoke.load_json_object(final_path, "Phase 11 final summary")
        if final_path.is_file()
        else None
    )
    identity = smoke.load_json_object(run / "identity.json", "Phase 11 identity")
    selected = summary.get("selected_step")
    lines = [
        "# Phase 11 GRPO held-out generalization",
        "",
        f"Experiment: `{identity['experiment_id']}` at source revision "
        f"`{identity['source_git_revision']}` (clean worktree).",
        f"Training status: **{summary['status']}**.",
        "",
        "## Frozen lineage",
        "",
    ]
    for key, value in identity["phase10"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        f"- anchor_suite_sha256: `{identity['anchor_suite_sha256']}`",
        f"- wheel_sha256: `{identity['wheel_sha256']}`",
        "",
    ]
    versions_path = run / "versions.json"
    if versions_path.is_file():
        versions = smoke.load_json_object(versions_path, "Phase 11 versions")
        lines += [
            "Software: "
            + ", ".join(
                f"{key} {versions.get(key)}" for key in ("torch", "trl", "vllm", "harbor", "peft")
            )
            + ".",
            "",
        ]
    signal_path = run / "preflight" / "grpo-signal.json"
    if signal_path.is_file():
        signal = smoke.load_json_object(signal_path, "GRPO signal")
        lines += [
            "## Frozen characterization",
            "",
            f"Among 32 predetermined train representatives, {signal['mixed_groups']} "
            f"groups were mixed, {signal['all_zero_groups']} all zero, and "
            f"{signal['all_one_groups']} all one.",
            "",
        ]
    if selected is not None:
        training = smoke.load_json_object(run / "training" / "training-result.json", "training")
        delta = smoke.load_json_object(run / "training" / "adapter-delta.json", "adapter delta")
        reward_groups = smoke.load_json_object(
            run / "training" / "reward-group-summary.json", "reward group summary"
        )
        lines += [
            "## Training",
            "",
            f"Fresh Qwen3-0.6B zero-init LoRA; {training['global_step']} optimizer steps, "
            f"{training['reward_groups']} four-rollout groups, "
            f"{training['training_rollouts']} rollouts.",
            "Every one of 160 train tasks had exactly two groups and eight rollouts.",
            f"Mixed reward-group fraction: {reward_groups['mixed_fraction']:.3f}; "
            f"adapter delta L2: {delta['delta_l2_norm']:.6f} across "
            f"{delta['changed_tensor_count']} changed tensors.",
            "",
        ]
        base_dev = smoke.load_json_object(
            run / "dev" / "baseline" / "task-summary.json", "dev baseline"
        )
        lines += [
            "## Development selection",
            "",
            "Policy | Prototype macro | Micro | Micro Wilson 95%",
            "--- | ---: | ---: | ---",
            f"Frozen baseline | {base_dev['prototype_macro_pass_rate']:.3f} | "
            f"{base_dev['micro_pass_rate']:.3f} | {base_dev['micro_wilson_95']}",
        ]
        decision = smoke.load_json_object(run / "selection-decision.json", "selection decision")
        for step, data in sorted(decision["candidates"].items()):
            lines.append(
                f"Step {step} | {data['prototype_macro_pass_rate']:.3f} | "
                f"{data['micro_pass_rate']:.3f} | {data['micro_wilson_95']}"
            )
        lines += [
            "",
            f"Selected checkpoint: step **{selected}**.",
            "Selection rule: highest prototype macro, then micro, then earlier step.",
            f"Canonical dev gate: **{'PASS' if summary['dev_gate_passed'] else 'FAIL'}**.",
            "",
        ]
        reps = smoke.load_json_object(
            run / "train-representatives-comparison.json", "train representative comparison"
        )
        lines += [
            "## Train representatives",
            "",
            f"Micro pass rate: {reps['baseline']['micro_pass_rate']:.3f} → "
            f"{reps['trained']['micro_pass_rate']:.3f}; "
            f"tool-call rate: {reps['baseline']['tool_call_rate']:.3f} → "
            f"{reps['trained']['tool_call_rate']:.3f}.",
            "",
        ]
    if final is None or final.get("status") != "PASS":
        lines += ["Canonical holdout remains unopened.", ""]
    else:
        anchors = smoke.load_json_object(
            run / "final" / "anchors" / "comparison.json", "anchor comparison"
        )
        lines += [
            "## Historical anchors",
            "",
            f"25 anchor tasks, four attempts per policy. Micro pass rate: "
            f"{anchors['baseline']['micro_pass_rate']:.3f} → "
            f"{anchors['trained']['micro_pass_rate']:.3f}.",
            "Anchor results are regression evidence, not the generalization test.",
            "",
        ]
        for archetype, base in anchors["baseline"]["archetypes"].items():
            trained_rate = anchors["trained"]["archetypes"][archetype]["pass_rate"]
            lines.append(f"- `{archetype}`: {base['pass_rate']:.3f} → {trained_rate:.3f}")
        lines.append("")
        comparison = smoke.load_json_object(
            run / "final" / "holdout" / "comparison.json", "holdout comparison"
        )
        lines += [
            "## Structural holdout",
            "",
            f"Holdout status: **{final['status']}**.",
            f"Micro pass rate: {comparison['baseline']['micro_pass_rate']:.3f} → "
            f"{comparison['trained']['micro_pass_rate']:.3f}.",
            f"Micro Wilson 95% intervals: {comparison['baseline']['micro_wilson_95']} → "
            f"{comparison['trained']['micro_wilson_95']}.",
            "Prototype macro pass rate: "
            f"{comparison['baseline']['prototype_macro_pass_rate']:.3f} → "
            f"{comparison['trained']['prototype_macro_pass_rate']:.3f}.",
            f"Prototypes improved/unchanged/worsened: {comparison['prototypes_improved']}/"
            f"{comparison['prototypes_unchanged']}/{comparison['prototypes_worsened']}.",
            f"Advisory generalization signal: **{final['generalization_signal_supported']}**; "
            "human review required.",
            "",
        ]
        for prototype, value in comparison["prototypes"].items():
            lines.append(f"- `{prototype}`: {value['baseline']:.3f} → {value['trained']:.3f}.")
        lines += [
            "",
            "## Behavioral audit",
            "",
            f"Recorded shortcut candidates: {len(final['obvious_shortcut_candidates'])}.",
            f"Tool-call rate: {comparison['baseline']['tool_call_rate']:.3f} → "
            f"{comparison['trained']['tool_call_rate']:.3f}.",
            "",
        ]
        for prototype, values in comparison["prototypes"].items():
            if values["delta"] <= 0:
                continue
            ids = [
                task_id
                for task_id, task in comparison["tasks"].items()
                if task["baseline"]["prototype_id"] == prototype
            ]
            before = sum(
                comparison["tasks"][task_id]["baseline"]["mean_write_count"] for task_id in ids
            ) / len(ids)
            after = sum(
                comparison["tasks"][task_id]["trained"]["mean_write_count"] for task_id in ids
            ) / len(ids)
            lines.append(
                f"- Improved `{prototype}`: mean writes/task {before:.2f} → "
                f"{after:.2f}; inspect per-task sequences and verifier components in evidence."
            )
        lines.append("")
    lines += [
        "## Limitations",
        "",
        "The scalar Harbor verifier supplied the only training reward. Per-task trajectories, "
        "target correctness, inspection-before-write rates, verifier components, and Wilson "
        "intervals are retained with the run evidence. Four dev families and three holdout "
        "prototypes do not support a precise significance claim. Human review is required.",
        "",
    ]
    path = REPO / "reports" / "phase-11-grpo-generalization.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "config-check",
            "train",
            "train-evidence",
            "selection-check",
            "final-eval",
            "final-evidence",
            "report",
        ),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "config-check":
        print(json.dumps(config_check(), indent=2, sort_keys=True))
    elif args.command == "train":
        print(train(args.run_id))
    elif args.command == "train-evidence":
        print(train_evidence(args.run_id))
    elif args.command == "selection-check":
        _, run = _resolve_run(args.run_id)
        print(json.dumps(selection_check(run), indent=2, sort_keys=True))
    elif args.command == "final-eval":
        print(final_eval(args.run_id))
    elif args.command == "final-evidence":
        print(final_evidence(args.run_id))
    else:
        print(report(args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
