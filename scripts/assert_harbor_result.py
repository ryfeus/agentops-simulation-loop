"""Assert one Harbor job's structured reward and optional agent trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import is_sha256
from agentops_demo.taskify.reproduction import TrialResult
from scripts.build_harbor_agent import load_manifest


def load_job_result(job_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads((job_dir / "result.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Harbor job result: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Harbor job result must be a JSON object")
    return value


def assert_result(
    job_dir: Path,
    expected_reward: float,
    expected_tools: list[str] | None = None,
    provenance_manifest: Path | None = None,
    candidate: str | None = None,
) -> dict[str, Any]:
    observed = read_trial_result("originating", job_dir)
    observed = with_expectations(observed, expected_reward, expected_tools)
    validate_trial(observed)
    trial = _load_trial_payload(job_dir)
    if (provenance_manifest is None) != (candidate is None):
        raise ValueError("provenance manifest and candidate must be provided together")
    if provenance_manifest is not None and candidate is not None:
        _assert_provenance(job_dir, trial, provenance_manifest, candidate)
    return trial


def _load_trial_payload(job_dir: Path) -> dict[str, Any]:
    trials = _load_trial_payloads(job_dir)
    if len(trials) != 1:
        raise AssertionError("expected exactly one Harbor trial result")
    return trials[0]


def _load_trial_payloads(job_dir: Path) -> list[dict[str, Any]]:
    """Load every retained result for a job without discarding failed attempts."""

    result = load_job_result(job_dir)
    trial_results = sorted(job_dir.glob("*/result.json"))
    total = result.get("n_total_trials")
    if not isinstance(total, int) or total < 1:
        raise AssertionError("Harbor job result must declare one or more trials")
    if len(trial_results) != total:
        raise AssertionError("Harbor job result is missing retained trial results")
    trials: list[dict[str, Any]] = []
    for path in trial_results:
        try:
            trial = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read Harbor trial result: {exc}") from exc
        if not isinstance(trial, dict):
            raise AssertionError("Harbor trial result must be an object")
        trials.append(trial)
    return trials


def read_trial_result(name: str, job_dir: Path) -> TrialResult:
    """Read observed structured Harbor evidence without asserting expectations."""

    trial = _load_trial_payload(job_dir)
    return _trial_result_from_payload(name, job_dir, trial)


def read_trial_results(name: str, job_dir: Path) -> list[TrialResult]:
    """Read every retained Harbor attempt, including partial or failed attempts."""

    return [
        _trial_result_from_payload(name, job_dir, trial) for trial in _load_trial_payloads(job_dir)
    ]


def _trial_result_from_payload(name: str, job_dir: Path, trial: dict[str, Any]) -> TrialResult:
    reward: float | None = None
    verifier = trial.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    raw_reward = rewards.get("reward") if isinstance(rewards, dict) else None
    if isinstance(raw_reward, int | float):
        reward = float(raw_reward)
    tool_calls: list[str] = []
    config_fingerprint: str | None = None
    provenance: dict[str, Any] = {}
    try:
        agent_result = _load_agent_log(job_dir, trial, "result.json")
        calls = agent_result.get("tool_calls")
        if isinstance(calls, list):
            tool_calls = [
                call["name"]
                for call in calls
                if isinstance(call, dict) and isinstance(call.get("name"), str)
            ]
        fingerprint = agent_result.get("agent_config_fingerprint")
        if isinstance(fingerprint, str) and is_sha256(fingerprint):
            config_fingerprint = fingerprint
        provenance = _load_agent_log(job_dir, trial, "provenance.json")
    except AssertionError:
        pass
    package_sha256 = provenance.get("package_sha256")
    model = provenance.get("model")
    return TrialResult(
        name=name,
        attempted=True,
        reward=None if trial.get("exception_info") is not None else reward,
        tool_calls=tool_calls,
        job_dir=str(job_dir),
        config_fingerprint=config_fingerprint,
        package_sha256=package_sha256 if is_sha256(package_sha256) else None,
        model=model if isinstance(model, str) else None,
    )


def with_expectations(
    trial: TrialResult, expected_reward: float, expected_tools: list[str] | None = None
) -> TrialResult:
    """Attach expected values and their comparison without discarding observations."""

    data = trial.model_dump(mode="json")
    data["expected_reward"] = expected_reward
    data["reward_matches"] = trial.reward == expected_reward
    if expected_tools is not None:
        data["expected_tool_calls"] = expected_tools
        data["tools_match"] = trial.tool_calls == expected_tools
    return TrialResult.model_validate(data)


def validate_trial(trial: TrialResult) -> None:
    """Fail only after an observed trial has been retained by the caller."""

    if trial.reward is None:
        raise AssertionError("Harbor trial has no numeric reward")
    if trial.reward_matches is not True:
        raise AssertionError(f"expected reward {trial.expected_reward}, received {trial.reward}")
    if trial.expected_tool_calls is not None and trial.tools_match is not True:
        raise AssertionError(
            f"expected tools {trial.expected_tool_calls}, received {trial.tool_calls}"
        )


def validate_runtime_provenance(
    trial: TrialResult, *, config: AgentConfig, package_sha256: str, model: str
) -> None:
    """Verify the installed agent emitted the exact requested runtime identity."""

    if trial.config_fingerprint != config.fingerprint():
        raise AssertionError("runtime config fingerprint does not match candidate")
    if trial.package_sha256 != package_sha256:
        raise AssertionError("runtime package SHA-256 does not match execution package")
    if trial.model != model:
        raise AssertionError("runtime model does not match requested model")


def _load_agent_log(job_dir: Path, trial: dict[str, Any], filename: str) -> dict[str, Any]:
    candidates: list[Path] = []
    trial_uri = trial.get("trial_uri")
    if isinstance(trial_uri, str):
        parsed = urlparse(trial_uri)
        path = Path(unquote(parsed.path if parsed.scheme == "file" else trial_uri))
        candidates.append(path / "agent" / filename)
    candidates.extend(job_dir.glob(f"*/agent/{filename}"))
    for candidate in candidates:
        if candidate.is_file():
            value = json.loads(candidate.read_text())
            if isinstance(value, dict):
                return value
    raise AssertionError(f"agent {filename} log was not found")


def _assert_provenance(
    job_dir: Path,
    trial: dict[str, Any],
    manifest_path: Path,
    candidate: str,
) -> None:
    manifest = load_manifest(manifest_path)
    source_revision = manifest.get("source_revision")
    package = manifest.get("package")
    candidates = manifest.get("candidates")
    expected = candidates.get(candidate) if isinstance(candidates, dict) else None
    package_sha256 = package.get("sha256") if isinstance(package, dict) else None
    if not isinstance(source_revision, str) or not source_revision:
        raise AssertionError("build manifest is missing source revision")
    if not is_sha256(package_sha256):
        raise AssertionError("build manifest is missing a valid package SHA-256")
    if not isinstance(expected, dict):
        raise AssertionError(f"build manifest is missing candidate {candidate!r}")

    config_path = expected.get("config_path")
    if not isinstance(config_path, str):
        raise AssertionError("build manifest is missing candidate config path")
    config = AgentConfig.model_validate_json(Path(config_path).read_text())
    expected_fingerprint = expected.get("fingerprint")
    expected_model = expected.get("model")
    if not is_sha256(expected_fingerprint) or expected_fingerprint != config.fingerprint():
        raise AssertionError("candidate fingerprint does not match generated config")
    if source_revision != config.agent.source_revision:
        raise AssertionError("source revision does not match generated config")
    config_model = f"{config.model.provider}/{config.model.model_id}"
    if expected_model != config_model:
        raise AssertionError("candidate model does not match generated config")

    provenance = _load_agent_log(job_dir, trial, "provenance.json")
    expected_provenance = {
        "agent_config_fingerprint": expected_fingerprint,
        "source_revision": source_revision,
        "package_sha256": package_sha256,
        "model": expected_model,
    }
    if provenance != expected_provenance:
        raise AssertionError(
            f"expected agent provenance {expected_provenance}, received {provenance}"
        )
    for field in ("agent_config_fingerprint", "package_sha256"):
        if not is_sha256(provenance[field]):
            raise AssertionError(f"agent provenance has invalid {field}")

    runtime_result = _load_agent_log(job_dir, trial, "result.json")
    if runtime_result.get("agent_config_fingerprint") != expected_fingerprint:
        raise AssertionError("runtime result fingerprint does not match candidate provenance")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--expected-reward", type=float, required=True)
    parser.add_argument("--expected-tools", nargs="*")
    parser.add_argument("--provenance-manifest", type=Path)
    parser.add_argument("--candidate")
    args = parser.parse_args(argv)
    trial = assert_result(
        args.job,
        args.expected_reward,
        args.expected_tools,
        args.provenance_manifest,
        args.candidate,
    )
    print(f"Task: {trial['task_name']}")
    print(f"Reward: {args.expected_reward}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
