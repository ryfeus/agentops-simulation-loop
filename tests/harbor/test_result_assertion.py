from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.assert_harbor_result import assert_result, load_job_result
from scripts.build_harbor_agent import write_manifest
from scripts.generate_harbor_configs import write_configs


def write_job(
    path: Path,
    *,
    reward: float | None = 1.0,
    tools: list[str] | None = None,
    exception: object = None,
    trial_count: int = 1,
    fingerprint: str | None = None,
    provenance: dict[str, str] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(json.dumps({"n_total_trials": trial_count}))
    for index in range(trial_count):
        trial_dir = path / f"trial-{index}"
        trial_dir.mkdir()
        verifier = {"rewards": {"reward": reward}} if reward is not None else {"rewards": {}}
        trial = {
            "task_name": "agentops-demo/disputed-refund",
            "trial_uri": trial_dir.resolve().as_uri(),
            "exception_info": exception,
            "verifier_result": verifier,
        }
        (trial_dir / "result.json").write_text(json.dumps(trial))
        if tools is not None or fingerprint is not None or provenance is not None:
            (trial_dir / "agent").mkdir()
        if tools is not None or fingerprint is not None:
            calls = [{"name": name, "arguments": {}} for name in tools or []]
            runtime_result = {
                "tool_calls": calls,
                "agent_config_fingerprint": fingerprint,
            }
            (trial_dir / "agent/result.json").write_text(json.dumps(runtime_result))
        if provenance is not None:
            (trial_dir / "agent/provenance.json").write_text(json.dumps(provenance))


def test_structured_result_and_tools_are_accepted(tmp_path: Path) -> None:
    write_job(tmp_path, reward=0.0, tools=["get_invoice", "refund_invoice"])
    trial = assert_result(tmp_path, 0.0, ["get_invoice", "refund_invoice"])
    assert trial["task_name"] == "agentops-demo/disputed-refund"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward": None},
        {"reward": 0.0},
        {"exception": {"type": "failure"}},
        {"trial_count": 2},
    ],
)
def test_unexpected_result_is_rejected(tmp_path: Path, kwargs: dict[str, object]) -> None:
    write_job(tmp_path, **kwargs)
    with pytest.raises(AssertionError):
        assert_result(tmp_path, 1.0)


def test_malformed_job_result_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text("not json")
    with pytest.raises(ValueError, match="cannot read Harbor job result"):
        load_job_result(tmp_path)


def test_unexpected_tool_trajectory_is_rejected(tmp_path: Path) -> None:
    write_job(tmp_path, tools=["get_invoice", "refund_invoice"])
    with pytest.raises(AssertionError, match="expected tools"):
        assert_result(tmp_path, 1.0, ["get_invoice", "escalate_dispute"])


def provenance_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    revision = "a" * 40
    config_dir = tmp_path / "configs"
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    wheel = package_dir / "project-1.0-py3-none-any.whl"
    wheel.write_bytes(b"candidate wheel")
    configs = write_configs(config_dir, revision)
    manifest_path = write_manifest(
        package_dir=package_dir,
        config_dir=config_dir,
        source_revision=revision,
        configs=configs,
    )
    manifest = json.loads(manifest_path.read_text())
    provenance = {
        "agent_config_fingerprint": manifest["candidates"]["bad"]["fingerprint"],
        "source_revision": revision,
        "package_sha256": manifest["package"]["sha256"],
        "model": "scripted/bad",
    }
    return manifest_path, provenance


def test_candidate_provenance_matches_build_manifest(tmp_path: Path) -> None:
    manifest_path, provenance = provenance_fixture(tmp_path)
    job_dir = tmp_path / "job"
    write_job(
        job_dir,
        reward=0.0,
        tools=["get_invoice", "refund_invoice"],
        fingerprint=provenance["agent_config_fingerprint"],
        provenance=provenance,
    )

    assert_result(
        job_dir,
        0.0,
        ["get_invoice", "refund_invoice"],
        manifest_path,
        "bad",
    )


def test_mismatched_candidate_provenance_is_rejected(tmp_path: Path) -> None:
    manifest_path, provenance = provenance_fixture(tmp_path)
    provenance["package_sha256"] = "0" * 64
    job_dir = tmp_path / "job"
    write_job(
        job_dir,
        reward=0.0,
        fingerprint=provenance["agent_config_fingerprint"],
        provenance=provenance,
    )

    with pytest.raises(AssertionError, match="expected agent provenance"):
        assert_result(job_dir, 0.0, provenance_manifest=manifest_path, candidate="bad")
