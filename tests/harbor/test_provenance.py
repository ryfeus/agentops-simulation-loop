from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentops_demo.harbor.provenance import is_sha256, sha256_file
from scripts.build_harbor_agent import discover_wheel, wheel_from_manifest, write_manifest
from scripts.generate_harbor_configs import write_configs


def test_file_sha256_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.whl"
    artifact.write_bytes(b"first artifact")

    first = sha256_file(artifact)
    assert is_sha256(first)
    assert sha256_file(artifact) == first

    artifact.write_bytes(b"changed artifact")
    assert sha256_file(artifact) != first


def test_exactly_one_wheel_is_discovered(tmp_path: Path) -> None:
    wheel = tmp_path / "project-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")

    assert discover_wheel(tmp_path) == wheel


def test_zero_wheels_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="found 0"):
        discover_wheel(tmp_path)


def test_multiple_wheels_are_rejected(tmp_path: Path) -> None:
    for name in ("project-1.0-py3-none-any.whl", "project-2.0-py3-none-any.whl"):
        (tmp_path / name).write_bytes(name.encode())

    with pytest.raises(RuntimeError, match="found 2"):
        discover_wheel(tmp_path)


def test_manifest_binds_source_wheel_and_candidate_configs(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    config_dir = tmp_path / "configs"
    package_dir.mkdir()
    wheel = package_dir / "project-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    revision = "a" * 40
    configs = write_configs(config_dir, revision)

    manifest_path = write_manifest(
        package_dir=package_dir,
        config_dir=config_dir,
        source_revision=revision,
        configs=configs,
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["source_revision"] == revision
    assert manifest["package"] == {"filename": wheel.name, "sha256": sha256_file(wheel)}
    assert manifest["candidates"]["bad"]["fingerprint"] == configs["bad"].fingerprint()
    assert manifest["candidates"]["correct"]["model"] == "scripted/correct"
    assert wheel_from_manifest(manifest_path) == wheel.resolve()
