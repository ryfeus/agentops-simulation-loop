from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.build_evaluator_lambda import build_package


def test_evaluator_package_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    one = build_package(output=first, manifest=tmp_path / "one.json", revision="a" * 40)
    two = build_package(output=second, manifest=tmp_path / "two.json", revision="a" * 40)
    assert first.read_bytes() == second.read_bytes()
    assert one["package"]["sha256"] == two["package"]["sha256"]
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert "handler.py" in archive.namelist()
        assert "agentops_demo/evaluation/dispute_policy.py" in archive.namelist()
