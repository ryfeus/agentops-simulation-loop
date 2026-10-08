"""Small provenance primitives shared by Harbor packaging and execution."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from pydantic import Field, model_validator

from agentops_demo.contracts._base import ContractModel


class PackageProvenance(ContractModel):
    """The exact clean Git checkout and wheel used for a Harbor execution."""

    source_revision: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worktree_clean: bool

    @model_validator(mode="after")
    def require_clean_worktree(self) -> PackageProvenance:
        if not self.worktree_clean:
            raise ValueError("execution package requires a clean worktree")
        return self


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file using bounded reads."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase hexadecimal SHA-256 digest."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_clean_worktree() -> None:
    """Require a strictly clean Git worktree for execution-package provenance."""

    result = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        raise RuntimeError("working tree must be clean for execution package provenance")


def current_revision() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    )
    revision = result.stdout.strip()
    if not revision:
        raise RuntimeError("Git HEAD is unavailable for execution package provenance")
    return revision


def build_clean_wheel(package_dir: Path) -> tuple[Path, PackageProvenance]:
    """Build exactly one wheel while binding it to an unchanged clean checkout."""

    require_clean_worktree()
    revision = current_revision()
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    subprocess.run(("uv", "build", "--wheel", "--out-dir", str(package_dir)), check=True)
    require_clean_worktree()
    if current_revision() != revision:
        raise RuntimeError("Git HEAD changed while building execution package")
    wheels = sorted(package_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"expected exactly one project wheel in {package_dir}, found {len(wheels)}"
        )
    wheel = wheels[0].resolve()
    return wheel, PackageProvenance(
        source_revision=revision,
        filename=wheel.name,
        sha256=sha256_file(wheel),
        worktree_clean=True,
    )
