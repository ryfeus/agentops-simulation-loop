"""Immutable stage identities, selection locks, and checksummed portable evidence."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rl.phase11.selection import adapter_files
from rl.phase12.common import REPO, digest, fingerprint, read, write
from rl.phase12.config import load
from rl.phase12.metrics import pilot_gate


def files(root: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in exclude for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("artifact tree contains symlink")
        if path.is_file():
            result[relative.as_posix()] = digest(path)
    return result


def verify_files(root: Path, expected: dict[str, str]) -> None:
    for relative, sha in expected.items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
            raise ValueError("unsafe artifact path")
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"missing or changed evidence: {relative}")


def verify_run(run: Path) -> dict[str, Any]:
    identity = read(run / "identity.json")
    if identity["experiment_sha256"] != fingerprint(load()):
        raise ValueError("run belongs to another experiment configuration")
    verify_files(run, identity["inputs"])
    verify_files(REPO, identity["runtime_sources"])
    if identity.get("worktree_clean") is not True:
        raise ValueError("run source was not clean")
    present = run / "datasets/train"
    if present.exists():
        verify_files(run / "datasets", identity["dataset_files"])
    elif not (run / "final-inputs.json").is_file():
        raise ValueError("training datasets are absent outside a final-only payload")
    from rl.phase12.calibration import validate

    validate(read(run / "calibration.json"), read(run / "corpus-manifest.json"))
    return identity


def eligible_methods(run: Path) -> list[str]:
    exp = load()
    base = read(run / "dev/baseline/summary.json")
    selected = []
    for method in ("sft", "grpo"):
        decision = read(run / f"{method}/42/selection.json")
        summary = read(run / f"{method}/42/dev/{decision['selected_step']}/summary.json")
        if pilot_gate(base, summary, exp):
            selected.append(method)
    return selected


def create_lock(run: Path) -> dict[str, Any]:
    identity = verify_run(run)
    if (run / "selection-lock.json").exists() or (run / "test-opened.json").exists():
        raise ValueError("selection already locked or test already opened")
    methods = eligible_methods(run)
    if not methods:
        raise ValueError("no method qualifies; leave final test unopened")
    protected = {"identity.json", "diagnostics/selection.json", "dev/baseline/summary.json"}
    selections = {}
    for method in methods:
        for seed in load()["seeds"]:
            prefix = f"{method}/{seed}"
            decision = read(run / prefix / "selection.json")
            step = decision["selected_step"]
            training = read(run / prefix / "training/training-result.json")
            if (
                training.get("execution_valid") is not True
                or training.get("method") != method
                or training.get("seed") != seed
            ):
                raise ValueError("selected checkpoint lacks valid training evidence")
            from rl.phase12.metrics import choose

            candidates = {
                int(candidate): read(run / prefix / f"dev/{candidate}/summary.json")
                for candidate in training["adapters"]
            }
            if choose(candidates) != step:
                raise ValueError("checkpoint selection does not follow the registered rule")
            adapter = run / prefix / f"training/adapters/{step}"
            if adapter_files(adapter) != training["adapters"][str(step)]:
                raise ValueError("adapter differs from the training artifact")
            selections[f"{method}/{seed}"] = {
                "step": step,
                "adapter": str(adapter.relative_to(run)),
            }
            protected.add(f"{prefix}/selection.json")
            protected.add(f"{prefix}/training/training-result.json")
            protected.update(f"{prefix}/dev/{p}" for p in files(run / prefix / "dev"))
            protected.update(str(p.relative_to(run)) for p in adapter.rglob("*") if p.is_file())
    protected.update(f"diagnostics/{p}" for p in files(run / "diagnostics"))
    protected.update(f"dev/baseline/{p}" for p in files(run / "dev/baseline"))
    lock = {
        "schema_version": "1",
        "experiment_sha256": identity["experiment_sha256"],
        "corpus_sha256": identity["corpus_sha256"],
        "source_revision": identity["source_revision"],
        "methods": methods,
        "selections": selections,
        "test_opened": False,
        "evidence": {p: digest(run / p) for p in sorted(protected)},
    }
    write(run / "selection-lock.json", lock)
    return lock


def verify_lock(run: Path) -> dict[str, Any]:
    lock = read(run / "selection-lock.json")
    identity = verify_run(run)
    if (
        lock["experiment_sha256"] != identity["experiment_sha256"]
        or lock["corpus_sha256"] != identity["corpus_sha256"]
        or lock["test_opened"] is not False
    ):
        raise ValueError("selection lock identity mismatch")
    verify_files(run, lock["evidence"])
    return lock


def export(run: Path, archive: Path) -> dict[str, Any]:
    if archive.exists() or archive.resolve().is_relative_to(run.resolve()):
        raise ValueError("export must be a new path outside the run")
    manifest = files(run, exclude=("datasets", "checkpoints", "__pycache__"))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as stream:
        for relative in manifest:
            stream.add(run / relative, arcname=relative, recursive=False)
    value = {"schema_version": "1", "archive_sha256": digest(archive), "files": manifest}
    write(archive.with_suffix(archive.suffix + ".json"), value)
    return value


def publish(archive: Path, destination: str) -> dict[str, Any]:
    """Explicit opt-in publication; never called from training or export."""
    from scripts.aws_context import verified_session

    target = urlparse(destination)
    if target.scheme != "s3" or not target.netloc or not target.path.strip("/"):
        raise ValueError("destination must be an explicit s3://bucket/prefix")
    manifest = read(archive.with_suffix(archive.suffix + ".json"))
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("export archive changed")
    session = verified_session()
    # Content addressing prevents overwriting a different experiment's evidence.
    key = f"{target.path.strip('/')}/{manifest['archive_sha256']}/{archive.name}"
    s3 = session.client("s3")
    s3.upload_file(
        str(archive),
        target.netloc,
        key,
        ExtraArgs={"Metadata": {"sha256": manifest["archive_sha256"]}},
    )
    s3.put_object(Bucket=target.netloc, Key=key + ".json", Body=json.dumps(manifest).encode())
    return {"uri": f"s3://{target.netloc}/{key}", "sha256": manifest["archive_sha256"]}
