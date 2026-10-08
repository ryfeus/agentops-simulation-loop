"""Load, validate, and render the declarative billing benchmark catalog."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from agentops_demo.contracts.scenario import Scenario
from agentops_demo.taskify.harbor_renderer import render_harbor_task
from agentops_demo.taskify.integrity import validate_harbor_task
from agentops_demo.validation.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = ROOT / "benchmarks" / "billing" / "scenarios"


class CatalogError(ValueError):
    def __init__(self, message: str, *, failures: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.failures = failures


class CatalogRenderError(CatalogError):
    """A failed catalog render with counts suitable for a suite-level report."""

    def __init__(
        self,
        message: str,
        *,
        catalog_count: int,
        rendered_count: int,
        failures: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.catalog_count = catalog_count
        self.rendered_count = rendered_count
        self.failures = failures


@dataclass(frozen=True)
class CatalogEntry:
    path: Path
    scenario: Scenario


def _scenario_file_count(root: Path) -> int:
    return len(list(root.rglob("scenario.yaml"))) if root.is_dir() else 0


def _replace_task_tree(staged: Path, output: Path) -> None:
    """Replace one complete task tree while retaining the old tree on failure."""

    backup = output.parent / f".{output.name}.previous-{uuid.uuid4().hex}"
    moved_previous = False
    try:
        if output.exists() or output.is_symlink():
            os.replace(output, backup)
            moved_previous = True
        os.replace(staged, output)
    except OSError:
        if moved_previous and not output.exists() and backup.exists():
            os.replace(backup, output)
        raise
    else:
        if backup.exists() or backup.is_symlink():
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup)
            else:
                backup.unlink()


def load_catalog(root: Path = DEFAULT_CATALOG) -> list[CatalogEntry]:
    if not root.is_dir():
        raise CatalogError(f"billing benchmark catalog is missing: {root}")
    files = sorted(root.rglob("scenario.yaml"))
    if not files:
        raise CatalogError(f"billing benchmark catalog is empty: {root}")
    entries: list[CatalogEntry] = []
    errors: list[str] = []
    for path in files:
        try:
            entries.append(CatalogEntry(path=path, scenario=load_scenario(path)))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"{path}: {exc}")

    seen_ids: dict[str, Path] = {}
    for entry in entries:
        previous = seen_ids.get(entry.scenario.id)
        if previous is not None:
            errors.append(
                f"{entry.path}: duplicate scenario ID {entry.scenario.id!r}; "
                f"first declared in {previous}"
            )
        else:
            seen_ids[entry.scenario.id] = entry.path
        if entry.scenario.benchmark is None:
            errors.append(f"{entry.path} is missing benchmark metadata")
        if entry.scenario.provenance.source.kind != "synthetic":
            errors.append(f"{entry.path} must be a synthetic static scenario")
        if entry.scenario.observed_failure is not None:
            errors.append(f"{entry.path} must not manufacture observed failure evidence")
    if errors:
        raise CatalogError(
            "billing benchmark catalog is invalid: " + "; ".join(errors),
            failures=tuple(errors),
        )
    return entries


async def render_catalog(root: Path, output: Path) -> list[CatalogEntry]:
    catalog_count = _scenario_file_count(root)
    try:
        entries = load_catalog(root)
    except (OSError, ValueError) as exc:
        raise CatalogRenderError(
            str(exc),
            catalog_count=catalog_count,
            rendered_count=0,
            failures=exc.failures or (str(exc),),
        ) from exc

    output_parent = output.parent
    if (output.exists() or output.is_symlink()) and (output.is_symlink() or not output.is_dir()):
        message = f"Harbor task destination must be a directory: {output}"
        raise CatalogRenderError(
            message,
            catalog_count=len(entries),
            rendered_count=0,
            failures=(message,),
        )
    try:
        output_parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        message = f"cannot create Harbor task destination parent {output_parent}: {exc}"
        raise CatalogRenderError(
            message,
            catalog_count=len(entries),
            rendered_count=0,
            failures=(message,),
        ) from exc
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.render-", dir=output_parent))
    rendered_count = 0
    active_task = "suite"
    try:
        for entry in entries:
            active_task = entry.scenario.id
            task = staged / entry.scenario.id
            await render_harbor_task(entry.scenario, task)
            validate_harbor_task(task)
            rendered_count += 1
        _replace_task_tree(staged, output)
    except Exception as exc:
        raise CatalogRenderError(
            f"failed to render Harbor task {active_task}: {exc}",
            catalog_count=len(entries),
            rendered_count=rendered_count,
            failures=(f"{active_task}: {exc}",),
        ) from exc
    finally:
        if staged.exists() or staged.is_symlink():
            if staged.is_dir() and not staged.is_symlink():
                shutil.rmtree(staged)
            else:
                staged.unlink()
    return entries


def render_catalog_sync(root: Path, output: Path) -> list[CatalogEntry]:
    return asyncio.run(render_catalog(root, output))
