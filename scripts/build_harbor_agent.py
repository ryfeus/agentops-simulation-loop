"""Build one provenance-bound Harbor candidate wheel and manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import (
    build_clean_wheel,
    is_sha256,
    sha256_file,
)
from scripts.generate_harbor_configs import (
    current_revision,
    require_clean_worktree,
    write_configs,
)

DEFAULT_CONFIG_DIR = Path(".harbor/configs")
DEFAULT_PACKAGE_DIR = Path(".harbor/package")
MANIFEST_NAME = "manifest.json"


def discover_wheel(package_dir: Path) -> Path:
    """Return the only wheel in a clean build directory."""

    wheels = sorted(package_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"expected exactly one project wheel in {package_dir}, found {len(wheels)}"
        )
    return wheels[0]


def write_manifest(
    *,
    package_dir: Path,
    config_dir: Path,
    source_revision: str,
    configs: dict[str, AgentConfig],
) -> Path:
    if any(config.agent.source_revision != source_revision for config in configs.values()):
        raise ValueError("candidate config revision does not match build revision")
    wheel = discover_wheel(package_dir)
    package_digest = sha256_file(wheel)
    if not is_sha256(package_digest):
        raise RuntimeError("built wheel digest is not lowercase SHA-256")
    manifest = {
        "source_revision": source_revision,
        "package": {"filename": wheel.name, "sha256": package_digest},
        "candidates": {
            mode: {
                "config_path": (config_dir / f"{mode}.json").as_posix(),
                "fingerprint": config.fingerprint(),
                "model": f"{config.model.provider}/{config.model.model_id}",
            }
            for mode, config in sorted(configs.items())
        },
    }
    manifest_path = package_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Harbor build manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Harbor build manifest must be a JSON object")
    return value


def wheel_from_manifest(path: Path) -> Path:
    manifest = load_manifest(path)
    package = manifest.get("package")
    filename = package.get("filename") if isinstance(package, dict) else None
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("Harbor build manifest has an invalid wheel filename")
    wheel = path.parent / filename
    if not wheel.is_file():
        raise FileNotFoundError(f"manifest wheel not found: {wheel}")
    return wheel.resolve()


def build(package_dir: Path, config_dir: Path) -> Path:
    require_clean_worktree()
    revision = current_revision()
    configs = write_configs(config_dir, revision)

    build_clean_wheel(package_dir)

    require_clean_worktree()
    if current_revision() != revision:
        raise RuntimeError("Git HEAD changed while building Harbor candidates")
    return write_manifest(
        package_dir=package_dir,
        config_dir=config_dir,
        source_revision=revision,
        configs=configs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--print-wheel", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.package_dir / MANIFEST_NAME
    if args.print_wheel:
        print(wheel_from_manifest(manifest_path))
        return 0

    written = build(args.package_dir, args.config_dir)
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
