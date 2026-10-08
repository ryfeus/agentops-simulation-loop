"""Create a manifest of retained Phase 11 evidence and adapter files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def manifest(run: Path) -> dict[str, object]:
    artifacts = []
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run)
        if (
            "checkpoints" in relative.parts
            or relative.name == "evidence-manifest.json"
            or relative.name.endswith(".raw.jsonl")
            or relative.suffix == ".log"
        ):
            continue
        artifacts.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    value = {"schema_version": "1", "artifacts": artifacts}
    (run / "evidence-manifest.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(manifest(args.run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
