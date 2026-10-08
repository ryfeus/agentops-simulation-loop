"""Build the reproducible, stdlib-only dispute-policy evaluator Lambda package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.generate_harbor_configs import current_revision, require_clean_worktree

OUTPUT = Path(".agentcore/evaluation/dispute-policy.zip")
PACKAGE_MANIFEST = Path(".agentcore/evaluation/package.json")
FILES = {
    Path("deploy/evaluator/handler.py"): "handler.py",
    Path("src/agentops_demo/__init__.py"): "agentops_demo/__init__.py",
    Path("src/agentops_demo/evaluation/__init__.py"): "agentops_demo/evaluation/__init__.py",
    Path("src/agentops_demo/evaluation/dispute_policy.py"): (
        "agentops_demo/evaluation/dispute_policy.py"
    ),
    Path("src/agentops_demo/evaluation/suite.py"): "agentops_demo/evaluation/suite.py",
}
HANDLERS = {
    "dispute_policy": Path("deploy/evaluator/handler.py"),
    "mutation_cardinality": Path("deploy/evaluator/mutation_cardinality_handler.py"),
    "tool_workflow": Path("deploy/evaluator/tool_workflow_handler.py"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_package(
    *,
    output: Path = OUTPUT,
    manifest: Path = PACKAGE_MANIFEST,
    revision: str,
    evaluator: str = "dispute_policy",
) -> dict[str, object]:
    if evaluator not in HANDLERS:
        raise ValueError(f"unsupported evaluator package {evaluator!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        files = dict(FILES)
        if evaluator == "dispute_policy":
            files[HANDLERS[evaluator]] = "handler.py"
        else:
            files[Path("deploy/evaluator/handler.py")] = "base_handler.py"
            files[HANDLERS[evaluator]] = "handler.py"
        for source, destination in sorted(files.items(), key=lambda item: item[1]):
            info = zipfile.ZipInfo(destination, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=9)
    digest = sha256_file(output)
    payload = {
        "schema_version": "1",
        "source_revision": revision,
        "evaluator": evaluator,
        "package": {
            "path": str(output),
            "sha256": digest,
            "sha256_base64": base64.b64encode(bytes.fromhex(digest)).decode(),
        },
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=PACKAGE_MANIFEST)
    parser.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--evaluator", choices=sorted(HANDLERS), default="dispute_policy")
    args = parser.parse_args(argv)
    if not args.allow_dirty:
        require_clean_worktree()
    payload = build_package(
        output=args.output,
        manifest=args.manifest,
        revision=current_revision(),
        evaluator=args.evaluator,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
