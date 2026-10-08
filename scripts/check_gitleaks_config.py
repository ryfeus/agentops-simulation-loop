"""Prove the public tokenizer exception does not hide another secret in its file."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def check(*, executable: str, root: Path) -> dict[str, object]:
    binary = shutil.which(executable)
    if binary is None:
        raise RuntimeError("Gitleaks executable is unavailable")
    config = root / ".gitleaks.toml"
    canonical = json.loads((root / "rl/phase12/experiment.json").read_text())
    # Computed synthetic probe, never a usable provider credential.
    canonical["api_key"] = hashlib.sha256(b"synthetic secret scanner regression probe").hexdigest()
    with tempfile.TemporaryDirectory(prefix="agentops-scanner-proof-") as temporary:
        fixture_root = Path(temporary)
        fixture = fixture_root / "rl/phase12/experiment.json"
        fixture.parent.mkdir(parents=True)
        fixture.write_text(json.dumps(canonical, indent=2) + "\n")
        report = fixture_root / "report.json"
        completed = subprocess.run(
            [
                binary,
                "dir",
                str(fixture_root),
                "--config",
                str(config),
                "--redact",
                "--no-banner",
                "--report-format",
                "json",
                "--report-path",
                str(report),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        findings = json.loads(report.read_text()) if report.exists() else []
        if (
            completed.returncode != 1
            or len(findings) != 1
            or findings[0]["RuleID"] != "generic-api-key"
        ):
            raise RuntimeError(
                "scanner must exempt only the public tokenizer revision and detect the probe"
            )
    return {"status": "PASS", "synthetic_secret_detected": True, "pinned_revision_excluded": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gitleaks", default="gitleaks")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(check(executable=args.gitleaks, root=args.root.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
