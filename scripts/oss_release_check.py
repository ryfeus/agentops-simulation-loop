"""Check a reviewed export tree without printing matched private values."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

SYNTHETIC_ACCOUNTS = {"000000000000", "111111111111", "123456789012", "210987654321"}
PATTERNS = {
    "private_home_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "private_repository": re.compile(r"github\.com/[^/\s]+/agentops-simulation-loop-aws"),
    "access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "cloud_instance": re.compile(r"\bi-[0-9a-f]{8,17}\b"),
    "cloud_bucket": re.compile(r"\bs3://[^/\s`\"\']+"),
}
ACCOUNT = re.compile(r"(?<![A-Za-z0-9.])[0-9]{12}(?![A-Za-z0-9])")
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
FORBIDDEN_PARTS = {
    ".git",
    ".agentcore",
    ".harbor",
    ".harbor-ec2",
    ".rl-smoke",
    ".taskify",
    ".demo",
    ".artifacts",
    ".terraform",
    ".venv",
    "node_modules",
    "plans",
}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2"}


def inspect(root: Path, paths: list[str], *, forbidden_values: list[str] | None = None) -> dict:
    findings = []
    root = root.resolve()
    for name in paths:
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts:
            findings.append({"path": name, "category": "unsafe_path"})
            continue
        if path.is_symlink():
            findings.append({"path": name, "category": "symlink"})
            continue
        if (
            any(part in FORBIDDEN_PARTS for part in relative.parts)
            or (relative.name.startswith(".env") and relative.name != ".env.example")
            or re.search(r"\.(?:tfstate(?:\..*)?|tfplan|db|pem|key|pt|pth|safetensors)$", name)
        ):
            findings.append({"path": name, "category": "runtime_or_private_artifact"})
        if not path.is_file():
            findings.append({"path": name, "category": "missing_file"})
            continue
        data = path.read_bytes()
        try:
            content = data.decode("utf-8")
        except UnicodeError:
            if path.suffix.lower() not in BINARY_SUFFIXES:
                findings.append({"path": name, "category": "unreviewed_binary"})
            continue
        for category, pattern in PATTERNS.items():
            matches = pattern.findall(content)
            if category == "private_home_path":
                matches = [value for value in matches if value != "/home/" + "user/"]
            elif category == "cloud_instance" and name == "tests/unit/test_harbor_ec2.py":
                matches = [
                    value
                    for value in matches
                    if value
                    not in {
                        "i-" + "0123456789abcdef0",
                        "i-" + "0fedcba9876543210",
                    }
                ]
            elif category == "cloud_bucket":
                matches = [
                    value
                    for value in matches
                    if value != "s3://" + "bucket" and not value.startswith("s3://" + "{")
                ]
            if matches:
                findings.append({"path": name, "category": category})
        if any(value not in SYNTHETIC_ACCOUNTS for value in ACCOUNT.findall(content)):
            findings.append({"path": name, "category": "unreviewed_account_id"})
        if any(value and value in content for value in (forbidden_values or [])):
            findings.append({"path": name, "category": "local_denylist"})
        if relative.parts[0] == "reports" and re.search(
            r'"(?:client_token|command_id|original_command_id|instance_id|bucket|local_archive)"\s*:',
            content,
        ):
            findings.append({"path": name, "category": "operational_report_receipt"})
        if path.suffix == ".md":
            for match in LINK.finditer(content):
                target = match.group(1).strip().split(" ", 1)[0].strip("<>")
                if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                linked = (path.parent / target).resolve()
                if not linked.is_relative_to(root) or not linked.exists():
                    findings.append({"path": name, "category": "broken_or_external_local_link"})
        if "plans/" in content and path.suffix == ".md":
            findings.append({"path": name, "category": "excluded_plan_reference"})
    return {
        "schema_version": "1",
        "status": "FAIL" if findings else "PASS",
        "files_checked": len(paths),
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tree", action="store_true", help="Scan a staging tree before committing")
    parser.add_argument("--denylist", type=Path, help="Local-only JSON array of forbidden values")
    args = parser.parse_args(argv)
    if args.tree:
        paths = sorted(
            str(path.relative_to(args.root))
            for path in args.root.rglob("*")
            if path.is_file() or path.is_symlink()
        )
    else:
        output = subprocess.check_output(["git", "-C", str(args.root), "ls-files", "-z"])
        paths = [value for value in output.decode().split("\0") if value]
    forbidden = json.loads(args.denylist.read_text()) if args.denylist else []
    if not isinstance(forbidden, list) or not all(isinstance(value, str) for value in forbidden):
        raise ValueError("denylist must be a JSON array of strings")
    result = inspect(args.root, paths, forbidden_values=forbidden)
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
