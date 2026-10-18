"""Download exact released artifacts, without changing the source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".artifacts/decision-model"))
    args = parser.parse_args()
    pins = json.loads(Path(__file__).with_name("sources.json").read_text())
    vendor = args.root / "vendor/strands-decider"
    if not vendor.exists():
        subprocess.run(["git", "clone", pins["upstream_repository"], str(vendor)], check=True)
    subprocess.run(
        ["git", "-C", str(vendor), "checkout", "--detach", pins["upstream_commit"]], check=True
    )
    checkpoint = args.root / "source-checkpoint"
    snapshot_download(
        pins["checkpoint_repo"], revision=pins["checkpoint_revision"], local_dir=str(checkpoint)
    )
    snapshot_download(pins["base_model"], revision=pins["base_revision"])
    config = json.loads((checkpoint / "strands_decider_config.json").read_text())
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    assert config["lora_r"] == pins["expected_lora_rank"]
    assert config["pointer_dim"] == pins["expected_pointer_dim"]
    assert provenance["base_model_revision"] == pins["base_revision"]
    files = {
        str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(checkpoint.rglob("*"))
        if p.is_file() and ".cache" not in p.parts
    }
    (args.root / "source-manifest.json").write_text(
        json.dumps({**pins, "files": files}, indent=2) + "\n"
    )
    print(checkpoint.resolve())


if __name__ == "__main__":
    main()
