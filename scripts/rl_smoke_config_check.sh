#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repository_root"
exec uv run --project rl --locked python rl/phase8a/config_check.py
