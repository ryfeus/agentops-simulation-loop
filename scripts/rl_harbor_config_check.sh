#!/usr/bin/env bash
set -Eeuo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
TRL_EXPERIMENTAL_SILENCE=1 PYTHONPATH="$repository_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  uv run --project "$repository_root/rl" --locked --extra harbor \
  python "$repository_root/rl/phase8b/config_check.py"
