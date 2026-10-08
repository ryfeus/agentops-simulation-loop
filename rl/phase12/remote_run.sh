#!/usr/bin/env bash
set -Eeuo pipefail
payload="${1:?extracted bundle directory required}"
stage="${2:?diagnose, demos, pilot, replicate, or final required}"
case "$stage" in diagnose|demos|pilot|replicate|final) ;; *) exit 2 ;; esac
cd "$payload"
uv sync --project rl --frozen --extra vllm --extra harbor
docker info >/dev/null
nvidia-smi >/dev/null
export PHASE8B_WHEEL_PATH="$(find "$payload/run/wheel" -maxdepth 1 -name '*.whl' -type f | head -1)"
test -f "$PHASE8B_WHEEL_PATH"
uv pip install --python rl/.venv/bin/python --no-deps "$PHASE8B_WHEEL_PATH"
export PHASE8B_TRIAL_ROOT="$payload/trials" TRL_EXPERIMENTAL_SILENCE=1
export PHASE8C_INFRA_RETRIES=2
mkdir -p "$payload/run/jobs"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$payload/run/jobs/gpu-$stage.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
rl/.venv/bin/python -m rl.phase12.runner execute --stage "$stage" --run "$payload/run"
