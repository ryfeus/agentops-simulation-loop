#!/usr/bin/env bash
set -Eeuo pipefail
payload="${1:?payload directory required}"; run="${2:?run directory required}"
runtime_root="$payload/rl"
mkdir -p "$run"
stage="BOOTSTRAP"
sampler_pid=""
finish() {
  status=$?
  if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; fi
  python3 - "$run/run-status.json" "$stage" "$status" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"1","stage":sys.argv[2],"exit_code":int(sys.argv[3])},indent=2)+"\n")
PY
}
trap finish EXIT
if ! command -v uv >/dev/null; then
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh | sh
  export PATH="${HOME:-/root}/.local/bin:${HOME:-/root}/.cargo/bin:$PATH"
fi
stage="DEPENDENCY_RESOLUTION"
(cd "$runtime_root" && uv sync --frozen --extra vllm --extra harbor)
stage="DOCKER_PREFLIGHT"
if ! command -v docker >/dev/null; then
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
  systemctl enable --now docker
fi
docker info >/dev/null
nvidia-smi > "$run/nvidia-smi.txt"
stage="GPU_PREFLIGHT"
uv run --project "$runtime_root" python - "$run/versions.json" <<'PY'
import json,sys
from importlib.metadata import version
from pathlib import Path
import torch
if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"1","gpu":torch.cuda.get_device_name(0),"torch":version("torch"),"trl":version("trl"),"vllm":version("vllm"),"harbor":version("harbor"),"peft":version("peft")},indent=2)+"\n")
PY
stage="PHASE11_TRAIN"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$run/gpu.csv" 2>&1 & sampler_pid=$!
export PHASE8B_WHEEL_PATH="$(find "$payload/wheel" -maxdepth 1 -name '*.whl' -type f | head -1)"
test -f "$PHASE8B_WHEEL_PATH"
uv pip install --python "$runtime_root/.venv/bin/python" --no-deps "$PHASE8B_WHEEL_PATH"
export PHASE8B_TRIAL_ROOT="${run}.trials" TRL_EXPERIMENTAL_SILENCE=1 PHASE8C_INFRA_RETRIES=2
cd "$payload"
"$runtime_root/.venv/bin/python" -m rl.phase11.remote train --payload "$payload" --run "$run" > "${run}.log" 2>&1
stage="COMPLETE"
