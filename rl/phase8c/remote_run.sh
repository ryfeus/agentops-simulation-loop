#!/usr/bin/env bash
# Remote Phase 8C runner. It evaluates frozen Qwen; it never trains or saves an adapter.
set -Eeuo pipefail

run_root=""
attempts_per_pass="${PHASE8C_ATTEMPTS_PER_PASS:-4}"
passes="${PHASE8C_PASSES:-1}"
task_ids="${PHASE8C_TASKS:-}"
seed="${PHASE8C_SEED:-20260920}"
infra_retries="${PHASE8C_INFRA_RETRIES:-2}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-root) run_root="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$run_root" ]] || { echo "--run-root is required" >&2; exit 2; }

runtime_root="$(cd "$(dirname "$0")/.." && pwd)"
remote_root="$(cd "$(dirname "$0")/../.." && pwd)"
dataset_root="$remote_root/dataset"
wheel="$(find "$remote_root/wheel" -maxdepth 1 -name '*.whl' -type f | head -1)"
mkdir -p "$run_root"
status_file="$run_root/run-status.json"
stage="BOOTSTRAP"
evaluation="NOT_RUN"
write_status() {
  python3 - "$status_file" "$stage" "$evaluation" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"1", "stage":sys.argv[2], "evaluation":sys.argv[3]}, indent=2, sort_keys=True) + "\n")
PY
}
trap write_status EXIT

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
docker --version > "$run_root/docker.txt"
docker info >> "$run_root/docker.txt"

stage="GPU_PREFLIGHT"
nvidia-smi > "$run_root/nvidia-smi.txt"
export PHASE8B_WHEEL_PATH="$wheel"
export PHASE8B_TRIAL_ROOT="$run_root/trials"
export TRL_EXPERIMENTAL_SILENCE=1
(cd "$remote_root" && uv run --project "$runtime_root" python - "$run_root/versions.json" <<'PY'
import json, platform, subprocess, sys
from importlib.metadata import version
from pathlib import Path
import torch
if not torch.cuda.is_available(): raise RuntimeError("torch.cuda.is_available() is False")
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"1", "python":sys.version, "kernel":platform.release(), "uv":subprocess.check_output(["uv","--version"],text=True).strip(), "gpu":torch.cuda.get_device_name(0), "torch":version("torch"), "torch_cuda":torch.version.cuda, "trl":version("trl"), "vllm":version("vllm"), "harbor":version("harbor")}, indent=2, sort_keys=True) + "\n")
PY
)

stage="ORACLE_GATE"
test -s "$dataset_root/execution-suite.json" || { echo "execution suite manifest is missing" >&2; exit 1; }
test -s "$remote_root/oracle-suite-check.json" || { echo "exact Oracle gate evidence is missing" >&2; exit 1; }
python3 - "$dataset_root/execution-suite.json" "$remote_root/oracle-suite-check.json" <<'PY'
import json, sys
suite, oracle = (json.load(open(path)) for path in sys.argv[1:])
assert oracle["status"] == "PASS" and oracle["execution_suite_sha256"] == suite["execution_suite_sha256"]
PY

stage="TRL_FROZEN_EVALUATE"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$run_root/gpu.csv" 2>&1 &
sampler_pid=$!
stop_sampler() { kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; }
trap 'stop_sampler; write_status' EXIT
if (cd "$remote_root" && uv run --project "$runtime_root" python -m rl.phase8c.evaluate_baseline --dataset-root "$dataset_root" --output "$run_root" --run-id "${PHASE8C_RUN_ID:?}" --attempts-per-pass "$attempts_per_pass" --passes "$passes" --task-ids "$task_ids" --seed "$seed" --infra-retries "$infra_retries") 2>&1 | tee "$run_root/evaluation.log"; then
  evaluation="PASS"
else
  evaluation="FAIL"
  exit 1
fi
stop_sampler
trap write_status EXIT
stage="COMPLETE"
