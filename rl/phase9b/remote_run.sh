#!/usr/bin/env bash
# Remote Phase 9b evaluator. It only evaluates a separately verified final adapter.
set -Eeuo pipefail
run_root=""; seed="${PHASE9B_SEED:-20260923}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-root) run_root="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$run_root" ]] || { echo "--run-root is required" >&2; exit 2; }
runtime_root="$(cd "$(dirname "$0")/.." && pwd)"
remote_root="$(cd "$(dirname "$0")/../.." && pwd)"
dataset_root="$remote_root/dataset"; adapter_root="$remote_root/source-adapter/adapter/final"
wheel="$(find "$remote_root/wheel" -maxdepth 1 -name '*.whl' -type f | head -1)"
mkdir -p "$run_root"; status_file="$run_root/run-status.json"; stage="BOOTSTRAP"; evaluation="NOT_RUN"
write_status() { python3 - "$status_file" "$stage" "$evaluation" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"1","stage":sys.argv[2],"evaluation":sys.argv[3]},indent=2,sort_keys=True)+"\n")
PY
}
trap write_status EXIT
if ! command -v uv >/dev/null; then
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh | sh
  export PATH="${HOME:-/root}/.local/bin:${HOME:-/root}/.cargo/bin:$PATH"
fi
stage="DEPENDENCY_RESOLUTION"; (cd "$runtime_root" && uv sync --frozen --extra vllm --extra harbor)
stage="DOCKER_PREFLIGHT"
if ! command -v docker >/dev/null; then apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io; systemctl enable --now docker; fi
docker --version > "$run_root/docker.txt"; docker info >> "$run_root/docker.txt"
stage="GPU_PREFLIGHT"; nvidia-smi > "$run_root/nvidia-smi.txt"
export PHASE8B_WHEEL_PATH="$wheel" PHASE8B_TRIAL_ROOT="$run_root/trials" TRL_EXPERIMENTAL_SILENCE=1 PHASE8C_INFRA_RETRIES=2
(cd "$remote_root" && uv run --project "$runtime_root" python - "$run_root/versions.json" <<'PY'
import json,platform,subprocess,sys
from importlib.metadata import version
from pathlib import Path
import torch
if not torch.cuda.is_available(): raise RuntimeError("torch.cuda.is_available() is False")
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"1","python":sys.version,"kernel":platform.release(),"uv":subprocess.check_output(["uv","--version"],text=True).strip(),"gpu":torch.cuda.get_device_name(0),"torch":version("torch"),"torch_cuda":torch.version.cuda,"trl":version("trl"),"vllm":version("vllm"),"harbor":version("harbor")},indent=2,sort_keys=True)+"\n")
PY
)
stage="ORACLE_GATE"
python3 - "$dataset_root/execution-suite.json" "$remote_root/oracle-suite-check.json" "$remote_root/task-set.json" "$remote_root/source-adapter/identity.json" "$remote_root/source-adapter/training-task-set.json" <<'PY'
import hashlib,json,sys
suite,oracle,tasks,source,source_tasks=(json.load(open(path)) for path in sys.argv[1:])
assert oracle["status"]=="PASS" and oracle["execution_suite_sha256"]==suite["execution_suite_sha256"]
assert tasks["execution_suite_sha256"]==suite["execution_suite_sha256"]
assert source["adapter_config_sha256"] and source["adapter_weights_sha256"] and source["source_revision"]
assert source["execution_suite_sha256"]==suite["execution_suite_sha256"]
assert hashlib.sha256(open(sys.argv[5],"rb").read()).hexdigest()==source["training_task_set_sha256"]
assert open(sys.argv[3],"rb").read()==open(sys.argv[5],"rb").read()
assert len(tasks["training_tasks"])==4 and len(tasks["regression_tasks"])==1
PY
stage="EVALUATION"; nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$run_root/gpu.csv" 2>&1 & sampler_pid=$!
stop_sampler() { kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; }
trap 'stop_sampler; write_status' EXIT
if (cd "$remote_root" && uv run --project "$runtime_root" python -m rl.phase9b.evaluate --dataset-root "$dataset_root" --output "$run_root" --run-id "${RL_SMOKE_RUN_ID:?}" --source-run "${PHASE9B_SOURCE_RUN:?}" --source-dir "$remote_root/source-adapter" --task-set "$remote_root/task-set.json" --seed "$seed") 2>&1 | tee "$run_root/evaluation.log"; then evaluation="PASS"; else evaluation="FAIL"; exit 1; fi
stop_sampler; trap write_status EXIT; stage="COMPLETE"
