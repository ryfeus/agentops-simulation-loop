#!/usr/bin/env bash
# Remote Phase 9 runner: fresh LoRA, native GRPO updates, Harbor on CPU-only Docker.
set -Eeuo pipefail

run_root=""
max_steps="${PHASE9_MAX_STEPS:-20}"
learning_rate="${PHASE9_LEARNING_RATE:-1e-5}"
seed="${PHASE9_SEED:-20260922}"
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
task_set="$remote_root/task-set.json"
mkdir -p "$run_root"
status_file="$run_root/run-status.json"
stage="BOOTSTRAP"
training="NOT_RUN"
write_status() {
  python3 - "$status_file" "$stage" "$training" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"1", "stage":sys.argv[2], "training":sys.argv[3]}, indent=2, sort_keys=True) + "\n")
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
test -s "$task_set" || { echo "Phase 9 task set is missing" >&2; exit 1; }
python3 - "$dataset_root/execution-suite.json" "$remote_root/oracle-suite-check.json" "$task_set" <<'PY'
import json, sys
suite, oracle, task_set = (json.load(open(path)) for path in sys.argv[1:])
assert oracle["status"] == "PASS" and oracle["execution_suite_sha256"] == suite["execution_suite_sha256"]
assert task_set["execution_suite_sha256"] == suite["execution_suite_sha256"]
PY

stage="TRL_HARBOR_ROLLOUT"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$run_root/gpu.csv" 2>&1 &
sampler_pid=$!
stop_sampler() { kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; }
trap 'stop_sampler; write_status' EXIT
if (cd "$remote_root" && uv run --project "$runtime_root" python -m rl.phase9.train_overfit --dataset-root "$dataset_root" --output "$run_root" --task-set "$task_set" --max-steps "$max_steps" --learning-rate "$learning_rate" --seed "$seed") 2>&1 | tee "$run_root/training.log"; then
  training="PASS"
else
  training="FAIL"
  exit 1
fi
python3 - "$run_root" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
artifacts = []
for relative_root in ("adapter/final", "checkpoints"):
    directory = root / relative_root
    for path in sorted(directory.rglob("*")) if directory.is_dir() else []:
        if path.is_file():
            artifacts.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(root / "artifacts-manifest.json").write_text(json.dumps({"schema_version": "1", "artifacts": artifacts}, indent=2, sort_keys=True) + "\n")
PY
stop_sampler
trap write_status EXIT
stage="COMPLETE"
