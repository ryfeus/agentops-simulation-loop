#!/usr/bin/env bash
# Remote Phase 8B runner. The policy runs on the host; Harbor sandboxes stay CPU-only.
set -Eeuo pipefail

run_root=""
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
[[ -d "$dataset_root/tasks/paid-refund-direct" ]] || { echo "single Harbor task is missing" >&2; exit 1; }
[[ -n "$wheel" ]] || { echo "clean project wheel is missing" >&2; exit 1; }

mkdir -p "$run_root"
status_file="$run_root/run-status.json"
stage="BOOTSTRAP"
preflight="NOT_RUN"
training="NOT_RUN"

write_status() {
  python3 - "$status_file" "$stage" "$preflight" "$training" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "1", "stage": sys.argv[2], "harbor_env_preflight": sys.argv[3],
    "training": sys.argv[4],
}, indent=2, sort_keys=True) + "\n")
PY
}
trap write_status EXIT

if ! command -v uv >/dev/null; then
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh | sh
  runtime_home="${HOME:-/root}"
  export PATH="$runtime_home/.local/bin:$runtime_home/.cargo/bin:$PATH"
fi
command -v uv >/dev/null

stage="DEPENDENCY_RESOLUTION"
(
  cd "$runtime_root"
  uv sync --frozen --extra vllm --extra harbor
)

stage="DOCKER_PREFLIGHT"
if ! command -v docker >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
  systemctl enable --now docker
fi
docker --version > "$run_root/docker.txt"
docker info >> "$run_root/docker.txt"

stage="GPU_PREFLIGHT"
nvidia-smi > "$run_root/nvidia-smi.txt"
imds_token="$(curl --fail --silent --show-error --request PUT \
  --header 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)"
export PHASE8B_EC2_INSTANCE_TYPE="$(curl --fail --silent --show-error \
  --header "X-aws-ec2-metadata-token: $imds_token" \
  http://169.254.169.254/latest/meta-data/instance-type)"
export PHASE8B_EC2_AMI_ID="$(curl --fail --silent --show-error \
  --header "X-aws-ec2-metadata-token: $imds_token" \
  http://169.254.169.254/latest/meta-data/ami-id)"
export PHASE8B_EC2_AVAILABILITY_ZONE="$(curl --fail --silent --show-error \
  --header "X-aws-ec2-metadata-token: $imds_token" \
  http://169.254.169.254/latest/meta-data/placement/availability-zone)"
export PHASE8B_WHEEL_PATH="$wheel"
export TRL_EXPERIMENTAL_SILENCE=1
(
  cd "$runtime_root"
  uv run python - "$run_root/versions.json" <<'PY'
import json, os, platform, subprocess, sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")
value = {
    "schema_version": "1", "timestamp_utc": datetime.now(UTC).isoformat(),
    "python": sys.version, "kernel": platform.release(),
    "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
    "gpu": torch.cuda.get_device_name(0), "torch": version("torch"),
    "torch_cuda": torch.version.cuda, "trl": version("trl"), "vllm": version("vllm"),
    "harbor": version("harbor"), "instance_type": os.environ["PHASE8B_EC2_INSTANCE_TYPE"],
    "ami_id": os.environ["PHASE8B_EC2_AMI_ID"],
    "availability_zone": os.environ["PHASE8B_EC2_AVAILABILITY_ZONE"],
}
value["nvidia_smi"] = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True
).strip()
Path(sys.argv[1]).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
)

stage="HARBOR_ENV_PREFLIGHT"
if (
  cd "$runtime_root"
  uv run python phase8b/env_smoke.py --task "$dataset_root/tasks/paid-refund-direct" \
    --summary "$run_root/harbor-env-preflight.json"
) 2>&1 | tee "$run_root/harbor-env-preflight.log"; then
  [[ -s "$run_root/harbor-env-preflight.json" ]] || { echo "missing Harbor preflight JSON" >&2; exit 1; }
  preflight="PASS"
else
  preflight="FAIL"
  exit 1
fi

stage="TRL_HARBOR_ROLLOUT"
nvidia-smi > "$run_root/nvidia-smi-before-training.txt"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 \
  > "$run_root/gpu.csv" 2>&1 &
sampler_pid=$!
stop_sampler() { kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; }
trap 'stop_sampler; write_status' EXIT
if (
  cd "$runtime_root"
  uv run python phase8b/train_harbor_smoke.py --dataset-root "$dataset_root" --output "$run_root/training" \
    --summary "$run_root/training-harbor.json" --vllm-gpu-memory-utilization 0.30
) 2>&1 | tee "$run_root/training-harbor.log"; then
  [[ -s "$run_root/training-harbor.json" ]] || { echo "missing Harbor training JSON" >&2; training="FAIL"; exit 1; }
  training="PASS"
else
  training="FAIL"
  exit 1
fi
stop_sampler
trap write_status EXIT
nvidia-smi > "$run_root/nvidia-smi-after-training.txt"
stage="COMPLETE"
