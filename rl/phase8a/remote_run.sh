#!/usr/bin/env bash
# Remote Phase 8A runner. It receives no AWS or source-control credentials.
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
mkdir -p "$run_root"
status_file="$run_root/run-status.json"
stage="BOOTSTRAP"
control="NOT_RUN"
vllm="NOT_RUN"
reload="NOT_RUN"

write_status() {
  python3 - "$status_file" "$stage" "$control" "$vllm" "$reload" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "1", "stage": sys.argv[2], "control": sys.argv[3],
    "vllm_colocate": sys.argv[4], "checkpoint_reload": sys.argv[5],
}, indent=2, sort_keys=True) + "\n")
PY
}
trap write_status EXIT

if ! command -v uv >/dev/null; then
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh | sh
  # AWS-RunShellScript does not guarantee HOME is exported. The DLAMI's SSM
  # commands run as root, which is also where uv's installer places binaries.
  runtime_home="${HOME:-/root}"
  export PATH="$runtime_home/.local/bin:$runtime_home/.cargo/bin:$PATH"
fi
command -v uv >/dev/null

stage="DEPENDENCY_RESOLUTION"
(
  cd "$runtime_root"
  uv sync --frozen --extra vllm
)

stage="GPU_PREFLIGHT"
nvidia-smi > "$run_root/nvidia-smi.txt"
imds_token="$(curl --fail --silent --show-error --request PUT \
  --header 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)"
export PHASE8A_EC2_INSTANCE_TYPE="$(curl --fail --silent --show-error \
  --header "X-aws-ec2-metadata-token: $imds_token" \
  http://169.254.169.254/latest/meta-data/instance-type)"
export PHASE8A_EC2_AMI_ID="$(curl --fail --silent --show-error \
  --header "X-aws-ec2-metadata-token: $imds_token" \
  http://169.254.169.254/latest/meta-data/ami-id)"
export PHASE8A_EC2_AVAILABILITY_ZONE="$(curl --fail --silent --show-error \
  --header "X-aws-ec2-metadata-token: $imds_token" \
  http://169.254.169.254/latest/meta-data/placement/availability-zone)"
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
    "os_release": Path("/etc/os-release").read_text(), "kernel": platform.release(),
    "python": sys.version, "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
    "torch": version("torch"), "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0),
    "instance_type": os.environ["PHASE8A_EC2_INSTANCE_TYPE"],
    "ami_id": os.environ["PHASE8A_EC2_AMI_ID"],
    "availability_zone": os.environ["PHASE8A_EC2_AVAILABILITY_ZONE"],
    "trl": version("trl"), "vllm": version("vllm"), "transformers": version("transformers"),
    "peft": version("peft"), "datasets": version("datasets"),
}
nvidia = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    text=True,
).strip()
value["nvidia_smi"] = nvidia
tensor = torch.tensor([6, 7], device="cuda")
value["cuda_tensor_result"] = int(tensor.prod().item())
Path(sys.argv[1]).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
)

stage="TRL_CONTROL"
nvidia-smi > "$run_root/nvidia-smi-before-control.txt"
if (
  cd "$runtime_root"
  uv run python phase8a/train_smoke.py --mode control --output "$run_root/control" \
    --summary "$run_root/training-control.json"
) 2>&1 | tee "$run_root/training-control.log"; then
  [[ -s "$run_root/training-control.json" ]] || {
    echo "control training did not produce its JSON summary" >&2
    control="FAIL"
    exit 1
  }
  control="PASS"
else
  control="FAIL"
  exit 1
fi
nvidia-smi > "$run_root/nvidia-smi-after-control.txt"

stage="VLLM_INIT"
nvidia-smi > "$run_root/nvidia-smi-before-vllm.txt"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 \
  > "$run_root/gpu.csv" 2>&1 &
sampler_pid=$!
stop_sampler() { kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; }
trap 'stop_sampler; write_status' EXIT
if (
  cd "$runtime_root"
  uv run python phase8a/train_smoke.py --mode vllm --output "$run_root/vllm" \
    --summary "$run_root/training-vllm.json" --vllm-gpu-memory-utilization 0.30
) 2>&1 | tee "$run_root/training-vllm.log"; then
  [[ -s "$run_root/training-vllm.json" ]] || {
    echo "vLLM training did not produce its JSON summary" >&2
    vllm="FAIL"
    exit 1
  }
  vllm="PASS"
else
  cp "$run_root/training-vllm.json" "$run_root/training-vllm-primary.json" 2>/dev/null || true
  cp "$run_root/training-vllm.log" "$run_root/training-vllm-primary.log"
  if grep -qiE 'out of memory|cuda oom' "$run_root/training-vllm-primary.log"; then
    stage="VLLM_OOM"
    if (
      cd "$runtime_root"
      uv run python phase8a/train_smoke.py --mode vllm --output "$run_root/vllm-fallback" \
        --summary "$run_root/training-vllm.json" --vllm-gpu-memory-utilization 0.25
    ) 2>&1 | tee "$run_root/training-vllm.log"; then
      [[ -s "$run_root/training-vllm.json" ]] || {
        echo "vLLM fallback did not produce its JSON summary" >&2
        vllm="FAIL"
        exit 1
      }
      vllm="PASS"
    else
      vllm="FAIL"
      exit 1
    fi
  else
    vllm="FAIL"
    exit 1
  fi
fi
stop_sampler
trap write_status EXIT
nvidia-smi > "$run_root/nvidia-smi-after-vllm.txt"

stage="CHECKPOINT_RELOAD"
adapter="$run_root/vllm/adapter"
[[ -d "$adapter" ]] || adapter="$run_root/vllm-fallback/adapter"
if (
  cd "$runtime_root"
  uv run python phase8a/reload_checkpoint.py --adapter "$adapter" \
    --summary "$run_root/checkpoint-reload.json"
) 2>&1 | tee "$run_root/checkpoint-reload.log"; then
  [[ -s "$run_root/checkpoint-reload.json" ]] || {
    echo "checkpoint reload did not produce its JSON summary" >&2
    reload="FAIL"
    exit 1
  }
  reload="PASS"
else
  reload="FAIL"
  exit 1
fi
stage="COMPLETE"
