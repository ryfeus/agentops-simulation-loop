#!/bin/bash
set -euo pipefail
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
root=${DECIDER_ROOT:-.artifacts/decision-model}
run=${DECIDER_NEXT_RUN:-.artifacts/decision-model-next/run}
mkdir -p "$run"
export DECIDER_NEXT_RUN="$run"
python=experiments/decision_model/.venv/bin/python
exec > "$run/train-stage.log" 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 5 > "$run/training-gpu.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
common=(--manifest "$root/next-v1/corpus/cohort-manifest.json" --tasks-root "$root/next-v1/corpus/tasks")
"$python" -m experiments.decision_model.next_train collect --checkpoint "$root/source-checkpoint" --data "$root/next-v1/data/train.jsonl" --dev-data "$root/next-v1/data/dev.jsonl" "${common[@]}" --output "$run/collection"
"$python" -m experiments.decision_model.model smoke --checkpoint "$root/source-checkpoint" --data "$run/collection/train.jsonl" --split train --steps 20 --gradient-checkpointing --output "$run/smoke"
"$python" - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['DECIDER_NEXT_RUN'])
s=json.loads((p/'smoke/smoke.json').read_text())
assert s['overfit_pass'] and s['base_frozen'] and all(s['changed'].values()),s
estimate=s['elapsed_seconds']/20*200*2+3600
(p/'runtime-estimate.json').write_text(json.dumps({'estimated_remaining_seconds':estimate,'basis':'20 smoke steps; maximum 200 steps at 2x observed time plus 1h evaluation'},indent=2)+'\n')
assert estimate < 10800,estimate
PY
"$python" -m experiments.decision_model.next_train train --checkpoint "$root/source-checkpoint" --data "$run/collection/train-dev.jsonl" "${common[@]}" --epochs 3 --max-steps 200 --gradient-checkpointing --output "$run/training"
"$python" -m experiments.decision_model.next_train calibrate --checkpoint "$root/source-checkpoint" --data "$root/next-v1/data/calibration.jsonl" "${common[@]}" --output "$run/calibration-released"
"$python" -m experiments.decision_model.next_train calibrate --checkpoint "$run/training/best" --data "$root/next-v1/data/calibration.jsonl" "${common[@]}" --output "$run/calibration-tuned"
"$python" - <<'PY'
import hashlib,json,datetime,os
from pathlib import Path
p=Path(os.environ['DECIDER_NEXT_RUN'])
files=['training/training.json','calibration-released/calibration.json','calibration-tuned/calibration.json']
result={'selection_complete':True,'completed_at':datetime.datetime.now(datetime.UTC).isoformat(),'evidence_sha256':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in files},'selection_uses':['train','dev','calibration'],'final_test_accessed':False}
(p/'selection-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
PY
