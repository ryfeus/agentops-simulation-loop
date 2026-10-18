#!/usr/bin/env bash
# Run from the repository root after prepare.py; all results stay ignored.
set -euo pipefail
phase="${1:?usage: run_gpu.sh smoke|train|evaluate|harbor}"
root=.artifacts/decision-model
run="$root/run"
python=experiments/decision_model/.venv/bin/python
export PYTHONPATH=.
mkdir -p "$run"
model=("$python" -m experiments.decision_model.model)
case "$phase" in
  smoke)
    "${model[@]}" evaluate --checkpoint "$root/source-checkpoint" --data "$root/dataset/test.jsonl" --split test --output "$run/untuned-test.jsonl"
    "${model[@]}" evaluate --checkpoint "$root/source-checkpoint" --data "$root/dataset/test.jsonl" --split test --reverse-options --limit 12 --output "$run/untuned-reverse.jsonl"
    "${model[@]}" smoke --checkpoint "$root/source-checkpoint" --data "$root/dataset/train.jsonl" --split train --steps 20 --gradient-checkpointing --output "$run/smoke"
    ;;
  train)
    "$python" - <<'PY'
import json
from pathlib import Path
p=Path('.artifacts/decision-model/run')
s=json.loads((p/'smoke/smoke.json').read_text())
assert s['overfit_pass'] and s['base_frozen'] and all(s['changed'].values())
# Conservative allowance: twice measured smoke-step time for the 87-step fit,
# plus one hour for setup/evaluation/transfer. Independent worker shutdown also applies.
estimate=s['elapsed_seconds']/20*87*2+3600
(p/'runtime-estimate.json').write_text(json.dumps({'estimated_remaining_seconds':estimate,'basis':'smoke throughput plus 1h evaluation allowance'},indent=2)+'\n')
assert estimate < 10800, f'Reduce scope: estimated remaining {estimate}s exceeds 3h'
PY
    "${model[@]}" train --checkpoint "$root/source-checkpoint" --data "$root/dataset/dataset.jsonl" --epochs 3 --max-steps 200 --gradient-checkpointing --output "$run/training"
    ;;
  evaluate)
    "${model[@]}" calibrate --checkpoint "$run/training/best" --data "$root/dataset/calibration.jsonl" --split calibration --output "$run/calibration.json"
    "${model[@]}" evaluate --checkpoint "$run/training/best" --data "$root/dataset/test.jsonl" --split test --calibration "$run/calibration.json" --output "$run/tuned-test.jsonl"
    "${model[@]}" evaluate --checkpoint "$run/training/best" --data "$root/dataset/test.jsonl" --split test --calibration "$run/calibration.json" --reverse-options --limit 12 --output "$run/tuned-reverse.jsonl"
    ;;
  harbor)
    wheel="$root/wheel/agentops_simulation_loop_aws-0.1.0-py3-none-any.whl"
    for arm in untuned tuned; do
      extra=()
      model_arm=rule
      if [[ "$arm" == untuned ]]; then
        model_arm=model
        extra=(--checkpoint "$root/source-checkpoint")
      elif [[ "$arm" == tuned ]]; then
        model_arm=model
        extra=(--checkpoint "$run/training/best" --calibration "$run/calibration.json")
      fi
      "$python" -m experiments.decision_model.harbor --dataset "$root/dataset/dataset.jsonl" --tasks "$root/tasks" --wheel "$wheel" --output "$run/harbor-$arm" --arm "$model_arm" --limit 12 --policy-mode enforced "${extra[@]}"
    done
    ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
