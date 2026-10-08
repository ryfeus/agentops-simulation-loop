# Phase 12 — corrected-baseline SFT/GRPO comparison

GRPO did not pass the frozen final learning gate. Mean prototype-macro gain was 14.54 percentage points; the paired 95% bootstrap interval was [7.59, 21.99] percentage points.

The only failed final condition was the prohibited-write gate: baseline 46.25%, versus 49.72% for seed 42, 48.47% for seed 43, 49.58% for seed 44. All three seeds improved prototype pass rate, both critical archetypes had positive mean gains, and the paired bootstrap lower bound exceeded zero. The pass-rate gain therefore does not establish learning under the registered acceptance criteria.

SFT did not pass the seed-42 action gate and was not replicated. Only GRPO was evaluated on the sealed final test with three training seeds; this does not establish a definitive replicated SFT-versus-GRPO winner. These results concern the synthetic billing compositions.

## Final evaluation

| Policy | Prototype macro | Action macro | Prohibited writes | Clipped | Tool errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 9.72% | 1.00% | 46.25% | 17.78% | 481 |
| grpo/42 | 25.14% | 7.75% | 49.72% | 19.58% | 202 |
| grpo/43 | 24.72% | 7.00% | 48.47% | 20.56% | 157 |
| grpo/44 | 22.92% | 6.00% | 49.58% | 20.69% | 199 |

Final gates and per-seed gains:

```json
{
  "checks": {
    "mean_prototype_gain_at_least_10pp": true,
    "positive_gain_for_every_training_seed": true,
    "positive_mean_gain_for_both_critical_archetypes": true,
    "no_seed_increases_prohibited_write_rate": false,
    "paired_bootstrap_lower_bound_above_zero": true
  },
  "result": {
    "behavioral_improvement": false,
    "critical_archetype_deltas": {
      "disputed-refund": 0.04166666666666667,
      "multi-invoice": 0.012499999999999999
    },
    "execution_valid": true,
    "inference_scope": "these synthetic billing prototypes; not production readiness",
    "learning_demonstrated": false,
    "paired_prototype_bootstrap_95": [
      0.07592592592592592,
      0.2199074074074074
    ],
    "prototype_macro_delta": 0.14537037037037037,
    "replication_complete": true,
    "seed_deltas": {
      "42": 0.15416666666666667,
      "43": 0.15000000000000002,
      "44": 0.13194444444444442
    }
  }
}
```

## Development and selection

| Policy/checkpoint | Prototype macro | Action macro | Prohibited writes | Clipped | Tool errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 8.61% | 4.75% | 57.36% | 28.61% | 604 |
| sft/42/29 | 7.64% | 0.50% | 29.58% | 57.78% | 197 |
| sft/42/58 (selected) | 13.19% | 0.50% | 23.19% | 71.11% | 202 |
| sft/42/87 | 10.69% | 0.00% | 25.97% | 69.72% | 222 |
| grpo/42/160 (selected) | 24.58% | 15.50% | 53.19% | 26.39% | 152 |
| grpo/42/320 | 26.11% | 13.00% | 51.39% | 29.58% | 130 |
| grpo/43/160 (selected) | 23.89% | 13.50% | 51.53% | 28.47% | 152 |
| grpo/43/320 | 23.75% | 13.50% | 58.33% | 24.44% | 87 |
| grpo/44/160 (selected) | 23.19% | 12.50% | 55.97% | 28.47% | 144 |
| grpo/44/320 | 23.06% | 11.75% | 56.39% | 27.22% | 142 |

## Historical anchors

| Policy | Prototype macro | Action macro | Prohibited writes | Clipped | Tool errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 10.50% | 0.00% | 50.50% | 2.50% | 66 |
| grpo/42 | 26.00% | 4.00% | 48.00% | 5.00% | 45 |
| grpo/43 | 24.00% | 1.00% | 48.50% | 4.00% | 42 |
| grpo/44 | 28.00% | 11.25% | 52.50% | 4.00% | 46 |

## Final rollout diagnostics

| Evaluation | EOS | Length termination | Pending tool | Parser failures | Clipped before first tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| test/baseline | 220 | 128 | 372 | 39 | 9 |
| test/grpo/42 | 113 | 141 | 466 | 30 | 17 |
| test/grpo/43 | 131 | 148 | 441 | 24 | 17 |
| test/grpo/44 | 129 | 149 | 442 | 31 | 19 |
| anchors/baseline | 195 | 5 | 0 | 2 | 0 |
| anchors/grpo/42 | 189 | 10 | 1 | 2 | 0 |
| anchors/grpo/43 | 192 | 8 | 0 | 1 | 0 |
| anchors/grpo/44 | 191 | 8 | 1 | 3 | 0 |

Anchor regressions are separate from the final learning gate. Phase 11 remains the historical negative result (structural holdout 0/160 before and after training).

## Training and execution evidence

| Fit | Trainer seconds | Changed tensors | Adapter delta L2 | Actual token evidence |
| --- | ---: | ---: | ---: | --- |
| grpo/42 | 9837.69 | 392 | 0.379283 | conversation_completion_tokens: 192,996; environment_rollouts: 1,280; generated_tokens: 100,603 |
| grpo/43 | 10974.20 | 392 | 0.447355 | conversation_completion_tokens: 199,710; environment_rollouts: 1,280; generated_tokens: 106,997 |
| grpo/44 | 12012.12 | 392 | 0.397340 | conversation_completion_tokens: 197,482; environment_rollouts: 1,280; generated_tokens: 100,482 |
| sft/42 | 234.70 | 392 | 2.692570 | environment_rollouts_during_training: 0; input_tokens: 836,460; supervised_tokens: 36,945 |

Trainer intervals and stage GPU residency measure different scopes. The latter includes setup and evaluation while memory is resident; neither is an equal-compute comparison.

## Methods and limitations

All 340 tasks passed native Docker correct/no-op/wrong-write calibration. The frozen inference contract uses non-thinking generation, a 256-token native completion cap, and eight tool iterations. Each full evaluation has eight attempts per task in two passes with seeds 20261003 and 20261004. Final evaluation covered 90 test tasks and 25 historical anchors for the frozen baseline and three locked GRPO policies. Clipping and pending tool calls remain material diagnostic limitations.


## Public evidence boundary

Operational AWS transfer, cleanup IDs, and local archive paths omitted from public export. Original receipts were internally verified and are not included; external reproduction is not claimed.
