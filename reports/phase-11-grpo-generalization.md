# Phase 11 GRPO held-out generalization

Run: `private run identifier omitted`. Experiment: `billing-phase11-v1` at source revision (private revision omitted) (clean worktree).
Training and final evaluation validity: **PASS**. The selected checkpoint improved on dev, but showed **no improvement on the structural holdout**. This is the plan's valid negative result: dev-guided improvement did not transfer to the final holdout.

## Frozen lineage

- corpus_sha256: `076e77f8139b34b84a76d25f35faba50da66fb9c3e525a6fcbf9bd3f363cdaea`
- execution_suite_sha256: `4596d59cec6df9832584bd3529589f78170c8a84b67260aeed96d61b3cadaa67`
- family_specs_sha256: `45534d90594cf89809f634b7d598c1063f1c6ba6c55fe97ec53075dc61f35c02`
- split_manifest_sha256: `7c6d75da46df4405afbbce3006e4e91c4944dfd1a5f7b380cb35deecf4835eed`
- anchor_suite_sha256: `cdb6a830e234557a5398e917514b5282ecb0afce60f868c95c49d5769ad7a489`
- wheel_sha256: `7a2a68602ac85849814ded004b00f2a5e0dc3d1bc3837842bb729314a153a6da`

Software: torch 2.13.0, trl 1.13.0, vllm 0.28.0, harbor 0.22.0, peft 0.21.0.

## Frozen characterization

Among 32 predetermined train representatives, 2 groups were mixed, 29 all zero, and 1 all one.

## Training

Fresh Qwen3-0.6B zero-init LoRA; 320 optimizer steps, 320 four-rollout groups, 1280 rollouts.
Every one of 160 train tasks had exactly two groups and eight rollouts.
Reward-group distribution: 57 mixed (17.8125%), 236 all zero, and 27 all one. Adapter delta L2: 0.263056 across 392 changed tensors.

## Development selection

Policy | Prototype macro | Micro | Micro Wilson 95%
--- | ---: | ---: | ---
Frozen baseline | 0.375 | 0.375 | [0.30374323210068377, 0.4521183166921926]
Step 160 | 0.412 | 0.412 | [0.3391443296506424, 0.48995875450437104]
Step 320 | 0.425 | 0.425 | [0.35104349819846475, 0.5024734310772612]

Selected checkpoint: step **320**.
Selection rule: highest prototype macro, then micro, then earlier step.
Canonical dev gate: **PASS**.

## Train representatives

Micro pass rate: 0.062 → 0.102; tool-call rate: 0.281 → 0.352.

## Historical anchors

25 anchor tasks, four attempts per policy. Micro pass rate: 0.160 → 0.170.
Anchor results are regression evidence, not the generalization test.

- `already-escalated`: 0.000 → 0.000
- `already-refunded`: 0.000 → 0.000
- `disputed-refund`: 0.000 → 0.000
- `duplicate-action`: 0.000 → 0.000
- `missing-invoice`: 0.500 → 0.500
- `multi-invoice`: 0.000 → 0.000
- `normal-refund`: 0.000 → 0.000
- `read-only`: 0.750 → 0.812
- `wrong-target`: 0.000 → 0.000

## Structural holdout

Holdout evaluation validity: **PASS**. The 20 tasks received eight attempts per policy, for 160 baseline and 160 trained rollouts, after the dev selection lock. The selected adapter was evaluated without further training.
Micro pass rate: 0/160 (0.000) → 0/160 (0.000).
Micro Wilson 95% intervals: [0.0, 0.02344619517150519] → [0.0, 0.02344619517150519].
Family macro pass rate: 0.000 → 0.000 across four structural families.
Prototype macro pass rate: 0.000 → 0.000.
Prototypes improved/unchanged/worsened: 0/3/0.
Advisory generalization signal: **False**; human review required.

- `disputed-direct-refund`: 0.000 → 0.000.
- `multi-two-policy-states`: 0.000 → 0.000.
- `paid-refund-money-back`: 0.000 → 0.000.

## Behavioral audit

Recorded shortcut candidates: 0.
Tool-call rate: 0.150 → 0.256. More tool use did not translate into any holdout passes.

## Limitations

The scalar Harbor verifier supplied the only training reward. Per-task trajectories, target correctness, inspection-before-write rates, verifier components, and Wilson intervals are retained in private evidence (not included). Four dev families and three holdout prototypes do not support a precise significance claim. Human review is required. No retraining is permitted under this experiment after inspecting the holdout.


## Public evidence boundary

Operational AWS transfer, cleanup IDs, and local archive paths omitted from public export. Original receipts were internally verified and are not included; external reproduction is not claimed.
