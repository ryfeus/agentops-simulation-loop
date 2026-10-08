# Phase 10 billing corpus

Generator: `billing-phase10-v1`

Corpus SHA: `076e77f8139b34b84a76d25f35faba50da66fb9c3e525a6fcbf9bd3f363cdaea`

Family specs SHA: `45534d90594cf89809f634b7d598c1063f1c6ba6c55fe97ec53075dc61f35c02`

Split manifest SHA: `7c6d75da46df4405afbbce3006e4e91c4944dfd1a5f7b380cb35deecf4835eed`

Execution suite SHA: `4596d59cec6df9832584bd3529589f78170c8a84b67260aeed96d61b3cadaa67`

40 families; 200 generated tasks; train 160, dev 20, holdout 20.

Oracle 200/200; family negative calibration 40/40.

Oracle cache reused: `True`; calibration cache reused: `True`. Both caches matched the exact corpus and execution-suite SHAs.

Full CPU render smoke: 200/200 valid. Family, prototype, normalized instruction, and behavior-signature cross-split checks: PASS (zero collisions).

Prototypes by split: `{"dev": 4, "holdout": 3, "train": 16}`. Within-split behavior signature reuse: `55` signature groups.

| Split | Family | Prototype | Archetype |
| --- | --- | --- | --- |
| dev | multi-read-and-refund | multi-read-and-refund | multi-invoice |
| dev | normal-open-adjustment | open-refund-direct | normal-refund |
| dev | readonly-disputed-status | status-disputed-readonly | read-only |
| dev | readonly-suspicious-review | readonly-suspicious-invoice | read-only |
| holdout | disputed-direct-guard | disputed-direct-refund | disputed-refund |
| holdout | disputed-protected-paid | disputed-direct-refund | disputed-refund |
| holdout | multi-two-policy-states | multi-two-policy-states | multi-invoice |
| holdout | normal-paid-indirect | paid-refund-money-back | normal-refund |

Archetypes: `{"already-escalated": 15, "already-refunded": 20, "disputed-refund": 30, "duplicate-action": 15, "missing-invoice": 15, "multi-invoice": 25, "normal-refund": 25, "read-only": 25, "wrong-target": 30}`

Difficulties: `{"adversarial": 60, "compound": 45, "easy": 45, "paraphrase": 50}`

Known limits: deterministic templates and billing-v1 tool policy; no model characterization or RL training.


## Public evidence boundary

Operational AWS transfer, cleanup IDs, and local archive paths omitted from public export. Original receipts were internally verified and are not included; external reproduction is not claimed.
