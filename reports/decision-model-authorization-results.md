> Public research summary. Private execution manifests, raw trajectories, checkpoints, operational receipts, and source revisions are omitted. Results below are historical measurements, not independently reproduced by this export.

# Invoice-specific authorization regression evaluation

The opt-in authorization change eliminated the six unwanted successful writes per history mode seen in the previous run. All three variants of the failing read-only audit task now pass in both modes. However, the conservative parser also withholds legitimate work: history-visible completion declined from **21/24 to 15/24**, with **12 required operations omitted**. This is a safety improvement with a completion regression, not a deployment-ready policy.

## Results

The same frozen 24 tasks across eight families were evaluated once per history mode. Both new arms use the same invoice-specific parser and execution guard; history visibility is their paired difference.

| Metric | Previous hidden | Guarded hidden | Previous visible | Guarded visible |
|---|---:|---:|---:|---:|
| Completed and native verifier passed | 7/24 | 8/24 | 21/24 | 15/24 |
| Terminal database-state invariants passed | 21/24 | 9/24 | 21/24 | 15/24 |
| Successful unwanted writes | 6 | **0** | 6 | **0** |
| Successful expected operations, out of 30 | 30 | 12 | 30 | 18 |
| LLM calls | 48 | 22 | 49 | 6 |
| Read-only audit tasks passed | 0/3 | **3/3** | 0/3 | **3/3** |

Within the new guarded comparison, visible history improves seven task pairs; none regress. Six improvements come from completing conditional escalations in the missing-target and paid-review families. One comes from recovering a missed inspection in the refunded/open-status family. The latter hidden-history trial completed its required refund and satisfied terminal-state invariants but failed the inspection trajectory requirement.

The guarded hidden arm blocked eight write requests before billing execution: five expected escalations whose required history condition could not be verified, and three duplicate-target escalations. A sixth expected escalation in the two history-sensitive families was never proposed. The visible arm made no blocked write requests. Zero blocks does not mean zero refusals: the planner often returned `no_action` before any write was attempted.

The unwanted read-only writes disappeared because the scoped planner stopped proposing them. Separate SQLite execution tests deliberately forced those writes through both rule and fallback paths and verified that the execution guard denied them without calling the billing service. Thus the paid traces demonstrate prevention by planning, while the tests exercise enforcement against bad proposals.

## What changed

`authorization.py` builds an invoice/action allowlist from the request. It recognizes explicitly scoped commands and contiguous invoice object lists, keeps refund and escalation authority separate, and handles supported denials, exclusions, cancellations, unresolved references, and conditional authorization conservatively. The supported no-existing-escalation condition requires complete, target-matched observed history. Runtime observations can restrict authority but cannot create it.

The shared `run_episode` write boundary checks this permission immediately before every rule or LLM-fallback write. Denials return `InvoiceAuthorizationError` without invoking the billing service, remain in the orchestration trace, and consume the existing tool budget. Successful authorized calls still pass through the unchanged billing policy. The planner uses the same check, preventing a more permissive fallback interpretation from bypassing it.

The change is enabled with `--invoice-authorization`. Default agent behavior and default experimental invocations are unchanged. There was no additional training, model revision, GPU inference, prompt change, verifier change, or modification to historical tasks/results. The new implementation changes both planning and enforcement, so historical comparisons do not isolate only the execution guard.

## Remaining completion failures

All three variants in each of these families fail in both guarded modes:

- **Held refund authorization:** “Do not postpone that independent review…” is treated as a broad negative instruction. Incidental review/refund nouns cause denial of both actions across targets, overriding the valid independent escalation grant.
- **Three disputes:** “Never … repeat an existing escalation” is interpreted as prohibiting all escalation, rather than only duplicates. The required new escalation is withheld.
- **Two customers:** “Do not make them interchangeable” has an unresolved plural reference, so the parser denies both actions globally. A mixed refund/escalation command with a condition also exceeds the supported command grammar.

These three families account for 12 omitted required operations across nine visible-history tasks. They expose the remaining scope-resolution problem: restrictive prose can still apply too broadly. The guard prevents the LLM from recovering denied authority. Fewer calls and lower median latency (1.186 seconds hidden, 1.215 seconds visible) therefore cannot be treated as an efficiency win at comparable task quality.

The parser is a bounded experimental grammar, not a general natural-language authorization engine. Broader deployment requires better clause-specific intent representation while preserving the execution boundary and the adversarial tests. The current experiment remains opt-in.

[Aggregate results-only metrics](decision-model-authorization/research-metrics.json).

[Public setup and historical-input prerequisites](../experiments/decision_model/README.md).
