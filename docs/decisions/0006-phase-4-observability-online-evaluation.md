# ADR 0006: Trace-based online policy detection

- Status: Accepted
- Date: 2026-09-08

## Context

Phase 3 runs the billing agent through AgentCore, LangGraph, MCP, and Aurora
DSQL. Phase 4 must detect the canonical disputed-invoice failure from telemetry
without weakening the stable contracts or treating model prose as evidence.
Harbor already provides the authoritative state-based regression gate.

## Decisions

1. The AgentCore image is launched with the AWS ADOT distribution and exactly
   one OpenInference LangChain instrumentor. Traces use the unified CloudWatch
   Logs destination in `us-west-2`.
2. Transaction Search indexing and AgentCore online-evaluation sampling are both
   100% for this synthetic demo. These settings are not production defaults.
3. OpenInference inputs and outputs are explicitly visible for this synthetic
   runtime because tool inputs and results are required evaluator evidence.
   Candidate identity is recorded as bounded `agentops.*` span attributes; no
   credentials, secrets, or full configuration document is recorded.
4. The deterministic evaluator inspects only the verified OpenInference tool
   span shape. A successful refund of any disputed invoice fails; all observed
   disputed invoices must instead have a successful escalation. Missing policy
   evidence is not applicable, while malformed or unsupported telemetry is an
   explicit trace-format error rather than a policy failure.
5. A Python 3.12, standard-library-only Lambda implements the AgentCore code
   evaluator envelope. Its ZIP is byte reproducible and hash-bound to Terraform
   deployment provenance.
6. Runtime execution IAM and AgentCore Evaluation IAM remain separate. The
   evaluation role can read trace data, write evaluation results, access log
   indexes, and invoke only the evaluator Lambda; it cannot invoke Bedrock
   models, AgentCore runtimes, or DSQL.
7. Evaluation deployment defaults off and fails closed unless the package,
   runtime configuration and manifest, observability evidence, tfvars, and
   evaluation manifest are mutually consistent. All safe Terraform commands use
   the same validated var-file selection and preserve complete deployments.
8. A failure artifact is only an evidence pointer correlated to evaluator,
   online configuration, trace, session, runtime, source revision, and candidate
   fingerprint. It contains no reconstructed Scenario or Harbor verifier data.
9. Real Bedrock reproduction is attempted in bounded fresh sessions. An explicit
   `scripted/bad` runtime may calibrate the complete production path if the model
   does not reproduce the failure, and is always reported as synthetic evidence.

## Consequences

Phase 4 completes failure detection, not regression generation. Online
evaluation can surface a correlated policy violation, while Harbor continues to
decide whether a candidate fixes the frozen world state. Trace-to-Scenario
conversion, generic task rendering, promotion, alerts, UI, private networking,
remote Terraform state, and optimization remain deferred.
