# ADR 0009: Guarded end-to-end evidence bundle

- Status: Accepted
- Date: 2026-09-11

## Decision

Phase 7 adds a guarded `live`/`replay` orchestrator around the existing
AgentCore, taskify, and Harbor interfaces. A run creates an ignored,
timestamp-addressed `.demo/runs/<run-id>/run.json` before any production
mutation. The account guard requires `123456789012` in `us-west-2`; preflight
also requires a clean checkout, the opt-in EC2 foundation, controller key/CIDR,
and no tagged workers for the requested run ID.

Live mode requires a Bedrock-backed deployed AgentConfig, uses the bounded existing
Bedrock failure trigger, and freezes its exact
failure evidence and correlated trace once. If the required failure does not
occur, the run fails; it never quietly changes to replay mode. Replay mode is
explicit and can copy only an accepted live run after validating the source
failure and trace hashes.

For the supported `get_invoice` then `refund_invoice` Bedrock trajectory,
taskify retains the original production provider, model, revision, and
AgentConfig fingerprint. It derives a separate `scripted/bad` AgentConfig for
no-network Harbor calibration and records its distinct fingerprint,
`trajectory_calibration` identity, expected tools, and derivation flag. This is
calibration, not an exact Bedrock replay. Unsupported Bedrock trajectories are
recorded as `UNSUPPORTED`; Harbor never receives network or Bedrock credentials.

The orchestrator dynamically locates its taskify Scenario, validates manifest
and source hashes, runs local Oracle/known-good/calibration gates (`1 / 1 / 0`),
then invokes the Phase 6 taskify-only parity and four-way scale gates. It copies
only EC2 gate reports into the bundle. A final report is accepted only when its
source IDs and hashes, Scenario digest, Harbor task digest, local result, parity,
four-of-four candidate gate, provenance, and zero-worker cleanup all validate.
An independent assertion reloads that evidence, stage hashes, and nested Phase 5/6
reports rather than trusting the final summary; it rejects credential field names.

Resume is conservative: it is allowed only at the same Git revision after every
previously completed stage's named artifact hashes validate. Existing IDs reject
by default and there is no force-resume. Candidate eligibility remains evidence
only; it does not deploy or promote any configuration.

## Consequences

Normal CI remains credential-free. Live and replay demonstrations remain a
manual, account-guarded operation using the operator's SSO session and existing
Phase 6 EC2 foundation. The run directory is ignored diagnostic material, while
the contracts and fixtures remain versioned and testable without AWS.

The isolated Harbor profile remains no-network and credential-free. A separately
opted-in Bedrock execution profile uses public agent egress only, a distinct task
digest, a no-network verifier, and a short-lived inference-only STS role. It does
not place credentials on EC2 instances, AgentCore, DSQL, or verifiers.
