# ADR 0008: Native ephemeral EC2 Harbor execution

- Status: Accepted
- Date: 2026-09-10

## Decision

Phase 6 leaves scenario semantics, taskify evidence, and the verifier unchanged.
It moves only Harbor's trial host to Harbor 0.22's native `ec2` environment.
Terraform owns an opt-in public foundation in `us-west-2`: a `10.77.0.0/24`
VPC, public subnet, IGW route, controller-CIDR SSH ingress, public key, and an
Ubuntu 24.04 x86_64 AMI. It does not own worker instances, a private key, IAM
profiles, NAT, DSQL connectivity, or production credentials.

The controller creates an ignored EC2 JobConfig per run. Harbor creates an
on-demand, public-IP, ephemeral agent environment for a trial, bootstraps Docker,
and deletes it afterward. Harbor may provision a separate verifier environment
when required. The task's agent and verifier containers remain no-network. The
controller must preflight the account, foundation, key pair, Harbor version, and
`RunInstances` DryRun permission before a trial can launch.

The Canonical Ubuntu image's default package source is HTTP while this execution
network permits the regional archive over HTTPS. The controller therefore
registers a tiny subclass of Harbor's native EC2 environment for its own process
that rewrites only Ubuntu archive source URLs to HTTPS before delegating to
Harbor's unmodified Docker bootstrap and trial lifecycle. This is a deliberately
narrow Harbor 0.22 compatibility shim: it replaces Harbor's private environment
registry entry only while the controller runs. Harbor is pinned to `0.22.x`; an
upgrade requires compatibility review before changing that private-registry use.

Only a `VALIDATED` Phase 5 report supplies a generated task by default. The
controller checks the canonical Scenario digest, taskify manifest, and a digest
over every rendered Harbor task file before building a wheel or contacting EC2.
An operator may explicitly select the hand-authored static task, which is
labelled as such, but static tasks cannot be used for parity because they lack
the local Phase 5 result. One clean, SHA-bound wheel is reused across a run.

Every invocation has an explicit purpose: Oracle smoke, bad calibration smoke,
three-control parity, or candidate gate. The runner records retained and
incomplete Harbor observations, runtime provenance, diagnostic worker IDs, and
cleanup in `candidate-gate.json` before selecting its process status. Oracle
and bad calibration can be accepted but never eligible; parity requires exact
local-to-EC2 agreement but never becomes eligible; candidate gates require all
requested trials to pass with homogeneous provenance and cleanup, and only then
set `accepted == eligible == true`. No result deploys, promotes, or otherwise
changes a candidate.

## Consequences

Normal CI remains AWS-free. EC2 operations require the existing expected-account
guard and explicit environment variables for an operator-owned local SSH key and
controller CIDR. Discovery and cleanup only target `Project=agentops-demo`,
`Phase=6` workers, and cleanup additionally requires an exact run ID or an
explicit age threshold. Bedrock and all networked replay remain outside Harbor.
