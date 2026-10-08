# Redistribution inventory

The repository retains its original MIT license and Rustem Feyzkhanov attribution. The tracked export contains project source, deterministic synthetic fixtures, documentation, manifests and lockfiles. No installed dependency trees, model weights, adapters, checkpoints, vendored repositories or runtime archives are included.

The core dependencies are fetched during installation rather than bundled in this source export. Installed package metadata reviewed during export identifies:

| Dependencies | Declared license |
| --- | --- |
| LangGraph, LangChain, Pydantic, AG-UI LangGraph | MIT |
| FastMCP, Harbor, OpenInference LangChain instrumentation, boto3, Aurora DSQL connector | Apache-2.0 |

Check upstream distributions and their license/NOTICE files when redistributing built images or installed dependencies. This table is an inventory of direct packages reviewed, not an assertion that every transitive package inherits the project's MIT license. Frontend and training dependencies retain their upstream terms; their source packages are not bundled here. Compatibility adapters extend upstream Harbor/TRL interfaces, and training environments fetch those libraries separately.

The optional model is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE), with Apache-2.0 terms at the pinned model revision. Model references and tokenizer/template hashes are retained; no weights or derived adapters are redistributed. Optional inference/training requires the operator to obtain model artifacts under their upstream terms.

Synthetic billing tasks derive from the checked-in generators and family specifications. The export preserves their corpus digests and does not introduce real customers, live traces or external datasets. Original research operational evidence is not included. A results-only report is not a redistribution of the omitted private evidence bundle.

## Reviewed scanner exceptions

- Account IDs `000000000000`, `111111111111`, `123456789012`, and `210987654321` are fictional test/example identifiers.
- The two EC2 IDs in `tests/unit/test_harbor_ec2.py` are fictional controller fixtures and are allowed only in that file.
- `s3://bucket/prefix` is a placeholder, and the URI formatter in `rl/phase12/artifacts.py` constructs the operator's destination dynamically.
- `/home/user/input` is a standard Harbor sandbox path, not a developer's private home directory.
- Gitleaks' sole rule-specific exception matches the exact public Qwen tokenizer revision line in the canonical experiment manifest. Other secrets remain scanned with default rules. CI also seeds a synthetic credential in that same file and requires detection.

The release checker reports only paths and categories. The local confidential denylist and original export/source revision mapping remain outside this repository. Gitleaks is an additional gate against real secrets in the exported tree and history.
