# Local quick start

Install Git, Make, Python 3.12 and uv. Use a fresh checkout with committed source files: Harbor packages bind their provenance to the current Git revision and require a clean worktree. Python supports the range in `pyproject.toml`; CI validates Python 3.12. Web checks use Node 22 and pnpm 11.19.0. Docker must be running for isolated trials.

```bash
uv sync --locked --all-groups
make demo MODE=bad
make demo MODE=correct
```

Each demo reseeds the local SQLite database at `data/billing.db` and runs the real local MCP HTTP boundary. Bad behavior calls `get_invoice` and `refund_invoice`; correct behavior calls `get_invoice` and `escalate_dispute`. Bad behavior demonstrates a policy violation intentionally: the permissive demo server permits it so the verifier can detect it.

```bash
make harbor-e2e
make taskify-fixture-harbor
```

Harbor runs Oracle, known bad, and correct candidates in Docker. Expected rewards are `1`, `0`, and `1`. Inspect `.harbor/jobs/` and `.harbor/package/manifest.json`. The fixture conversion uses checked-in synthetic traces, writes `.taskify/disputed-refund-000000000000/`, and asserts Oracle/known-good passes and originating failure. Its `reproduction.json` must say `VALIDATED`.

For browser chat:

```bash
make chat
make chat AGENTCORE_CANDIDATE=scripted-bad
```

The supervised browser UI uses localhost ports 3000, 8080, and 8000. Chat preserves an existing database. Use `make demo-reset` only when you intend to replace its billing state. `make chat-cli` uses the same scripted-correct default. Bedrock requires an explicit candidate, model ID, and credentials.

## Troubleshooting

- Dirty-worktree error: commit intended edits before provenance-bound Harbor tests; do not bypass cleanliness checks.
- Docker connection error: start Docker and confirm the current Docker context points to its local daemon. Docker jobs may need platform emulation on macOS/ARM; Ubuntu CI is the release reference.
- Port collision: stop a prior local chat/MCP process before running the demo. The scripted demo also accepts `BILLING_MCP_PORT`.
- Missing packages: run the locked install command above. Advanced AWS and web checks use extras; `uv sync --locked --all-groups --all-extras` installs them without requiring credentials.

Run `make check` for lint, web checks, ordinary credential-free tests, scenario checks, and runtime dependency checks. See [contributing](../CONTRIBUTING.md) for the full matrix.
