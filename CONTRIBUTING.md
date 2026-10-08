# Contributing

Use Python 3.12 and uv; web development uses Node 22 and pnpm 11.19.0. Docker is required for Harbor/Taskify execution. Preserve the MIT attribution and keep changes focused. AWS access and GPU runs are not required for ordinary pull requests.

```bash
uv sync --locked --all-groups --all-extras
make check
make benchmark-billing-generated-check benchmark-billing-phase10-check
make benchmark-billing-phase10-render-smoke
make rl-phase12-config-check rl-phase12-render-smoke
make tf-validate
uv run python -m scripts.oss_release_check
```

Before provenance-bound Docker checks, commit intended source changes so execution packages reference a clean revision:

```bash
make harbor-e2e
make taskify-fixture-harbor
make agentcore-image-check
```

`make web-check` checks frontend lint, types and build. Default CI uses credential-free Ubuntu jobs; never require cloud secrets on external PRs. Keep training dependencies in the separate `rl/` environment and avoid unrelated GPU lock updates. Dependabot covers the root Python environment and GitHub Actions. Frontend updates must preserve the pinned pnpm 11 lockfile; verify updater compatibility before enabling automatic frontend lock changes.

Fixture changes require an explicit semantic reason. Preserve frozen corpus/scenario fingerprints and seeds; use existing generator/check commands and review changed artifacts together. Do not regenerate frozen experiment manifests to conceal a failure or alter a registered learning gate. Include meaningful validation and note unavailable checks accurately.

Never put real customer data, account identifiers, credentials, private connection information, runtime archives, or operational receipts in issues, tests or PRs. Use documented fictional test identifiers. The release checker and complementary secret scan must pass before publication. See [security](SECURITY.md) for private disclosure.
