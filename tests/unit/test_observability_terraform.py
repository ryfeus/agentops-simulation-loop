from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_transaction_search_policy_includes_exact_runtime_log_group() -> None:
    root = (ROOT / "infra/terraform/main.tf").read_text()
    module = (ROOT / "infra/terraform/modules/observability/main.tf").read_text()

    assert 'runtime_log_group = local.runtime_enabled ? local.runtime_log_group : ""' in root
    assert "log-group:${var.runtime_log_group}:*" in module
    assert 'var.runtime_log_group == "" ? []' in module


def test_sensitive_capture_requires_explicit_synthetic_opt_in() -> None:
    variables = (ROOT / "infra/terraform/variables.tf").read_text()
    runtime = (ROOT / "infra/terraform/main.tf").read_text()
    block = variables.split('variable "synthetic_demo_trace_content_capture"', 1)[1]
    assert "default     = false" in block
    for name in (
        "AGENTOPS_TRACE_CONTENT_ENABLED",
        "AGENTOPS_WORLD_SNAPSHOT_ENABLED",
        "AGENTOPS_WORLD_SNAPSHOT_REQUIRED",
    ):
        line = next(line for line in runtime.splitlines() if name in line)
        assert "tostring(var.synthetic_demo_trace_content_capture)" in line
    for name in ("OPENINFERENCE_HIDE_INPUTS", "OPENINFERENCE_HIDE_OUTPUTS"):
        line = next(line for line in runtime.splitlines() if name in line)
        assert "tostring(!var.synthetic_demo_trace_content_capture)" in line
    for root in ("infra/terraform", "infra/terraform-rl-smoke"):
        provider = (ROOT / root / "providers.tf").read_text()
        assert "allowed_account_ids = [var.expected_aws_account_id]" in provider
