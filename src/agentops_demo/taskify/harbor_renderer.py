"""Render a billing-v1 Scenario into an isolated Harbor task directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from agentops_demo.contracts.scenario import Scenario
from agentops_demo.validation.scenario import dump_scenario_yaml
from scripts.generate_harbor_seed import render_seed


class HarborRenderError(ValueError):
    pass


ROOT = Path(__file__).resolve().parents[3]
STATIC_TASK = ROOT / "benchmarks" / "disputed-refund"


async def render_harbor_task(scenario: Scenario, output: Path) -> Path:
    """Render only the supported billing invariant vocabulary into a Harbor task."""

    supported = {
        "no_refund",
        "must_refund",
        "no_escalation",
        "must_escalate",
        "invoice_status",
        "unchanged",
    }
    if any(invariant.type not in supported for invariant in scenario.expected_invariants):
        raise HarborRenderError("scenario contains unsupported billing invariants")
    if output.exists():
        shutil.rmtree(output)
    (output / "environment").mkdir(parents=True)
    (output / "solution").mkdir()
    (output / "tests").mkdir()
    scenario_path = output / "scenario.yaml"
    scenario_path.write_text(dump_scenario_yaml(scenario), encoding="utf-8")
    (output / "instruction.md").write_text(scenario.instruction + "\n", encoding="utf-8")
    (output / "environment" / "seed.sql").write_text(
        await render_seed(scenario_path), encoding="utf-8"
    )
    for relative in (
        "environment/Dockerfile",
        "environment/requirements.txt",
        "tests/Dockerfile",
        "tests/test.sh",
        "tests/verify.py",
    ):
        source = STATIC_TASK / relative
        destination = output / relative
        destination.write_bytes(source.read_bytes())
    (output / "solution" / "solve.sh").write_text(_oracle_script(scenario), encoding="utf-8")
    hidden = {
        "scenario_id": scenario.id,
        "expected_invariants": [
            item.model_dump(mode="json") for item in scenario.expected_invariants
        ],
        "initial_state": scenario.initial_state.model_dump(mode="json"),
        "benchmark": scenario.benchmark.model_dump(mode="json") if scenario.benchmark else None,
    }
    (output / "tests" / "scenario.json").write_text(
        json.dumps(hidden, sort_keys=True) + "\n", encoding="utf-8"
    )
    source = scenario.provenance.source
    config = scenario.provenance.originating_agent_config
    metadata = {
        "scenario_id": scenario.id,
        "category": "billing",
        "failure_type": "policy_violation" if source.kind == "trace" else "static_benchmark",
        "source_kind": source.kind,
        "source_trace_id": source.trace_ref.rsplit("/", 1)[-1] if source.trace_ref else "",
        "source_session_id": (
            source.trace_ref.split("/trace/", 1)[0].rsplit("/", 1)[-1] if source.trace_ref else ""
        ),
        "source_evaluator": source.evaluation.evaluator if source.evaluation else "",
        "source_candidate_fingerprint": config.fingerprint() if config else "",
    }
    metadata_toml = "\n".join(f"{key} = {value!r}" for key, value in metadata.items())
    (output / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.4"',
                'artifacts = ["/app/data/billing.db", "/app/data/agent-run.json"]',
                "",
                "[task]",
                f'name = "agentops-taskify/{scenario.id}"',
                'version = "0.1.0"',
                'description = "Taskified billing policy regression"',
                'keywords = ["billing", "policy", "taskify"]',
                "",
                "[metadata]",
                metadata_toml,
                "",
                "[agent]",
                "timeout_sec = 120",
                'network_mode = "no-network"',
                'user = "root"',
                "",
                "[verifier]",
                "timeout_sec = 60",
                'environment_mode = "separate"',
                'network_mode = "no-network"',
                "",
                "[verifier.environment]",
                'network_mode = "no-network"',
                "",
                "[environment]",
                'network_mode = "no-network"',
                "build_timeout_sec = 600",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


def _oracle_script(scenario: Scenario) -> str:
    refunds = {record.invoice_id for record in scenario.initial_state.refunds}
    escalations = {record.invoice_id for record in scenario.initial_state.escalations}
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "python - <<'PY'",
        "import json, sqlite3",
        "",
        "with sqlite3.connect('/app/data/billing.db') as connection:",
    ]
    refunded: set[str] = set()
    escalated: set[str] = set()
    statuses: dict[str, str] = {}
    for invariant in scenario.expected_invariants:
        if invariant.type == "no_refund" and invariant.invoice_id in refunds:
            # no_refund means no *new* refund; existing records are valid.
            continue
        if invariant.type == "must_refund":
            if invariant.invoice_id in refunds:
                raise HarborRenderError(
                    f"initial state is unsatisfiable: {invariant.invoice_id} already has a refund"
                )
            refunded.add(invariant.invoice_id)
        elif invariant.type == "must_escalate":
            if invariant.invoice_id in escalations:
                raise HarborRenderError(
                    "initial state is unsatisfiable: "
                    f"{invariant.invoice_id} already has an escalation"
                )
            escalated.add(invariant.invoice_id)
        elif invariant.type == "invoice_status":
            statuses[invariant.invoice_id] = invariant.status
    for invoice in sorted(refunded):
        refund = f"refund-{invoice}"
        lines.extend(
            [
                "    connection.execute(",
                "        'INSERT OR IGNORE INTO refunds '",
                "        '(id, invoice_id, reason) VALUES (?, ?, ?)',",
                f"        ({refund!r}, {invoice!r}, 'Oracle refund'),",
                "    )",
                "    connection.execute(\"UPDATE invoices SET status = 'refunded' WHERE id = ?\",",
                f"                       ({invoice!r},))",
            ]
        )
    for invoice in sorted(escalated):
        escalation = f"escalation-{invoice}"
        lines.extend(
            [
                "    connection.execute(",
                "        'INSERT OR IGNORE INTO escalations '",
                "        '(id, invoice_id, reason) VALUES (?, ?, ?)',",
                f"        ({escalation!r}, {invoice!r}, 'Oracle specialist review'),",
                "    )",
                "    connection.execute(",
                "        \"UPDATE invoices SET status = 'disputed' WHERE id = ?\",",
                f"        ({invoice!r},),",
                "    )",
            ]
        )
    for invoice, status in sorted(statuses.items()):
        lines.extend(
            [
                '    connection.execute("UPDATE invoices SET status = ? WHERE id = ?",',
                f"                       ({status!r}, {invoice!r}))",
            ]
        )
    if not refunded and not escalated and not statuses:
        lines.append("    pass")
    if scenario.benchmark is not None:
        required_inspections = scenario.benchmark.trajectory.required_inspections
        inspection_ids = sorted(
            {invoice.id for invoice in scenario.initial_state.invoices} | set(required_inspections)
        )
        calls = [
            {"name": "get_invoice", "arguments": {"invoice_id": invoice_id}}
            for invoice_id in inspection_ids
        ]
        calls.extend(
            {
                "name": "refund_invoice",
                "arguments": {"invoice_id": invoice, "reason": "Oracle refund"},
            }
            for invoice in sorted(refunded)
        )
        calls.extend(
            {
                "name": "escalate_dispute",
                "arguments": {"invoice_id": invoice, "reason": "Oracle specialist review"},
            }
            for invoice in sorted(escalated)
        )
        lines.extend(
            [
                "with open('/app/data/agent-run.json', 'w', encoding='utf-8') as artifact:",
                f"    json.dump({{'tool_calls': {calls!r}}}, artifact, sort_keys=True)",
            ]
        )
    lines.extend(["PY", ""])
    return "\n".join(lines)
