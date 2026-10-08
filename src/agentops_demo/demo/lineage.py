"""Cross-phase identity checks for frozen demo evidence."""

from __future__ import annotations

from agentops_demo.demo.contracts import LineageSummary, ProductionFailureIdentity
from agentops_demo.scale.contracts import CandidateGate
from agentops_demo.taskify.integrity import scenario_sha256
from agentops_demo.taskify.reproduction import ReproductionReport
from agentops_demo.validation.scenario import load_scenario


def validate_lineage(
    source: ProductionFailureIdentity,
    scenario_path: str,
    local: ReproductionReport,
    parity: CandidateGate,
    scale: CandidateGate,
) -> LineageSummary:
    scenario = load_scenario(scenario_path)
    digest = scenario_sha256(scenario)
    task_digest = local.harbor_task.sha256 if local.harbor_task else ""
    values = [
        local.source.trace_id
        == source.trace_id
        == parity.benchmark.trace_id
        == scale.benchmark.trace_id,
        local.source.session_id
        == source.session_id
        == parity.benchmark.session_id
        == scale.benchmark.session_id,
        local.source.source_revision == source.source_revision,
        local.source.agent_config_fingerprint == source.agent_config_fingerprint,
        parity.benchmark.scenario_sha256 == digest == scale.benchmark.scenario_sha256,
        parity.benchmark.harbor_task_sha256 == task_digest == scale.benchmark.harbor_task_sha256,
    ]
    return LineageSummary(
        trace_id=source.trace_id,
        session_id=source.session_id,
        scenario_sha256=digest,
        harbor_task_sha256=task_digest,
        passed=all(values),
    )
