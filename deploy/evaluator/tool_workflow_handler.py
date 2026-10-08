"""AgentCore Lambda entry point for deterministic tool-workflow checks."""

from typing import Any

from base_handler import evaluate_handler

from agentops_demo.evaluation.suite import evaluate_tool_workflow

EVALUATOR_NAME = "ToolWorkflow"


def handler(event: Any, _context: Any) -> dict[str, object]:
    return evaluate_handler(
        event,
        evaluator_name=EVALUATOR_NAME,
        evaluator=evaluate_tool_workflow,
    )
