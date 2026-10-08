"""AgentCore Lambda entry point for deterministic mutation-cardinality checks."""

from typing import Any

from base_handler import evaluate_handler

from agentops_demo.evaluation.suite import evaluate_mutation_cardinality

EVALUATOR_NAME = "MutationCardinality"


def handler(event: Any, _context: Any) -> dict[str, object]:
    return evaluate_handler(
        event,
        evaluator_name=EVALUATOR_NAME,
        evaluator=evaluate_mutation_cardinality,
    )
