from __future__ import annotations

import json

import pytest

from scripts.read_online_evaluation_results import parse_evaluation_result, wait_for_result


def result_message() -> str:
    return json.dumps(
        {
            "name": "gen_ai.evaluation.result",
            "traceId": "trace-1",
            "timeUnixNano": "123",
            "onlineEvaluationConfigId": "online-1",
            "attributes": {
                "gen_ai.evaluation.name": "DisputePolicyCompliance",
                "gen_ai.evaluation.result.label": "FAIL",
                "gen_ai.evaluation.result.score": 0.0,
                "session.id": "session-1",
                "aws.bedrock_agentcore.evaluator.arn": (
                    "arn:aws:bedrock-agentcore:us-west-2:123456789012:evaluator/eval-1"
                ),
                "aws.bedrock_agentcore.online_evaluation_config.arn": (
                    "arn:aws:bedrock-agentcore:us-west-2:123456789012:"
                    "online-evaluation-config/online-1"
                ),
                "aws.bedrock_agentcore.online_evaluation_config.name": "DisputePolicyOnline",
            },
        }
    )


def test_parse_evaluation_result() -> None:
    result = parse_evaluation_result(result_message())
    assert result["label"] == "FAIL"
    assert result["value"] == 0.0
    assert result["trace_id"] == "trace-1"
    assert result["session_id"] == "session-1"
    assert result["online_evaluation_config_id"] == "online-1"
    assert result["online_evaluation_config_name"] == "DisputePolicyOnline"


def test_parse_live_error_result() -> None:
    payload = json.loads(result_message())
    attributes = payload["attributes"]
    attributes.pop("gen_ai.evaluation.result.label")
    attributes.pop("gen_ai.evaluation.result.score")
    attributes["error.type"] = "TRACE_FORMAT_UNSUPPORTED"
    attributes["error.message"] = "unsupported telemetry"
    result = parse_evaluation_result(payload)
    assert result["label"] is None
    assert result["error_type"] == "TRACE_FORMAT_UNSUPPORTED"
    assert result["error_message"] == "unsupported telemetry"


def test_malformed_result_fails_closed() -> None:
    with pytest.raises(ValueError, match="not an evaluation result"):
        parse_evaluation_result("{}")


class FakeLogs:
    def filter_log_events(self, **_kwargs):
        return {"events": [{"message": result_message()}]}


def test_wait_requires_exact_correlation() -> None:
    result = wait_for_result(
        log_group="group",
        evaluator_id="eval-1",
        online_evaluation_config_id="online-1",
        session_id="session-1",
        trace_id="trace-1",
        start_time=0,
        timeout=0.1,
        client=FakeLogs(),
    )
    assert result["label"] == "FAIL"
