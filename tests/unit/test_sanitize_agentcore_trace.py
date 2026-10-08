from __future__ import annotations

import json
from pathlib import Path

from scripts.sanitize_agentcore_trace import sanitize_trace

FIXTURE = Path("tests/fixtures/agentcore/langgraph_billing_trace.json")


def test_committed_fixture_preserves_real_semantics_without_request_metadata() -> None:
    payload = json.loads(FIXTURE.read_text())
    spans = payload["evaluationInput"]["sessionSpans"]
    names = [span["name"] for span in spans]
    assert names == [
        "ChatBedrockConverse",
        "model",
        "get_invoice",
        "tools",
        "ChatBedrockConverse",
        "model",
        "escalate_dispute",
        "tools",
        "ChatBedrockConverse",
        "model",
        "billing_agent",
        "POST /invocations",
    ]
    tools = {
        span["attributes"].get("tool.name"): span["attributes"]
        for span in spans
        if span["attributes"].get("tool.name")
    }
    assert tools["get_invoice"]["input.value"] == "inv-123"
    assert json.loads(tools["get_invoice"]["output.value"])["status"] == "disputed"
    assert json.loads(tools["escalate_dispute"]["output.value"]) == {
        "escalation_id": "escalation-inv-123",
        "invoice_id": "inv-123",
        "success": True,
    }
    serialized = FIXTURE.read_text()
    for forbidden in (
        "aws.auth",
        "aws.request_id",
        "RequestId",
        "x-amzn-requestid",
        "http.",
        "net.peer",
        "123456789012",
    ):
        assert forbidden not in serialized


def test_sanitizer_maps_ids_and_drops_unselected_spans() -> None:
    raw = {
        "evaluationInput": {
            "sessionSpans": [
                {
                    "name": "ignored HTTP",
                    "spanId": "ignored",
                    "traceId": "live-trace",
                    "scope": {"name": "opentelemetry.instrumentation.http"},
                    "attributes": {"aws.auth.account.access_key": "secret"},
                },
                {
                    "name": "get_invoice",
                    "spanId": "child",
                    "parentSpanId": "root",
                    "traceId": "live-trace",
                    "scope": {
                        "name": "openinference.instrumentation.langchain",
                        "version": "0.1.74",
                    },
                    "status": {"code": "STATUS_CODE_OK"},
                    "attributes": {
                        "openinference.span.kind": "TOOL",
                        "tool.name": "get_invoice",
                        "input.value": "inv-123",
                        "output.value": json.dumps(
                            {
                                "artifact": {
                                    "structured_content": {
                                        "result": {"id": "inv-123", "status": "disputed"}
                                    }
                                },
                                "response_metadata": {"RequestId": "secret"},
                            }
                        ),
                    },
                },
                {
                    "name": "POST /invocations",
                    "spanId": "root",
                    "parentSpanId": "outside",
                    "traceId": "live-trace",
                    "scope": {"name": "opentelemetry.instrumentation.fastapi"},
                    "attributes": {
                        "session.id": "live-session",
                        "agentops.agent_config_fingerprint": "live-fingerprint",
                        "agentops.billing_policy_mode": "permissive",
                        "agentops.harness_framework": "langgraph",
                        "agentops.harness_version": "v1",
                        "agentops.model_id": "model",
                        "agentops.model_provider": "bedrock",
                        "agentops.prompt_version": "billing-v1",
                        "agentops.source_revision": "live-revision",
                        "agentops.tools_version": "billing-mcp-v1",
                        "http.request.header.authorization": "secret",
                    },
                },
            ]
        }
    }
    sanitized = sanitize_trace(raw)
    spans = sanitized["evaluationInput"]["sessionSpans"]
    assert [span["name"] for span in spans] == ["get_invoice", "POST /invocations"]
    assert spans[0]["parentSpanId"] == spans[1]["spanId"]
    serialized = json.dumps(sanitized)
    assert "secret" not in serialized
    assert "live-trace" not in serialized
    assert "live-session" not in serialized
