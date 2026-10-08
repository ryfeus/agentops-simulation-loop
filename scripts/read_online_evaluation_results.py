"""Read strict, normalized AgentCore online-evaluation result events from CloudWatch."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentops_demo.evaluation.trace_reader import parse_json_record
from scripts.evaluation_deployment import EVALUATION_MANIFEST, _load_object


def _otel_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in (
            "stringValue",
            "doubleValue",
            "intValue",
            "boolValue",
            "string_value",
            "double_value",
            "int_value",
            "bool_value",
        ):
            if key in value:
                return value[key]
    return value


def _attributes(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("attributes", {})
    if isinstance(raw, Mapping):
        return {str(key): _otel_value(value) for key, value in raw.items()}
    if isinstance(raw, list):
        result: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, Mapping) and isinstance(item.get("key"), str):
                result[item["key"]] = _otel_value(item.get("value"))
        return result
    return {}


def _first(record: Mapping[str, Any], attributes: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in attributes:
            return attributes[key]
        if key in record:
            return record[key]
    return None


def parse_evaluation_result(message: str | Mapping[str, Any]) -> dict[str, Any]:
    record = parse_json_record(message)
    if record.get("name") != "gen_ai.evaluation.result":
        raise ValueError("CloudWatch record is not an evaluation result")
    attributes = _attributes(record)
    evaluator = _first(
        record,
        attributes,
        "gen_ai.evaluation.name",
        "evaluatorName",
        "evaluator_name",
    )
    evaluator_id = _first(
        record,
        attributes,
        "gen_ai.evaluation.evaluator_id",
        "evaluatorId",
        "evaluator_id",
    )
    evaluator_arn = _first(
        record,
        attributes,
        "aws.bedrock_agentcore.evaluator.arn",
        "evaluatorArn",
        "evaluator_arn",
    )
    if evaluator_id is None and isinstance(evaluator_arn, str):
        evaluator_id = evaluator_arn.rpartition("/")[2]
    label = _first(
        record,
        attributes,
        "gen_ai.evaluation.result.label",
        "gen_ai.evaluation.score.label",
        "gen_ai.evaluation.label",
        "label",
    )
    value = _first(
        record,
        attributes,
        "gen_ai.evaluation.result.score",
        "gen_ai.evaluation.score.value",
        "value",
    )
    trace_id = _first(record, attributes, "traceId", "trace_id", "gen_ai.evaluation.trace_id")
    session_id = _first(record, attributes, "session.id", "sessionId", "session_id")
    timestamp = _first(record, attributes, "timestamp", "timeUnixNano", "time_unix_nano")
    online_config_id = _first(record, attributes, "onlineEvaluationConfigId")
    online_config_arn = _first(
        record,
        attributes,
        "aws.bedrock_agentcore.online_evaluation_config.arn",
    )
    if online_config_id is None and isinstance(online_config_arn, str):
        online_config_id = online_config_arn.rpartition("/")[2]
    online_config_name = _first(
        record,
        attributes,
        "aws.bedrock_agentcore.online_evaluation_config.name",
    )
    error_type = _first(record, attributes, "error.type", "errorCode", "error_code")
    error_message = _first(record, attributes, "error.message", "errorMessage", "error_message")
    explanation = _first(
        record,
        attributes,
        "gen_ai.evaluation.result.explanation",
        "gen_ai.evaluation.explanation",
        "explanation",
    )
    if not all(
        isinstance(item, str) and item
        for item in (
            evaluator,
            evaluator_id,
            trace_id,
            session_id,
            online_config_id,
            online_config_name,
        )
    ):
        raise ValueError("online evaluation result is missing required identity fields")
    if label is None and not isinstance(error_type, str):
        raise ValueError("online evaluation result has neither a label nor an error")
    if label is not None and not isinstance(label, str):
        raise ValueError("online evaluation result label is invalid")
    if explanation is not None and not isinstance(explanation, str):
        raise ValueError("online evaluation result explanation is invalid")
    if value is not None and not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("online evaluation result value is not numeric") from exc
    return {
        "evaluator": evaluator,
        "evaluator_id": evaluator_id,
        "evaluator_arn": evaluator_arn,
        "online_evaluation_config_id": online_config_id,
        "online_evaluation_config_name": online_config_name,
        "label": label,
        "value": value,
        "trace_id": trace_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "error_type": error_type,
        "error_message": error_message,
        "explanation": explanation,
    }


def read_results(
    *, log_group: str, start_time: float, client: Any | None = None
) -> list[dict[str, Any]]:
    if client is None:
        from scripts.aws_context import verified_session

        client = verified_session().client("logs", region_name="us-west-2")
    response = client.filter_log_events(
        logGroupName=log_group,
        startTime=int(start_time * 1000),
    )
    results = []
    for event in response.get("events", []):
        try:
            results.append(parse_evaluation_result(event["message"]))
        except (KeyError, ValueError):
            continue
    return results


def wait_for_result(
    *,
    log_group: str,
    evaluator_id: str,
    online_evaluation_config_id: str,
    session_id: str,
    trace_id: str,
    start_time: float,
    timeout: float = 600.0,
    client: Any | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for result in read_results(log_group=log_group, start_time=start_time, client=client):
            if (
                result["evaluator_id"] == evaluator_id
                and result["online_evaluation_config_id"] == online_evaluation_config_id
                and result["session_id"] == session_id
                and result["trace_id"] == trace_id
            ):
                return result
        time.sleep(10)
    raise TimeoutError(
        f"online evaluation timed out for evaluator={evaluator_id} "
        f"online_config={online_evaluation_config_id} session={session_id} "
        f"trace={trace_id} log_group={log_group}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = _load_object(EVALUATION_MANIFEST, "evaluation manifest")
    log_group = manifest["online_evaluation"]["result_log_group"]
    results = read_results(log_group=log_group, start_time=time.time() - args.minutes * 60)
    serialized = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
