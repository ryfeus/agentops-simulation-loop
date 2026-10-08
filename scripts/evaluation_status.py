"""Print the deployed Phase 4 evaluator and online-evaluation status."""

from __future__ import annotations

import json

from scripts.evaluation_deployment import EVALUATION_MANIFEST, _load_object


def main() -> int:
    from scripts.aws_context import verified_session

    manifest = _load_object(EVALUATION_MANIFEST, "evaluation manifest")
    client = verified_session().client("bedrock-agentcore-control", region_name="us-west-2")
    evaluator = client.get_evaluator(evaluatorId=manifest["evaluator"]["id"])
    online = client.get_online_evaluation_config(
        onlineEvaluationConfigId=manifest["online_evaluation"]["id"]
    )
    payload = {
        "evaluator": {
            "id": evaluator.get("evaluatorId"),
            "status": evaluator.get("status"),
            "level": evaluator.get("level"),
        },
        "online_evaluation": {
            "id": online.get("onlineEvaluationConfigId"),
            "status": online.get("status"),
            "execution_status": online.get("executionStatus"),
            "sampling_percentage": manifest["online_evaluation"]["sampling_percentage"],
            "result_log_group": manifest["online_evaluation"]["result_log_group"],
        },
        "runtime": manifest["runtime"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
