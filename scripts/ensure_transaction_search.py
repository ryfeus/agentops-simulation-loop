"""Preview, explicitly apply, or restore account-wide Transaction Search settings."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from scripts.aws_context import REGION, expected_account_id, verified_session

POLICY_NAME = "agentops-demo-transaction-search"
RECEIPT_ROOT = Path(".agentcore/observability")
ACKNOWLEDGEMENT = "AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE"


def _acknowledge() -> None:
    if os.getenv(ACKNOWLEDGEMENT) != "yes":
        raise RuntimeError(f"set {ACKNOWLEDGEMENT}=yes to change account-wide settings")


def _state(xray: Any) -> dict[str, Any]:
    destination = xray.get_trace_segment_destination()
    rules = xray.get_indexing_rules().get("IndexingRules", [])
    rule = next((item for item in rules if item.get("Name") == "Default"), {})
    percentage = rule.get("Rule", {}).get("Probabilistic", {}).get("DesiredSamplingPercentage")
    return {
        "destination": destination.get("Destination"),
        "status": destination.get("Status"),
        "indexing_percentage": percentage,
    }


def _settings(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state[key] for key in ("destination", "indexing_percentage")}


def _validate_settings(settings: dict[str, Any]) -> None:
    if settings.get("destination") not in {"XRay", "CloudWatchLogs"}:
        raise RuntimeError("cannot safely restore an unknown trace destination")
    percentage = settings.get("indexing_percentage")
    if not isinstance(percentage, (float, int)) or not math.isfinite(percentage):
        raise RuntimeError("cannot safely restore an unknown indexing percentage")
    if not 0 <= percentage <= 100:
        raise ValueError("indexing percentage must be between 0 and 100")


def _save(path: Path, receipt: dict[str, Any], *, create: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("receipt must not be a symlink")
    temporary = path if create else path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not create:
            os.replace(temporary, path)
    finally:
        if not create:
            temporary.unlink(missing_ok=True)


def _wait_active(xray: Any, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while _state(xray)["status"] != "ACTIVE":
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Transaction Search did not become ACTIVE; retain receipt for recovery"
            )
        time.sleep(2)


def _transition(
    xray: Any, path: Path, receipt: dict[str, Any], target: dict[str, Any], timeout: float
) -> None:
    current = _settings(_state(xray))
    recorded = receipt["current"]
    pending = receipt.get("pending")
    # A crash after the API call but before receipt persistence is recoverable only
    # when the observable state is exactly the journaled before/after state.
    if current != recorded and current != pending:
        raise RuntimeError("observability state changed concurrently; refusing to guess")
    for key in ("destination", "indexing_percentage"):
        observed = _state(xray)
        if _settings(observed) != current:
            raise RuntimeError("observability state changed concurrently; refusing to guess")
        if current[key] == target[key]:
            continue
        if observed["status"] != "ACTIVE":
            raise RuntimeError("trace destination is transitioning; retry after it becomes ACTIVE")
        after = {**current, key: target[key]}
        receipt["current"] = current
        receipt["pending"] = after
        _save(path, receipt)
        if key == "destination":
            xray.update_trace_segment_destination(Destination=target[key])
            _wait_active(xray, timeout)
        else:
            xray.update_indexing_rule(
                Name="Default",
                Rule={"Probabilistic": {"DesiredSamplingPercentage": target[key]}},
            )
        if _settings(_state(xray)) != after:
            raise RuntimeError("observable settings differ from requested state; retain receipt")
        current = after
        receipt["current"] = current
        receipt["pending"] = None
        _save(path, receipt)
    receipt["current"] = current
    receipt["pending"] = None
    _save(path, receipt)


def ensure(
    *,
    check_only: bool = False,
    apply: bool = False,
    timeout: float = 600.0,
    indexing_percentage: float | None = None,
    receipt_path: Path | None = None,
    session: Any | None = None,
) -> dict[str, object]:
    if apply and check_only:
        raise ValueError("apply and check are mutually exclusive")
    if apply:
        _acknowledge()
        if indexing_percentage is None:
            raise RuntimeError("an explicit indexing percentage is required")
    if indexing_percentage is not None and (
        not math.isfinite(indexing_percentage) or not 0 <= indexing_percentage <= 100
    ):
        raise ValueError("indexing percentage must be between 0 and 100")
    session = verified_session(session=session)
    account = expected_account_id()
    xray = session.client("xray")
    state = _state(xray)
    policies = session.client("logs").describe_resource_policies().get("resourcePolicies", [])
    policy_present = any(item.get("policyName") == POLICY_NAME for item in policies)
    result = {
        "account": account,
        "region": REGION,
        **state,
        "resource_policy_present": policy_present,
    }
    if check_only:
        if state["destination"] != "CloudWatchLogs":
            raise RuntimeError("trace destination is not CloudWatchLogs")
        if state["status"] != "ACTIVE" or not policy_present:
            raise RuntimeError("Transaction Search must be ACTIVE with its resource policy")
        if indexing_percentage is not None and state["indexing_percentage"] != indexing_percentage:
            raise RuntimeError("Transaction Search indexing differs from the expected percentage")
    if apply:
        if not policy_present:
            raise RuntimeError(
                f"required CloudWatch Logs resource policy {POLICY_NAME!r} is missing"
            )
        before = _settings(state)
        _validate_settings(before)
        if state["status"] != "ACTIVE":
            raise RuntimeError("trace destination must be ACTIVE before applying changes")
        target = {"destination": "CloudWatchLogs", "indexing_percentage": indexing_percentage}
        path = receipt_path or RECEIPT_ROOT / f"{account}-{REGION}-{uuid.uuid4().hex}.json"
        receipt = {
            "schema_version": "1",
            "account": account,
            "region": REGION,
            "before": before,
            "target": target,
            "current": before,
            "pending": None,
            "restored": False,
        }
        _save(path, receipt, create=True)
        _transition(xray, path, receipt, target, timeout)
        result.update(_state(xray))
        result["receipt"] = str(path)
    return result


def restore(path: Path, *, session: Any | None = None, timeout: float = 600.0) -> dict[str, Any]:
    _acknowledge()
    if path.is_symlink():
        raise RuntimeError("receipt must not be a symlink")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise RuntimeError("receipt must be owned by the operator with mode 0600")
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") != "1" or receipt.get("region") != REGION:
        raise RuntimeError("receipt schema or region differs")
    if receipt.get("account") != expected_account_id():
        raise RuntimeError("receipt belongs to a different account")
    for name in ("before", "target", "current"):
        _validate_settings(receipt[name])
    if receipt.get("pending") is not None:
        _validate_settings(receipt["pending"])
    session = verified_session(session=session)
    _transition(session.client("xray"), path, receipt, receipt["before"], timeout)
    receipt["restored"] = True
    _save(path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--restore", type=Path)
    parser.add_argument("--indexing-percentage", type=float)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    result = (
        restore(args.restore)
        if args.restore
        else ensure(
            check_only=args.check,
            apply=args.apply,
            indexing_percentage=args.indexing_percentage,
            receipt_path=args.receipt,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
