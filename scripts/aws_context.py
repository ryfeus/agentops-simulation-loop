"""Fail-closed operator identity checks for opt-in AWS controllers."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

REGION = "us-west-2"


def expected_account_id(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    value = env.get("EXPECTED_AWS_ACCOUNT_ID", "")
    if not re.fullmatch(r"[0-9]{12}", value):
        raise ValueError("EXPECTED_AWS_ACCOUNT_ID must be set to exactly 12 digits")
    return value


def require_account(sts_client: Any, expected: str | None = None) -> str:
    desired = expected_account_id(
        None if expected is None else {"EXPECTED_AWS_ACCOUNT_ID": expected}
    )
    actual = sts_client.get_caller_identity().get("Account")
    if actual != desired:
        raise RuntimeError(f"refusing AWS operation in account {actual}; expected {desired}")
    return desired


def verified_session(*, session: Any | None = None, session_factory: Any | None = None) -> Any:
    """Validate configuration before loading credentials; support offline injected sessions."""
    expected = expected_account_id()
    if session is None:
        if session_factory is None:
            import boto3

            session_factory = boto3.Session
        session = session_factory(
            profile_name=os.getenv("AWS_PROFILE", "default"), region_name=REGION
        )
    require_account(session.client("sts"), expected)
    return session
