from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path("scripts/with_aws_credentials.sh").resolve()


def fake_aws(tmp_path: Path, account_id: str = "123456789012") -> Path:
    executable = tmp_path / "aws"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2" = "sts get-caller-identity" ]; then\n'
        f"  printf '%s\\n' '{{\"Account\":\"{account_id}\"}}'\n"
        'elif [ "$1 $2" = "configure export-credentials" ]; then\n'
        "  printf '%s\\n' 'export AWS_ACCESS_KEY_ID=fake-access' "
        "'export AWS_SECRET_ACCESS_KEY=fake-secret' 'export AWS_SESSION_TOKEN=fake-token'\n"
        "else\n"
        "  exit 9\n"
        "fi\n"
    )
    executable.chmod(0o755)
    return executable


def environment(tmp_path: Path) -> dict[str, str]:
    value = os.environ.copy()
    value["PATH"] = f"{tmp_path}:{value['PATH']}"
    value.pop("EXPECTED_AWS_ACCOUNT_ID", None)
    return value


@pytest.mark.parametrize("account_id", [None, "", "123", "abcdefghijkl", "12345678901x"])
def test_wrapper_rejects_missing_or_invalid_expected_account(
    tmp_path: Path, account_id: str | None
) -> None:
    fake_aws(tmp_path)
    env = environment(tmp_path)
    if account_id is not None:
        env["EXPECTED_AWS_ACCOUNT_ID"] = account_id
    result = subprocess.run(
        [str(WRAPPER), "/usr/bin/true"], capture_output=True, text=True, env=env
    )
    assert result.returncode == 1
    assert "EXPECTED_AWS_ACCOUNT_ID must be set to exactly 12 digits" in result.stderr


def test_wrapper_rejects_caller_account_mismatch(tmp_path: Path) -> None:
    fake_aws(tmp_path, account_id="111111111111")
    env = environment(tmp_path)
    env["EXPECTED_AWS_ACCOUNT_ID"] = "123456789012"
    result = subprocess.run(
        [str(WRAPPER), "/usr/bin/true"], capture_output=True, text=True, env=env
    )
    assert result.returncode == 1
    assert "refusing AWS operation in account 111111111111" in result.stderr


def test_wrapper_exports_credentials_profile_and_fixed_region(tmp_path: Path) -> None:
    fake_aws(tmp_path)
    env = environment(tmp_path)
    env.update(
        {
            "EXPECTED_AWS_ACCOUNT_ID": "123456789012",
            "AWS_PROFILE": "default",
            "AWS_REGION": "eu-west-1",
            "AWS_DEFAULT_REGION": "eu-west-1",
        }
    )
    result = subprocess.run([str(WRAPPER), "/usr/bin/env"], capture_output=True, text=True, env=env)
    assert result.returncode == 0
    child_environment = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    assert child_environment["AWS_PROFILE"] == "default"
    assert child_environment["AWS_SDK_LOAD_CONFIG"] == "1"
    assert child_environment["AWS_REGION"] == "us-west-2"
    assert child_environment["AWS_DEFAULT_REGION"] == "us-west-2"
    assert child_environment["AWS_ACCESS_KEY_ID"] == "fake-access"
