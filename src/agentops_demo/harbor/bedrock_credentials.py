"""Ephemeral, non-serializable credentials for the opt-in Harbor Bedrock profile."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import Field, SecretStr

from agentops_demo.contracts._base import ContractModel


class TemporaryBedrockCredentials(ContractModel):
    access_key_id: SecretStr
    secret_access_key: SecretStr
    session_token: SecretStr
    expiration: datetime
    region: str = "us-west-2"
    role_arn: str = Field(min_length=1)

    def credentials_file(self) -> str:
        return (
            "[default]\n"
            f"aws_access_key_id={self.access_key_id.get_secret_value()}\n"
            f"aws_secret_access_key={self.secret_access_key.get_secret_value()}\n"
            f"aws_session_token={self.session_token.get_secret_value()}\n"
        )


def assume_bedrock_role(
    sts_client: Any, *, role_arn: str, duration_seconds: int
) -> TemporaryBedrockCredentials:
    if not role_arn.startswith("arn:") or not 900 <= duration_seconds <= 3600:
        raise ValueError("Bedrock role ARN or session duration is invalid")
    response = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName="agentops-harbor-bedrock",
        DurationSeconds=duration_seconds,
    )
    values = response.get("Credentials", {})
    expiration = values.get("Expiration")
    if not isinstance(expiration, datetime) or expiration <= datetime.now(UTC):
        raise RuntimeError("STS returned expired Bedrock credentials")
    return TemporaryBedrockCredentials(
        access_key_id=values["AccessKeyId"],
        secret_access_key=values["SecretAccessKey"],
        session_token=values["SessionToken"],
        expiration=expiration,
        role_arn=role_arn,
    )
