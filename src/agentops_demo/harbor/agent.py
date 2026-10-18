"""Harbor installed-agent adapter for the Phase 1 LangGraph runtime."""

from __future__ import annotations

import ast
import json
import shlex
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agentops_demo.agent.graph import validate_runtime_config
from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import is_sha256, sha256_file

REMOTE_CONFIG = "/installed-agent/agent-config.json"
REMOTE_INSTRUCTION = "/installed-agent/instruction.md"
REMOTE_AWS_CREDENTIALS = "/installed-agent/aws/credentials"
REMOTE_BEDROCK_TOKEN = "/installed-agent/aws/bedrock-token"
MCP_URL = "http://127.0.0.1:8000/mcp"


class LangGraphBillingAgent(BaseInstalledAgent):
    """Install and run the exact configured project wheel inside Harbor."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "langgraph-billing"

    def __init__(
        self,
        logs_dir: Path,
        *,
        agent_config_path: str,
        package_path: str,
        bedrock_credentials_path: str | None = None,
        bedrock_token_path: str | None = None,
        billing_policy_mode: str = "permissive",
        inference_temperature: float | None = None,
        inference_max_tokens: int | None = None,
        inference_total_attempts: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(logs_dir, **kwargs)
        if billing_policy_mode not in {"permissive", "enforced"}:
            raise ValueError("unsupported billing policy mode")
        self.billing_policy_mode = billing_policy_mode
        self.inference_environment = {}
        if inference_temperature is not None:
            temperature = float(inference_temperature)
            if not 0 <= temperature <= 1:
                raise ValueError("temperature must be between zero and one")
            self.inference_environment["BILLING_MODEL_TEMPERATURE"] = str(temperature)
        for key, value in (
            ("BILLING_MODEL_MAX_TOKENS", inference_max_tokens),
            ("BILLING_MODEL_TOTAL_ATTEMPTS", inference_total_attempts),
        ):
            if value is not None:
                if int(value) < 1:
                    raise ValueError("inference limits must be positive")
                self.inference_environment[key] = str(int(value))
        self.agent_config_path = Path(agent_config_path).resolve()
        self.package_path = Path(package_path).resolve()
        self.bedrock_credentials_path = (
            Path(bedrock_credentials_path).resolve() if bedrock_credentials_path else None
        )
        self.bedrock_token_path = Path(bedrock_token_path).resolve() if bedrock_token_path else None
        if self.bedrock_credentials_path and self.bedrock_token_path:
            raise ValueError("choose either Bedrock token or AWS credentials, not both")
        if self.bedrock_token_path and not self.bedrock_token_path.is_file():
            raise FileNotFoundError("Bedrock token file not found")
        if not self.agent_config_path.is_file():
            raise FileNotFoundError(f"candidate config not found: {self.agent_config_path}")
        if not self.package_path.is_file():
            raise FileNotFoundError(f"candidate package not found: {self.package_path}")
        if self.bedrock_credentials_path and not self.bedrock_credentials_path.is_file():
            raise FileNotFoundError("Bedrock credential file not found")
        wheel_parts = self.package_path.name.removesuffix(".whl").split("-")
        if self.package_path.suffix != ".whl" or len(wheel_parts) not in {5, 6}:
            raise ValueError("candidate package must have a valid wheel filename")
        _require_atif_wheel(self.package_path)
        if self.bedrock_token_path:
            with ZipFile(self.package_path) as wheel:
                try:
                    model_source = wheel.read("agentops_demo/agent/model.py").decode()
                except KeyError as exc:
                    raise ValueError("rebuild the wheel for Bedrock token-file support") from exc
            if "AWS_BEARER_TOKEN_BEDROCK_FILE" not in model_source:
                raise ValueError("rebuild the wheel for Bedrock token-file support")
        self.remote_wheel = f"/installed-agent/{self.package_path.name}"

        self.agent_config = AgentConfig.model_validate_json(self.agent_config_path.read_text())
        validate_runtime_config(self.agent_config)
        self.agent_config_fingerprint = self.agent_config.fingerprint()
        if not is_sha256(self.agent_config_fingerprint):
            raise ValueError("candidate configuration fingerprint is not lowercase SHA-256")
        self.package_sha256 = sha256_file(self.package_path)
        if not is_sha256(self.package_sha256):
            raise ValueError("candidate package digest is not lowercase SHA-256")
        expected_model = f"{self.agent_config.model.provider}/{self.agent_config.model.model_id}"
        if self.model_name != expected_model:
            raise ValueError(
                f"Harbor model {self.model_name!r} does not match AgentConfig {expected_model!r}"
            )
        self.provenance = {
            "agent_config_fingerprint": self.agent_config_fingerprint,
            "source_revision": self.agent_config.agent.source_revision,
            "package_sha256": self.package_sha256,
            "model": expected_model,
            "billing_policy_mode": self.billing_policy_mode,
            "inference_environment": self.inference_environment,
        }

    async def install(self, environment: BaseEnvironment) -> None:
        await environment.upload_file(self.package_path, self.remote_wheel)
        await environment.upload_file(self.agent_config_path, REMOTE_CONFIG)
        await self.exec_as_agent(
            environment,
            command=f"python -m pip install --no-deps {shlex.quote(self.remote_wheel)}",
        )
        await self.exec_as_agent(environment, command="mkdir -p /logs/agent")
        serialized_provenance = json.dumps(self.provenance, sort_keys=True)
        await self.exec_as_agent(
            environment,
            command=(
                f"printf '%s\\n' {shlex.quote(serialized_provenance)} >/logs/agent/provenance.json"
            ),
        )
        await self.exec_as_agent(environment, command=_start_mcp_command(self.billing_policy_mode))
        await self.exec_as_agent(environment, command=_readiness_command(), timeout_sec=20)
        if self.bedrock_credentials_path or self.bedrock_token_path:
            await self.exec_as_agent(
                environment,
                command="mkdir -p /installed-agent/aws && chmod 700 /installed-agent/aws",
            )
            local_path = self.bedrock_token_path or self.bedrock_credentials_path
            remote_path = (
                REMOTE_BEDROCK_TOKEN if self.bedrock_token_path else REMOTE_AWS_CREDENTIALS
            )
            await environment.upload_file(local_path, remote_path)
            await self.exec_as_agent(environment, command="chmod 600 " + remote_path)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="agentops-harbor-") as temporary:
            instruction_path = Path(temporary) / "instruction.md"
            instruction_path.write_text(instruction)
            await environment.upload_file(instruction_path, REMOTE_INSTRUCTION)

        command = shlex.join(
            [
                "python",
                "-m",
                "agentops_demo.cli.run_once",
                "--config",
                REMOTE_CONFIG,
                "--instruction-file",
                REMOTE_INSTRUCTION,
                "--mcp-url",
                MCP_URL,
                "--output",
                "/logs/agent/result.json",
                "--trajectory",
                "/logs/agent/trajectory.json",
            ]
        )
        # Keep the legacy agent diagnostic location while exposing the exact
        # trajectory as a Harbor task artifact for structural verification.
        command += " && cp /logs/agent/result.json /app/data/agent-run.json"
        environment_values = None
        if self.bedrock_credentials_path:
            environment_values = {
                "AWS_SHARED_CREDENTIALS_FILE": REMOTE_AWS_CREDENTIALS,
                "AWS_PROFILE": "default",
                "AWS_REGION": "us-west-2",
                "AWS_DEFAULT_REGION": "us-west-2",
            }
        elif self.bedrock_token_path:
            environment_values = {
                "AWS_BEARER_TOKEN_BEDROCK_FILE": REMOTE_BEDROCK_TOKEN,
                "AWS_REGION": "us-west-2",
                "AWS_DEFAULT_REGION": "us-west-2",
                "AWS_EC2_METADATA_DISABLED": "true",
            }
        if self.inference_environment:
            environment_values = {**(environment_values or {}), **self.inference_environment}
        try:
            await self.exec_as_agent(environment, command=command, env=environment_values)
        finally:
            if self.bedrock_credentials_path:
                await self.exec_as_agent(environment, command="rm -f " + REMOTE_AWS_CREDENTIALS)
            elif self.bedrock_token_path:
                await self.exec_as_agent(environment, command="rm -f " + REMOTE_BEDROCK_TOKEN)
        context.metadata = self.provenance.copy()


def _require_atif_wheel(path: Path) -> None:
    """Reject an older installed CLI before starting an incompatible trial."""

    try:
        with ZipFile(path) as wheel:
            wheel.getinfo("agentops_demo/harbor/trajectory.py")
            cli = ast.parse(wheel.read("agentops_demo/cli/run_once.py"))
    except (BadZipFile, KeyError, SyntaxError) as exc:
        raise ValueError(
            "candidate wheel lacks ATIF trajectory support; "
            "rebuild the wheel and restart demo setup"
        ) from exc
    if not any(
        isinstance(node, ast.Constant) and node.value == "--trajectory" for node in ast.walk(cli)
    ):
        raise ValueError(
            "candidate wheel CLI lacks --trajectory; rebuild the wheel and restart demo setup"
        )


def _start_mcp_command(policy_mode: str = "permissive") -> str:
    if policy_mode not in {"permissive", "enforced"}:
        raise ValueError("unsupported billing policy mode")
    return (
        "BILLING_DATABASE_PATH=/app/data/billing.db "
        f"BILLING_POLICY_MODE={policy_mode} "
        "BILLING_MCP_HOST=127.0.0.1 "
        "BILLING_MCP_PORT=8000 "
        "nohup python -m agentops_demo.mcp.billing_server "
        ">/logs/agent/mcp.log 2>&1 </dev/null & "
        "echo $! >/logs/agent/mcp.pid"
    )


def _readiness_command() -> str:
    probe = (
        "import sys,urllib.request;"
        "r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=1);"
        "sys.exit(0 if r.status==200 else 1)"
    )
    return (
        "for attempt in $(seq 1 150); do "
        f"python -c {shlex.quote(probe)} && exit 0; "
        "sleep 0.1; "
        "done; exit 1"
    )
