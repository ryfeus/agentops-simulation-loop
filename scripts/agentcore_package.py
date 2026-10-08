"""Build and validate the provenance-bound AgentCore image package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agentops_demo.contracts.agent_config import AgentConfig
from agentops_demo.harbor.provenance import is_sha256, sha256_file
from scripts.dsql_admin import terraform_outputs
from scripts.generate_harbor_configs import current_revision, require_clean_worktree

CONFIG_PATH = Path(".agentcore/config/agent-config.json")
PACKAGE_DIR = Path(".agentcore/package")
IMAGE_CHECK_DIR = Path(".agentcore/image-check")
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"
REQUIREMENTS_PATH = Path("deploy/agentcore/requirements.txt")
OBSERVABILITY_PACKAGES = {
    "aws-opentelemetry-distro",
    "openinference-instrumentation-langchain",
}


def load_config(path: Path = CONFIG_PATH) -> AgentConfig:
    try:
        return AgentConfig.model_validate_json(path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read AgentCore config {path}: {exc}") from exc


def verify_runtime_requirements() -> None:
    with tempfile.NamedTemporaryFile() as temporary:
        subprocess.run(
            (
                "uv",
                "export",
                "--frozen",
                "--no-dev",
                "--no-group",
                "harbor",
                "--extra",
                "bedrock",
                "--extra",
                "dsql",
                "--extra",
                "aws",
                "--no-emit-project",
                "--no-hashes",
                "--no-header",
                "--output-file",
                temporary.name,
            ),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if Path(temporary.name).read_bytes() != REQUIREMENTS_PATH.read_bytes():
            raise RuntimeError("AgentCore runtime requirements are stale")
    requirement_names = {
        line.split("==", 1)[0]
        for line in REQUIREMENTS_PATH.read_text().splitlines()
        if line and not line.startswith((" ", "#")) and "==" in line
    }
    missing = OBSERVABILITY_PACKAGES - requirement_names
    if missing:
        raise RuntimeError(f"AgentCore observability requirements are missing: {sorted(missing)}")
    langchain_instrumentors = {
        name for name in requirement_names if "instrumentation-langchain" in name
    }
    if langchain_instrumentors != {"openinference-instrumentation-langchain"}:
        raise RuntimeError(
            f"expected exactly the OpenInference LangChain instrumentor: {langchain_instrumentors}"
        )


def build_wheel(output_dir: Path) -> Path:
    """Build exactly one project wheel into the requested ignored directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for wheel in output_dir.glob("*.whl"):
        wheel.unlink()
    subprocess.run(("uv", "build", "--wheel", "--out-dir", str(output_dir)), check=True)
    wheels = list(output_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one AgentCore wheel, found {len(wheels)}")
    return wheels[0]


def docker_build_command(*, wheel: Path, tag: str, push: bool) -> list[str]:
    """Build the production ARM64 image, optionally exporting it to a registry."""

    command = [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/arm64",
        "--file",
        "deploy/agentcore/Dockerfile",
        "--build-arg",
        f"WHEEL_FILE={wheel}",
        "--build-arg",
        f"RUNTIME_REQUIREMENTS={REQUIREMENTS_PATH}",
        "--tag",
        tag,
    ]
    command.extend(["--push"] if push else ["--output=type=cacheonly"])
    command.append(".")
    return command


def build_image_check() -> Path:
    """Build the real image locally without config, Terraform, AWS, or a registry push."""

    verify_runtime_requirements()
    wheel = build_wheel(IMAGE_CHECK_DIR)
    subprocess.run(
        docker_build_command(wheel=wheel, tag="agentops-demo-agentcore:image-check", push=False),
        check=True,
    )
    return wheel


def build_and_push() -> Path:
    from scripts.aws_context import verified_session

    verified_session()
    require_clean_worktree()
    revision = current_revision()
    config = load_config()
    if config.agent.source_revision != revision:
        raise ValueError("AgentCore config revision does not match Git HEAD")
    verify_runtime_requirements()
    wheel = build_wheel(PACKAGE_DIR)

    outputs = terraform_outputs()
    repository_url = str(outputs["ecr_repository_url"])
    repository_name = str(outputs["ecr_repository_name"])
    registry = repository_url.split("/", 1)[0]
    password = subprocess.run(
        (
            "aws",
            "--profile",
            os.getenv("AWS_PROFILE", "default"),
            "ecr",
            "get-login-password",
            "--region",
            "us-west-2",
        ),
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(
        ("docker", "login", "--username", "AWS", "--password-stdin", registry),
        input=password,
        check=True,
    )
    tagged_uri = f"{repository_url}:{revision}"
    subprocess.run(docker_build_command(wheel=wheel, tag=tagged_uri, push=True), check=True)
    digest = subprocess.run(
        (
            "aws",
            "--profile",
            os.getenv("AWS_PROFILE", "default"),
            "ecr",
            "describe-images",
            "--region",
            "us-west-2",
            "--repository-name",
            repository_name,
            "--image-ids",
            f"imageTag={revision}",
            "--query",
            "imageDetails[0].imageDigest",
            "--output",
            "text",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not digest.startswith("sha256:") or not is_sha256(digest.removeprefix("sha256:")):
        raise RuntimeError(f"ECR returned an invalid image digest: {digest!r}")

    manifest = {
        "source_revision": revision,
        "wheel": {"filename": wheel.name, "sha256": sha256_file(wheel)},
        "agent_config": {
            "fingerprint": config.fingerprint(),
            "model": f"{config.model.provider}/{config.model.model_id}",
        },
        "container": {
            "repository": repository_url,
            "tag": revision,
            "digest": digest,
            "uri": f"{repository_url}@{digest}",
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    require_clean_worktree()
    return MANIFEST_PATH


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read AgentCore manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("AgentCore manifest must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    if args.build_only:
        wheel = build_image_check()
        print(f"Built AgentCore image from {wheel}")
        return 0
    path = build_and_push()
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
