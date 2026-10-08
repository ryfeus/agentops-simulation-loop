"""SSM-only controller for the disposable Phase 8A GPU compatibility smoke."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY_ROOT / ".rl-smoke"
DEFAULT_OUTPUTS = DEFAULT_ROOT / "terraform-outputs.json"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Compact JSON/log evidence is intentionally kept well below SSM's command-output
# boundary.  Adapter archives are transferred separately and may use the larger,
# still bounded size below; callers record the actual fallback decision.
EVIDENCE_CHUNK_BYTES = 8 * 1024
ARTIFACT_CHUNK_BYTES = 16 * 1024
# Backwards-compatible name used by the upload path and older callers.
CHUNK_BYTES = EVIDENCE_CHUNK_BYTES
REMOTE_ROOT = "/tmp/agentops-rl-smoke"
REDACTION_PATTERNS = (
    re.compile(r"(?i)(aws_(?:access_key_id|secret_access_key|session_token)\s*[=:]\s*)\S+"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
)


class SmokeError(RuntimeError):
    """A clear operator-facing error which must not be reported as a passing smoke."""


@dataclass(frozen=True)
class SsmResult:
    status: str
    stdout: str
    stderr: str
    response_code: int

    @property
    def succeeded(self) -> bool:
        return self.status == "Success" and self.response_code == 0


def validate_run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError("RL_SMOKE_RUN_ID must be 1..128 safe filename characters")
    return value


def terraform_var_args(environ: Mapping[str, str] | None = None) -> list[str]:
    env = environ or os.environ
    values = (
        ("RL_SMOKE_INSTANCE_TYPE", "instance_type"),
        ("RL_SMOKE_AMI_ID", "ami_id"),
        ("RL_SMOKE_AVAILABILITY_ZONE", "availability_zone"),
    )
    args: list[str] = []
    for env_name, variable in values:
        value = env.get(env_name, "").strip()
        if not value:
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", value):
            raise ValueError(f"{env_name} must use only letters, digits, dots, and hyphens")
        if env_name == "RL_SMOKE_AMI_ID" and not re.fullmatch(r"ami-[0-9a-f]+", value):
            raise ValueError("RL_SMOKE_AMI_ID must be an EC2 AMI ID")
        args.append(f"-var={variable}={value}")
    return args


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def terraform_output_value(outputs: Mapping[str, Any], name: str) -> str:
    entry = outputs.get(name)
    value = entry.get("value") if isinstance(entry, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Terraform output {name!r} is missing or invalid")
    return value


def parse_ssm_invocation(value: Mapping[str, Any]) -> SsmResult:
    return SsmResult(
        status=str(value.get("Status", "Unknown")),
        stdout=str(value.get("StandardOutputContent", "")),
        stderr=str(value.get("StandardErrorContent", "")),
        response_code=int(value.get("ResponseCode", -1)),
    )


def classify_failure(text: str, stage: str = "") -> str:
    normalized = f"{stage}\n{text}".lower()
    checks = (
        ("AWS_CAPACITY", ("insufficientinstancecapacity", "quota", "vcpus")),
        ("SSM_UNAVAILABLE", ("ssm", "managed instance", "connection lost")),
        ("GPU_DRIVER", ("nvidia-smi", "driver/library version mismatch", "no devices were found")),
        ("CUDA", ("torch.cuda.is_available() is false", "cuda error", "cuda is not available")),
        ("DEPENDENCY_RESOLUTION", ("uv sync", "resolution", "no solution found")),
        (
            "MODEL_DOWNLOAD",
            ("from_pretrained", "huggingface", "model download", "repository not found"),
        ),
        ("VLLM_OOM", ("out of memory", "cuda oom")),
        ("VLLM_INIT", ("vllm", "colocate")),
        ("VLLM_GENERATION", ("generation", "generate")),
        ("TRL_CONTROL", ("grpotrainer", "trl_control")),
        ("GRPO_BACKWARD", ("backward", "optimizer")),
        ("CHECKPOINT_SAVE", ("save_model", "adapter")),
        ("CHECKPOINT_RELOAD", ("checkpoint_reload", "reload_checkpoint", "adapter_config")),
    )
    for name, markers in checks:
        if any(marker in normalized for marker in markers):
            return name
    return "UNKNOWN"


def redact_text(value: str) -> str:
    redacted = value
    for pattern in REDACTION_PATTERNS:
        if pattern.pattern.startswith("(?i)"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redacted_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        try:
            original = source.read_bytes()
            text = original.decode("utf-8")
            value = json.loads(text)
            redacted_value = _redact_value(value)
            # Some JSON evidence is itself a byte-exact lineage input. Preserve
            # its original representation when it contains nothing to redact.
            if redacted_value == value and redact_text(text) == text:
                if source != destination:
                    destination.write_bytes(original)
                return
            destination.write_text(json.dumps(redacted_value, indent=2, sort_keys=True) + "\n")
            return
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    destination.write_text(redact_text(source.read_text(errors="replace")))


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in value.items()}
    return value


def package_payload() -> tuple[bytes, str]:
    files = (
        REPOSITORY_ROOT / "rl" / "pyproject.toml",
        REPOSITORY_ROOT / "rl" / "uv.lock",
        REPOSITORY_ROOT / "rl" / "phase8a" / "train_smoke.py",
        REPOSITORY_ROOT / "rl" / "phase8a" / "reload_checkpoint.py",
        REPOSITORY_ROOT / "rl" / "phase8a" / "remote_run.sh",
    )
    missing = [str(path.relative_to(REPOSITORY_ROOT)) for path in files if not path.is_file()]
    if missing:
        raise SmokeError(f"isolated RL payload is incomplete: {', '.join(missing)}")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(REPOSITORY_ROOT / "rl")))
    payload = stream.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def payload_chunks(payload: bytes, chunk_size: int = CHUNK_BYTES) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return [
        base64.b64encode(payload[index : index + chunk_size]).decode()
        for index in range(0, len(payload), chunk_size)
    ]


def _aws_clients() -> tuple[Any, Any, Any]:
    from scripts.aws_context import verified_session

    session = verified_session()
    return session.client("sts"), session.client("ec2"), session.client("ssm")


def require_account(sts_client: Any) -> None:
    from scripts.aws_context import require_account as verify

    try:
        verify(sts_client)
    except (ValueError, RuntimeError) as exc:
        raise SmokeError(str(exc)) from exc


def send_shell(
    ssm_client: Any, instance_id: str, commands: list[str], *, timeout: int = 3600
) -> str:
    response = ssm_client.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands},
        TimeoutSeconds=timeout,
    )
    command = response.get("Command", {})
    command_id = command.get("CommandId")
    if not isinstance(command_id, str) or not command_id:
        raise SmokeError("SSM SendCommand returned no command ID")
    return command_id


def wait_command(
    ssm_client: Any, instance_id: str, command_id: str, *, timeout: int = 7200
) -> SsmResult:
    deadline = time.monotonic() + timeout
    last: Mapping[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            current = ssm_client.get_command_invocation(
                CommandId=command_id, InstanceId=instance_id
            )
        except ssm_client.exceptions.InvocationDoesNotExist:
            time.sleep(2)
            continue
        last = current
        status = str(current.get("Status", ""))
        if status in {"Success", "Cancelled", "TimedOut", "Failed", "Cancelling"}:
            return parse_ssm_invocation(current)
        time.sleep(5)
    detail = parse_ssm_invocation(last or {})
    raise SmokeError(f"SSM command {command_id} timed out; last status was {detail.status}")


def run_shell(
    ssm_client: Any, instance_id: str, commands: list[str], *, timeout: int = 7200
) -> SsmResult:
    command_id = send_shell(ssm_client, instance_id, commands, timeout=timeout)
    return wait_command(ssm_client, instance_id, command_id, timeout=timeout)


def wait_for_instance(
    ec2_client: Any, instance_id: str, *, timeout: int = 900
) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        instances = [
            item
            for reservation in response.get("Reservations", [])
            for item in reservation.get("Instances", [])
        ]
        if not instances:
            raise SmokeError(f"EC2 instance {instance_id} was not found")
        instance = instances[0]
        state = instance.get("State", {}).get("Name")
        if state == "running":
            return instance
        if state in {"shutting-down", "terminated", "stopping", "stopped"}:
            raise SmokeError(f"EC2 instance is {state}")
        time.sleep(5)
    raise SmokeError(
        "EC2 did not reach running state; check capacity, quota, and availability zone"
    )


def wait_for_ssm(ssm_client: Any, instance_id: str, *, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = ssm_client.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        )
        information = response.get("InstanceInformationList", [])
        if information and information[0].get("PingStatus") == "Online":
            return
        time.sleep(5)
    raise SmokeError("SSM_UNAVAILABLE: instance did not become online")


def upload_payload(
    ssm_client: Any,
    instance_id: str,
    run_id: str,
    payload: bytes,
    digest: str,
    *,
    parallel_chunks: bool = False,
) -> str:
    remote_dir = f"{REMOTE_ROOT}/{validate_run_id(run_id)}"
    payload_path = f"{remote_dir}/payload.tar.gz"
    quoted_path = shlex.quote(payload_path)
    initialize_file = (
        f"truncate -s {len(payload)} {quoted_path}" if parallel_chunks else f": > {quoted_path}"
    )
    initial = run_shell(
        ssm_client,
        instance_id,
        [
            f"install -d -m 700 {shlex.quote(remote_dir)}",
            initialize_file,
        ],
    )
    if not initial.succeeded:
        raise SmokeError(f"SSM bootstrap failed: {initial.stderr}")
    chunks = payload_chunks(payload)
    if parallel_chunks:
        # Non-overlapping, pre-sized offsets let bounded SSM commands run in
        # parallel without relying on S3, SSH, or remote credentials.
        for start in range(0, len(chunks), 8):
            batch = list(enumerate(chunks[start : start + 8], start))

            def write_chunk(item: tuple[int, str]) -> tuple[int, SsmResult]:
                index, chunk = item
                command = (
                    f"printf '%s' '{chunk}' | base64 -d | "
                    f"dd of={quoted_path} bs=1 seek={index * CHUNK_BYTES} "
                    "conv=notrunc status=none"
                )
                return index, run_shell(ssm_client, instance_id, [command], timeout=600)

            with ThreadPoolExecutor(max_workers=len(batch)) as workers:
                results = list(workers.map(write_chunk, batch))
            for index, result in results:
                if not result.succeeded:
                    raise SmokeError(f"SSM payload chunk {index} failed: {result.stderr}")
    else:
        for index, chunk in enumerate(chunks):
            result = run_shell(
                ssm_client,
                instance_id,
                [f"printf '%s' '{chunk}' | base64 -d >> {quoted_path}"],
                timeout=600,
            )
            if not result.succeeded:
                raise SmokeError(f"SSM payload chunk {index} failed: {result.stderr}")
    verify = run_shell(
        ssm_client,
        instance_id,
        [
            f"printf '%s  %s\\n' '{digest}' {quoted_path} | sha256sum -c -",
            f"tar -xzf {quoted_path} -C {shlex.quote(remote_dir)}",
        ],
    )
    if not verify.succeeded:
        raise SmokeError(f"SSM payload verification failed: {verify.stderr}")
    return remote_dir


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably publish small controller state without exposing partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def remote_file_metadata(ssm_client: Any, instance_id: str, path: str) -> tuple[int, str]:
    metadata = run_shell(
        ssm_client, instance_id, [f"wc -c < {path}; sha256sum {path} | cut -d' ' -f1"]
    )
    if not metadata.succeeded:
        raise SmokeError(f"could not read remote evidence: {metadata.stderr}")
    lines = metadata.stdout.splitlines()
    if len(lines) < 2:
        raise SmokeError("remote evidence metadata was incomplete")
    try:
        size = int(lines[-2].strip())
    except ValueError as exc:
        raise SmokeError("remote evidence size was invalid") from exc
    digest = lines[-1].strip()
    if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SmokeError("remote evidence metadata was invalid")
    return size, digest


def _read_remote_file(
    ssm_client: Any,
    instance_id: str,
    path: str,
    *,
    chunk_size: int = EVIDENCE_CHUNK_BYTES,
) -> bytes:
    if chunk_size < 1 or chunk_size > ARTIFACT_CHUNK_BYTES:
        raise ValueError("remote evidence chunk size is outside the bounded SSM range")
    size, digest = remote_file_metadata(ssm_client, instance_id, path)
    data = bytearray()
    for offset in range(0, size, chunk_size):
        count = min(chunk_size, size - offset)
        result = run_shell(
            ssm_client,
            instance_id,
            [f"dd if={path} bs=1 skip={offset} count={count} status=none | base64 -w0"],
            timeout=600,
        )
        if not result.succeeded:
            raise SmokeError(f"remote evidence chunk at {offset} failed: {result.stderr}")
        try:
            chunk = base64.b64decode(result.stdout.strip(), validate=True)
        except ValueError as exc:
            raise SmokeError(f"remote evidence chunk at {offset} was not valid base64") from exc
        if len(chunk) != count:
            raise SmokeError(f"remote evidence chunk at {offset} had an unexpected length")
        data.extend(chunk)
    payload = bytes(data)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise SmokeError("remote evidence checksum mismatch")
    return payload


def artifact_manifest_files(value: Mapping[str, Any], prefix: str) -> dict[str, dict[str, Any]]:
    """Return a validated relative inventory below one artifact-manifest prefix."""

    root = PurePosixPath(prefix.rstrip("/"))
    if not root.parts or root.is_absolute() or any(part in {"", ".", ".."} for part in root.parts):
        raise SmokeError("artifact manifest prefix is unsafe")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise SmokeError("artifact manifest has no artifacts list")
    result: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise SmokeError("artifact manifest entry is invalid")
        raw_path, size, digest = item.get("path"), item.get("size"), item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(size, int) or size < 0:
            raise SmokeError("artifact manifest entry has invalid path or size")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SmokeError("artifact manifest entry has invalid SHA-256")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise SmokeError("artifact manifest contains an unsafe path")
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not relative.parts:
            raise SmokeError("artifact manifest entry identifies a directory")
        name = relative.as_posix()
        if name in result:
            raise SmokeError("artifact manifest contains duplicate paths")
        result[name] = {"size": size, "sha256": digest}
    if not result:
        raise SmokeError(f"artifact manifest has no files below {prefix}")
    return result


def _extract_verified_archive(
    archive_bytes: bytes, destination: Path, expected_files: Mapping[str, Mapping[str, Any]]
) -> None:
    """Safely extract one exact regular-file archive and verify every member."""

    if destination.exists():
        raise SmokeError(f"artifact destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = set(expected_files)
    temporary = Path(tempfile.mkdtemp(prefix="rl-artifact-", dir=destination.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = archive.getmembers()
            names = {member.name for member in members}
            if names != expected:
                raise SmokeError("artifact archive members differ from the manifest inventory")
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    not member.isreg()
                    or path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or member.name not in expected
                ):
                    raise SmokeError("artifact archive contains an unsafe member")
                metadata = expected_files[member.name]
                if member.size != int(metadata["size"]):
                    raise SmokeError("artifact archive member size differs from the manifest")
                source = archive.extractfile(member)
                if source is None:
                    raise SmokeError("artifact archive member could not be read")
                target = temporary.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if digest != metadata["sha256"]:
                    raise SmokeError("artifact archive member checksum differs from the manifest")
        extracted = {
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file()
        }
        if extracted != expected:
            raise SmokeError("artifact extraction produced an unexpected file inventory")
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def download_remote_directory(
    ssm_client: Any,
    instance_id: str,
    remote_directory: str,
    destination: Path,
    expected_files: Mapping[str, Mapping[str, Any]],
    *,
    journal_path: Path | None = None,
) -> dict[str, Any]:
    """Fetch a manifest-pinned remote directory as one safe, resumable archive."""

    if not expected_files:
        raise SmokeError("artifact download requires a non-empty expected inventory")
    names = sorted(expected_files)
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise SmokeError("artifact download has an unsafe member name")
    archive = f"{remote_directory.rstrip('/')}/.rl-artifacts.tar.gz"
    members = " ".join(shlex.quote(name) for name in names)
    created = run_shell(
        ssm_client,
        instance_id,
        [
            "cd "
            f"{shlex.quote(remote_directory)} && tar -czf {shlex.quote(archive)} "
            f"--no-recursion {members}",
            f"wc -c < {shlex.quote(archive)}; sha256sum {shlex.quote(archive)} | cut -d' ' -f1",
        ],
        timeout=3600,
    )
    if not created.succeeded:
        raise SmokeError(f"could not archive remote artifacts: {created.stderr}")
    lines = created.stdout.splitlines()
    if len(lines) < 2:
        raise SmokeError("remote artifact archive metadata was incomplete")
    try:
        archive_size = int(lines[-2].strip())
    except ValueError as exc:
        raise SmokeError("remote artifact archive size was invalid") from exc
    archive_sha256 = lines[-1].strip()
    if archive_size < 1 or not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        raise SmokeError("remote artifact archive checksum was invalid")
    if journal_path is None:
        archive_bytes = _read_remote_file(
            ssm_client, instance_id, archive, chunk_size=ARTIFACT_CHUNK_BYTES
        )
        transfer = {
            "chunk_size": ARTIFACT_CHUNK_BYTES,
            "resumed": False,
            "fallback_to_evidence_chunk_size": False,
        }
    else:
        archive_bytes, transfer = _download_remote_archive_resumable(
            ssm_client,
            instance_id,
            archive,
            archive_size=archive_size,
            archive_sha256=archive_sha256,
            journal_path=journal_path,
        )
    if (
        len(archive_bytes) != archive_size
        or hashlib.sha256(archive_bytes).hexdigest() != archive_sha256
    ):
        raise SmokeError("remote artifact archive checksum mismatch")
    _extract_verified_archive(archive_bytes, destination, expected_files)
    return {
        "archive_size": archive_size,
        "archive_sha256": archive_sha256,
        "file_count": len(expected_files),
        **transfer,
    }


def _download_remote_archive_resumable(
    ssm_client: Any,
    instance_id: str,
    archive: str,
    *,
    archive_size: int,
    archive_sha256: str,
    journal_path: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Resume a verified archive download at exact SSM chunk boundaries.

    The final archive digest remains authoritative.  The journal only allows a
    later invocation to reuse already decoded bounded chunks for the same
    remote archive identity; changed remote metadata discards that partial data.
    """

    part_path = journal_path.with_name(f".{journal_path.name}.{archive_sha256}.part")
    previous: Mapping[str, Any] = {}
    try:
        previous = load_json_object(journal_path, "artifact transfer journal")
    except ValueError:
        previous = {}
    completed: dict[int, str] = {}
    chunk_size = ARTIFACT_CHUNK_BYTES
    resumed = False
    if (
        previous.get("archive_sha256") == archive_sha256
        and previous.get("archive_size") == archive_size
        and previous.get("chunk_size") in {EVIDENCE_CHUNK_BYTES, ARTIFACT_CHUNK_BYTES}
        and part_path.is_file()
    ):
        chunk_size = int(previous["chunk_size"])
        raw_completed = previous.get("completed_chunks", {})
        if isinstance(raw_completed, Mapping):
            for raw_offset, digest in raw_completed.items():
                try:
                    offset = int(raw_offset)
                except (TypeError, ValueError):
                    continue
                if (
                    0 <= offset < archive_size
                    and isinstance(digest, str)
                    and re.fullmatch(r"[0-9a-f]{64}", digest)
                ):
                    completed[offset] = digest
            resumed = bool(completed)
    else:
        part_path.unlink(missing_ok=True)

    def write_journal(*, fallback: bool = False, complete: bool = False) -> None:
        _atomic_json(
            journal_path,
            {
                "schema_version": "1",
                "archive_path": archive,
                "archive_size": archive_size,
                "archive_sha256": archive_sha256,
                "chunk_size": chunk_size,
                "completed_chunks": {str(key): completed[key] for key in sorted(completed)},
                "fallback_to_evidence_chunk_size": fallback,
                "complete": complete,
            },
        )

    def fetch_with(size: int) -> None:
        nonlocal chunk_size
        chunk_size = size
        if not part_path.exists():
            with part_path.open("wb") as stream:
                stream.truncate(archive_size)
        pending = [
            (offset, min(chunk_size, archive_size - offset))
            for offset in range(0, archive_size, chunk_size)
            if offset not in completed
        ]
        # SSM invocation latency dominates a 16 KiB stdout chunk.  A bounded
        # batch preserves the same stdout limit while avoiding one five-second
        # polling interval per chunk.
        for start in range(0, len(pending), 8):
            batch = pending[start : start + 8]

            def read_chunk(item: tuple[int, int]) -> tuple[int, int, SsmResult]:
                offset, count = item
                result = run_shell(
                    ssm_client,
                    instance_id,
                    [f"dd if={archive} bs=1 skip={offset} count={count} status=none | base64 -w0"],
                    timeout=600,
                )
                return offset, count, result

            with ThreadPoolExecutor(max_workers=len(batch)) as workers:
                results = list(workers.map(read_chunk, batch))
            failure: SmokeError | None = None
            for offset, count, result in results:
                if not result.succeeded:
                    failure = SmokeError(
                        f"remote artifact chunk at {offset} failed: {result.stderr}"
                    )
                    continue
                try:
                    chunk = base64.b64decode(result.stdout.strip(), validate=True)
                except ValueError as exc:
                    failure = SmokeError(f"remote artifact chunk at {offset} was not valid base64")
                    failure.__cause__ = exc
                    continue
                if len(chunk) != count:
                    failure = SmokeError(
                        f"remote artifact chunk at {offset} had an unexpected length"
                    )
                    continue
                with part_path.open("r+b") as stream:
                    stream.seek(offset)
                    stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                completed[offset] = hashlib.sha256(chunk).hexdigest()
                write_journal()
            if failure is not None:
                raise failure

    fallback = False
    try:
        fetch_with(chunk_size)
    except SmokeError:
        if chunk_size != ARTIFACT_CHUNK_BYTES:
            raise
        # A standard-output transport limit is recoverable without touching the
        # source adapter.  Restart on the smaller evidence-safe boundary.
        completed.clear()
        part_path.unlink(missing_ok=True)
        fallback = True
        fetch_with(EVIDENCE_CHUNK_BYTES)
    archive_bytes = part_path.read_bytes()
    if (
        len(archive_bytes) != archive_size
        or hashlib.sha256(archive_bytes).hexdigest() != archive_sha256
    ):
        raise SmokeError("resumed remote artifact archive checksum mismatch")
    write_journal(fallback=fallback, complete=True)
    part_path.unlink(missing_ok=True)
    return archive_bytes, {
        "chunk_size": chunk_size,
        "resumed": resumed,
        "fallback_to_evidence_chunk_size": fallback,
    }


def collect_evidence(ssm_client: Any, instance_id: str, run_id: str, destination: Path) -> None:
    remote_dir = f"{REMOTE_ROOT}/{run_id}/run"
    archive = f"{remote_dir}/evidence.tar.gz"
    compact = (
        "cd " + remote_dir + " && "
        'for file in *.log; do [ -f "$file" ] && tail -c 65536 "$file" '
        '> "$file.compact" && mv "$file.compact" "$file"; done; '
        "find . -maxdepth 1 -type f \\( -name '*.json' -o -name '*.txt' "
        "-o -name '*.csv' -o -name '*.log' \\) "
        "-printf '%f\\0' | tar --null --files-from=- -czf " + archive
    )
    result = run_shell(ssm_client, instance_id, [compact])
    if not result.succeeded:
        raise SmokeError(f"failed to compact remote evidence: {result.stderr}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(
        fileobj=io.BytesIO(_read_remote_file(ssm_client, instance_id, archive)), mode="r:gz"
    ) as bundle:
        for member in bundle.getmembers():
            if member.isdir() or "/" in member.name or member.name.startswith("."):
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            target = destination / member.name
            target.write_bytes(source.read())
            redacted_copy(target, target)

    # The compact archive is the normal path. Fetch any required result that
    # was absent from it separately, using the same bounded/checksummed SSM
    # transfer. This keeps a successful runtime gate from losing its compact
    # record because of an archive edge case.
    for name in (
        "run-status.json",
        "versions.json",
        "training-control.json",
        "training-vllm.json",
        "checkpoint-reload.json",
    ):
        target = destination / name
        if target.is_file():
            continue
        try:
            target.write_bytes(_read_remote_file(ssm_client, instance_id, f"{remote_dir}/{name}"))
        except SmokeError:
            continue
        redacted_copy(target, target)


def git_provenance() -> tuple[str, bool]:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return revision, not bool(status.strip())


def structured_training_failure(destination: Path) -> tuple[str, str, str] | None:
    """Return the first failed trainer result's mode, phase, and error text."""

    for name in ("training-control.json", "training-vllm.json"):
        path = destination / name
        if not path.is_file():
            continue
        result = load_json_object(path, name)
        if result.get("status") != "FAIL":
            continue
        mode = str(result.get("mode", "control"))
        phase = str(result.get("failure_phase", ""))
        error = str(result.get("error", ""))
        return mode, phase, error
    return None


def structured_failure_class(mode: str, phase: str, error: str) -> str:
    """Map the runner's stable phases before falling back to log text heuristics."""

    specific = classify_failure(error)
    if specific in {"GPU_DRIVER", "CUDA", "DEPENDENCY_RESOLUTION", "MODEL_DOWNLOAD", "VLLM_OOM"}:
        return specific
    if phase == "CHECKPOINT_SAVE":
        return "CHECKPOINT_SAVE"
    if phase == "TRAIN":
        return "VLLM_GENERATION" if mode == "vllm" else "GRPO_BACKWARD"
    if phase in {"CONFIG", "TRAINER_INIT"}:
        return "VLLM_INIT" if mode == "vllm" else "TRL_CONTROL"
    return specific


def write_summary(
    destination: Path, outputs: Mapping[str, Any], *, remote_result: SsmResult | None
) -> dict[str, Any]:
    run_status = (
        load_json_object(destination / "run-status.json", "remote run status")
        if (destination / "run-status.json").is_file()
        else {}
    )
    versions = (
        load_json_object(destination / "versions.json", "versions")
        if (destination / "versions.json").is_file()
        else {}
    )
    revision, clean = git_provenance()
    control = str(run_status.get("control", "NOT_RUN"))
    vllm = str(run_status.get("vllm_colocate", "NOT_RUN"))
    reload = str(run_status.get("checkpoint_reload", "NOT_RUN"))
    reload_evidence = destination / "checkpoint-reload.json"
    reload_recorded = False
    if reload_evidence.is_file():
        try:
            reload_recorded = (
                load_json_object(reload_evidence, "checkpoint reload").get("status") == "PASS"
            )
        except ValueError:
            reload_recorded = False
    passed = (
        control == vllm == reload == "PASS"
        and (remote_result is None or remote_result.succeeded)
        and reload_recorded
    )
    text = "\n".join((remote_result.stdout, remote_result.stderr)) if remote_result else ""
    value: dict[str, Any] = {
        "schema_version": "1",
        "phase": "8A",
        "status": "PASS" if passed else "FAIL",
        "git_revision": revision,
        "worktree_clean": clean,
        "canonical_acceptance": passed and clean,
        "instance_type": terraform_output_value(outputs, "instance_type"),
        "ami_id": terraform_output_value(outputs, "ami_id"),
        "availability_zone": terraform_output_value(outputs, "availability_zone"),
        "gpu": versions.get("gpu"),
        "model_id": "Qwen/Qwen3-0.6B",
        "trl_version": versions.get("trl"),
        "vllm_version": versions.get("vllm"),
        "control": control,
        "vllm_colocate": vllm,
        "checkpoint_reload": reload,
        "checkpoint_reload_evidence": "PASS" if reload_recorded else "MISSING_OR_INVALID",
    }
    if not passed:
        structured = structured_training_failure(destination)
        if structured is not None:
            mode, phase, error = structured
            value["failure_mode"] = mode
            value["failure_phase"] = phase
            value["failure_class"] = structured_failure_class(mode, phase, error)
        elif reload == "PASS" and not reload_recorded:
            value["failure_class"] = "CHECKPOINT_RELOAD"
        else:
            value["failure_class"] = classify_failure(text, str(run_status.get("stage", "")))
    (destination / "summary.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def latest_run_path(root: Path = DEFAULT_ROOT) -> Path:
    return root / "latest-run.json"


def resolve_run_id(requested: str | None, root: Path = DEFAULT_ROOT) -> str:
    if requested:
        return validate_run_id(requested)
    pointer = latest_run_path(root)
    if pointer.is_file():
        return validate_run_id(str(load_json_object(pointer, "latest run").get("run_id", "")))
    return validate_run_id("phase8a-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S"))


def run_smoke(outputs_path: Path = DEFAULT_OUTPUTS, requested_run_id: str | None = None) -> Path:
    outputs = load_json_object(outputs_path, "RL smoke Terraform outputs")
    instance_id = terraform_output_value(outputs, "instance_id")
    run_id = resolve_run_id(requested_run_id)
    destination = DEFAULT_ROOT / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    latest_run_path().write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    sts, ec2, ssm = _aws_clients()
    require_account(sts)
    wait_for_instance(ec2, instance_id)
    wait_for_ssm(ssm, instance_id)
    payload, digest = package_payload()
    remote_dir = upload_payload(ssm, instance_id, run_id, payload, digest)
    result = run_shell(
        ssm,
        instance_id,
        [f"bash {remote_dir}/phase8a/remote_run.sh --run-root {remote_dir}/run"],
        timeout=7200,
    )
    try:
        collect_evidence(ssm, instance_id, run_id, destination)
    finally:
        summary = write_summary(destination, outputs, remote_result=result)
    if not result.succeeded or summary["status"] != "PASS":
        raise SmokeError(f"Phase 8A run failed; evidence: {destination}")
    return destination


def status(outputs_path: Path = DEFAULT_OUTPUTS) -> dict[str, Any]:
    outputs = load_json_object(outputs_path, "RL smoke Terraform outputs")
    instance_id = terraform_output_value(outputs, "instance_id")
    sts, ec2, ssm = _aws_clients()
    require_account(sts)
    instance = wait_for_instance(ec2, instance_id, timeout=30)
    info = ssm.describe_instance_information(
        Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
    )
    value = {
        "instance_id": instance_id,
        "ec2_state": instance.get("State", {}).get("Name"),
        "ssm_ping_status": (info.get("InstanceInformationList") or [{}])[0].get("PingStatus"),
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return value


def evidence(requested_run_id: str | None = None) -> Path:
    run_id = resolve_run_id(requested_run_id)
    path = DEFAULT_ROOT / "runs" / run_id / "summary.json"
    if not path.is_file():
        raise SmokeError(f"evidence is missing for run {run_id}; run aws-rl-smoke-run first")
    print(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    run = subparsers.choices["run"]
    run.add_argument("--run-id")
    evidence_parser = subparsers.add_parser("evidence")
    evidence_parser.add_argument("--run-id")
    subparsers.add_parser("terraform-args")
    args = parser.parse_args(argv)
    if args.command == "run":
        print(run_smoke(args.outputs, args.run_id))
    elif args.command == "status":
        status(args.outputs)
    elif args.command == "terraform-args":
        print(" ".join(terraform_var_args()))
    else:
        evidence(args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
