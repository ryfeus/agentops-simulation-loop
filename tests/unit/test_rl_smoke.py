from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path

import pytest

from scripts import rl_smoke


def outputs() -> dict[str, dict[str, str]]:
    return {
        "instance_id": {"value": "i-123"},
        "instance_type": {"value": "g6.2xlarge"},
        "ami_id": {"value": "ami-123"},
        "availability_zone": {"value": "us-west-2a"},
        "private_ip": {"value": "10.78.0.10"},
    }


def test_run_id_and_terraform_overrides_are_safe() -> None:
    assert rl_smoke.validate_run_id("phase8a-20260917_01") == "phase8a-20260917_01"
    with pytest.raises(ValueError, match="safe filename"):
        rl_smoke.validate_run_id("../../bad")
    assert rl_smoke.terraform_var_args(
        {
            "RL_SMOKE_INSTANCE_TYPE": "g6.2xlarge",
            "RL_SMOKE_AMI_ID": "ami-123abc",
            "RL_SMOKE_AVAILABILITY_ZONE": "us-west-2b",
        }
    ) == [
        "-var=instance_type=g6.2xlarge",
        "-var=ami_id=ami-123abc",
        "-var=availability_zone=us-west-2b",
    ]
    with pytest.raises(ValueError, match="AMI ID"):
        rl_smoke.terraform_var_args({"RL_SMOKE_AMI_ID": "not-an-ami"})
    with pytest.raises(ValueError, match="letters, digits"):
        rl_smoke.terraform_var_args({"RL_SMOKE_INSTANCE_TYPE": "g6.2xlarge; rm"})


def test_output_and_ssm_result_parsing_fail_closed() -> None:
    assert rl_smoke.terraform_output_value(outputs(), "instance_id") == "i-123"
    with pytest.raises(ValueError, match="missing or invalid"):
        rl_smoke.terraform_output_value({}, "instance_id")
    result = rl_smoke.parse_ssm_invocation(
        {"Status": "Success", "ResponseCode": 0, "StandardOutputContent": "ok"}
    )
    assert result.succeeded
    assert not rl_smoke.parse_ssm_invocation({"Status": "Failed", "ResponseCode": 1}).succeeded


def test_chunk_encoding_round_trips_and_evidence_is_redacted(tmp_path: Path) -> None:
    payload = b"phase8a" * 2000
    chunks = rl_smoke.payload_chunks(payload, chunk_size=257)
    assert b"".join(base64.b64decode(chunk) for chunk in chunks) == payload
    source = tmp_path / "input.txt"
    source.write_text("AWS_SECRET_ACCESS_KEY=super-secret\nhf_abcdefghijklmnopqrstuv\n")
    rl_smoke.redacted_copy(source, source)
    assert "super-secret" not in source.read_text()
    assert "[REDACTED]" in source.read_text()


def test_json_evidence_preserves_safe_lineage_bytes_and_redacts_secrets(tmp_path: Path) -> None:
    source = tmp_path / "task-set.json"
    exact_bytes = b'{ "training_tasks": [ { "task_id": "readonly-before-action" } ] }\n'
    source.write_bytes(exact_bytes)
    destination = tmp_path / "retained.json"
    rl_smoke.redacted_copy(source, destination)
    assert destination.read_bytes() == exact_bytes

    source.write_text('{"token":"hf_abcdefghijklmnopqrstuv"}\n')
    rl_smoke.redacted_copy(source, destination)
    assert "hf_abcdefghijklmnopqrstuv" not in destination.read_text()
    assert "[REDACTED]" in destination.read_text()


def test_parallel_payload_upload_uses_bounded_nonoverlapping_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(256)) * 289
    digest = hashlib.sha256(payload).hexdigest()
    calls: list[list[str]] = []

    def fake_run_shell(
        _client: object, _instance: str, commands: list[str], **_kwargs: object
    ) -> rl_smoke.SsmResult:
        calls.append(commands)
        return rl_smoke.SsmResult("Success", "", "", 0)

    monkeypatch.setattr(rl_smoke, "run_shell", fake_run_shell)
    remote = rl_smoke.upload_payload(
        object(), "i-123", "phase9b-test", payload, digest, parallel_chunks=True
    )
    assert remote.endswith("/phase9b-test")
    assert f"truncate -s {len(payload)}" in calls[0][1]
    chunks = rl_smoke.payload_chunks(payload)
    written: dict[int, str] = {}
    for commands in calls[1:-1]:
        match = re.search(r"seek=(\d+)", commands[0])
        assert match is not None
        offset = int(match.group(1))
        assert offset not in written
        assert "conv=notrunc" in commands[0]
        written[offset] = commands[0]
    assert set(written) == {index * rl_smoke.CHUNK_BYTES for index in range(len(chunks))}
    for index, chunk in enumerate(chunks):
        assert chunk in written[index * rl_smoke.CHUNK_BYTES]
    assert digest in calls[-1][0]
    assert "sha256sum -c" in calls[-1][0]


@pytest.mark.parametrize(
    ("text", "stage", "expected"),
    [
        ("InsufficientInstanceCapacity", "", "AWS_CAPACITY"),
        ("torch.cuda.is_available() is False", "", "CUDA"),
        ("CUDA out of memory", "VLLM_OOM", "VLLM_OOM"),
        ("anything", "CHECKPOINT_RELOAD", "CHECKPOINT_RELOAD"),
    ],
)
def test_failure_classification(text: str, stage: str, expected: str) -> None:
    assert rl_smoke.classify_failure(text, stage) == expected


def test_payload_is_locked_and_contains_no_repository_or_credentials() -> None:
    payload, digest = rl_smoke.package_payload()
    assert len(digest) == 64
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = archive.getnames()
        contents = b"".join(archive.extractfile(name).read() for name in names)
    assert "uv.lock" in names
    assert "phase8a/remote_run.sh" in names
    assert all(not name.startswith(".git") for name in names)
    assert b"AWS_ACCESS_KEY_ID" not in contents
    assert b"EXPECTED_AWS_ACCOUNT_ID" not in contents


def test_reload_script_invokes_its_cli_entry_point() -> None:
    source = Path("rl/phase8a/reload_checkpoint.py").read_text()
    assert 'if __name__ == "__main__":' in source
    assert "raise SystemExit(main())" in source


def test_resumable_artifact_transfer_reuses_verified_chunks_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        content = os.urandom(20000)
        info = tarfile.TarInfo("adapter_model.safetensors")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    archive_bytes = payload.getvalue()
    expected = {
        "adapter_model.safetensors": {
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    }
    calls: list[int] = []
    fail_after_first_small = {"value": True}

    def fake_run_shell(*_: object, **__: object) -> rl_smoke.SsmResult:
        command = _[2][0]
        if "tar -czf" in command:
            metadata = f"{len(archive_bytes)}\n{hashlib.sha256(archive_bytes).hexdigest()}\n"
            return rl_smoke.SsmResult("Success", metadata, "", 0)
        match = re.search(r"skip=(\d+) count=(\d+)", command)
        assert match
        offset, count = (int(value) for value in match.groups())
        calls.append(offset)
        if count > rl_smoke.EVIDENCE_CHUNK_BYTES:
            return rl_smoke.SsmResult("Failed", "", "stdout limit", 1)
        if offset and fail_after_first_small["value"]:
            return rl_smoke.SsmResult("Failed", "", "interrupted", 1)
        return rl_smoke.SsmResult(
            "Success", base64.b64encode(archive_bytes[offset : offset + count]).decode(), "", 0
        )

    monkeypatch.setattr(rl_smoke, "run_shell", fake_run_shell)
    journal = tmp_path / "adapter-transfer.journal.json"
    with pytest.raises(rl_smoke.SmokeError, match="remote artifact chunk"):
        rl_smoke.download_remote_directory(
            object(), "i-test", "/remote", tmp_path / "adapter", expected, journal_path=journal
        )
    saved = json.loads(journal.read_text())
    assert saved["chunk_size"] == rl_smoke.EVIDENCE_CHUNK_BYTES
    assert "0" in saved["completed_chunks"]

    fail_after_first_small["value"] = False
    result = rl_smoke.download_remote_directory(
        object(), "i-test", "/remote", tmp_path / "adapter", expected, journal_path=journal
    )
    assert result["resumed"] is True
    assert result["fallback_to_evidence_chunk_size"] is False
    assert (tmp_path / "adapter" / "adapter_model.safetensors").read_bytes() == content


def test_summary_requires_all_three_successes(tmp_path: Path) -> None:
    (tmp_path / "run-status.json").write_text(
        json.dumps({"control": "PASS", "vllm_colocate": "PASS", "checkpoint_reload": "PASS"})
    )
    (tmp_path / "versions.json").write_text(
        json.dumps({"gpu": "NVIDIA L4", "trl": "1.13.0", "vllm": "0.28.0"})
    )
    (tmp_path / "checkpoint-reload.json").write_text(json.dumps({"status": "PASS"}))
    summary = rl_smoke.write_summary(tmp_path, outputs(), remote_result=None)
    assert summary["status"] == "PASS"
    assert summary["checkpoint_reload_evidence"] == "PASS"
    assert json.loads((tmp_path / "summary.json").read_text())["gpu"] == "NVIDIA L4"


def test_summary_keeps_dirty_success_noncanonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rl_smoke, "git_provenance", lambda: ("a" * 40, False))
    (tmp_path / "run-status.json").write_text(
        json.dumps({"control": "PASS", "vllm_colocate": "PASS", "checkpoint_reload": "PASS"})
    )
    (tmp_path / "checkpoint-reload.json").write_text(json.dumps({"status": "PASS"}))
    summary = rl_smoke.write_summary(tmp_path, outputs(), remote_result=None)
    assert summary["status"] == "PASS"
    assert summary["canonical_acceptance"] is False


def test_summary_fails_closed_when_reload_record_is_missing(tmp_path: Path) -> None:
    (tmp_path / "run-status.json").write_text(
        json.dumps({"control": "PASS", "vllm_colocate": "PASS", "checkpoint_reload": "PASS"})
    )
    summary = rl_smoke.write_summary(tmp_path, outputs(), remote_result=None)
    assert summary["status"] == "FAIL"
    assert summary["checkpoint_reload_evidence"] == "MISSING_OR_INVALID"
    assert summary["failure_class"] == "CHECKPOINT_RELOAD"


def test_summary_prefers_structured_training_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rl_smoke, "git_provenance", lambda: ("a" * 40, True))
    (tmp_path / "run-status.json").write_text(
        json.dumps({"control": "PASS", "vllm_colocate": "FAIL", "checkpoint_reload": "NOT_RUN"})
    )
    (tmp_path / "training-vllm.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "mode": "vllm",
                "failure_phase": "TRAIN",
                "error": "unexpected generation failure",
            }
        )
    )
    summary = rl_smoke.write_summary(tmp_path, outputs(), remote_result=None)
    assert summary["failure_mode"] == "vllm"
    assert summary["failure_phase"] == "TRAIN"
    assert summary["failure_class"] == "VLLM_GENERATION"


def test_terraform_stack_is_disposable_ssm_only() -> None:
    root = Path("infra/terraform-rl-smoke")
    main = (root / "main.tf").read_text()
    variables = (root / "variables.tf").read_text()
    assert (
        "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
        in main
    )
    assert "AmazonSSMManagedInstanceCore" in main
    assert "ingress" not in main
    assert "delete_on_termination = true" in main
    assert "encrypted             = true" in main
    assert 'volume_type           = "gp3"' in main
    assert 'default = "g6.2xlarge"' in variables
