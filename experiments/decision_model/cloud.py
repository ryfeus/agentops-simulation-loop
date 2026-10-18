"""Small SSM/S3 transport for this bounded experiment; uses existing Terraform worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(os.environ.get("DECIDER_CLOUD_ROOT", ".artifacts/decision-model"))
REMOTE = os.environ.get("DECIDER_REMOTE_ROOT", "/opt/decision-model")


def bucket_name(account_id: str, instance_id: str) -> str:
    """Build a portable name only from validated operator identities."""
    import re

    from scripts.aws_context import expected_account_id

    account = expected_account_id({"EXPECTED_AWS_ACCOUNT_ID": account_id})
    if not re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id):
        raise ValueError("invalid worker instance ID")
    return f"agentops-decider-{account}-{instance_id[2:]}"


def clients():
    from botocore.config import Config

    from scripts.aws_context import verified_session

    session = verified_session()
    return session.client("ssm"), session.client(
        "s3",
        endpoint_url="https://s3.us-west-2.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def state():
    return json.loads((ROOT / "cloud.json").read_text())


def send(command: str, label: str, timeout: int = 3600):
    ssm, _ = clients()
    cloud = state()
    result = ssm.send_command(
        InstanceIds=[cloud["instance_id"]],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command], "executionTimeout": [str(timeout)]},
    )
    ident = result["Command"]["CommandId"]
    (ROOT / f"{label}-command.json").write_text(
        json.dumps(
            {"id": ident, "label": label, "started": datetime.now(UTC).isoformat()}, indent=2
        )
        + "\n"
    )
    print(json.dumps({"id": ident, "label": label}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init").add_argument("--outputs", type=Path, required=True)
    p = sub.add_parser("push")
    p.add_argument("--archive", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--timeout", type=int, default=3600)
    sub.add_parser("status").add_argument("--label", required=True)
    sub.add_parser("fetch")
    sub.add_parser("download")
    sub.add_parser("delete-bucket")
    args = parser.parse_args()
    ssm, s3 = clients()
    if args.command == "init":
        if (ROOT / "cloud.json").exists():
            raise ValueError("This run already has cloud resources; reuse its recorded identity")
        outputs = json.loads(args.outputs.read_text())
        instance = outputs["instance_id"]["value"]
        from scripts.aws_context import expected_account_id

        bucket = bucket_name(expected_account_id(), instance)
        s3.create_bucket(
            Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
        )
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
            },
        )
        (ROOT / "cloud.json").write_text(
            json.dumps(
                {
                    "instance_id": instance,
                    "bucket": bucket,
                    "created": datetime.now(UTC).isoformat(),
                    "remote": REMOTE,
                },
                indent=2,
            )
            + "\n"
        )
        print(instance)
    elif args.command == "push":
        cloud = state()
        key = "payload.tar.gz"
        s3.upload_file(str(args.archive), cloud["bucket"], key)
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": cloud["bucket"], "Key": key}, ExpiresIn=900
        )
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        send(
            f"set -eu; mkdir -p {REMOTE}; cd {REMOTE}; "
            f"curl -fsSL {shlex.quote(url)} -o /tmp/decision-payload.tar.gz; "
            f"echo '{digest}  /tmp/decision-payload.tar.gz' | sha256sum -c -; "
            "tar -xzf /tmp/decision-payload.tar.gz",
            "push",
        )
    elif args.command == "run":
        if not 1 <= args.timeout <= 14400:
            raise ValueError("command timeout exceeds experiment cap")
        send(args.script.read_text(), args.label, args.timeout)
    elif args.command == "status":
        ident = json.loads((ROOT / f"{args.label}-command.json").read_text())["id"]
        result = ssm.get_command_invocation(CommandId=ident, InstanceId=state()["instance_id"])
        clean = {
            k: result.get(k)
            for k in (
                "Status",
                "ResponseCode",
                "ExecutionStartDateTime",
                "ExecutionElapsedTime",
                "ExecutionEndDateTime",
                "StandardOutputContent",
                "StandardErrorContent",
            )
        }
        (ROOT / f"{args.label}-status.json").write_text(json.dumps(clean, indent=2) + "\n")
        print(json.dumps(clean, indent=2))
    elif args.command == "fetch":
        cloud = state()
        fetch_paths = json.loads(
            os.environ.get(
                "DECIDER_FETCH_PATHS",
                '[".artifacts/decision-model/run",'
                '".artifacts/decision-model/source-manifest.json",'
                '".artifacts/decision-model/worker.log"]',
            )
        )
        if (
            not isinstance(fetch_paths, list)
            or not fetch_paths
            or not all(
                isinstance(path, str)
                and path.startswith(".artifacts/")
                and ".." not in Path(path).parts
                for path in fetch_paths
            )
        ):
            raise ValueError("Fetch paths must be experiment-relative artifact paths")
        url = s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": cloud["bucket"], "Key": "results.tar.gz"},
            ExpiresIn=1800,
        )
        send(
            f"set -eu; cd {REMOTE}; tar -czf /tmp/decision-results.tar.gz "
            f"{shlex.join(fetch_paths)}; "
            "sha256sum /tmp/decision-results.tar.gz; "
            "status=$(curl -fsS -X PUT --upload-file /tmp/decision-results.tar.gz "
            f"-o /tmp/decision-upload-response -w '%{{http_code}}' {shlex.quote(url)}); "
            'test "$status" = 200 || { cat /tmp/decision-upload-response; exit 1; }',
            "fetch",
            1200,
        )
    elif args.command == "download":
        dest = ROOT / "results.tar.gz"
        s3.download_file(state()["bucket"], "results.tar.gz", str(dest))
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        fetched = json.loads((ROOT / "fetch-status.json").read_text())
        if fetched["Status"] != "Success":
            raise ValueError("Verify a successful fetch before downloading results")
        expected = fetched["StandardOutputContent"].split()[0]
        if digest != expected:
            raise ValueError("Downloaded results do not match the worker's SHA-256")
        print("sha256", digest)
        # Validate paths before extraction; archive comes only from our owned worker.
        with tarfile.open(dest) as archive:
            archive.extractall(ROOT / "download", filter="data")
    elif args.command == "delete-bucket":
        bucket = state()["bucket"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        s3.delete_bucket(Bucket=bucket)
        (ROOT / "bucket-cleanup.json").write_text(
            json.dumps({"bucket": bucket, "deleted": True}) + "\n"
        )


if __name__ == "__main__":
    main()
