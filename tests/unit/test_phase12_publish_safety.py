from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from rl.phase12.artifacts import publish


def archive_fixture(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    archive.write_bytes(b"synthetic archive bytes")
    manifest = {
        "schema_version": "1",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {},
    }
    archive.with_suffix(".gz.json").write_text(json.dumps(manifest))
    return archive


def test_publish_missing_account_does_not_create_session(monkeypatch, tmp_path):
    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT_ID", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(Session=lambda **kwargs: pytest.fail("AWS session created")),
    )
    with pytest.raises(ValueError, match="EXPECTED_AWS_ACCOUNT_ID"):
        publish(archive_fixture(tmp_path), "s3://bucket/synthetic")


@pytest.mark.parametrize("account", ["123456789012", "210987654321"])
def test_publish_uses_same_verified_profile_for_s3(monkeypatch, tmp_path, account):
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", account)
    monkeypatch.setenv("AWS_PROFILE", "research-sandbox")
    services, profiles, writes = [], [], []
    s3 = SimpleNamespace(
        upload_file=lambda *args, **kwargs: writes.append("upload"),
        put_object=lambda **kwargs: writes.append("manifest"),
    )

    def client(service):
        services.append(service)
        return (
            SimpleNamespace(get_caller_identity=lambda: {"Account": account})
            if service == "sts"
            else s3
        )

    def factory(**kwargs):
        profiles.append(kwargs)
        return SimpleNamespace(client=client)

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            Session=factory, client=lambda *args: pytest.fail("unverified default client")
        ),
    )
    publish(archive_fixture(tmp_path), "s3://bucket/synthetic")
    assert profiles == [{"profile_name": "research-sandbox", "region_name": "us-west-2"}]
    assert services == ["sts", "s3"]
    assert writes == ["upload", "manifest"]


def test_publish_mismatch_prevents_s3_mutation(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPECTED_AWS_ACCOUNT_ID", "123456789012")
    services = []

    def client(service):
        services.append(service)
        return SimpleNamespace(get_caller_identity=lambda: {"Account": "210987654321"})

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(Session=lambda **kwargs: SimpleNamespace(client=client)),
    )
    with pytest.raises(RuntimeError, match="refusing AWS"):
        publish(archive_fixture(tmp_path), "s3://bucket/synthetic")
    assert services == ["sts"]
