from __future__ import annotations

from scripts.oss_release_check import inspect


def test_synthetic_accounts_allowed_and_unknown_account_rejected(tmp_path):
    path = tmp_path / "example.txt"
    path.write_text("123456789012 210987654321")
    assert inspect(tmp_path, [path.name])["status"] == "PASS"
    path.write_text("987654" + "321098")
    result = inspect(tmp_path, [path.name])
    assert result["status"] == "FAIL"
    assert result["findings"] == [{"path": path.name, "category": "unreviewed_account_id"}]
    assert "987654" not in str(result)


def test_local_denylist_does_not_disclose_values(tmp_path):
    path = tmp_path / "example.txt"
    path.write_text("synthetic-secret-marker")
    result = inspect(tmp_path, [path.name], forbidden_values=["synthetic-secret-marker"])
    assert result["findings"] == [{"path": path.name, "category": "local_denylist"}]
    assert "synthetic-secret-marker" not in str(result)


def test_symlinks_runtime_artifacts_and_broken_links_rejected(tmp_path):
    (tmp_path / "link").symlink_to("/does-not-exist")
    (tmp_path / "state.tfstate").write_text("{}")
    (tmp_path / "README.md").write_text("[missing](docs/missing.md)")
    result = inspect(tmp_path, ["link", "state.tfstate", "README.md"])
    assert {item["category"] for item in result["findings"]} == {
        "symlink",
        "runtime_or_private_artifact",
        "broken_or_external_local_link",
    }


def test_existing_local_links_resolve(tmp_path):
    (tmp_path / "README.md").write_text("[license](LICENSE) [site](https://example.org)")
    (tmp_path / "LICENSE").write_text("synthetic license fixture")
    assert inspect(tmp_path, ["README.md", "LICENSE"])["status"] == "PASS"


def test_operational_report_receipts_are_not_portable_metrics(tmp_path):
    folder = tmp_path / "reports"
    folder.mkdir()
    (folder / "result.json").write_text('{"client_token": "synthetic-marker"}')
    result = inspect(tmp_path, ["reports/result.json"])
    assert result["findings"] == [
        {"path": "reports/result.json", "category": "operational_report_receipt"}
    ]
