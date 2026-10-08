from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from harbor.models.job.config import JobConfig

from agentops_demo.harbor.provenance import PackageProvenance
from agentops_demo.scale.contracts import (
    BenchmarkIdentity,
    CandidateIdentity,
    CleanupSummary,
    Ec2ExecutionConfig,
    Ec2Trial,
)
from agentops_demo.scale.gate import candidate_gate
from scripts import run_harbor_ec2
from scripts.cleanup_harbor_ec2_workers import eligible_for_cleanup
from scripts.generate_harbor_ec2_config import execution_config, job_config
from scripts.generate_harbor_ec2_tfvars import payload
from scripts.harbor_ec2_controller import route_has_igw, worker_filters
from scripts.harbor_ec2_preflight import preflight
from scripts.run_harbor_ec2_job import install_https_ec2_environment


def private_key(tmp_path: Path) -> Path:
    key = tmp_path / "harbor"
    key.write_text("private material is never read into tfvars")
    key.with_name("harbor.pub").write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI test@operator\n")
    return key


def outputs() -> dict[str, dict[str, str]]:
    return {
        "aws_region": {"value": "us-west-2"},
        "harbor_vpc_id": {"value": "vpc-123"},
        "harbor_subnet_id": {"value": "subnet-123"},
        "harbor_security_group_id": {"value": "sg-123"},
        "harbor_key_name": {"value": "agentops-demo-dev-harbor"},
        "harbor_ami_id": {"value": "ami-123"},
    }


def env(key: Path) -> dict[str, str]:
    return {
        "HARBOR_EC2_SSH_PRIVATE_KEY": str(key),
        "HARBOR_EC2_CONTROLLER_CIDR": "198.51.100.10/32",
        "EXPECTED_AWS_ACCOUNT_ID": "123456789012",
    }


def test_tfvars_are_public_key_only_and_validate_cidr(tmp_path: Path) -> None:
    key = private_key(tmp_path)
    value = payload(env(key))
    assert value["harbor_ec2_enabled"] is True
    assert value["harbor_ssh_public_key"].startswith("ssh-ed25519")
    assert "private material" not in str(value)
    with pytest.raises(ValueError, match="every address"):
        payload({**env(key), "HARBOR_EC2_CONTROLLER_CIDR": "0.0.0.0/0"})


def test_execution_config_and_job_config_have_fixed_ec2_contract(tmp_path: Path) -> None:
    key = private_key(tmp_path)
    execution = execution_config(
        outputs=outputs(), run_id="run-1", task_source="static", environ=env(key)
    )
    assert execution.root_volume_size_gb == 30
    assert execution.tags["RunId"] == "run-1"
    config = job_config(
        execution=execution,
        task=tmp_path,
        jobs_dir=tmp_path / "jobs",
        job_name="oracle",
        model="oracle",
        ssh_key_path=key,
    )
    kwargs = config["environment"]["kwargs"]
    assert config["environment"]["type"] == "ec2"
    assert kwargs["launch_mode"] == "ephemeral"
    assert kwargs["use_public_ip"] is True
    assert kwargs["root_volume_type"] == "gp3"
    assert kwargs["bootstrap_docker"] is True
    assert config["verifier_timeout_multiplier"] == 12.0
    assert JobConfig.model_validate(config).environment.type == "ec2"


def test_execution_rejects_bad_attempts_or_region(tmp_path: Path) -> None:
    key = private_key(tmp_path)
    with pytest.raises(ValueError, match=r"1\.\.4"):
        execution_config(
            outputs=outputs(),
            run_id="run",
            task_source="static",
            environ={**env(key), "HARBOR_EC2_ATTEMPTS": "5"},
        )
    invalid = outputs()
    invalid["aws_region"] = {"value": "us-east-1"}
    with pytest.raises(ValueError, match="aws_region"):
        execution_config(outputs=invalid, run_id="run", task_source="static", environ=env(key))


def execution() -> Ec2ExecutionConfig:
    return Ec2ExecutionConfig(
        vpc_id="vpc-1",
        subnet_id="subnet-1",
        security_group_id="sg-1",
        key_name="key",
        ami_id="ami-1",
        run_id="run-1",
        task_source="static",
        tags={"Project": "agentops-demo", "Phase": "6", "RunId": "run-1", "ManagedBy": "harbor"},
    )


def package() -> PackageProvenance:
    return PackageProvenance(
        source_revision="a" * 40,
        filename="candidate.whl",
        sha256="b" * 64,
        worktree_clean=True,
    )


def benchmark() -> BenchmarkIdentity:
    return BenchmarkIdentity(task_source="static", harbor_task_sha256="c" * 64)


def cleanup(passed: bool = True) -> CleanupSummary:
    return CleanupSummary(workers_remaining=[] if passed else ["i-left"], passed=passed)


def candidate() -> CandidateIdentity:
    return CandidateIdentity(
        name="known-good",
        model="scripted/correct",
        agent_config_fingerprint="d" * 64,
        package_sha256="b" * 64,
    )


def test_oracle_and_calibration_are_accepted_but_never_eligible() -> None:
    oracle = candidate_gate(
        run_id="run-1",
        purpose="oracle_smoke",
        task_source="static",
        execution=execution(),
        benchmark=benchmark(),
        execution_package=package(),
        candidate=None,
        trials=[
            Ec2Trial(
                name="oracle", requested_reward=1, reward=1, completed=True, provenance_matches=True
            )
        ],
        cleanup=cleanup(),
        local_ec2_parity=None,
    )
    bad = candidate_gate(
        run_id="run-1",
        purpose="calibration_smoke",
        task_source="static",
        execution=execution(),
        benchmark=benchmark(),
        execution_package=package(),
        candidate=None,
        trials=[
            Ec2Trial(
                name="scripted-bad",
                requested_reward=0,
                reward=0,
                completed=True,
                provenance_matches=True,
            )
        ],
        cleanup=cleanup(),
        local_ec2_parity=None,
    )
    assert (oracle.accepted, oracle.eligible) == (True, None)
    assert (bad.accepted, bad.eligible) == (True, None)


def test_candidate_gate_fails_closed_for_missing_trial_or_provenance() -> None:
    scaled = execution().model_copy(update={"attempts": 4, "concurrency": 4})
    trials = [
        Ec2Trial(
            name="known-good",
            requested_reward=1,
            reward=1,
            completed=True,
            provenance_matches=True,
            config_fingerprint="d" * 64,
            package_sha256="b" * 64,
            model="scripted/correct",
        )
        for _ in range(3)
    ]
    gate = candidate_gate(
        run_id="run-1",
        purpose="candidate_gate",
        task_source="static",
        execution=scaled,
        benchmark=benchmark(),
        execution_package=package(),
        candidate=candidate(),
        trials=trials,
        cleanup=cleanup(),
        local_ec2_parity=None,
    )
    assert (gate.accepted, gate.eligible) == (False, False)
    assert any("retained 3" in reason for reason in gate.failure_reasons)


def test_cleanup_filters_only_exact_phase_and_run_tags() -> None:
    now = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    tags = [
        {"Key": "Project", "Value": "agentops-demo"},
        {"Key": "Phase", "Value": "6"},
        {"Key": "RunId", "Value": "run-1"},
        {"Key": "ManagedBy", "Value": "harbor"},
    ]
    instances = [
        {"InstanceId": "i-good", "Tags": tags},
        {"InstanceId": "i-other", "Tags": [*tags[:-2], {"Key": "RunId", "Value": "run-2"}]},
        {"InstanceId": "i-wrong", "Tags": [{"Key": "Project", "Value": "other"}]},
    ]
    assert eligible_for_cleanup(instances, run_id="run-1", older_than_hours=None, now=now) == [
        "i-good"
    ]


def test_worker_filter_and_igw_route_are_narrow() -> None:
    assert worker_filters("run-1")[-1] == {"Name": "tag:RunId", "Values": ["run-1"]}
    assert route_has_igw([{"GatewayId": "igw-1", "DestinationCidrBlock": "0.0.0.0/0"}])
    assert not route_has_igw([{"GatewayId": "nat-1", "DestinationCidrBlock": "0.0.0.0/0"}])


def test_ec2_runner_registers_https_bootstrap_environment() -> None:
    from harbor.environments.factory import _ENVIRONMENT_REGISTRY
    from harbor.models.environment_type import EnvironmentType

    install_https_ec2_environment()
    entry = _ENVIRONMENT_REGISTRY[EnvironmentType.EC2]
    assert entry.module == "agentops_demo.harbor.ec2"
    assert entry.class_name == "HttpsBootstrapEC2Environment"


def test_scale_retains_every_harbor_trial_result(tmp_path: Path) -> None:
    job = tmp_path / "known-good"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"n_total_trials": 2}))
    for name in ("attempt-one", "attempt-two"):
        trial = job / name
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
        )

    retained = run_harbor_ec2._retained_trials(
        name="known-good",
        job_dir=job,
        expected_reward=1.0,
        config=None,
        package_sha256=None,
        model="scripted/correct",
    )

    assert len(retained) == 2
    assert all(trial.completed and trial.reward == 1.0 for trial in retained)


def test_worker_ids_include_harbor_job_log_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "job.log").write_text(
        "Launched EC2 instance i-0123456789abcdef0\nTerminated EC2 instance i-0fedcba9876543210\n"
    )
    assert run_harbor_ec2._worker_ids(tmp_path) == [
        "i-0123456789abcdef0",
        "i-0fedcba9876543210",
    ]


class FakeSts:
    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": "123456789012"}


class FakeEc2:
    def describe_subnets(self, **_: object) -> dict[str, object]:
        return {"Subnets": [{"MapPublicIpOnLaunch": True}]}

    def describe_route_tables(self, **_: object) -> dict[str, object]:
        return {
            "RouteTables": [
                {"Routes": [{"GatewayId": "igw-1", "DestinationCidrBlock": "0.0.0.0/0"}]}
            ]
        }

    def describe_security_groups(self, **_: object) -> dict[str, object]:
        return {}

    def run_instances(self, **_: object) -> None:
        raise ClientError({"Error": {"Code": "DryRunOperation", "Message": "ok"}}, "RunInstances")


def test_preflight_uses_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = private_key(tmp_path)
    monkeypatch.setattr("scripts.harbor_ec2_preflight.version", lambda _: "0.22.0")
    result = preflight(
        sts_client=FakeSts(), ec2_client=FakeEc2(), outputs=outputs(), environ=env(key)
    )
    assert result["harbor_ami_id"] == "ami-123"
