from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentops_demo.billing.sqlite_repository import initialize_database
from rl.phase8b import sandbox_billing_bridge as bridge
from rl.phase12 import artifacts, calibration, config, corpus, diagnostics, metrics, runner
from rl.phase12.common import REPO, digest, fingerprint, read, write
from rl.phase12.demonstrations import conversation, masked_examples, teacher_calls, validate
from rl.phase12.sampling import bind_request_seeds
from scripts import rl_finetuning_comparison as controller


@pytest.fixture(scope="module")
def data():
    return corpus.scenarios()


def test_frozen_corpus_and_independent_coverage(data):
    frozen = corpus.manifest(data)
    assert frozen == corpus.check()
    assert {r: len(v) for r, v in data.items()} == {"train": 160, "dev": 90, "test": 90}
    assert len(corpus.representatives(frozen["roles"]["train"])) == 32
    assert max(t["required_tool_calls"] for r in frozen["roles"].values() for t in r.values()) == 7


def test_cross_split_structure_and_instruction_leaks_rejected(data):
    poisoned = copy.deepcopy(data)
    train = next(iter(data["train"].values()))
    key = next(iter(poisoned["dev"]))
    poisoned["dev"][key] = train.model_copy(update={"id": key})
    with pytest.raises(ValueError):
        corpus.manifest(poisoned)


def test_training_transport_redacts_final_test_answers(data):
    frozen = corpus.manifest(data)
    transported = corpus.transport_manifest(frozen)
    assert transported["corpus_sha256"] == frozen["corpus_sha256"]
    assert transported["roles"]["train"] == frozen["roles"]["train"]
    assert transported["redacted_roles"] == ["test"]
    for key, metadata in transported["roles"]["test"].items():
        assert "expected_operations" not in metadata
        assert "expected_write_targets" not in metadata
        assert metadata["scenario_sha256"] == frozen["roles"]["test"][key]["scenario_sha256"]


def test_model_revision_and_template_must_be_pinned(tmp_path):
    exp = config.load()
    assert len(exp["model"]["revision"]) == 40
    assert len(exp["model"]["chat_template_sha256"]) == 64
    exp["model"]["revision"] = "main"
    write(tmp_path / "exp.json", exp)
    with pytest.raises(ValueError):
        config.load(tmp_path / "exp.json")


def test_vllm_request_streams_are_reproducible_distinct_and_pass_seeded():
    def streams(seed):
        recorded = []
        engine = SimpleNamespace(
            generate=lambda prompts, *, sampling_params, **kwargs: recorded.append(
                [p.seed for p in sampling_params]
            )
        )
        original = SimpleNamespace(seed=None)
        bind_request_seeds(engine, seed)
        engine.generate(["same"] * 4, sampling_params=original)
        engine.generate(["same"] * 4, sampling_params=original)
        assert original.seed is None
        return recorded

    first = streams(42)
    assert first == streams(42)
    assert first != streams(43)
    assert len({s for group in first for s in group}) == 8


def test_config_check_never_imports_gpu_runtime(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "trl", "transformers", "boto3"}:
            raise AssertionError(f"unexpected runtime import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert controller.config_check()["training_performed"] is False


def arm(ids, fraction):
    return {
        "task_ids": ids,
        "attempts": 4,
        "execution_valid": True,
        "diagnostics": {"clipped_fraction": fraction, "rollouts": len(ids) * 4},
    }


def test_budget_selection_conditional_fallback_and_saturation():
    exp = config.load()
    evidence = {
        "thinking-256": arm(["a"], 0.8),
        "nonthinking-256": arm(["a"], 0.1),
        "nonthinking-1024": arm(["a"], 0.04),
    }
    assert diagnostics.select_budget(evidence, ["a"], 7, exp)["budget"] == 1024
    evidence["nonthinking-1024"] = arm(["a"], 0.2)
    with pytest.raises(ValueError, match="2048"):
        diagnostics.select_budget(evidence, ["a"], 7, exp)
    evidence["nonthinking-2048"] = arm(["a"], 0.01)
    assert diagnostics.select_budget(evidence, ["a"], 7, exp)["budget"] == 2048
    evidence["nonthinking-2048"] = arm(["a"], 0.5)
    with pytest.raises(ValueError, match="NO_QUALIFYING"):
        diagnostics.select_budget(evidence, ["a"], 7, exp)


def test_diagnostic_rejects_leakage_and_short_iteration_limit():
    e = {k: arm(["test-task"], 0) for k in ("thinking-256", "nonthinking-256", "nonthinking-1024")}
    with pytest.raises(ValueError, match="train-only"):
        diagnostics.select_budget(e, ["train-task"], 1, config.load())
    with pytest.raises(ValueError, match="iteration"):
        diagnostics.select_budget(e, ["test-task"], 8, config.load())


def test_completion_diagnostics_distinguish_clipping_parser_and_domain_error():
    messages = [
        {"role": "assistant", "content": '<tool_call>{"name":'},
        {"role": "tool", "content": '{"ok": false, "error": "missing"}'},
    ]
    d = diagnostics.inspect_completion([3, 4], messages, "raw", {0}, 2, [])
    assert d["termination_reason"] == "length"
    assert d["clipped_before_first_tool"]
    assert d["parser_failures"] == d["tool_errors"] == 1


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join(
            f"<{m['role']}>" + json.dumps(m, sort_keys=True) + "</end>" for m in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())


def demo_record():
    call = {"name": "get_invoice", "arguments": {"invoice_id": "inv-1"}}
    return {
        "task_id": "train",
        "reward": 1.0,
        "source": "executed-billing-tools",
        "complete": True,
        "truncated": False,
        "tool_calls": [call],
        "messages": [
            {"role": "user", "content": "Look up inv-1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": call}],
            },
            {"role": "tool", "name": "get_invoice", "content": '{"ok": true}'},
            {"role": "assistant", "content": "Checked."},
        ],
    }


def test_sft_masks_context_but_teaches_calls_and_final_answer():
    record = demo_record()
    validate(record, {"train"})
    examples = masked_examples(record, FakeTokenizer(), [], 4096)
    assert len(examples) == 2
    for example in examples:
        first = next(i for i, label in enumerate(example["labels"]) if label != -100)
        assert first > 0
        assert example["labels"][first:] == example["input_ids"][first:]
        target = bytes(example["labels"][first:]).decode()
        assert '"role": "tool"' not in target and '"role": "user"' not in target
    assert "get_invoice" in bytes(v for v in examples[0]["labels"] if v != -100).decode()
    with pytest.raises(ValueError, match="truncated"):
        masked_examples(record, FakeTokenizer(), [], 10)


@pytest.mark.parametrize(
    "change",
    [
        {"task_id": "test"},
        {"truncated": True},
        {"reward": 0},
        {"source": "oracle-sql"},
        {"complete": False},
    ],
)
def test_sft_rejects_unverified_demonstrations(change):
    with pytest.raises(ValueError):
        validate({**demo_record(), **change}, {"train"})


def test_sft_masks_token_ids_with_transformers_mapping_default():
    """Transformers 5 returns a mapping unless flat token IDs are requested."""

    class MappingDefaultTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, return_dict=True, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            return {"input_ids": ids} if return_dict else ids

    examples = masked_examples(demo_record(), MappingDefaultTokenizer(), [], 4096)
    assert len(examples) == 2
    for example in examples:
        assert all(isinstance(value, int) for value in example["input_ids"])
        first = next(i for i, label in enumerate(example["labels"]) if label != -100)
        assert example["labels"][:first] == [-100] * first
        assert example["labels"][first:] == example["input_ids"][first:]


def test_idle_docker_shutdown_preserves_native_cleanup(monkeypatch):
    """The optimization must retain native cleanup and leave tool execution alone."""
    import sys

    class Parent:
        async def _stop(self):
            if self._env is not None:
                try:
                    await self._env.stop(delete=True)
                finally:
                    self._env = None

    class Docker:
        def __init__(self):
            self.commands = []
            self.cleaned = False

        async def _run_docker_compose_command(self, command, *, timeout_sec=None):
            self.commands.append((command, timeout_sec))

        async def stop(self, *, delete):
            assert delete
            await self._run_docker_compose_command(["exec", "main", "flush-logs"])
            await self._run_docker_compose_command(
                ["down", "--rmi", "local", "--volumes", "--remove-orphans"], timeout_sec=180
            )
            self.cleaned = True

    monkeypatch.setitem(
        sys.modules,
        "rl.phase8c.baseline_env",
        SimpleNamespace(BaselineBillingHarborEnv=Parent),
    )
    spec = importlib.util.spec_from_file_location(
        "phase12_shutdown_test", REPO / "rl/phase12/environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    docker = Docker()
    env = module.ComparisonEnv()
    env._env, env._environment_type = docker, "docker"
    asyncio.run(env._stop())
    assert env._env is None and docker.cleaned
    assert docker.commands == [
        (["exec", "main", "flush-logs"], None),
        (["down", "--timeout", "1", "--rmi", "local", "--volumes", "--remove-orphans"], 180),
    ]


@pytest.mark.parametrize(
    "operation,arguments",
    [
        ("get_invoice", {"invoice_id": 123}),
        ("get_invoice", {"invoice_id": ""}),
        ("refund_invoice", {"invoice_id": "inv-1", "reason": ""}),
    ],
)
def test_bridge_argument_rejections_remain_policy_errors(
    tmp_path, monkeypatch, data, operation, arguments
):
    from rl.phase12 import runtime

    scenario = next(iter(data["train"].values()))
    database, trajectory = tmp_path / "billing.db", tmp_path / "agent-run.json"
    asyncio.run(initialize_database(database, scenario.initial_state))

    class Parent:
        def _invoke(self, name, args):
            payload = base64.b64encode(json.dumps(args, sort_keys=True).encode()).decode()
            return self._exec(shlex.join([name, payload]))

    class LocalBridge:
        async def exec(self, command, *, timeout_sec):
            import os

            context = {
                **os.environ,
                "BILLING_DATABASE_PATH": str(database),
                "AGENT_RUN_PATH": str(trajectory),
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "rl/phase8b/sandbox_billing_bridge.py"),
                    *shlex.split(command),
                ],
                env=context,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            return SimpleNamespace(
                stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
            )

    monkeypatch.setitem(
        sys.modules,
        "rl.phase8c.baseline_env",
        SimpleNamespace(BaselineBillingHarborEnv=Parent),
    )
    spec = importlib.util.spec_from_file_location(
        "phase12_rejected_bridge_test", REPO / "rl/phase12/environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.ComparisonEnv()
    env._env, env._run = LocalBridge(), asyncio.run
    env._phase12_observations = []
    invoice = scenario.initial_state.invoices[0].id
    good = env._invoke("get_invoice", {"invoice_id": invoice})
    assert json.loads(good)["ok"] is True
    before = trajectory.read_bytes()
    rejected = env._invoke(operation, arguments)
    assert trajectory.read_bytes() == before
    assert env._phase12_observations[-1]["response"] == rejected
    assert env._phase12_observations[-1]["return_code"] == 2
    assert env._phase12_observations[-1]["bridge_rejected"] is True
    env.reward = 0.0
    env._phase12_generation = diagnostics.inspect_completion(
        [0], [{"role": "tool", "content": rejected}], rejected, {0}, 256, []
    )
    env._baseline_evidence = lambda: {
        "tool_calls": read(trajectory)["tool_calls"],
        "verifier_diagnostics": {},
    }
    output = tmp_path / "rollouts.jsonl"
    reward = runtime.reward_capture(output, {"task": {}})
    assert reward(environments=[env], completions=[[]], task_id=["task"]) == [0.0]
    captured = json.loads(output.read_text())
    assert captured["tool_errors"] == 1
    assert captured["tool_calls"] == read(trajectory)["tool_calls"]
    # Successful bridge calls must still be backed by matching trajectory data.
    env._baseline_evidence = lambda: {"tool_calls": [], "verifier_diagnostics": {}}
    with pytest.raises(ValueError, match="executed-tool evidence"):
        reward(environments=[env], completions=[[]], task_id=["task"])
    assert read(tmp_path / "evidence-mismatch.json")["expected_tool_calls"]


def test_repeated_fetch_excludes_its_generated_transfer_archive(tmp_path, monkeypatch):
    import shutil

    from scripts import rl_harbor_generalization

    remote, run = tmp_path / "remote", tmp_path / "retained"
    (remote / "run").mkdir(parents=True)
    write(remote / "run/result.json", {"execution_valid": True})
    write(
        run / "dispatch/diagnose.json",
        {"remote": str(remote), "instance_id": "worker", "command_id": "command"},
    )
    ssm = SimpleNamespace(
        get_command_invocation=lambda **kwargs: {"Status": "Success", "ResponseCode": 0}
    )
    monkeypatch.setattr(rl_harbor_generalization, "_aws", lambda: (ssm, "worker", {}))

    def shell(client, instance, commands, **kwargs):
        result = subprocess.run(commands[0], shell=True, capture_output=True, text=True)
        return SimpleNamespace(succeeded=result.returncode == 0)

    inventories = []

    def download(client, instance, source, target, inventory, **kwargs):
        inventories.append(inventory)
        # The downloader creates this archive inside the remote evidence root.
        (Path(source) / ".rl-artifacts.tar.gz").write_bytes(b"transport archive")
        for relative in inventory:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(source) / relative, destination)

    monkeypatch.setattr(controller.smoke, "run_shell", shell)
    monkeypatch.setattr(controller.smoke, "_read_remote_file", lambda c, i, p: Path(p).read_text())
    monkeypatch.setattr(controller.smoke, "download_remote_directory", download)
    assert controller.fetch(run, "diagnose")["evidence_fetched"]
    assert controller.fetch(run, "diagnose")["evidence_fetched"]
    assert inventories[0] == inventories[1]
    assert set(inventories[1]) == {"result.json"}
    assert read(run / "result.json") == {"execution_valid": True}


def summary(rate=0.0, action=0.0, prohibited=0.0):
    return {
        "execution_valid": True,
        "tasks": {f"t{i}": {} for i in range(90)},
        "prototypes": {f"p{i}": {"pass_rate": rate} for i in range(18)},
        "prototype_macro_pass_rate": rate,
        "action_macro_pass_rate": action,
        "prohibited_write_rate": prohibited,
        "archetypes": {
            a: {"pass_rate": action if a in corpus.ACTION_ARCHETYPES else rate}
            for a in corpus.ARCHETYPES
        },
    }


def test_read_only_gain_cannot_pass_pilot_and_checkpoint_priority():
    base = summary(0.2, 0.1)
    readonly = summary(0.6, 0.1)
    action = summary(0.4, 0.3)
    assert not metrics.pilot_gate(base, readonly, config.load())
    assert metrics.pilot_gate(base, action, config.load())
    assert metrics.choose({160: readonly, 320: action}) == 320
    assert metrics.choose({160: action, 320: action}) == 160
    action["prohibited_write_rate"] = 0.01
    assert not metrics.pilot_gate(base, action, config.load())


def test_replicated_gate_requires_all_seeds_and_cluster_improvement():
    exp = config.load()
    good = {s: summary(0.6, 0.6) for s in exp["seeds"]}
    result = metrics.final_result(summary(0.2, 0.2), good, exp)
    assert result["learning_demonstrated"]
    assert result["paired_prototype_bootstrap_95"] == pytest.approx([0.4, 0.4])
    with pytest.raises(ValueError, match="three"):
        metrics.final_result(summary(), {42: summary()}, exp)
    good[44] = summary(0.1, 0.1)
    assert not metrics.final_result(summary(0.2, 0.2), good, exp)["learning_demonstrated"]


def test_prototype_bootstrap_not_repeated_attempt_count():
    exp = config.load()
    varied = summary(0.5, 0.5)
    for i in range(18):
        varied["prototypes"][f"p{i}"]["pass_rate"] = float(i % 2)
    result = metrics.final_result(summary(0.4, 0.4), {s: varied for s in exp["seeds"]}, exp)
    assert result["paired_prototype_bootstrap_95"][0] < 0
    assert not result["learning_demonstrated"]


def test_reward_exposure_and_zero_variance():
    records = [
        {"task_id": str(i), "archetype": "disputed-refund", "reward": 0.0}
        for _ in range(2)
        for i in range(160)
        for _ in range(4)
    ]
    assert metrics.reward_groups(records)["by_archetype"] == {"disputed-refund": {"all-zero": 320}}
    records[0]["task_id"] = "outside"
    with pytest.raises(ValueError):
        metrics.reward_groups(records)


def test_export_is_checksummed_and_never_publishes(tmp_path):
    run = tmp_path / "run"
    write(run / "result.json", {"execution_valid": False})
    write(run / "datasets/test/secret.json", {"instruction": "test"})
    archive = tmp_path / "evidence.tar.gz"
    result = artifacts.export(run, archive)
    assert result["archive_sha256"] == digest(archive)
    with tarfile.open(archive) as stream:
        assert stream.getnames() == ["result.json"]
    with pytest.raises(ValueError):
        artifacts.verify_files(run, {"../escape": "0" * 64})
    write(run / "result.json", {})
    with pytest.raises(ValueError, match="changed"):
        artifacts.verify_files(run, result["files"])


def test_missing_calibration_is_not_execution_success():
    with pytest.raises(ValueError):
        calibration.validate({"execution_valid": True}, corpus.check())


def test_locked_run_cannot_fit_again(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "verify_run", lambda run: {})
    write(tmp_path / "selection-lock.json", {})
    with pytest.raises(ValueError, match="locked"):
        runner.execute("pilot", tmp_path)


def test_training_pack_excludes_test_contents(tmp_path, monkeypatch):
    run = tmp_path / "run"
    for name in (
        "datasets/train/scenario.yaml",
        "datasets/dev/scenario.yaml",
        "datasets/test/scenario.yaml",
    ):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text(name)
    monkeypatch.setattr(artifacts, "verify_run", lambda run: {"source_revision": "abc"})
    monkeypatch.setattr(controller.smoke, "git_provenance", lambda: ("abc", True))
    result = controller.pack(run, "pilot", tmp_path / "payload.tar.gz")
    assert "run/datasets/train/scenario.yaml" in result["files"]
    assert not any("datasets/test/" in p or "prototypes.json" in p for p in result["files"])


def test_controller_stages_with_simulated_workers_only(tmp_path, monkeypatch, data):
    """Exercise the orchestration without downloading or invoking any model."""
    from rl.phase11.selection import adapter_files
    from rl.phase12.common import append

    frozen = corpus.check()
    run = tmp_path / "run"
    run.mkdir()
    for role in ("train", "dev"):
        write(run / f"datasets/{role}/metadata.json", frozen["roles"][role])
    write(run / "corpus-manifest.json", frozen)
    identity = {
        "experiment_sha256": fingerprint(config.load()),
        "corpus_sha256": frozen["corpus_sha256"],
        "source_revision": "abc",
    }
    write(run / "identity.json", identity)
    monkeypatch.setattr(artifacts, "verify_run", lambda run: identity)
    monkeypatch.setattr(runner, "verify_run", lambda run: identity)
    launched = []

    def worker(run, kind, args):
        launched.append((kind, args))
        output = Path(args["output"])
        if kind == "train":
            adapters = {}
            for step in [10, 20, 30] if args["method"] == "sft" else [160, 320]:
                adapter = output / f"adapters/{step}"
                write(adapter / "adapter_config.json", {"fake": True})
                (adapter / "adapter_model.safetensors").write_bytes(
                    b"test fixture, not model weights"
                )
                adapters[str(step)] = adapter_files(adapter)
            write(
                output / "training-result.json",
                {
                    "execution_valid": True,
                    "adapters": adapters,
                    "method": args["method"],
                    "seed": args["seed"],
                    "inference_sha256": config.inference_hash(config.load(), args["budget"]),
                },
            )
            return
        assert kind == "evaluate"
        metadata = read(Path(args["root"]) / "metadata.json")
        for key in args["selected"]:
            for attempt in range(4):
                append(
                    output / "rollouts.jsonl",
                    {
                        "task_id": key,
                        "attempt": attempt,
                        "reward": float(args.get("adapter") is not None),
                        "tool_calls": [],
                        "tool_call_count": 0,
                        "verifier_components": {},
                        "verifier_diagnostics": {},
                        "clipped": False,
                        "clipped_before_first_tool": False,
                        "completion_token_count": 10,
                        "parser_failures": 0,
                        "tool_errors": 0,
                        "termination_reason": "eos",
                        "archetype": metadata[key]["archetype"],
                    },
                )
        adapter = Path(args["adapter"]) if args.get("adapter") else None
        write(
            output / "result.json",
            {
                "execution_valid": True,
                "global_step": 0,
                "optimizer_created": False,
                "training_performed": False,
                "seed": args["seed"],
                "adapter_files": adapter_files(adapter) if adapter else None,
                "inference_sha256": config.inference_hash(
                    config.load(), args["budget"], args["thinking"]
                ),
            },
        )

    monkeypatch.setattr(runner, "launch", worker)
    demonstrations = [{"task_id": key} for key in sorted(data["train"])]
    for record in demonstrations:
        append(run / "demonstrations/demonstrations.jsonl", record)
    write(
        run / "demonstrations/manifest.json",
        {"execution_valid": True, "records_sha256": fingerprint(demonstrations)},
    )
    runner.execute("diagnose", run)
    assert runner.budget_for(run) == 256
    assert all(set(args["selected"]) <= set(data["train"]) for _, args in launched)
    runner.execute("pilot", run)
    assert artifacts.eligible_methods(run) == ["sft", "grpo"]
    runner.execute("replicate", run)
    lock = artifacts.create_lock(run)
    assert len(lock["selections"]) == 6
    with pytest.raises(ValueError, match="locked"):
        runner.execute("pilot", run)
    controller.prepare_final(run)
    # A final-only payload must remain runnable without training task contents.
    import shutil

    shutil.rmtree(run / "datasets/train")
    shutil.rmtree(run / "datasets/dev")
    runner.execute("final", run)
    assert all(
        result["learning_demonstrated"] for result in read(run / "final/result.json").values()
    )
    prior = len(launched)
    runner.execute("final", run)
    assert len(launched) == prior, "completed passes must be reused rather than regenerated"
    selected = next(iter(lock["selections"].values()))
    (run / selected["adapter"] / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        artifacts.verify_lock(run)


def test_mismatched_inference_invalidates_method_comparison():
    base, candidate = summary(), summary(0.9, 0.9)
    base["inference_sha256"] = "a"
    candidate["inference_sha256"] = "b"
    with pytest.raises(ValueError, match="inference"):
        metrics.pilot_gate(base, candidate, config.load())


def test_final_pack_omits_train_dev_and_demonstrations(tmp_path, monkeypatch):
    run = tmp_path / "run"
    for role in ("train", "dev", "test", "anchors"):
        write(run / f"datasets/{role}/metadata.json", {})
    write(run / "demonstrations/manifest.json", {})
    write(run / "final-inputs.json", {})
    monkeypatch.setattr(artifacts, "verify_run", lambda run: {"source_revision": "abc"})
    monkeypatch.setattr(artifacts, "verify_lock", lambda run: {})
    monkeypatch.setattr(controller.smoke, "git_provenance", lambda: ("abc", True))
    result = controller.pack(run, "final", tmp_path / "final.tar.gz")
    assert "run/datasets/test/metadata.json" in result["files"]
    assert "run/datasets/anchors/metadata.json" in result["files"]
    assert not any(
        p.startswith(("run/datasets/train/", "run/datasets/dev/", "run/demonstrations/"))
        for p in result["files"]
    )


def test_request_seed_policy_is_part_of_inference_identity(monkeypatch):
    from rl.phase12 import sampling

    before = config.inference_hash(config.load(), 1024)
    monkeypatch.setattr(sampling, "SEED_POLICY", "changed")
    assert before != config.inference_hash(config.load(), 1024)


def test_every_fresh_scenario_solvable_through_real_tools_and_noop_fails(
    data, tmp_path, monkeypatch
):
    """CPU contract calibration only: no Harbor containers, model, or saved demonstrations."""
    path = REPO / "benchmarks/disputed-refund/tests/verify.py"
    spec = importlib.util.spec_from_file_location("phase12_contract_verifier", path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    class LocalEnvironment:
        def reset(self, *, task_dir):
            self.scenario = current[0]
            self.database = tmp_path / f"{self.scenario.id}-{len(list(tmp_path.glob('*.db')))}.db"
            self.trace = self.database.with_suffix(".json")
            asyncio.run(initialize_database(self.database, self.scenario.initial_state))
            monkeypatch.setenv("BILLING_DATABASE_PATH", str(self.database))
            monkeypatch.setenv("AGENT_RUN_PATH", str(self.trace))
            verifier.AGENT_RESULT_PATH = self.trace
            verifier._load_spec = lambda: {
                "expected_invariants": [
                    i.model_dump(mode="json") for i in self.scenario.expected_invariants
                ],
                "initial_state": self.scenario.initial_state.model_dump(mode="json"),
                "benchmark": self.scenario.benchmark.model_dump(mode="json"),
            }
            return self.scenario.instruction

        def __getattr__(self, name):
            return lambda **kwargs: json.dumps(asyncio.run(bridge.invoke(name, kwargs)))

        @property
        def reward(self):
            self.components, _ = verifier.evaluate_database(self.database)
            return self.components["reward"]

        def _baseline_evidence(self):
            return {
                "tool_calls": read(self.trace)["tool_calls"],
                "verifier_components": self.components,
            }

    current = [None]
    env = LocalEnvironment()
    for scenario in [*data["dev"].values(), *data["test"].values()]:
        current[0] = scenario
        record = conversation(env, scenario, tmp_path)
        validate(record, {scenario.id})
        assert record["tool_calls"] == teacher_calls(scenario)
        env.reset(task_dir="")
        assert env.reward == 0.0, scenario.id
        protected = next(
            i.invoice_id for i in scenario.expected_invariants if i.type == "unchanged"
        )
        env.refund_invoice(invoice_id=protected, reason="Wrong write")
        assert env.reward == 0.0, scenario.id
