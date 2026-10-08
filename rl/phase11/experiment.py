"""Frozen Phase 11 identity and source validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentops_demo.benchmark.phase10_variants import PHASE10, check, load_phase10_corpus

EXPERIMENT_PATH = Path(__file__).with_name("experiment.json")
EXPECTED_COUNTS = {"train": (160, 32, 16), "dev": (20, 4, 4), "holdout": (20, 4, 3)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_experiment() -> dict[str, Any]:
    value = json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "1" or value.get("experiment_id") != "billing-phase11-v1":
        raise ValueError("unsupported Phase 11 experiment")
    if value.get("model_id") != "Qwen/Qwen3-0.6B":
        raise ValueError("Phase 11 model changed")
    if value.get("training") != {
        "steps": 320,
        "exposures_per_task": 2,
        "num_generations": 4,
        "learning_rate": 1e-5,
        "seed": 20260925,
    } or value.get("checkpoints") != [160, 320]:
        raise ValueError("Phase 11 canonical training settings changed")
    if value.get("evaluation") != {
        "train_representative_attempts": 4,
        "dev_attempts": 8,
        "anchor_attempts": 4,
        "holdout_attempts": 8,
    }:
        raise ValueError("Phase 11 canonical evaluation settings changed")
    return value


def validate_lineage(*, root: Path = PHASE10, suite_sha256: str) -> dict[str, Any]:
    experiment = load_experiment()
    if check(root)["status"] != "CURRENT":
        raise ValueError("Phase 10 generated corpus is stale")
    roles = {role: load_phase10_corpus(role, root) for role in EXPECTED_COUNTS}
    expected = experiment["phase10"]
    actual = {
        "corpus_sha256": roles["train"]["corpus_sha256"],
        "family_specs_sha256": roles["train"]["family_specs_sha256"],
        "split_manifest_sha256": roles["train"]["split_manifest_sha256"],
        "execution_suite_sha256": suite_sha256,
    }
    if actual != expected:
        raise ValueError(f"Phase 10 lineage differs from Phase 11 v1: {actual}")
    for role, (tasks, families, prototypes) in EXPECTED_COUNTS.items():
        value = roles[role]
        if (len(value["task_ids"]), len(value["family_ids"]), len(value["prototype_ids"])) != (
            tasks,
            families,
            prototypes,
        ):
            raise ValueError(f"Phase 10 {role} counts changed")
    for index, left in enumerate(EXPECTED_COUNTS):
        for right in list(EXPECTED_COUNTS)[index + 1 :]:
            for key in ("task_ids", "family_ids", "prototype_ids"):
                if set(roles[left][key]) & set(roles[right][key]):
                    raise ValueError(f"Phase 10 {key} overlap between {left} and {right}")
            if set(roles[left]["behavior_signatures"].values()) & set(
                roles[right]["behavior_signatures"].values()
            ):
                raise ValueError(f"Phase 10 behavior overlap between {left} and {right}")
    representatives = sorted(f"{family}-v00" for family in roles["train"]["family_ids"])
    if len(representatives) != 32 or not set(representatives) <= set(roles["train"]["task_ids"]):
        raise ValueError("Phase 11 train representatives changed")
    return {
        "experiment": experiment,
        "roles": roles,
        "representatives": representatives,
        "lineage": actual,
        "experiment_sha256": sha256(EXPERIMENT_PATH),
    }
