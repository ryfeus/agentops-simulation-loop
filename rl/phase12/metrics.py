"""Behavioral comparison with paired prototype resampling, not rollout pseudoreplication."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from rl.phase11.compare import summarize as summarize_existing
from rl.phase12.corpus import ACTION_ARCHETYPES
from rl.phase12.diagnostics import aggregate


def summarize(
    rows: list[dict[str, Any]], metadata: dict[str, Any], attempts: int
) -> dict[str, Any]:
    result = summarize_existing(rows, metadata, attempts)
    prohibited = 0
    failures: Counter[str] = Counter()
    for row in rows:
        expected = metadata[row["task_id"]]["expected_operations"]
        inspections: set[str] = set()
        writes: Counter[str] = Counter()
        bad = False
        for call in row["tool_calls"]:
            invoice = call["arguments"].get("invoice_id", "")
            if call["name"] == "get_invoice":
                inspections.add(invoice)
            else:
                writes[invoice] += 1
                bad |= (
                    expected.get(invoice) != call["name"]
                    or invoice not in inspections
                    or writes[invoice] > 1
                )
        prohibited += int(bad)
        if row["reward"] == 0:
            failures[
                row.get("verifier_diagnostics", {}).get("trajectory_error", "state_invariant")
            ] += 1
    actions = [
        result["archetypes"][key]["pass_rate"]
        for key in ACTION_ARCHETYPES
        if key in result["archetypes"]
    ]
    result.update(
        {
            "action_macro_pass_rate": mean(actions) if actions else 0.0,
            "prohibited_write_rate": prohibited / len(rows),
            "prohibited_write_rollouts": prohibited,
            "diagnostics": aggregate(rows),
            "failure_categories": dict(failures),
            "execution_valid": True,
            "attempts": attempts,
            "task_ids": sorted(metadata),
        }
    )
    return result


def choose(candidates: dict[int, dict[str, Any]]) -> int:
    if not candidates or any(c.get("execution_valid") is not True for c in candidates.values()):
        raise ValueError("cannot select an incomplete candidate")
    return min(
        candidates,
        key=lambda step: (
            -candidates[step]["action_macro_pass_rate"],
            -candidates[step]["prototype_macro_pass_rate"],
            step,
        ),
    )


def pilot_gate(base: dict[str, Any], candidate: dict[str, Any], exp: dict[str, Any]) -> bool:
    if set(base["tasks"]) != set(candidate["tasks"]) or not all(
        c.get("execution_valid") is True for c in (base, candidate)
    ):
        raise ValueError("invalid paired pilot evidence")
    for key in ("inference_sha256", "seed_schedule", "attempts"):
        if base.get(key) != candidate.get(key):
            raise ValueError("policies were not evaluated with the same inference contract")
    critical = exp["gates"]["critical_archetypes"]
    return (
        candidate["action_macro_pass_rate"] - base["action_macro_pass_rate"]
        >= exp["gates"]["pilot_action_delta"] - 1e-12
        and all(
            candidate["archetypes"][a]["pass_rate"] > base["archetypes"][a]["pass_rate"]
            for a in critical
        )
        and candidate["prohibited_write_rate"] <= base["prohibited_write_rate"]
    )


def final_result(
    base: dict[str, Any], candidates: dict[int, dict[str, Any]], exp: dict[str, Any]
) -> dict[str, Any]:
    if set(candidates) != set(exp["seeds"]):
        raise ValueError("all three training seeds are required for replication")
    if not base["execution_valid"] or any(not c["execution_valid"] for c in candidates.values()):
        raise ValueError("invalid evaluation cannot establish learning")
    prototypes = sorted(base["prototypes"])
    if len(prototypes) != 18 or len(base["tasks"]) != 90:
        raise ValueError("final evaluation requires the complete 18-prototype test suite")
    if any(set(c["tasks"]) != set(base["tasks"]) for c in candidates.values()):
        raise ValueError("final policies have different task coverage")
    for candidate in candidates.values():
        for key in ("inference_sha256", "seed_schedule", "attempts"):
            if candidate.get(key) != base.get(key):
                raise ValueError("final policies have different inference contracts")
    seed_deltas = {
        str(s): c["prototype_macro_pass_rate"] - base["prototype_macro_pass_rate"]
        for s, c in candidates.items()
    }
    # Each unit keeps all tasks, attempts, and all three training seeds together.
    cluster_deltas = [
        mean(
            c["prototypes"][p]["pass_rate"] - base["prototypes"][p]["pass_rate"]
            for c in candidates.values()
        )
        for p in prototypes
    ]
    rng = random.Random(exp["gates"]["bootstrap_seed"])
    samples = sorted(
        mean(rng.choices(cluster_deltas, k=len(cluster_deltas)))
        for _ in range(exp["gates"]["bootstrap_samples"])
    )
    interval = [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]
    delta = mean(cluster_deltas)
    critical = {
        a: mean(c["archetypes"][a]["pass_rate"] for c in candidates.values())
        - base["archetypes"][a]["pass_rate"]
        for a in exp["gates"]["critical_archetypes"]
    }
    improved = (
        delta >= exp["gates"]["final_prototype_delta"] - 1e-12
        and all(d > 0 for d in seed_deltas.values())
        and all(d > 0 for d in critical.values())
        and all(
            c["prohibited_write_rate"] <= base["prohibited_write_rate"] for c in candidates.values()
        )
        and interval[0] > 0
    )
    return {
        "execution_valid": True,
        "behavioral_improvement": improved,
        "replication_complete": True,
        "learning_demonstrated": improved,
        "prototype_macro_delta": delta,
        "paired_prototype_bootstrap_95": interval,
        "seed_deltas": seed_deltas,
        "critical_archetype_deltas": critical,
        "inference_scope": "these synthetic billing prototypes; not production readiness",
    }


def reward_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1280:
        raise ValueError("GRPO must emit exactly 1280 training rollouts")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    exposure: Counter[str] = Counter()
    for offset in range(0, len(rows), 4):
        group = rows[offset : offset + 4]
        if len({r["task_id"] for r in group}) != 1:
            raise ValueError("GRPO group crosses task boundaries")
        rewards = {r["reward"] for r in group}
        kind = "mixed" if len(rewards) > 1 else "all-one" if 1.0 in rewards else "all-zero"
        counts[group[0]["archetype"]][kind] += 1
        exposure[group[0]["task_id"]] += 1
    if len(exposure) != 160 or set(exposure.values()) != {2}:
        raise ValueError("GRPO exact two-exposure invariant failed")
    return {
        "by_archetype": {key: dict(value) for key, value in counts.items()},
        "groups": 320,
        "task_exposure": dict(exposure),
    }
