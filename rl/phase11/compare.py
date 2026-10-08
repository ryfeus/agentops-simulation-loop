"""Coverage-checked policy comparisons and behavior summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from rl.phase8c.report import wilson_interval


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], attempts: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id not in metadata or row.get("reward") not in (0, 0.0, 1, 1.0):
            raise ValueError("evaluation has an unknown task or nonbinary reward")
        grouped[task_id].append(row)
    if set(grouped) != set(metadata):
        raise ValueError("evaluation task coverage is incomplete")
    tasks = {}
    for task_id, items in sorted(grouped.items()):
        logical = [row.get("attempt") for row in items]
        if sorted(logical) != list(range(attempts)):
            raise ValueError(f"evaluation attempts incomplete for {task_id}")
        passes = sum(int(row["reward"]) for row in items)
        sequences = Counter(
            " → ".join(
                str(call.get("name", ""))
                for call in row.get("tool_calls", [])
                if isinstance(call, dict)
            )
            for row in items
        )
        components = {}
        for key in sorted({key for row in items for key in row.get("verifier_components", {})}):
            components[key] = _mean(
                [float(row.get("verifier_components", {}).get(key, 0)) for row in items]
            )
        expected_targets = set(metadata[task_id].get("expected_write_targets", []))
        writes = []
        inspected_before_write = []
        unsupported = 0
        for row in items:
            inspected: set[str] = set()
            row_writes = []
            for call in row.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name", ""))
                arguments = call.get("arguments", {})
                invoice_id = (
                    str(arguments.get("invoice_id", "")) if isinstance(arguments, dict) else ""
                )
                if name == "get_invoice":
                    inspected.add(invoice_id)
                elif name in {"refund_invoice", "escalate_dispute"}:
                    row_writes.append(invoice_id)
                    inspected_before_write.append(invoice_id in inspected)
                else:
                    unsupported += 1
            writes.extend(row_writes)
        tasks[task_id] = {
            **metadata[task_id],
            "task_id": task_id,
            "attempts": attempts,
            "passes": passes,
            "pass_rate": passes / attempts,
            "wilson_95": wilson_interval(passes, attempts),
            "tool_call_rate": _mean([float(bool(row.get("tool_call_count", 0))) for row in items]),
            "mean_write_count": _mean(
                [
                    sum(
                        call.get("name") in {"refund_invoice", "escalate_dispute"}
                        for call in row.get("tool_calls", [])
                        if isinstance(call, dict)
                    )
                    for row in items
                ]
            ),
            "write_count": len(writes),
            "target_correctness_rate": (
                sum(target in expected_targets for target in writes) / len(writes)
                if writes
                else None
            ),
            "inspection_before_write_rate": _mean(
                [float(value) for value in inspected_before_write]
            ),
            "unsupported_tool_attempts_recorded": unsupported,
            "tool_sequence_histogram": dict(sorted(sequences.items())),
            "verifier_component_rates": components,
        }

    def grouped_rates(key: str) -> dict[str, dict[str, Any]]:
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in tasks.values():
            by[str(item[key])].append(item)
        return {
            name: {
                "task_count": len(values),
                "passes": sum(v["passes"] for v in values),
                "attempts": sum(v["attempts"] for v in values),
                "pass_rate": sum(v["passes"] for v in values) / sum(v["attempts"] for v in values),
            }
            for name, values in sorted(by.items())
        }

    family, prototype, archetype = (
        grouped_rates(key) for key in ("family_id", "prototype_id", "archetype")
    )
    all_passes = sum(item["passes"] for item in tasks.values())
    all_attempts = len(tasks) * attempts
    return {
        "tasks": tasks,
        "families": family,
        "prototypes": prototype,
        "archetypes": archetype,
        "micro_pass_rate": all_passes / all_attempts,
        "micro_wilson_95": wilson_interval(all_passes, all_attempts),
        "family_macro_pass_rate": _mean([v["pass_rate"] for v in family.values()]),
        "prototype_macro_pass_rate": _mean([v["pass_rate"] for v in prototype.values()]),
        "tool_call_rate": _mean([v["tool_call_rate"] for v in tasks.values()]),
    }


def compare(baseline: dict[str, Any], trained: dict[str, Any]) -> dict[str, Any]:
    if set(baseline["tasks"]) != set(trained["tasks"]):
        raise ValueError("policy task sets differ")
    tasks = {
        key: {
            "baseline": baseline["tasks"][key],
            "trained": trained["tasks"][key],
            "absolute_delta": trained["tasks"][key]["pass_rate"]
            - baseline["tasks"][key]["pass_rate"],
        }
        for key in sorted(baseline["tasks"])
    }
    prototypes = {
        key: {
            "baseline": baseline["prototypes"][key]["pass_rate"],
            "trained": trained["prototypes"][key]["pass_rate"],
            "delta": trained["prototypes"][key]["pass_rate"]
            - baseline["prototypes"][key]["pass_rate"],
        }
        for key in sorted(baseline["prototypes"])
    }
    return {
        "baseline": baseline,
        "trained": trained,
        "tasks": tasks,
        "prototypes": prototypes,
        "micro_delta": trained["micro_pass_rate"] - baseline["micro_pass_rate"],
        "prototype_macro_delta": trained["prototype_macro_pass_rate"]
        - baseline["prototype_macro_pass_rate"],
        "prototypes_improved": sum(v["delta"] > 0 for v in prototypes.values()),
        "prototypes_unchanged": sum(v["delta"] == 0 for v in prototypes.values()),
        "prototypes_worsened": sum(v["delta"] < 0 for v in prototypes.values()),
    }
