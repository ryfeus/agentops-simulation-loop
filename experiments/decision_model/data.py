"""Observed-state decision snapshots from verified, historical training demonstrations.

Teacher actions provide gold labels independently of the deliberately conservative rule
baseline. Hidden pre-existing escalations are not exposed by get_invoice; their teacher
no-op is therefore relabeled defer, never presented as an observable no_action label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ACTIONS = ("refund", "escalate", "no_action", "defer")
DESCRIPTIONS = {
    "refund": "Request a refund for the observed target invoice.",
    "escalate": "Request specialist dispute escalation for the observed target invoice.",
    "no_action": "The observed state and request require no further action for this invoice.",
    "defer": "Insufficient reliable observations or unclear intent; return to the existing agent.",
}
SERIALIZATION_VERSION = "billing-observed-v1"
INSTRUCTIONS = "What should happen next for this invoice? Choose exactly one action."
QUESTION = INSTRUCTIONS
OPTIONS = [[action, DESCRIPTIONS[action]] for action in ACTIONS]
POLICY = (
    "Inspect the target invoice before deciding. Refund a paid invoice when a refund or "
    "billing correction is requested. Disputed invoices must be escalated rather than "
    "refunded. Do not repeat a successfully completed refund or escalation. A missing "
    "invoice or an already refunded invoice requires no action. Information-only requests "
    "require no write. Use defer for incomplete observations, unclear requests, or history "
    "needed but not observed. get_invoice does not report prior escalation records. "
    "Decide for this target only; other invoices may still need work."
)
SOURCE_DIR = Path(".rl-smoke/phase12/runs/billing-phase12-v1-evidence-fix")
# Fixed prototype allocation before augmentation. Multiple family IDs share prototypes.
# All source tasks are historical TRAIN tasks; historical dev/test are never opened.
HELDOUT_GROUPS = {
    "dev": {"paid-refund-direct", "disputed-owner-override"},
    "calibration": {"target-similar-invoice-ids", "disputed-user-pressure"},
    "test": {
        "multi-paid-and-disputed",
        "already-escalated-direct",
        "missing-status",
        "status-paid-readonly",
    },
}


def serialize_state(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_state(
    request: str,
    target_invoice: str,
    observation: dict[str, Any] | None,
    prior_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Only supplied, actually observed facts belong in this input; no source metadata."""
    return copy.deepcopy(
        {
            "request": request,
            "target_invoice": target_invoice,
            "observation": observation,
            "prior_actions": prior_actions,
            "policy": POLICY,
        }
    )


def _completed(state: dict[str, Any]) -> bool:
    target = state["target_invoice"]
    for action in state["prior_actions"]:
        name = action.get("name")
        if name not in {"refund_invoice", "escalate_dispute"}:
            continue
        if action.get("arguments", {}).get("invoice_id") != target:
            continue
        result = action.get("result", {})
        payload = result.get("refund" if name == "refund_invoice" else "escalation", {})
        if (
            result.get("ok") is True
            and payload.get("success") is True
            and payload.get("invoice_id") == target
        ):
            return True
    return False


def _readonly(request: str) -> bool:
    text = request.split("\n")[0].lower()
    return any(
        s in text
        for s in (
            "what is the status",
            "tell me the status",
            "only for information",
            "for now just inspect",
            "changed status",
            "report its current billing status",
            "report the status",
            "make no changes",
            "a status answer",
        )
    )


def _excluded(request: str, target: str) -> bool:
    text = request.split("\n")[0].lower()
    escaped = re.escape(target.lower())
    if re.search(rf"(?:protecting|do not refund|leave) {escaped}(?:\b|$)", text):
        return True
    only = re.search(r"only (inv-\d+) has the wrong charge", text)
    if only and only.group(1) != target.lower():
        return True
    ids = set(re.findall(r"inv-\d+", text))
    return "that bill alone" in text and bool(ids) and target.lower() not in ids


def rule_action(state: dict[str, Any]) -> str:
    """Conservative structured baseline; uses no teacher labels or scenario metadata."""
    obs = state.get("observation")
    if _completed(state):
        return "no_action"
    if not obs:
        return "defer"
    if obs.get("ok") is False:
        return "no_action" if obs.get("error_type") == "InvoiceNotFoundError" else "defer"
    invoice = obs.get("invoice", {})
    if obs.get("ok") is not True or invoice.get("id") != state["target_invoice"]:
        return "defer"
    status = invoice.get("status")
    if status == "refunded":
        return "no_action"
    request = state["request"].split("\n")[0].lower()
    if _readonly(request) or _excluded(request, state["target_invoice"]):
        return "no_action"
    # These requests require escalation history absent from the tool observation.
    if any(
        s in request
        for s in (
            "already been sent",
            "already reached review",
            "prior specialist",
            "existing escalation",
            "check its history",
            "following up on specialist",
            "escalate",
            "the escalation concerns",
        )
    ):
        return "defer"
    if status == "disputed":
        return "escalate"
    if status == "paid" and re.search(r"refund|reverse|correct|wrong charge", request):
        return "refund"
    return "defer"


def prohibited_actions(state: dict[str, Any]) -> list[str]:
    obs = state.get("observation") or {}
    if _completed(state) or obs.get("error_type") == "InvoiceNotFoundError":
        return ["refund", "escalate"]
    status = obs.get("invoice", {}).get("status")
    if (
        status == "refunded"
        or _readonly(state["request"])
        or _excluded(state["request"], state["target_invoice"])
    ):
        return ["refund", "escalate"]
    return ["refund"] if status == "disputed" else []


def _validate_demonstration(record: dict[str, Any], allowed: dict[str, Any]) -> None:
    if not (
        record.get("task_id") in allowed
        and record.get("reward") == 1.0
        and record.get("source") == "executed-billing-tools"
        and record.get("complete") is True
        and record.get("truncated") is False
    ):
        raise ValueError("unverified or non-training demonstration")
    components = record.get("verifier_components", {})
    if not components or any(value != 1.0 for value in components.values()):
        raise ValueError("demonstration has failed verifier components")
    messages = record["messages"]
    if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        raise ValueError("incomplete conversation")
    calls = []
    for index in range(1, len(messages) - 1, 2):
        assistant, tool = messages[index : index + 2]
        call = assistant["tool_calls"][0]["function"]
        if assistant["role"] != "assistant" or tool["role"] != "tool":
            raise ValueError("unpaired tool observation")
        if tool["name"] != call["name"]:
            raise ValueError("tool observation mismatch")
        result = json.loads(tool["content"])
        if result.get("ok") is not True and not (
            call["name"] == "get_invoice" and result.get("error_type") == "InvoiceNotFoundError"
        ):
            raise ValueError("unexpected teacher tool failure")
        calls.append(call)
    if not calls or calls != record["tool_calls"]:
        raise ValueError("tool trajectory differs from evidence")


def split_groups(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        group: next((split for split, groups in HELDOUT_GROUPS.items() if group in groups), "train")
        for group in sorted({entry["prototype_id"] for entry in metadata.values()})
    }


def remap_options(row: dict[str, Any], order: list[int]) -> dict[str, Any]:
    if sorted(order) != list(range(len(ACTIONS))):
        raise ValueError("option order must be a permutation")
    result = copy.deepcopy(row)
    result["options"] = [row["options"][i] for i in order]
    result["label"] = order.index(row["label"])
    return result


def extract_records(
    demonstrations: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    groups = split_groups(metadata)
    result = []
    rng = random.Random(seed)
    for record in sorted(demonstrations, key=lambda row: row["task_id"]):
        _validate_demonstration(record, metadata)
        task = record["task_id"]
        group = metadata[task]["prototype_id"]
        split = groups[group]
        request = record["messages"][0]["content"]
        observations: dict[str, dict[str, Any]] = {}
        prior: list[dict[str, Any]] = []
        calls = record["tool_calls"]
        # Teacher future actions are labels only; never put them into model state.
        teacher_actions = {
            call["arguments"]["invoice_id"]: (
                "refund" if call["name"] == "refund_invoice" else "escalate"
            )
            for call in calls
            if call["name"] in {"refund_invoice", "escalate_dispute"}
        }

        def add(
            state: dict[str, Any],
            action: str,
            boundary: str,
            index: int,
            reason: str,
            task: str = task,
            group: str = group,
            split: str = split,
        ) -> None:
            row = {
                "kind": "choice",
                "state": serialize_state(state),
                "instructions": INSTRUCTIONS,
                "options": [[a, DESCRIPTIONS[a]] for a in ACTIONS],
                "label": ACTIONS.index(action),
                "example_id": f"{task}:{index}:{boundary}",
                "task_id": task,
                "group_id": group,
                "family_id": metadata[task]["family_id"],
                "split": split,
                "label_action": action,
                "label_basis": reason,
                "boundary": boundary,
                "source_message_index": index,
                "historical_split": "train",
                "serialization_version": SERIALIZATION_VERSION,
                "prohibited_actions": prohibited_actions(state),
            }
            if split == "train":
                order = list(range(len(ACTIONS)))
                rng.shuffle(order)
                row = remap_options(row, order)
            result.append(row)

        for index in range(1, len(record["messages"]) - 1, 2):
            call = record["messages"][index]["tool_calls"][0]["function"]
            observation = json.loads(record["messages"][index + 1]["content"])
            target = call["arguments"]["invoice_id"]
            if call["name"] == "get_invoice":
                observations[target] = observation
                state = make_state(request, target, observation, prior)
                if target in teacher_actions:
                    action, basis = teacher_actions[target], "verified_teacher_action"
                elif (
                    observation.get("error_type") == "InvoiceNotFoundError"
                    or observation.get("invoice", {}).get("status") == "refunded"
                    or _readonly(request)
                    or _excluded(request, target)
                ):
                    action, basis = "no_action", "verified_teacher_observable_noop"
                else:
                    action, basis = "defer", "teacher_noop_requires_unobserved_history"
                add(state, action, "after_inspection", index + 1, basis)
                add(
                    make_state(request, target, None, prior),
                    "defer",
                    "missing_observation",
                    index + 1,
                    "observation_ablation",
                )
            else:
                prior.append({**copy.deepcopy(call), "result": observation})
                state = make_state(request, target, observations[target], prior)
                if not _completed(state):
                    raise ValueError("teacher write did not observably succeed")
                add(
                    state,
                    "no_action",
                    "after_completed_action",
                    index + 1,
                    "observed_successful_action",
                )
    resolve_observable_conflicts(result)
    validate_records(result)
    return result, groups


def resolve_observable_conflicts(records: list[dict[str, Any]]) -> None:
    """Identical observations with differing labels must abstain symmetrically."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[row["state"]].append(row)
    for bucket in buckets.values():
        if len({row["label_action"] for row in bucket}) > 1:
            for row in bucket:
                row["original_label_action"] = row["label_action"]
                row["label_action"] = "defer"
                row["label"] = next(
                    i for i, option in enumerate(row["options"]) if option[0] == "defer"
                )
                row["label_basis"] = "conflicting_observable_labels"


def validate_records(records: list[dict[str, Any]]) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    identifiers = set()
    for row in records:
        group_splits[row["group_id"]].add(row["split"])
        if row["example_id"] in identifiers:
            raise ValueError("duplicate example identity")
        identifiers.add(row["example_id"])
        if row["kind"] != "choice" or row["historical_split"] != "train":
            raise ValueError("invalid source split or upstream kind")
        if sorted(option[0] for option in row["options"]) != sorted(ACTIONS):
            raise ValueError("invalid option set")
        if row["options"][row["label"]][0] != row["label_action"]:
            raise ValueError("label mapping mismatch")
        state = json.loads(row["state"])
        if set(state) != {"request", "target_invoice", "observation", "prior_actions", "policy"}:
            raise ValueError("non-observed metadata in model state")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("prototype split leakage")


def baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions, latencies = [], []
    confusion = {a: dict.fromkeys(ACTIONS, 0) for a in ACTIONS}
    prohibited = 0
    for row in records:
        state = json.loads(row["state"])
        start = time.perf_counter()
        prediction = rule_action(state)
        latencies.append((time.perf_counter() - start) * 1000)
        confusion[row["label_action"]][prediction] += 1
        prohibited += prediction in row["prohibited_actions"]
        predictions.append(
            {
                "example_id": row["example_id"],
                "task_id": row["task_id"],
                "gold": row["label_action"],
                "prediction": prediction,
                "action": prediction,
                "group_id": row["group_id"],
                "split": row["split"],
                "label_action": row["label_action"],
                "probabilities": {a: float(a == prediction) for a in ACTIONS},
                "confidence": 1.0,
                "input_tokens": None,
                "latency_ms": latencies[-1],
                "error": None,
            }
        )
    recalls, f1s = {}, []
    for action in ACTIONS:
        tp = confusion[action][action]
        support = sum(confusion[action].values())
        predicted = sum(confusion[a][action] for a in ACTIONS)
        recalls[action] = tp / support if support else None
        f1s.append(2 * tp / (support + predicted) if support + predicted else 0.0)
    ordered = sorted(latencies)
    return {
        "count": len(records),
        "task_count": len({r["task_id"] for r in records}),
        "group_count": len({r["group_id"] for r in records}),
        "accuracy": sum(p["gold"] == p["prediction"] for p in predictions) / len(records),
        "macro_f1": sum(f1s) / len(ACTIONS),
        "per_class_recall": recalls,
        "confusion_matrix": confusion,
        "prohibited_proposals": prohibited,
        "defer_rate": sum(p["prediction"] == "defer" for p in predictions) / len(records),
        "warm_latency_ms_p50": ordered[int((len(ordered) - 1) * 0.5)],
        "warm_latency_ms_p95": ordered[int((len(ordered) - 1) * 0.95)],
        "predictions": predictions,
    }


def build(source: Path, output: Path, *, seed: int = 17) -> dict[str, Any]:
    demonstrations_path = source / "demonstrations" / "demonstrations.jsonl"
    metadata_path = source / "corpus-manifest.json"
    demonstrations = [json.loads(line) for line in demonstrations_path.read_text().splitlines()]
    metadata = json.loads(metadata_path.read_text())["roles"]["train"]
    records, groups = extract_records(demonstrations, metadata, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split in ("train", "dev", "calibration", "test"):
        subset = [row for row in records if row["split"] == split]
        _write_jsonl(output / f"{split}.jsonl", subset)
        manifests[split] = {
            "rows": len(subset),
            "classes": dict(Counter(row["label_action"] for row in subset)),
            "groups": len({r["group_id"] for r in subset}),
        }
    _write_jsonl(output / "dataset.jsonl", records)
    # Only deterministic replay is evaluated here; it cannot influence model selection.
    scores = {
        split: baseline([r for r in records if r["split"] == split])
        for split in manifests
        if manifests[split]["rows"]
    }
    (output / "rule-baseline.json").write_text(json.dumps(scores, indent=2) + "\n")
    _write_jsonl(
        output / "rule-predictions.jsonl",
        [row for score in scores.values() for row in score["predictions"]],
    )
    visible_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        visible_splits[row["state"]].add(row["split"])
    manifest = {
        "observable_state_audit": {
            "unique_states": len(visible_splits),
            "cross_split_exact_duplicates": sum(len(v) > 1 for v in visible_splits.values()),
            "conflict_rows_deferred": sum(
                r["label_basis"] == "conflicting_observable_labels" for r in records
            ),
            "method": "Exact full model-state equality; does not certify semantic identifiability.",
        },
        "serialization_version": SERIALIZATION_VERSION,
        "seed": seed,
        "source_demonstrations": str(demonstrations_path),
        "demonstrations": len(demonstrations),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (demonstrations_path, metadata_path)
        },
        "dataset_sha256": hashlib.sha256((output / "dataset.jsonl").read_bytes()).hexdigest(),
        "group_to_split": groups,
        "splits": manifests,
        "historical_source_split": "train",
        "label_bases": dict(Counter(r["label_basis"] for r in records)),
        "limitations": [
            "Limited in-distribution diagnostic; all 16 prototypes were historically train.",
            "Calibration and dev each contain only two independent prototypes.",
            "Missing observations are ablations; completed actions use actual tool results.",
            "get_invoice hides prior escalations: teacher no-ops needing history become defer.",
            "Visible states cannot universally distinguish hidden prior escalation; text shortcuts "
            "may explain accuracy. Replay labels do not establish general billing safety.",
            "Prototype grouping preserves families but does not prove semantic independence.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
