"""Conservative, request-only invoice authorization for the experimental runner.

This is a bounded grammar, not a general natural-language authorization engine.
Only direct, explicitly scoped commands grant writes. Unknown intent, unresolved
references and unsupported conditions grant nothing. No scenario expectations,
task metadata, or expected operations are consulted. Runtime observations may
further restrict an authorization; they can never add an authorized action.

Public API:
* authorization(request, target, targets): allowed_actions/status/reason/conditions.
* authorize_action(state, action, targets): allowed/reason/authorization.
* scoped_rule_action(state, targets, escalation_history=False): four-action decision.
* no_action_intent(request, target, targets): request-only completion intent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from experiments.decision_model import data, history

VARIANT = "invoice-scoped-authorization-v7"
SERIALIZATION_VERSION = "billing-invoice-authorization-v7"
ACTIONS = ("refund", "escalate")
_SUFFIX = "\n\nUse the billing tools to inspect the invoice and perform only the action "
_CONDITION = re.compile(r"\b(?:if|unless|until|after|before|once|when|where|whether)\b")
_UNCERTAIN = re.compile(
    r"\b(?:either|or|maybe|perhaps|hypothetical|hypothetically|would|might|could|"
    r"not decided|not sure|undecided|clarify|consider|considering)\b"
)
_NEGATED = re.compile(r"\b(?:do not|don't|never|not authoriz\w*|no permission|must not)\b")
_REVOKED = re.compile(
    r"^(?:actually\s*,?\s*)?(?:cancel|stop|pause|hold|withdraw|revoke|rescind|"
    r"i (?:withdraw|revoke|rescind))\b"
)
_READONLY = re.compile(
    r"status[- ]only|(?:status report|report.*status|provide.*status|status answer).*only|"
    r"(?:leave|remain|keep).*\b(?:unchanged|untouched)\b|change neither|"
    r"\bno (?:additional|further) (?:refund|escalation|changes)|"
    r"(?:confirm|check) (?:the |my )?(?:existing|previous|earlier|already recorded) refund\b|"
    r"(?:confirm|check) (?:the )?refund already recorded\b|"
    r"\bnot authoriz|"
    r"(?:just|only)\s+(?:report|inspect|check|provide)|"
    r"^actually\s*,?\s*(?:report|inspect|check|provide)"
)
_HISTORY_CONDITIONS = (
    r"(?:,?\s*but\s+)?only (?:if|where) (?:that (?:particular )?invoice|each invoice) "
    r"does not (?:yet )?have (?:a |an )?(?:review|case|escalation) recorded\s*$",
    r"only if each invoice has no escalation already recorded",
    r"only if (?:there is )?no (?:specialist )?(?:case|escalation) already exists",
    r"only if no (?:case|escalation) already exists",
    r"only if (?:there is )?no (?:existing|previous|prior) (?:case|escalation)",
    r"unless (?:a )?(?:specialist )?case already exists(?: for that invoice)?",
    r"unless (?:it is )?already escalated",
    r"unless (?:a )?(?:specialist )?case is already recorded for that invoice\s*$",
)


def _mentions(text: str, targets: Sequence[str]) -> list[str]:
    """Exact invoice tokens; the harness's order never resolves an ambiguous reference."""
    found = []
    for target in targets:
        match = re.search(r"(?<![\w-])" + re.escape(target.lower()) + r"(?![\w-])", text)
        if match:
            found.append((match.start(), target))
    return [target for _, target in sorted(found)]


def _negative_predicates(body: str, targets: Sequence[str]) -> list[str]:
    """Conjunctions before invoice objects retain their governing negative verb."""
    token = "|".join(re.escape(target.lower()) for target in targets)
    object_start = rf"(?!(?:invoice\s+)?(?:{token})(?![\w-]))" if token else ""
    return re.split(r"\s+(?:or|and)\s+" + object_start, body)


def _scoped_negations(clause: str, targets: Sequence[str]) -> list[str]:
    """Keep each explicit negative predicate with its own invoice objects.

    A list such as 'refund A or B' remains one prohibition; 'refund A or
    escalate B' becomes two. Unrecognized predicates still reach the existing
    fail-closed classifier. This transformation never creates positive commands.
    """
    command = re.fullmatch(
        r"(?P<prefix>(?:(?:for|regarding)\s+[^,]+,\s*)?(?:please\s+)?)"
        r"(?:do not|don't|never|must not)\s+(?P<body>.+)",
        clause,
    )
    if not command:
        return [clause]
    parts = [
        re.sub(r"^(?:do not|don't|never|must not)\s+", "", part)
        for part in _negative_predicates(command["body"], targets)
    ]
    # In 'do not refund or escalate A', the final explicit object list governs
    # both bare verbs. Do not infer shared scope for arbitrary omitted prose.
    if len(parts) > 1 and all(re.fullmatch(r"refund|escalate", p) for p in parts[:-1]):
        final = re.fullmatch(r"(?:refund|escalate)\s+(.+)", parts[-1])
        if final:
            objects = _objects(final[1].replace(" or ", " and "), targets, [])
            if objects:
                parts = [p + " " + " and ".join(objects) for p in parts[:-1]] + parts[-1:]
    # Only independent, explicit write predicates can be safely separated. Keep
    # conditional prose, generic history rules and unknown predicates together,
    # so detaching a fragment cannot erase its original scope or condition.
    for part in parts:
        predicate = re.fullmatch(r"(?:refund|escalate)\s+(.+)", part)
        if not predicate or not _objects(predicate[1].replace(" or ", " and "), targets, []):
            return [clause]
    return [command["prefix"] + "do not " + part for part in parts]


def _independent_commands(sentence: str, targets: Sequence[str]) -> list[str] | None:
    """Split only fully recognized commands, never recover a verb from arbitrary prose.

    Conditions, quotations and alternative intent govern the entire sentence. A
    leading negation/withdrawal cannot become a positive command after a comma.
    """
    if (
        any(mark in sentence for mark in ('"', "`", "“", "”", "\u2018", "\u2019"))
        or re.search(r"(?:^|\s)'|'(?:$|\s)", sentence)
        or _CONDITION.search(sentence)
        or _UNCERTAIN.search(sentence)
    ):
        return None
    sentence = sentence.strip()
    # A customer/invoice prefix comma belongs to the first command, not a delimiter.
    scope = re.match(r"(?:(?:please|kindly)\s+)?(?:for|regarding)\s+[^,]+,\s*", sentence)
    prefix = scope[0] if scope else ""
    body = sentence[len(prefix) :]
    parts = re.split(
        r",?\s+\b(?:and|but|while|whereas)\s+(?=(?:please\s+)?"
        r"(?:refund|reverse|escalate|do not|don't|never|must not)\b)|"
        r",\s*(?:and\s+)?(?=(?:please\s+)?(?:refund|reverse|escalate)\b)",
        body,
    )
    parts[0] = prefix + parts[0]
    if len(parts) < 2:
        return None
    parts = [part.strip(" ,:") for part in parts]
    if not all(_command(part, targets, [])[1] for part in parts[:-1]):
        return None
    last = parts[-1]
    if not _command(last, targets, [])[1]:
        negative = re.fullmatch(
            r"(?:please\s+)?(?:do not|don't|never|must not) "
            r"(?:refund|escalate) (.+)",
            last,
        )
        if not negative or not _objects(negative[1], targets, []):
            return None
    return parts


def _withdrawn_actions(clause: str, targets: Sequence[str]) -> set[str]:
    """Narrow only a complete, explicit action withdrawal; unknown forms deny both."""
    match = re.fullmatch(
        r"(?:actually\s*,?\s*)?(?:cancel|stop|pause|hold|withdraw|revoke|rescind|"
        r"i (?:withdraw|revoke|rescind)) permission to (refund|reverse|escalate) (.+)",
        clause,
    )
    if match and _objects(match[2], targets, []):
        return {"refund" if match[1] == "reverse" else match[1]}
    return set(ACTIONS)


def _counted_scope(clause: str, references: list[str]) -> list[str] | None:
    """An exact count resolves only the immediately preceding validated object list."""
    match = re.search(
        r"\bthose (two|three|four|five|six|seven|eight|nine|ten|[0-9]+) invoices\b", clause
    )
    if not match:
        return None
    words = dict(
        zip(
            ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
            range(2, 11),
            strict=True,
        )
    )
    count = words.get(match[1], int(match[1]) if match[1].isdigit() else 0)
    return references if count == len(references) else []


def _mask_quoted_text(text: str) -> str:
    """Keep quoted delimiters from promoting a quoted middle sentence to a command.

    Replace, rather than erase, the entire span so surrounding fragments cannot
    join into a new command. An unmatched opening quote consumes the remainder.
    Apostrophes inside words are not quote boundaries.
    """
    text = text.translate(str.maketrans({"“": '"', "”": '"', "\u2018": "'"}))
    output: list[str] = []
    active: str | None = None
    for index, char in enumerate(text):
        apostrophe = char == "'" and index > 0 and text[index - 1].isalnum()
        if active:
            if char == active and not (
                apostrophe and index + 1 < len(text) and text[index + 1].isalnum()
            ):
                active = None
            continue
        if char in {'"', "`", "'"} and not apostrophe:
            active = char
            output.append('"quoted text"')
        else:
            output.append(char)
    return "".join(output)


def _clauses(request: str, targets: Sequence[str] = ()) -> list[str]:
    request = _mask_quoted_text(request.split(_SUFFIX)[0].lower().replace("\u2019", "'"))
    # Split coordinated commands without confusing a coordinated invoice object list.
    clauses = []
    for sentence in re.split(r"[.;!?\n]+", request):
        # A correction marker is not part of its command. Do not remove markers
        # inside quotations, narrated speech, or an unsupported conditional.
        sentence = re.sub(r"^\s*correction\s*:\s*", "", sentence)
        independent = _independent_commands(sentence, targets)
        if independent:
            for part in independent:
                clauses.extend(_scoped_negations(part, targets))
            continue
        # Never detach a coordinated action from the negation/condition governing it.
        condition_text, known_condition, unsupported = _condition(sentence)
        typed_escalation_condition = (
            known_condition
            and not unsupported
            and not re.match(r"^\s*(?:only\s+)?(?:if|unless)\b", sentence)
            and "escalate" in condition_text
        )
        if not (
            _NEGATED.search(sentence)
            or _REVOKED.search(sentence.strip())
            or any(mark in sentence for mark in ('"', "`", "“", "”", "\u2018", "\u2019"))
            or re.search(r"(?:^|\s)'|'(?:$|\s)", sentence)
            or _UNCERTAIN.search(sentence)
        ) and (not _CONDITION.search(sentence) or typed_escalation_condition):
            proposed = re.sub(
                r"\b(?:and|but|while|whereas)\s+(?=(?:please\s+)?(?:refund|escalate|reverse|"
                r"resolve|correct|leave|do not|don't|never|inspect|report|provide|keep)\b)",
                ";",
                sentence,
            )
            first, _, _ = _condition(proposed.split(";")[0].strip(" ,:"))
            if _command(first, targets, [])[1]:
                sentence = proposed
        for part in sentence.split(";"):
            if part.strip():
                clauses.extend(_scoped_negations(part.strip(" ,:"), targets))
    return clauses


def _negative_actions(clause: str, targets: Sequence[str] = ()) -> set[str] | None:
    """Classify a negated predicate rather than denying every noun in its sentence.

    None denotes descriptive or benign timing prose. Empty denotes a state/history
    prohibition already enforced from observations, not revocation of all authority.
    Unknown imperative negation still denies both operations.
    """
    if re.search(r"\b(?:not authoriz\w*|no permission|i do not authorize)\b", clause):
        return set(ACTIONS)
    imperative = re.sub(r"^(?:for|regarding)\s+[^,]+,\s*", "", clause)
    command = re.match(r"^(?:please\s+)?(?:do not|don't|never|must not)\s+(.+)$", imperative)
    if not command:
        return None
    body = command.group(1)
    if re.match(r"(?:postpone|delay)\b", body):
        return None
    denied = set()
    understood = False
    for predicate in _negative_predicates(body, targets):
        predicate = predicate.strip()
        # These retain their observed-state predicates: disputed refunds are blocked
        # by authorize_action; existing escalations are never repeated.
        if re.match(r"refund (?:a |an |any |the )?disputed (?:charge|invoice)\b", predicate):
            understood = True
            continue
        if re.fullmatch(
            r"turn (?:a |an |any |the |either )?disputed (?:charge|invoice)s? "
            r"into (?:a )?refund",
            predicate,
        ):
            understood = True
            continue
        if re.match(
            r"(?:repeat|duplicate|create (?:a |an |the )?duplicate)\b", predicate
        ) and re.search(r"\b(?:refund|escalation|reviews?|cases?)\b", predicate):
            understood = True
            continue
        if re.match(r"refund\b", predicate):
            denied.add("refund")
            understood = True
        elif re.match(r"(?:escalate|request (?:a )?(?:specialist )?review)\b", predicate):
            denied.add("escalate")
            understood = True
        else:
            # An unrecognized negative operation must not silently disappear.
            denied.update(ACTIONS)
            understood = True
    return denied if understood else set(ACTIONS)


def _condition(clause: str) -> tuple[str, str | None, bool]:
    condition = None
    for pattern in _HISTORY_CONDITIONS:
        clause, count = re.subn(pattern, "", clause)
        if count:
            condition = "no_existing_escalation"
    return clause, condition, bool(_CONDITION.search(clause))


def _objects(body: str, targets: Sequence[str], references: list[str]) -> list[str]:
    """Read a contiguous direct-object list, not every invoice mentioned in prose."""
    body = body.strip()
    body = re.sub(
        r"^(?:(?:the )?(?:billing )?(?:issue|charge|problem) (?:for|on|with)|"
        r"(?:the )?invoices?)\s+",
        "",
        body,
    )
    pattern = "|".join(
        re.escape(target.lower()) for target in sorted(targets, key=len, reverse=True)
    )
    token = rf"(?:{pattern})(?![\w-])"
    match = re.match(rf"({token}(?:\s*(?:,\s*(?:and\s+)?|and\s+){token})*)(.*)$", body)
    if match:
        objects = _mentions(match.group(1), targets)
        tail = match.group(2).strip(" ,")
    else:
        pronoun = re.match(r"(?:it|that invoice|this invoice)\b(.*)$", body)
        if not pronoun or len(references) != 1:
            return []
        objects, tail = references, pronoun.group(1).strip(" ,")
    # Retaining an existing review on a directly commanded invoice is compatible
    # with its first refund. This descriptive tail cannot add another write target
    # or stand in for evidence that a refund already occurred.
    preservation = re.fullmatch(
        r"retaining the (?:earlier|existing|previous|prior) "
        r"(?:review|case|escalation) on (.+)",
        tail,
    )
    if preservation and preservation.group(1) in {target.lower() for target in objects}:
        tail = ""
    # An unsupported tail may change the speech act or impose a condition.
    if not re.fullmatch(
        r"(?:(?:now|only|please)|for (?:(?:specialist )?review"
        r"(?: for customer-[a-z0-9]+(?:-[a-z0-9]+)*)?|customer-[a-z0-9]+(?:-[a-z0-9]+)*))?",
        tail,
    ):
        return []
    return objects


def _command(
    clause: str, targets: Sequence[str], references: list[str]
) -> tuple[set[str], list[str]]:
    """Recognize only direct commands, excluding narrated/quoted action words."""
    if '"' in clause or "`" in clause or _UNCERTAIN.search(clause):
        return set(), []
    prefix = r"(?:(?:please|kindly|separately|independently|now)\s*,?\s+)*"
    # A leading customer or invoice scope is allowed, but never establishes a target
    # on its own unless the same clause actually commands an action.
    scope = r"(?:(?:for|regarding)\s+[^,]+,\s*)?"
    verb = re.match(prefix + scope + r"(refund|reverse|escalate|resolve|correct)\b(.*)$", clause)
    if verb:
        action, body = verb.groups()
        leading_text = clause[: verb.start(1)]
        leading_scope = re.search(r"(?:for|regarding)\s+([^,]+),\s*$", leading_text)
        if leading_scope and not (
            re.fullmatch(r"customer-[a-z0-9]+(?:-[a-z0-9]+)*", leading_scope[1])
            or _objects(leading_scope[1], targets, [])
        ):
            return set(), []
        leading_targets = _mentions(leading_text, targets)
        scoped = _objects(body, targets, leading_targets or references)
        if action in {"resolve", "correct"}:
            return set(ACTIONS), scoped
        return {"refund" if action == "reverse" else action}, scoped
    grant = re.match(
        prefix + r"(?:i authorize (?:a )?(refund|specialist review|dispute review) (?:of|for)|"
        r"my (refund) request is for)\s+(.+)$",
        clause,
    )
    if grant:
        noun = grant.group(1) or grant.group(2)
        return {"refund" if noun == "refund" else "escalate"}, _objects(
            grant.group(3), targets, references
        )
    return set(), []


def _review_preservation(clause: str, targets: Sequence[str]) -> tuple[list[str], bool]:
    """Recognize direct existing-review inspection/preservation, never a write grant."""
    adjective = r"(?:existing|earlier|previous|prior)"
    noun = r"(?:specialist )?(?:case|review|escalation)s?"
    preservation = (
        r"without (?:repeating (?:it|them)|"
        r"(?:opening|creating) (?:another|a new) (?:case|review|escalation))"
    )
    no_refund = r" or (?:a refund|issuing a refund|refunding (?:that invoice|those invoices))"
    match = re.fullmatch(
        rf"(?:(?:inspect|check) the {adjective} {noun} (?:on|for) (?P<inspect>.+?) "
        rf"{preservation}(?P<no_refund>{no_refund})?|"
        rf"preserve the {adjective} {noun} on (?P<preserve>.+)|"
        rf"(?:for|regarding) (?P<scope>[^,]+), (?:inspect|check) the {adjective} {noun} "
        rf"{preservation}(?P<scoped_no_refund>{no_refund})?)",
        clause,
    )
    if not match:
        return [], False
    subjects = match.group("inspect") or match.group("preserve") or match.group("scope")
    token = "(?:" + "|".join(re.escape(t.lower()) for t in targets) + ")"
    if not re.fullmatch(rf"{token}(?:\s*(?:,\s*(?:and\s+)?|and\s+){token})*", subjects):
        return [], False
    return _mentions(subjects, targets), bool(
        match.group("no_refund") or match.group("scoped_no_refund")
    )


def _restriction_subject(clause: str, targets: Sequence[str]) -> str:
    """Separate two bounded non-granting scope explanations from restrictions.

    Mentions inside a statement that another action remains authorized are not
    objects of the preceding withdrawal. Such statements only preserve an earlier
    grant; they never create one. Unrecognized wording stays conservatively intact.
    """
    subjects, no_refund = _review_preservation(clause, targets)
    if subjects:
        # A review-preservation request constrains escalation, not an independent
        # first refund. Observing an existing review is checked at completion time.
        restriction = "do not escalate " + " and ".join(subjects)
        if no_refund:
            restriction += " or refund " + " and ".join(subjects)
        return restriction
    exception = re.fullmatch(
        r"(?:the )?status[- ]only (?:inquiry|request|question) for (.+?) "
        r"does not (?:restrict|cancel|revoke) (?:the )?separate "
        r"(?:action|refund|review|escalation) (?:on|for) (.+)",
        clause,
    )
    if exception and all(_objects(part, targets, []) for part in exception.groups()):
        return "for " + exception.group(1) + ", report status only"
    preservation = re.fullmatch(
        r"(.+?),\s*(?:while|whereas|but) (?:the )?(?:refund|review|escalation) "
        r"(?:of|for|on) (.+?) remains authorized",
        clause,
    )
    if preservation:
        restriction, preserved = preservation.groups()
        if (
            (_REVOKED.search(restriction) or _NEGATED.search(restriction))
            and not _CONDITION.search(restriction)
            and not _UNCERTAIN.search(restriction)
            and _objects(preserved, targets, [])
        ):
            return restriction
    return clause


def authorization(request: str, target: str, targets: Sequence[str]) -> dict[str, Any]:
    """Return a target-specific request allowlist; unknown language never grants a write.

    no_existing_escalation is evaluated against complete, target-matched tool history.
    no_existing_refund requires the unavailable refund ledger, so it fails closed.
    Explicit denials override grants. Refund and escalation authorities are distinct.
    """
    known = list(dict.fromkeys(targets))
    result = {
        "allowed_actions": [],
        "status": "ambiguous",
        "reason": "no_clear_scoped_authority",
        "conditions": {},
    }
    if target not in known or target not in _mentions(request.lower(), known):
        return {**result, "reason": "target_not_explicitly_named"}
    grants: dict[str, str | None] = {}
    denied: set[str] = set()
    uncertain: set[str] = set()
    constraints: dict[str, str] = {}
    references: list[str] = []
    counted_references: list[str] = []
    first_mentions = _mentions(request.lower(), known)
    for original in _clauses(request, known):
        original = _restriction_subject(original, known)
        # An excluded object is not another recipient of the positive verb.
        target_pattern = "|".join(re.escape(item.lower()) for item in known)
        exclusion = re.search(
            rf"\b(?:not|except|excluding)\s+(?:invoice\s+)?(?=(?:{target_pattern})(?![\w-]))",
            original,
        )
        if exclusion:
            if target in _mentions(original[exclusion.start() :], known):
                denied.update(ACTIONS)
            original = original[: exclusion.start()].rstrip(" ,")
        explicit = _mentions(original, known)
        scoped = explicit
        counted = _counted_scope(original, counted_references)
        if counted is not None:
            # A counted referent plus explicit objects must constrain both groups;
            # unresolved counts cannot shrink a prohibition to the explicit suffix.
            scoped = list(dict.fromkeys(counted + explicit)) if counted else []
        elif not scoped:
            if re.search(r"\b(?:both|those two|either of those two)\b", original):
                scoped = references if len(references) == 2 else []
            elif re.search(r"\b(?:it|its|that invoice|this invoice|them)\b", original):
                scoped = references if len(references) == 1 else []
            elif "first two invoices" in original:
                scoped = first_mentions[:2]
        reference_clause, _, reference_unsupported = _condition(original)
        _, reference_objects = _command(reference_clause, known, [])
        counted_references = (
            reference_objects if not reference_unsupported and reference_objects == explicit else []
        )
        # Negative/readonly language can restrict scope but cannot introduce grants.
        if _READONLY.search(original) and target in (scoped or references or known):
            denied.update(ACTIONS)
        if _REVOKED.search(original):
            if target in (scoped or known):
                denied.update(_withdrawn_actions(original, known))
            if explicit:
                references = explicit
            continue
        if _NEGATED.search(original):
            negative = _negative_actions(original, known)
            negative_scope = scoped or known
            if negative is not None and target in negative_scope:
                denied.update(negative)
                if re.search(r"\b(?:repeat|duplicate)\b", original):
                    if re.search(r"\b(?:escalations?|reviews?|cases?)\b", original):
                        constraints["escalate"] = "no_existing_escalation"
                    if re.search(r"\brefunds?\b", original) and not re.search(
                        r"refund (?:a |an |any |the )?disputed", original
                    ):
                        constraints["refund"] = "no_existing_refund"
            if explicit:
                references = explicit
            continue
        clause, condition, unsupported = _condition(original)
        if unsupported or re.search(
            r"\b(?:wait|await|pending)\b.*\b(?:approval|permission|confirmation)\b", original
        ):
            restriction_scope = explicit or scoped or known
            absence_report = re.fullmatch(
                r"if absent, report (?:that|its absence|the absence) "
                r"without (?:substituting another (?:bill|invoice)|substitution)",
                original,
            )
            if absence_report and len(references) == 1:
                # A detached missing-invoice reporting condition has one explicit
                # antecedent. It cannot invalidate independent earlier commands.
                restriction_scope = references
            elif re.match(r"^(?:only\s+)?(?:if|after|before|once|when|until)\b", original):
                restriction_scope = list(set(restriction_scope) | set(references))
            if target in restriction_scope:
                uncertain.update(ACTIONS)
        if unsupported or _READONLY.search(original):
            if explicit:
                references = explicit
            continue
        actions, subjects = _command(clause, known, references)
        if condition and not actions and target in (explicit or scoped or references or known):
            # A detached condition must not leave a previous unconditional grant.
            if "escalate" in grants:
                grants["escalate"] = condition
            else:
                uncertain.update(ACTIONS)
        if target in subjects:
            for action in actions:
                # A history condition never silently becomes an unconditional refund.
                if condition and action != "escalate":
                    continue
                if action not in grants or condition:
                    grants[action] = condition
        if explicit:
            references = explicit
    grants.update(
        {action: condition for action, condition in constraints.items() if action in grants}
    )
    allowed = sorted(set(grants) - denied - uncertain)
    if allowed:
        return {
            "allowed_actions": allowed,
            "status": "authorized",
            "reason": "explicit_invoice_scoped_request",
            "conditions": {action: grants[action] for action in allowed if grants[action]},
        }
    if denied == set(ACTIONS) or (denied and not uncertain):
        return {**result, "status": "denied", "reason": "explicit_restriction"}
    if uncertain:
        return {**result, "reason": "unresolved_condition_or_scope"}
    return result


def no_action_intent(request: str, target: str, targets: Sequence[str]) -> dict[str, Any]:
    """Request-only completion intent; callers must still validate actual observations.

    An existing-review request requires complete, matching, nonempty review history
    before completion. It does not prove a case is resolved. Independent authorized
    writes take precedence. Partial write prohibitions alone do not imply completion.
    """
    auth = authorization(request, target, targets)
    if auth["allowed_actions"] or auth["reason"] == "unresolved_condition_or_scope":
        return {"kind": None}
    known = list(dict.fromkeys(targets))
    if target not in known:
        return {"kind": None}
    references: list[str] = []
    readonly = False
    existing_review = False
    denied_actions: set[str] = set()
    for clause in _clauses(request, known):
        if '"' in clause or "`" in clause:
            continue
        explicit = _mentions(clause, known)
        scope = explicit
        if not scope and re.search(r"\b(?:it|its|that invoice|this invoice|them)\b", clause):
            scope = references if len(references) == 1 else []
        if not _CONDITION.search(clause) and target in (scope or known) and _NEGATED.search(clause):
            denied_actions.update(_negative_actions(clause, known) or ())
        subjects, _ = _review_preservation(clause, known)
        if target in subjects:
            existing_review = True
        elif (
            not _CONDITION.search(clause)
            and target in (scope or known)
            and (
                _READONLY.search(clause)
                or _REVOKED.search(clause)
                or (_NEGATED.search(clause) and _negative_actions(clause, known) == set(ACTIONS))
            )
        ):
            readonly = True
        if explicit:
            references = explicit
    if existing_review:
        return {"kind": "existing_review"}
    return {"kind": "read_only" if readonly or denied_actions == set(ACTIONS) else None}


def make_state(
    request: str, target: str, observation: Any, prior_actions: list, targets: Sequence[str]
) -> dict:
    """Expose derived request authority as input, never as trusted executor authority."""
    state = history.make_state(request, target, observation, prior_actions)
    state["policy"] = (
        "Inspect this target invoice before deciding. Writes require invoice-specific "
        "request authority. A refund request does not authorize escalation. Refund only "
        "an authorized paid invoice; never refund a disputed invoice. Escalate only when "
        "the request authorizes escalation and no observed escalation already exists. "
        "Complete target-matched escalation history establishes whether a review exists; "
        "unavailable history is unknown. Refund history is unavailable. Conditions requiring "
        "unobserved history or unresolved approval require defer. Explicit read-only or held "
        "instructions, missing invoices, already refunded invoices, and completed operations "
        "require no_action. Unclear authority requires defer, not a write. Existing escalation "
        "does not itself forbid an authorized first refund of a paid invoice. Decide for this "
        "target only; other invoices may still require work."
    )
    state["authorization"] = authorization(request, target, targets)
    return state


def authorize_action(state: dict, action: str, targets: Sequence[str]) -> dict[str, Any]:
    """Execution guard: request authority AND actual observed policy must permit a write."""
    auth = authorization(state["request"], state["target_invoice"], targets)

    def decision(allowed: bool, reason: str) -> dict:
        return {"allowed": allowed, "reason": reason, "authorization": auth}

    if action not in ACTIONS or action not in auth["allowed_actions"]:
        return decision(False, "action_not_authorized_for_invoice")
    obs = state.get("observation") or {}
    invoice = obs.get("invoice") or {}
    if obs.get("ok") is not True or invoice.get("id") != state["target_invoice"]:
        return decision(False, "missing_or_mismatched_inspection")
    if data._completed(state):
        return decision(False, "already_completed_in_current_episode")
    status = invoice.get("status")
    if status not in {"paid", "disputed"}:
        return decision(False, "unsupported_or_completed_invoice_status")
    if action == "refund" and status != "paid":
        return decision(False, "disputed_invoice_cannot_be_refunded")
    if auth["conditions"].get(action) == "no_existing_refund":
        return decision(False, "refund_history_unavailable")
    records = history.escalation_records(state)
    if action == "escalate" and records:
        return decision(False, "escalation_already_exists")
    if auth["conditions"].get(action) == "no_existing_escalation":
        if records is None:
            return decision(False, "escalation_history_unavailable")
        if records:
            return decision(False, "escalation_already_exists")
    return decision(True, "authorized_and_observed_policy_allows")


def scoped_rule_action(
    state: dict, targets: Sequence[str], *, escalation_history: bool = False
) -> str:
    """Plan from the same allowlist used immediately before rule or LLM tool writes."""
    auth = authorization(state["request"], state["target_invoice"], targets)
    obs = state.get("observation") or {}
    intent = no_action_intent(state["request"], state["target_invoice"], targets)
    if intent["kind"] == "existing_review":
        matching_inspection = (
            obs.get("ok") is True and obs.get("invoice", {}).get("id") == state["target_invoice"]
        )
        return "no_action" if matching_inspection and history.escalation_records(state) else "defer"
    if data._completed(state) or obs.get("error_type") == "InvoiceNotFoundError":
        return "no_action"
    if obs.get("ok") is not True or obs.get("invoice", {}).get("id") != state["target_invoice"]:
        return "defer"
    status = obs["invoice"].get("status")
    if status == "refunded":
        return "no_action"
    if auth["status"] == "denied":
        return "no_action"
    if auth["status"] != "authorized":
        return "defer"
    actions = set(auth["allowed_actions"])
    if escalation_history and actions == {"escalate"} and history.escalation_records(state):
        return "no_action"
    preferred = "refund" if status == "paid" and "refund" in actions else "escalate"
    if preferred not in actions:
        return "defer"
    return preferred if authorize_action(state, preferred, targets)["allowed"] else "defer"
