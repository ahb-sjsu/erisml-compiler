"""Pattern-based maxim extractor.

A Kantian maxim is *the action under a particular description* — the
principle on which the agent is acting. Kant's universalizability
test operates on the maxim, not on the act. ("I lie when the lie
spares feelings" is a different maxim from "I lie when convenient,"
and they may universalize differently.)

The pre-v0.8.x heuristic derived `action_kind` from the dominant
`ethical_fact.kind`. That's a coarse proxy: it produces e.g.
`action_kind="deceive"` whenever a deception fact fires, regardless
of whether the prose actually frames the act as deception or as
self-protection. This module replaces that proxy with a pattern-
based reader of verb phrases + purpose clauses.

The output remains a `Maxim` Pydantic model (defined in
`projections/substrate.py`). The extractor produces both the
`action_kind` (verb-phrase normalised) and the `purpose` (what the
agent is trying to achieve), and constructs a human-readable
`description` of the form `"<action_kind> to <purpose>"`.

What this is:
  - Deterministic, offline, runs in tens of microseconds.
  - Reads regex pattern libraries for verbs + purpose clauses.
  - Identifies the agent (first-person pronoun → "self";
    profession noun → that stakeholder; explicit name → resolve
    against the stakeholder list).
  - Falls back to the kind-based heuristic only when prose
    extraction finds nothing.

What this is NOT:
  - Not LLM-based. An LLM extractor with maxim prompting is a
    natural Tier 3 upgrade (future work).
  - Not the canonical solution. Real maxim extraction requires
    natural-language inference (the agent's *implicit* purpose
    often isn't in the prose at all). v1 extracts what's stated;
    where the prose doesn't surface the purpose, we record
    `purpose=None` rather than guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from erisml_compiler.projections.substrate import Maxim


# ---------------------------------------------------- verb pattern library
#
# Each pattern maps to a normalised `action_kind` string consistent
# with the action_kinds used by the deontic universalizability gate.
# Order matters: more specific patterns first.

_VERB_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Deception family
    (re.compile(r"\b(lied?|lying|deceiv(?:e|ed|ing)|misled?|conceal(?:ed|ing)?|"
                r"withhold(?:ing)?|withheld|hide|hid|hidden|cover(?:ed)? up)\b",
                re.IGNORECASE), "deceive"),
    # Promise / commitment family
    (re.compile(r"\b(vow(?:ed|s)?|promise[ds]?|swore|swear|pledge[ds]?|"
                r"undertake[ns]?|undertook|contracted to)\b",
                re.IGNORECASE), "make_or_keep_commitment"),
    # Breaking promises
    (re.compile(r"\b(broke|breaking|break)\s+(?:my|the|a|her|his)?\s*(?:promise|vow|oath|word)\b",
                re.IGNORECASE), "break_commitment"),
    # Protection / care family
    (re.compile(r"\b(protect(?:ed|ing|s)?|shelter(?:ed|ing|s)?|shield(?:ed|ing|s)?|"
                r"defend(?:ed|ing|s)?|safeguard(?:ed|ing|s)?|"
                r"care for|caring for|look(?:ed|ing)? after)\b",
                re.IGNORECASE), "protect"),
    # Harm family
    (re.compile(r"\b(harm(?:ed|ing|s)?|hurt(?:ing|s)?|injur(?:ed|ing|es)?|"
                r"kill(?:ed|ing|s)?|murder(?:ed|ing|s)?|attack(?:ed|ing|s)?)\b",
                re.IGNORECASE), "inflict_harm"),
    # Coercion family
    (re.compile(r"\b(coerc(?:ed|ing|es)?|threat(?:en)?(?:ed|ing|s)?|compel(?:led|s|ling)?|"
                r"force[ds]?|forc(?:ing|es))\b",
                re.IGNORECASE), "coerce"),
    # Externality / risk imposition
    (re.compile(r"\b(impos(?:e|ed|ing)|exposed?|exposing|risk(?:ed|ing|s)?)\s+"
                r"(?:.*\s)?(?:on|upon|to)\s+(?:non[- ]consenting|the public|"
                r"third part(?:y|ies)|bystanders?)\b",
                re.IGNORECASE), "impose_externality"),
    # Disclosure / whistleblowing
    (re.compile(r"\b(disclos(?:e|ed|ing)|expos(?:e|ed|ing)|reveal(?:ed|ing|s)?|"
                r"report(?:ed|ing|s)?\s+(?:to)?|blow(?:ing)? the whistle)\b",
                re.IGNORECASE), "disclose"),
    # Cheating
    (re.compile(r"\b(cheat(?:ed|ing|s)?|defraud(?:ed|ing|s)?|swindl(?:e|ed|ing)|"
                r"steal(?:ing|s)?|stole|stolen)\b",
                re.IGNORECASE), "cheat"),
    # Refusal / abstention
    (re.compile(r"\b(refus(?:e|ed|ing)|decline[ds]?|declining|abstain(?:ed|ing|s)?)\b",
                re.IGNORECASE), "refuse"),
    # Help / aid
    (re.compile(r"\b(help(?:ed|ing|s)?|aid(?:ed|ing|s)?|assist(?:ed|ing|s)?|"
                r"support(?:ed|ing|s)?)\b",
                re.IGNORECASE), "help"),
    # Instrumental use (caught after more specific patterns above)
    (re.compile(r"\b(use|using|used|exploit(?:ed|ing|s)?|"
                r"instrumentalis(?:e|ed|ing))\b",
                re.IGNORECASE), "use_as_means"),
]


# ---------------------------------------------------- purpose clause patterns

_PURPOSE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:in order |so as |so that |so |in order|so it would |so it could |to )"
               r"(?:to |that )?([a-zA-Z][^,.;!?\n]{3,80})",
               re.IGNORECASE),
    re.compile(r"\bfor the sake of\s+([a-zA-Z][^,.;!?\n]{3,60})", re.IGNORECASE),
    re.compile(r"\bto (protect|save|spare|shield|prevent|stop|avoid|hide|conceal|warn|honor|"
               r"honour|keep|maintain|preserve|achieve|secure|advance|fulfil|fulfill|defend|"
               r"comply with|escape|expose|disclose|reveal)\s+([a-zA-Z][^,.;!?\n]{0,60})",
               re.IGNORECASE),
]


# ---------------------------------------------------- first-person / agent patterns

_FIRST_PERSON = re.compile(r"\b(I|me|my|mine|we|us|our|ours)\b")
_PROFESSION = re.compile(
    r"\b(?:as (?:a |an |the )?)?(doctor|nurse|physician|lawyer|attorney|engineer|"
    r"officer|soldier|employee|teacher|professor|priest|minister|judge|analyst)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------- mere-means proxy patterns

_MERE_MEANS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:use|using|used|exploit(?:ed|ing|s)?|instrumentalis(?:e|ed|ing))\s+"
               r"(?:the |her |his |their |these )?([a-zA-Z][\w\s]{0,40})\s+"
               r"(?:as|for|to)\s+(?:a |an |the )?(means|tool|leverage|gain|profit)\b",
               re.IGNORECASE),
]


@dataclass(frozen=True)
class MaximExtractionEvidence:
    """Debug-trail for the maxim extraction step."""

    matched_verb: str | None
    matched_verb_action_kind: str | None
    matched_purpose_phrase: str | None
    agent_evidence: str | None  # "first_person" | "profession:doctor" | "stakeholder:<id>"
    mere_means_hits: list[str]


def extract_maxim(
    text: str,
    *,
    stakeholders: list | None = None,
    fallback_action_kind: str | None = None,
) -> tuple[Maxim | None, MaximExtractionEvidence]:
    """Extract a maxim from prose. Returns (maxim, evidence_for_audit).

    `stakeholders` is the extractor's stakeholder list (Pydantic
    Stakeholder objects); used to resolve named agents.
    `fallback_action_kind` is what to use when no verb pattern fires
    (e.g. the kind-heuristic's result). When None and no pattern
    fires, returns (None, evidence).
    """
    if not text:
        return None, MaximExtractionEvidence(None, None, None, None, [])

    # 1. Find the dominant action verb.
    matched_verb: str | None = None
    action_kind: str | None = None
    for pattern, kind in _VERB_PATTERNS:
        m = pattern.search(text)
        if m:
            matched_verb = m.group(0)
            action_kind = kind
            break

    if action_kind is None:
        if fallback_action_kind is None:
            return None, MaximExtractionEvidence(None, None, None, None, [])
        action_kind = fallback_action_kind

    # 2. Find a purpose clause if present.
    matched_purpose: str | None = None
    purpose: str | None = None
    for pattern in _PURPOSE_PATTERNS:
        m = pattern.search(text)
        if m:
            matched_purpose = m.group(0)
            purpose = (m.group(1) if pattern.groups >= 1 else "").strip()
            purpose = _normalise_purpose(purpose)
            if purpose:
                break

    # 3. Identify the agent.
    agent_id, agent_evidence = _identify_agent(text, stakeholders or [])

    # 4. Mere-means evidence: who does the prose say is being used?
    mere_means_subjects: dict[str, str] = {}
    mere_means_hits: list[str] = []
    for pattern in _MERE_MEANS_PATTERNS:
        for m in pattern.finditer(text):
            hit = m.group(0)
            mere_means_hits.append(hit)
            target = (m.group(1) or "").strip().lower()
            # Try to map target to a stakeholder id.
            sid = _resolve_target_to_stakeholder(target, stakeholders or [])
            if sid:
                mere_means_subjects[sid] = "mere_means"

    # 5. Construct the description.
    parts = [action_kind.replace("_", " ")]
    if purpose:
        parts.append(f"to {purpose}")
    description = " ".join(parts)

    maxim = Maxim(
        description=description,
        agent_id=agent_id,
        action_kind=action_kind,
        purpose=purpose,
        treats_persons_as=mere_means_subjects,
    )
    evidence = MaximExtractionEvidence(
        matched_verb=matched_verb,
        matched_verb_action_kind=action_kind if matched_verb else None,
        matched_purpose_phrase=matched_purpose,
        agent_evidence=agent_evidence,
        mere_means_hits=mere_means_hits,
    )
    return maxim, evidence


# ---------------------------------------------------- helpers


def _normalise_purpose(phrase: str) -> str | None:
    """Trim filler words and articles from a purpose phrase."""
    if not phrase:
        return None
    p = phrase.strip().rstrip(".,;:!?")
    p = re.sub(r"^(?:the |a |an |my |our )", "", p, flags=re.IGNORECASE)
    p = p.strip()
    return p or None


def _identify_agent(text: str, stakeholders: list) -> tuple[str | None, str | None]:
    """Find a stakeholder id to attribute the maxim's agency to."""
    # First-person prose → "self" (matches RuleExtractor's default stakeholder id).
    if _FIRST_PERSON.search(text):
        if any(getattr(s, "id", None) == "self" for s in stakeholders):
            return "self", "first_person"
        return "self", "first_person"

    # Profession mention → match the first stakeholder with that role/label.
    m = _PROFESSION.search(text)
    prof_label: str | None = None
    if m:
        prof = m.group(1).lower()
        prof_label = prof
        for s in stakeholders:
            label_l = (getattr(s, "label", "") or "").lower()
            if prof in label_l:
                return s.id, f"profession:{prof}"

    # Otherwise, the first stakeholder with role='agent' wins.
    # When a profession was named but couldn't be resolved, attribute
    # to the agent-role stakeholder and note it in the evidence string.
    for s in stakeholders:
        roles = getattr(s, "roles", None) or []
        if "agent" in roles:
            tag = f"agent-role:{s.id}"
            if prof_label:
                tag = f"profession:{prof_label}->{tag}"
            return s.id, tag

    if prof_label:
        return None, f"profession:{prof_label} (no matching stakeholder, no agent-role fallback)"
    return None, None


def _resolve_target_to_stakeholder(target: str, stakeholders: list) -> str | None:
    """Match a free-text target phrase to a stakeholder id by label."""
    if not target:
        return None
    target_l = target.lower()
    for s in stakeholders:
        label_l = (getattr(s, "label", "") or "").lower()
        if label_l and label_l in target_l:
            return s.id
        sid_l = (getattr(s, "id", "") or "").lower()
        if sid_l and sid_l in target_l:
            return s.id
    return None
