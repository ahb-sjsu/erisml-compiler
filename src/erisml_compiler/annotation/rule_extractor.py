"""RuleExtractor: Tier 2 deterministic pattern-based extraction.

This is the **production Tier-2 extractor**: regex + keyword pattern library,
no network, no LLM. It is honest about its limits -- it can only detect what
its rules know how to find -- but it generalises beyond the three example
texts, runs offline, and is fully deterministic.

The rules are organised by concern:
    - stakeholder detection: pronouns, named-entity-like patterns
    - commitment detection: "vow", "promise", "swore", "contracted to"
    - coercion detection: "threat", "must", "will kill", "or else"
    - externality detection: collective harm targets ("village", "town", "city", "population")
    - role-duty detection: profession nouns (doctor, lawyer, employee)

Phase 2 will augment this with a learned classifier; Phase 3 will replace it
with an LLM extractor for cases where rules fail.
"""

from __future__ import annotations

import re

from erisml_compiler.annotation.base import Extractor, ExtractorResult
from erisml_compiler.ir.schemas import (
    Commitment,
    EthicalFact,
    Event,
    Stakeholder,
)

# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

COMMITMENT_VERBS = re.compile(
    r"\b(vow(?:ed|s)?|promise[ds]?|swore|swear|pledge[ds]?|"
    r"undertake[ns]?|undertook|contracted to)\b",
    flags=re.IGNORECASE,
)

COERCION_KEYWORDS = re.compile(
    r"\b(threat(?:en)?(?:ed|s)?|or else|will (?:kill|murder|destroy|punish|"
    r"harm|hurt)|forced (?:to|us)|under (?:duress|threat)|compel(?:led|s)?)\b",
    flags=re.IGNORECASE,
)

DECEPTION_KEYWORDS = re.compile(
    r"\b(lied?|lying|deceiv(?:ed|ing)?|conceal(?:ed|ing)?|hide|hid|" r"hidden|withheld|misled?)\b",
    flags=re.IGNORECASE,
)

COLLECTIVE_TARGETS = re.compile(
    r"\b(village|town|city|community|population|public|company|firm|"
    r"organisation|organization|society|nation|people)\b",
    flags=re.IGNORECASE,
)

PROFESSION_KEYWORDS = re.compile(
    r"\b(doctor|physician|nurse|lawyer|attorney|priest|clergy|"
    r"employee|officer|teacher|engineer|analyst)\b",
    flags=re.IGNORECASE,
)

VULNERABILITY_KEYWORDS = re.compile(
    r"\b(refugees?|innocents?|children|elderly|disabled|patients?|" r"hidden|defenseless)\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def _graph_from_extractor_result(result):
    """Build a MoralGraph from an in-progress ExtractorResult. Mirrors
    `ir.graph.graph_from_flat` but takes the result object directly so
    we don't need a full CompilerIR. Future LLM extractors that
    construct the graph natively will skip this entirely."""
    from erisml_compiler.ir.graph import graph_from_flat
    from types import SimpleNamespace

    pseudo_ir = SimpleNamespace(
        stakeholders=result.stakeholders,
        events=result.events,
        commitments=result.commitments,
        norms=result.norms,
        ethical_facts=result.ethical_facts,
        relations=result.relations,
    )
    return graph_from_flat(pseudo_ir)


class RuleExtractor(Extractor):
    """Pattern-based extractor. Deterministic, offline, generalisable in
    proportion to its rule library."""

    name = "rule"

    def extract(self, document, segments) -> ExtractorResult:
        result = ExtractorResult(extractor_metadata={"extractor": "rule"})

        # Generate a single 'speaker' / 'self' stakeholder per document.
        # (Phase 2: replace with proper NER.)
        result.stakeholders.append(
            Stakeholder(
                id="self",
                label="Document narrator/subject",
                type="individual",
                roles=["agent"],
                agency="full",
                vulnerability="moderate",
                consent_status="n/a",
                source_spans=[seg.segment_id + f":0-{len(seg.text)}" for seg in segments[:1]],
                confidence=0.6,
                requires_review=True,
            )
        )

        evt_counter = 0
        fact_counter = 0

        for seg in segments:
            seg_span_full = f"{seg.segment_id}:0-{len(seg.text)}"

            # Detect collective targets (third-party externality).
            for m in COLLECTIVE_TARGETS.finditer(seg.text):
                label = m.group(0).lower()
                sid = f"collective_{label}_{seg.segment_id}"
                if not any(s.id == sid for s in result.stakeholders):
                    result.stakeholders.append(
                        Stakeholder(
                            id=sid,
                            label=m.group(0),
                            type="community",
                            roles=["nonconsenting_third_party"],
                            agency="collective_limited",
                            vulnerability="high",
                            consent_status="not_obtained",
                            source_spans=[seg_span_full],
                            confidence=0.7,
                            requires_review=True,
                        )
                    )

            # Detect vulnerable parties.
            for m in VULNERABILITY_KEYWORDS.finditer(seg.text):
                sid = f"vulnerable_{m.group(0).lower()}_{seg.segment_id}"
                if not any(s.id == sid for s in result.stakeholders):
                    result.stakeholders.append(
                        Stakeholder(
                            id=sid,
                            label=m.group(0),
                            type="group",
                            roles=["patient", "dependent"],
                            agency="incapacitated",
                            vulnerability="extreme",
                            consent_status="n/a",
                            source_spans=[seg_span_full],
                            confidence=0.7,
                            requires_review=True,
                        )
                    )

            # Detect professions -> role_duty commitment.
            for m in PROFESSION_KEYWORDS.finditer(seg.text):
                profession = m.group(0).lower()
                evt_counter += 1
                cid = f"commitment_role_{profession}"
                if not any(c.id == cid for c in result.commitments):
                    result.commitments.append(
                        Commitment(
                            id=cid,
                            type="role_duty",
                            holder="self",
                            beneficiary=None,
                            content=f"role_duty_of_{profession}",
                            voluntariness="voluntary",
                            status="active_but_defeasible",
                            legitimacy="prima_facie_valid",
                            defeasibility_conditions=["higher_duty_conflict"],
                            source_spans=[seg_span_full],
                        )
                    )

            # Detect commitment verbs -> vow/promise.
            for m in COMMITMENT_VERBS.finditer(seg.text):
                evt_counter += 1
                eid = f"evt_{evt_counter:03d}"
                cid = f"commitment_{evt_counter:03d}"
                result.events.append(
                    Event(
                        id=eid,
                        time_index=evt_counter - 1,
                        type="commitment_made",
                        actor="self",
                        content=m.group(0).lower(),
                        source_spans=[seg_span_full],
                    )
                )
                result.commitments.append(
                    Commitment(
                        id=cid,
                        type="vow",
                        holder="self",
                        content=seg.text[:80],
                        voluntariness="voluntary",
                        created_at_event=eid,
                        status="active_but_defeasible",
                        legitimacy="prima_facie_valid",
                        defeasibility_conditions=["higher_duty_conflict"],
                        source_spans=[seg_span_full],
                    )
                )

            # Detect coercion.
            if COERCION_KEYWORDS.search(seg.text):
                fact_counter += 1
                result.ethical_facts.append(
                    EthicalFact(
                        id=f"fact_{fact_counter:03d}",
                        kind="coercion",
                        subjects=["self"],
                        description=(
                            f"Coercive language detected: '{seg.text[:80]}...'"
                            if len(seg.text) > 80
                            else f"Coercive language detected: '{seg.text}'"
                        ),
                        severity="grave",
                        confidence=0.7,
                        source_spans=[seg_span_full],
                    )
                )
                fact_counter += 1
                result.ethical_facts.append(
                    EthicalFact(
                        id=f"fact_{fact_counter:03d}",
                        kind="legitimacy",
                        subjects=[],
                        description="Authority issuing threats has defeasible or void legitimacy.",
                        severity="grave",
                        confidence=0.7,
                        source_spans=[seg_span_full],
                    )
                )

            # Detect collective harm -> externality fact.
            collective_present = COLLECTIVE_TARGETS.search(seg.text)
            harm_signal = re.search(
                r"\b(kill|murder|harm|destroy|hurt|threaten)\b",
                seg.text,
                flags=re.IGNORECASE,
            )
            if collective_present and harm_signal:
                fact_counter += 1
                collective_label = collective_present.group(0).lower()
                result.ethical_facts.append(
                    EthicalFact(
                        id=f"fact_{fact_counter:03d}",
                        kind="externality",
                        subjects=[
                            s.id for s in result.stakeholders if collective_label in s.label.lower()
                        ],
                        description=(
                            f"Catastrophic non-consensual risk to " f"{collective_label}."
                        ),
                        severity="catastrophic",
                        confidence=0.75,
                        source_spans=[seg_span_full],
                    )
                )

            # Detect deception.
            if DECEPTION_KEYWORDS.search(seg.text):
                fact_counter += 1
                result.ethical_facts.append(
                    EthicalFact(
                        id=f"fact_{fact_counter:03d}",
                        kind="deception",
                        subjects=["self"],
                        description="Deception or concealment language detected.",
                        severity="moderate",
                        confidence=0.65,
                        source_spans=[seg_span_full],
                    )
                )

            # Detect vulnerable -> care.
            if VULNERABILITY_KEYWORDS.search(seg.text):
                fact_counter += 1
                result.ethical_facts.append(
                    EthicalFact(
                        id=f"fact_{fact_counter:03d}",
                        kind="care",
                        subjects=[
                            s.id
                            for s in result.stakeholders
                            if s.vulnerability in ("high", "extreme")
                        ],
                        description="Protective duty toward vulnerable parties detected.",
                        severity="grave",
                        confidence=0.7,
                        source_spans=[seg_span_full],
                    )
                )

        # No canonical_form mapping at the rule layer; left to Phase 2
        # learned canonicaliser. Tag as 'unknown' to surface this.
        result.canonical_form = None

        # DAG-native emission: build the graph directly from the
        # flat lists we just populated. Conceptually this makes the
        # rule extractor graph-emitting at its result boundary; the
        # orchestrator will skip its own promotion step. The flat lists
        # remain populated for backward compat.
        result.graph = _graph_from_extractor_result(result)
        return result
