"""MockExtractor: hand-curated IR for the three example texts.

This is **fixture data**, not a general extractor. It exists so the rest of
the pipeline can be tested, exercised, and demonstrated without LLM API
costs. Raises UnknownDocumentError if given a document outside its
fixture set.

Selection is by SHA-256 of the raw text so renaming a file does not change
behaviour but editing the text does.
"""
from __future__ import annotations

import hashlib
from typing import Callable

from erisml_compiler.annotation.base import (
    Extractor,
    ExtractorResult,
    UnknownDocumentError,
)
from erisml_compiler.ir.schemas import (
    Commitment,
    Conflict,
    EthicalFact,
    Event,
    Stakeholder,
)


# ---------------------------------------------------------------------------
# Fixture: Nazi attic (spec section 28)
# ---------------------------------------------------------------------------


def _nazi_attic() -> ExtractorResult:
    stakeholders = [
        Stakeholder(
            id="speaker", label="Villager (speaker)", type="individual",
            roles=["agent", "vow_holder", "protector"],
            agency="full", vulnerability="moderate",
            consent_status="n/a", source_spans=["seg_001:0-40"], confidence=0.95,
        ),
        Stakeholder(
            id="hidden_refugees", label="Hidden refugees", type="group",
            roles=["patient", "beneficiary", "dependent"],
            agency="incapacitated", vulnerability="extreme",
            consent_status="obtained", source_spans=["seg_001:30-65"], confidence=0.92,
        ),
        Stakeholder(
            id="nazis", label="Nazi soldiers", type="institution",
            roles=["coercer", "authority"],
            agency="full", vulnerability="low",
            consent_status="n/a", source_spans=["seg_001:50-78"], confidence=0.97,
        ),
        Stakeholder(
            id="village", label="The village", type="community",
            roles=["nonconsenting_third_party"],
            agency="collective_limited", vulnerability="high",
            consent_status="not_obtained", source_spans=["seg_002:60-77"], confidence=0.88,
        ),
    ]
    events = [
        Event(
            id="evt_001", time_index=0, type="vow_made", actor="speaker",
            target="hidden_refugees",
            content="conceal hiding place",
            source_spans=["seg_001:0-78"],
        ),
        Event(
            id="evt_002", time_index=1, type="threat_uttered", actor="nazis",
            target="village",
            content="murder entire village if lied to",
            source_spans=["seg_002:0-90"],
        ),
        Event(
            id="evt_003", time_index=2, type="demand_made", actor="nazis",
            target="speaker",
            content="reveal location of hidden refugees",
            source_spans=["seg_001:50-78"],
        ),
    ]
    commitments = [
        Commitment(
            id="commitment_001", type="vow", holder="speaker",
            beneficiary="hidden_refugees",
            content="conceal_hiding_place",
            voluntariness="voluntary",
            created_at_event="evt_001",
            status="active_but_defeasible",
            legitimacy="prima_facie_valid",
            defeasibility_conditions=["catastrophic_nonconsensual_externality"],
            source_spans=["seg_001:0-78"],
        ),
    ]
    ethical_facts = [
        EthicalFact(
            id="fact_001", kind="legitimacy",
            subjects=["nazis"],
            description="Authority is coercive and tyrannical; legitimacy void.",
            severity="catastrophic", confidence=0.95,
            source_spans=["seg_002:0-90"],
        ),
        EthicalFact(
            id="fact_002", kind="coercion",
            subjects=["speaker", "village"],
            description="Soldiers threaten murderous reprisal against village.",
            severity="catastrophic", confidence=0.95,
            source_spans=["seg_002:0-90"],
        ),
        EthicalFact(
            id="fact_003", kind="externality",
            subjects=["village"],
            description="Catastrophic non-consensual risk imposed on village.",
            severity="catastrophic", confidence=0.92,
            source_spans=["seg_002:0-90"],
        ),
        EthicalFact(
            id="fact_004", kind="care",
            subjects=["hidden_refugees"],
            description="Speaker is actively protecting vulnerable hidden refugees.",
            severity="grave", confidence=0.93,
            source_spans=["seg_001:0-78"],
        ),
        EthicalFact(
            id="fact_005", kind="deception",
            subjects=["speaker", "nazis"],
            description="Deception toward illegitimate murderous authority is permitted.",
            severity="moderate", confidence=0.90,
            source_spans=["seg_001:0-78"],
        ),
    ]
    conflicts = [
        Conflict(
            id="conflict_001",
            name="Protective duty vs imposed catastrophic village risk",
            stakeholders=["hidden_refugees", "village", "speaker"],
            dimensions=["care_protection", "third_party_externality", "vow_fidelity"],
            severity="catastrophic",
            resolution_status="tragic",
            requires_escalation=True,
            description="Truthful disclosure murders refugees; deceptive disclosure risks village reprisal.",
        )
    ]
    return ExtractorResult(
        stakeholders=stakeholders,
        events=events,
        commitments=commitments,
        ethical_facts=ethical_facts,
        conflicts=conflicts,
        canonical_form="coercive_murderous_interrogation_with_collective_reprisal",
        extractor_metadata={"fixture": "nazi_attic"},
    )


# ---------------------------------------------------------------------------
# Fixture: medical confidentiality
# ---------------------------------------------------------------------------


def _medical_confidentiality() -> ExtractorResult:
    stakeholders = [
        Stakeholder(
            id="doctor", label="Treating physician", type="individual",
            roles=["agent", "vow_holder"],
            agency="full", vulnerability="low",
            consent_status="n/a", source_spans=["seg_001:0-50"], confidence=0.96,
        ),
        Stakeholder(
            id="patient", label="Patient", type="individual",
            roles=["patient", "beneficiary"],
            agency="full", vulnerability="moderate",
            consent_status="obtained", source_spans=["seg_001:0-90"], confidence=0.95,
        ),
        Stakeholder(
            id="threatened_party", label="Identifiable third party at risk", type="individual",
            roles=["nonconsenting_third_party", "victim"],
            agency="full", vulnerability="high",
            consent_status="not_obtained", source_spans=["seg_002:0-100"], confidence=0.90,
        ),
        Stakeholder(
            id="public", label="General public", type="collective",
            roles=["beneficiary", "bystander"],
            agency="collective_limited", vulnerability="moderate",
            consent_status="n/a", source_spans=["seg_003:0-60"], confidence=0.80,
        ),
    ]
    events = [
        Event(
            id="evt_001", time_index=0, type="confidentiality_established",
            actor="doctor", target="patient",
            content="medical confidentiality obligation",
            source_spans=["seg_001:0-90"],
        ),
        Event(
            id="evt_002", time_index=1, type="threat_disclosed",
            actor="patient", target="threatened_party",
            content="patient discloses intent to harm identifiable third party",
            source_spans=["seg_002:0-100"],
        ),
    ]
    commitments = [
        Commitment(
            id="commitment_001", type="role_duty", holder="doctor",
            beneficiary="patient",
            content="maintain_medical_confidentiality",
            voluntariness="voluntary", created_at_event="evt_001",
            status="active_but_defeasible",
            legitimacy="fully_legitimate",
            defeasibility_conditions=["higher_duty_conflict"],
            source_spans=["seg_001:0-90"],
        ),
    ]
    ethical_facts = [
        EthicalFact(
            id="fact_001", kind="role_duty",
            subjects=["doctor", "patient"],
            description="Doctor owes professional confidentiality.",
            severity="grave", confidence=0.96, source_spans=["seg_001:0-90"],
        ),
        EthicalFact(
            id="fact_002", kind="harm",
            subjects=["threatened_party"],
            description="Imminent serious harm to identifiable third party.",
            severity="grave", confidence=0.90, source_spans=["seg_002:0-100"],
        ),
        EthicalFact(
            id="fact_003", kind="externality",
            subjects=["threatened_party"],
            description="Confidentiality-protected silence imposes risk on non-consenting third party.",
            severity="grave", confidence=0.88, source_spans=["seg_002:0-100"],
        ),
        EthicalFact(
            id="fact_004", kind="care",
            subjects=["threatened_party"],
            description="Protective duty to identifiable potential victim is triggered.",
            severity="grave", confidence=0.90, source_spans=["seg_002:0-100"],
        ),
    ]
    conflicts = [
        Conflict(
            id="conflict_001",
            name="Confidentiality vs duty to warn",
            stakeholders=["doctor", "patient", "threatened_party"],
            dimensions=["vow_fidelity", "third_party_externality", "care_protection"],
            severity="grave",
            resolution_status="unresolved",
            requires_escalation=False,
            description="Tarasoff-style conflict: confidentiality defeasible by duty to warn.",
        )
    ]
    return ExtractorResult(
        stakeholders=stakeholders,
        events=events,
        commitments=commitments,
        ethical_facts=ethical_facts,
        conflicts=conflicts,
        canonical_form="professional_privilege_versus_duty_to_warn",
        extractor_metadata={"fixture": "medical_confidentiality"},
    )


# ---------------------------------------------------------------------------
# Fixture: whistleblower
# ---------------------------------------------------------------------------


def _whistleblower() -> ExtractorResult:
    stakeholders = [
        Stakeholder(
            id="employee", label="Employee with knowledge", type="individual",
            roles=["agent", "vow_holder"],
            agency="full", vulnerability="moderate",
            consent_status="n/a", source_spans=["seg_001:0-80"], confidence=0.95,
        ),
        Stakeholder(
            id="institution", label="Employing institution", type="institution",
            roles=["authority", "agent"],
            agency="full", vulnerability="low",
            consent_status="n/a", source_spans=["seg_001:0-80"], confidence=0.93,
        ),
        Stakeholder(
            id="public", label="Affected public", type="collective",
            roles=["nonconsenting_third_party", "victim"],
            agency="collective_limited", vulnerability="high",
            consent_status="not_obtained", source_spans=["seg_002:0-100"], confidence=0.91,
        ),
        Stakeholder(
            id="regulators", label="Regulators / oversight", type="institution",
            roles=["authority"],
            agency="full", vulnerability="low",
            consent_status="n/a", source_spans=["seg_003:0-60"], confidence=0.88,
        ),
    ]
    events = [
        Event(
            id="evt_001", time_index=0, type="loyalty_established",
            actor="employee", target="institution",
            content="employment loyalty implicit",
            source_spans=["seg_001:0-80"],
        ),
        Event(
            id="evt_002", time_index=1, type="wrongdoing_discovered",
            actor="employee", target="institution",
            content="employee discovers institutional wrongdoing harming public",
            source_spans=["seg_002:0-100"],
        ),
    ]
    commitments = [
        Commitment(
            id="commitment_001", type="role_duty", holder="employee",
            beneficiary="institution",
            content="institutional_loyalty",
            voluntariness="voluntary", created_at_event="evt_001",
            status="active_but_defeasible",
            legitimacy="defeasible",
            defeasibility_conditions=["vow_to_commit_wrong", "catastrophic_nonconsensual_externality"],
            source_spans=["seg_001:0-80"],
        ),
    ]
    ethical_facts = [
        EthicalFact(
            id="fact_001", kind="role_duty",
            subjects=["employee", "institution"],
            description="Employee owes institutional loyalty.",
            severity="moderate", confidence=0.85, source_spans=["seg_001:0-80"],
        ),
        EthicalFact(
            id="fact_002", kind="harm",
            subjects=["public"],
            description="Institutional wrongdoing causes serious public harm.",
            severity="grave", confidence=0.90, source_spans=["seg_002:0-100"],
        ),
        EthicalFact(
            id="fact_003", kind="externality",
            subjects=["public"],
            description="Silence imposes harm on non-consenting public.",
            severity="grave", confidence=0.88, source_spans=["seg_002:0-100"],
        ),
        EthicalFact(
            id="fact_004", kind="truth",
            subjects=["employee", "public"],
            description="Truth-telling duty to non-consenting affected parties activated.",
            severity="grave", confidence=0.88, source_spans=["seg_003:0-60"],
        ),
        EthicalFact(
            id="fact_005", kind="legitimacy",
            subjects=["institution"],
            description="Institutional authority defeasible due to wrongdoing.",
            severity="moderate", confidence=0.85, source_spans=["seg_002:0-100"],
        ),
    ]
    conflicts = [
        Conflict(
            id="conflict_001",
            name="Institutional loyalty vs public truth-telling",
            stakeholders=["employee", "institution", "public"],
            dimensions=["vow_fidelity", "epistemic_quality", "third_party_externality"],
            severity="grave",
            resolution_status="unresolved",
            requires_escalation=True,
            description="Loyalty defeasible by harm-to-public; whistleblowing channel exists.",
        )
    ]
    return ExtractorResult(
        stakeholders=stakeholders,
        events=events,
        commitments=commitments,
        ethical_facts=ethical_facts,
        conflicts=conflicts,
        canonical_form="institutional_loyalty_versus_public_truth_telling",
        extractor_metadata={"fixture": "whistleblower"},
    )


# ---------------------------------------------------------------------------
# Registry, keyed by SHA-256 of the raw text
# ---------------------------------------------------------------------------

# Fixtures keyed by leading-substring match on text content so the registry
# survives whitespace edits to the example files. We compute the key from a
# normalised prefix.

def _normalised_key(text: str) -> str:
    return " ".join(text.lower().split())[:80]


_FIXTURE_KEY_PREFIXES: dict[str, Callable[[], ExtractorResult]] = {
    "i vowed that there are no jews hiding in the attic": _nazi_attic,
    "dr. m has been treating a patient": _medical_confidentiality,
    "as a financial analyst at the firm": _whistleblower,
}


class MockExtractor(Extractor):
    """Returns hand-curated IR for the three example texts. Raises
    UnknownDocumentError on any other input."""

    name = "mock"

    def extract(self, document, segments) -> ExtractorResult:
        key = _normalised_key(document.raw_text)
        for prefix, fn in _FIXTURE_KEY_PREFIXES.items():
            if key.startswith(prefix):
                return fn()
        raise UnknownDocumentError(
            f"MockExtractor has no fixture for document {document.doc_id!r}. "
            f"The MockExtractor is fixture data, not a general extractor. "
            f"For other inputs, use --extractor rule or --extractor llm."
        )
