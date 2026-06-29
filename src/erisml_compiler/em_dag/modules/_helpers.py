"""Shared helpers for EM evaluators.

These helpers count and aggregate ethical facts of given kinds from the IR.
They keep individual EMs short and consistent. All arithmetic here is fixed
sign convention: positive scores indicate the dimension is being respected
(harm avoided, autonomy honored); negative scores indicate violation.

Two read paths:
  - **Graph-native** (preferred): when `ir.graph` is populated, helpers
    query the typed MoralGraph directly. FACT nodes carry the
    EthicalFact payload; COMMITMENT nodes carry the Commitment payload;
    STAKEHOLDER nodes carry the Stakeholder payload. The fact-kind
    filter reads node.labels (where promote.py sets the kind string).
  - **Flat fallback**: when no graph is attached, fall back to reading
    `ir.ethical_facts` / `ir.commitments` / `ir.stakeholders` directly.

Both paths produce identical EthicalFact / Commitment / Stakeholder
objects (the graph's payload IS the model_dump of the original), so
the 10 EM modules behave identically regardless of which path runs.
The graph path is preferred because it expresses *what the data
actually is* (typed edges + nodes) rather than *what list it lived in*.
"""

from __future__ import annotations

from erisml_compiler.ir.schemas import (
    Commitment,
    CompilerIR,
    DimensionScore,
    EthicalFact,
    EthicalFactKind,
    Stakeholder,
)

SEVERITY_WEIGHT = {
    "minor": 0.2,
    "moderate": 0.5,
    "grave": 0.85,
    "catastrophic": 1.0,
}


# ----------------------------------------------------------------- graph-native readers


def _facts_from_graph(graph, kind: EthicalFactKind) -> list[EthicalFact]:
    """Read FACT nodes whose label matches `kind` and rebuild EthicalFact."""
    from erisml_compiler.ir.graph import NodeKind

    out: list[EthicalFact] = []
    kind_lower = kind.lower()
    for node in graph.nodes_of_kind(NodeKind.FACT):
        if kind_lower in [lbl.lower() for lbl in (node.labels or [])]:
            if node.payload:
                out.append(EthicalFact.model_validate(node.payload))
    return out


def _commitments_from_graph(graph) -> list[Commitment]:
    """Read all COMMITMENT nodes back into Commitment objects."""
    from erisml_compiler.ir.graph import NodeKind

    out: list[Commitment] = []
    for node in graph.nodes_of_kind(NodeKind.COMMITMENT):
        if node.payload:
            out.append(Commitment.model_validate(node.payload))
    return out


def _stakeholders_from_graph(graph) -> list[Stakeholder]:
    from erisml_compiler.ir.graph import NodeKind

    out: list[Stakeholder] = []
    for node in graph.nodes_of_kind(NodeKind.STAKEHOLDER):
        if node.payload:
            out.append(Stakeholder.model_validate(node.payload))
    return out


# ----------------------------------------------------------------- public API


def facts_of_kind(ir: CompilerIR, kind: EthicalFactKind) -> list[EthicalFact]:
    """Return ethical facts of a given kind.

    Graph-native when `ir.graph` is populated; flat fallback otherwise.
    """
    if getattr(ir, "graph", None) is not None:
        return _facts_from_graph(ir.graph, kind)
    return [f for f in ir.ethical_facts if f.kind == kind]


def severity_score(fact: EthicalFact) -> float:
    """Map a fact's severity to a magnitude in [0, 1]."""
    if not fact.severity:
        return 0.5
    return SEVERITY_WEIGHT.get(fact.severity, 0.5)


def aggregate_negative(
    facts: list[EthicalFact],
    explanation_prefix: str = "",
) -> DimensionScore:
    """Aggregate facts into a negative-valued DimensionScore (violation
    detected, score moves toward -1).

    The aggregation is: max severity over contributing facts, with confidence
    averaged. If no facts, returns a neutral 0.0 score.
    """
    if not facts:
        return DimensionScore(
            value=0.0,
            confidence=1.0,
            uncertainty=0.0,
            direction="neutral",
            source_spans=[],
            explanation=f"{explanation_prefix}No relevant facts detected.",
        )
    magnitudes = [severity_score(f) for f in facts]
    max_mag = max(magnitudes)
    mean_conf = sum(f.confidence for f in facts) / len(facts)
    spans: list[str] = []
    for f in facts:
        spans.extend(f.source_spans)
    descriptions = "; ".join(f.description for f in facts[:3])
    return DimensionScore(
        value=-max_mag,
        confidence=mean_conf,
        uncertainty=1.0 - mean_conf,
        direction="negative",
        source_spans=spans,
        explanation=f"{explanation_prefix}{descriptions}",
    )


def aggregate_positive(
    facts: list[EthicalFact],
    explanation_prefix: str = "",
) -> DimensionScore:
    """Aggregate facts into a positive-valued DimensionScore (good behaviour
    detected, score moves toward +1)."""
    if not facts:
        return DimensionScore(
            value=0.0,
            confidence=1.0,
            uncertainty=0.0,
            direction="neutral",
            source_spans=[],
            explanation=f"{explanation_prefix}No relevant facts detected.",
        )
    magnitudes = [severity_score(f) for f in facts]
    max_mag = max(magnitudes)
    mean_conf = sum(f.confidence for f in facts) / len(facts)
    spans: list[str] = []
    for f in facts:
        spans.extend(f.source_spans)
    descriptions = "; ".join(f.description for f in facts[:3])
    return DimensionScore(
        value=+max_mag,
        confidence=mean_conf,
        uncertainty=1.0 - mean_conf,
        direction="positive",
        source_spans=spans,
        explanation=f"{explanation_prefix}{descriptions}",
    )


def active_commitments(ir: CompilerIR) -> list[Commitment]:
    """Active or fulfilled commitments.

    Graph-native when `ir.graph` is populated; flat fallback otherwise.
    """
    if getattr(ir, "graph", None) is not None:
        comms = _commitments_from_graph(ir.graph)
    else:
        comms = list(ir.commitments)
    return [c for c in comms if c.status in ("active", "active_but_defeasible", "fulfilled")]


def violated_commitments(ir: CompilerIR) -> list[Commitment]:
    if getattr(ir, "graph", None) is not None:
        comms = _commitments_from_graph(ir.graph)
    else:
        comms = list(ir.commitments)
    return [c for c in comms if c.status == "violated"]


def stakeholders(ir: CompilerIR) -> list[Stakeholder]:
    """Stakeholder list — graph-native when available.

    EM modules that need stakeholder enumeration (CareEM, ExternalityEM)
    should call this helper instead of reading `ir.stakeholders` directly,
    so the graph becomes the source of truth.
    """
    if getattr(ir, "graph", None) is not None:
        return _stakeholders_from_graph(ir.graph)
    return list(ir.stakeholders)


def stakeholders_with_role(ir: CompilerIR, role: str) -> list[Stakeholder]:
    """Stakeholders whose `roles` list contains `role`."""
    return [s for s in stakeholders(ir) if role in (s.roles or [])]


def vulnerable_stakeholders(ir: CompilerIR) -> list[Stakeholder]:
    """Stakeholders with vulnerability in {'high', 'extreme'}."""
    return [s for s in stakeholders(ir) if s.vulnerability in ("high", "extreme")]


def nonconsenting_third_party_ids(ir: CompilerIR) -> set[str]:
    """Stakeholder ids identified as non-consenting third parties.

    Graph-native: read STAKEHOLDER nodes with `nonconsenting_third_party`
    label, OR target of an IMPOSES_ON edge without a paired CONSENTS_TO
    edge. The latter is the typed-edge way of saying 'bears the act
    without consent', which is the actual semantic relationship.
    """
    sids: set[str] = set()
    graph = ir.graph
    if graph is not None:
        from erisml_compiler.ir.graph import EdgeKind, NodeKind

        for n in graph.nodes_of_kind(NodeKind.STAKEHOLDER):
            labels = [lbl.lower() for lbl in (n.labels or [])]
            if "nonconsenting_third_party" in labels:
                sids.add(
                    (n.payload.get("id") if n.payload else None)
                    or n.id.removeprefix("stakeholder:")
                )
        # Augment with IMPOSES_ON-without-CONSENTS_TO targets.
        for e in graph.edges_of_kind(EdgeKind.IMPOSES_ON):
            has_consent = graph.has_edge(e.dst, e.src, kind=EdgeKind.CONSENTS_TO)
            if not has_consent:
                sids.add(e.dst.removeprefix("stakeholder:"))
        return sids

    # Flat fallback.
    for s in ir.stakeholders:
        if "nonconsenting_third_party" in (s.roles or []):
            sids.add(s.id)
        elif s.consent_status in ("not_obtained", "coerced") and "bystander" in (s.roles or []):
            sids.add(s.id)
    return sids
