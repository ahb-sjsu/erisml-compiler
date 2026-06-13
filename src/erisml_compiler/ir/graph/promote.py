"""Promote flat-list CompilerIR fields to a typed MoralGraph.

v0 strategy: leave the extractors emitting flat lists; this module
promotes those lists into the typed graph at Pass 7.5. Future
versions will have extractors emit graph nodes+edges directly,
making this promotion redundant.

Mapping from flat -> graph:

  Stakeholder           -> Node(kind=stakeholder, payload=Stakeholder.model_dump())
  Event                 -> Node(kind=act,         payload=Event.model_dump())
                           + Edge(performs)   stakeholder.id -> event.id  (when there's an agent role)
                           + Edge(coerces)    event.actor -> target       (when type involves coercion)
  Commitment            -> Node(kind=commitment, payload=Commitment.model_dump())
                           + Edge(holds_commitment)    holder -> commitment
                           + Edge(commitment_binds)    commitment -> beneficiary  (if set)
  EthicalFact           -> Node(kind=fact, payload=EthicalFact.model_dump())
                           + Edge(fact_subject)        fact -> each subject
                           + Edge(imposes_on)          (act -> subject) for harm/externality kinds
                           + Edge(consents_to{...given:false})  subject -> act (negated) for nonconsent kinds
                           + Edge(treats_as{role:mere_means})   act -> subject for instrumental kinds
  Norm                  -> Node(kind=norm, payload=Norm.model_dump())
  Maxim (derived)       -> Node(kind=maxim, payload={action_kind, purpose, ...})
                           + Edge(under_maxim) act -> maxim
                           + Edge(treats_as)   act -> stakeholder per maxim.treats_persons_as

Stakeholder labels (`agent`, `coercer`, `authority`, `vulnerable`,
`nonconsenting_third_party`) carry over to MoralNode.labels for query
convenience.
"""
from __future__ import annotations

from typing import Any

from erisml_compiler.ir.graph.container import MoralGraph
from erisml_compiler.ir.graph.schema import EdgeKind, MoralEdge, MoralNode, NodeKind


# Same kind sets used by projections/substrate.py to interpret
# extractor output. Keep in sync.
_DECEIT_KINDS = {"deception", "lie", "concealment"}
_COERCION_KINDS = {"coercion", "threat", "duress"}
_EXTERNALITY_KINDS = {"externality", "imposed_harm", "third_party_risk"}
_INSTRUMENTAL_KINDS = {"instrumental_use", "exploitation"}
_HARM_KINDS = {"harm", "physical_harm", "psychological_harm"} | _EXTERNALITY_KINDS


def _fact_kind_str(fact) -> str:
    k = getattr(fact, "kind", None) or getattr(fact, "type", "")
    return (k.value if hasattr(k, "value") else str(k)).lower()


def _node_id(kind: str, local_id: str) -> str:
    return f"{kind}:{local_id}"


def graph_from_flat(ir) -> MoralGraph:
    """Build a MoralGraph from the existing CompilerIR flat fields.

    Idempotent: calling it twice on the same IR yields equal graphs.
    """
    g = MoralGraph()

    # ----------- stakeholders -> nodes -----------
    for s in ir.stakeholders:
        roles = list(getattr(s, "roles", None) or [])
        nid = _node_id("stakeholder", s.id)
        g.add_node(
            MoralNode(
                id=nid,
                kind=NodeKind.STAKEHOLDER,
                payload=s.model_dump() if hasattr(s, "model_dump") else dict(s),
                labels=sorted({*(r.lower() for r in roles)}),
            )
        )

    # ----------- events -> act nodes (+ performs edges) -----------
    for ev in ir.events:
        act_id = _node_id("act", ev.id)
        g.add_node(
            MoralNode(
                id=act_id,
                kind=NodeKind.ACT,
                payload=ev.model_dump() if hasattr(ev, "model_dump") else dict(ev),
            )
        )
        # If an actor / agent is named on the event, draw `performs`.
        actor = getattr(ev, "actor", None) or getattr(ev, "agent", None) or getattr(ev, "subject", None)
        if actor:
            g.add_edge(
                MoralEdge(
                    src=_node_id("stakeholder", actor),
                    dst=act_id,
                    kind=EdgeKind.PERFORMS,
                )
            )

    # ----------- commitments -> nodes + binding edges -----------
    for i, c in enumerate(ir.commitments):
        cid = _node_id("commitment", f"c{i}")
        g.add_node(
            MoralNode(
                id=cid,
                kind=NodeKind.COMMITMENT,
                payload=c.model_dump() if hasattr(c, "model_dump") else dict(c),
            )
        )
        holder = getattr(c, "holder", None)
        beneficiary = getattr(c, "beneficiary", None)
        if holder:
            g.add_edge(
                MoralEdge(
                    src=_node_id("stakeholder", holder),
                    dst=cid,
                    kind=EdgeKind.HOLDS_COMMITMENT,
                    payload={"type": getattr(c, "type", None)},
                )
            )
        if beneficiary:
            g.add_edge(
                MoralEdge(
                    src=cid,
                    dst=_node_id("stakeholder", beneficiary),
                    kind=EdgeKind.COMMITMENT_BINDS,
                )
            )

    # ----------- norms -> nodes (no edges yet) -----------
    for i, n in enumerate(ir.norms):
        nid = _node_id("norm", f"n{i}")
        g.add_node(
            MoralNode(
                id=nid,
                kind=NodeKind.NORM,
                payload=n.model_dump() if hasattr(n, "model_dump") else dict(n),
            )
        )

    # ----------- ethical_facts -> fact nodes + interpretive edges -----------
    # The act we attribute imposes_on / treats_as / coerces to: the
    # rule extractor doesn't link facts to specific events, so we use
    # the first event (most likely the situation's primary act) as a
    # fallback. Future extractors should link facts to events directly.
    primary_act_id = (
        _node_id("act", ir.events[0].id) if ir.events else None
    )

    for f in ir.ethical_facts:
        fid = _node_id("fact", f.id)
        g.add_node(
            MoralNode(
                id=fid,
                kind=NodeKind.FACT,
                payload=f.model_dump() if hasattr(f, "model_dump") else dict(f),
                labels=[_fact_kind_str(f)],
            )
        )
        subjects = list(getattr(f, "subjects", []) or [])
        for sid in subjects:
            stake_id = _node_id("stakeholder", sid)
            g.add_edge(MoralEdge(src=fid, dst=stake_id, kind=EdgeKind.FACT_SUBJECT))

            if primary_act_id is not None:
                g.add_edge(
                    MoralEdge(
                        src=primary_act_id,
                        dst=fid,
                        kind=EdgeKind.SURFACES_FACT,
                    )
                )

                fk = _fact_kind_str(f)
                if fk in _HARM_KINDS:
                    g.add_edge(
                        MoralEdge(
                            src=primary_act_id,
                            dst=stake_id,
                            kind=EdgeKind.IMPOSES_ON,
                            payload={
                                "severity": getattr(f, "severity", None),
                                "confidence": float(getattr(f, "confidence", 1.0)),
                                "fact_id": fid,
                            },
                        )
                    )
                    # Non-consent is the *absence* of CONSENTS_TO edges.
                    # We don't add a negated edge; the deontic gate
                    # interprets missing consent as non-consent.
                if fk in _COERCION_KINDS:
                    # Coercion target -> the act's primary actor.
                    actor = next(
                        (
                            n.payload.get("id") or n.id.removeprefix("stakeholder:")
                            for n in g.nodes
                            if n.kind == NodeKind.STAKEHOLDER and "agent" in n.labels
                        ),
                        None,
                    )
                    if actor:
                        g.add_edge(
                            MoralEdge(
                                src=stake_id,
                                dst=_node_id("stakeholder", actor),
                                kind=EdgeKind.COERCES,
                                payload={"severity": getattr(f, "severity", None)},
                            )
                        )
                if fk in _INSTRUMENTAL_KINDS:
                    g.add_edge(
                        MoralEdge(
                            src=primary_act_id,
                            dst=stake_id,
                            kind=EdgeKind.TREATS_AS,
                            payload={"role": "mere_means"},
                        )
                    )

    # ----------- maxim (derived) -> node + under_maxim + treats_as edges -----------
    # Mirror the same heuristic that lived in substrate.py's
    # `_derive_maxim`. The graph carries it as a proper node now.
    if primary_act_id is not None and ir.ethical_facts:
        kinds = [_fact_kind_str(f) for f in ir.ethical_facts]
        if any(k in _DECEIT_KINDS for k in kinds):
            action_kind = "deceive"
        elif any(k in _COERCION_KINDS for k in kinds):
            action_kind = "coerce_or_be_coerced"
        elif any(k in _EXTERNALITY_KINDS for k in kinds):
            action_kind = "impose_externality"
        elif ir.commitments:
            action_kind = "make_or_keep_commitment"
        else:
            action_kind = "act_under_norm"

        purpose = None
        if "care" in kinds:
            purpose = "protect_vulnerable"
        elif "legitimacy" in kinds:
            purpose = "act_under_authority"

        maxim_id = "maxim:m0"
        g.add_node(
            MoralNode(
                id=maxim_id,
                kind=NodeKind.MAXIM,
                payload={"action_kind": action_kind, "purpose": purpose},
            )
        )
        g.add_edge(
            MoralEdge(src=primary_act_id, dst=maxim_id, kind=EdgeKind.UNDER_MAXIM)
        )

    return g
