"""`MoralSubstrate` — the descriptive layer projections read from.

The substrate contains everything the extractor surfaced about *who did
what to whom, with what authority, under what existing commitments* —
the framework-neutral-up-to-extraction-categories layer. Three
substrate-specific concepts are first-class here that don't have
direct CompilerIR equivalents (yet):

  - `Maxim` — the action under its description. Kantian
    universalizability tests operate on the maxim, not on the act.
  - `ConsentState` — explicit per-stakeholder consent record. Maps
    `autonomy_consent` from a continuous dimension to a categorical
    gate input.
  - `AuthorityLegitimacy` — per-authority procedural standing.
    Same shape: from continuous `legitimacy_trust` to categorical
    gate input.

For v0 these are populated by `substrate_from_ir(ir)` using rule-based
heuristics over the existing `CompilerIR` fields. Future versions
should extract maxims, consent states, and authority legitimacy as
explicit extractor outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from erisml_compiler.ir.schemas import (
    Commitment,
    Document,
    EthicalFact,
    Event,
    Norm,
    Relation,
    Segment,
    Stakeholder,
)


class Maxim(BaseModel):
    """The action under a particular description.

    Kantian universalizability operates on maxims, not on acts. The
    same physical act can be intended under multiple maxims (e.g.
    "lie to save a life" vs "lie when convenient"), and the
    universalizability test depends on which maxim is at stake.
    """

    model_config = ConfigDict(frozen=True)

    description: str
    """Natural-language phrasing of the maxim."""
    agent_id: str | None = None
    """Stakeholder id of the agent whose maxim this is."""
    action_kind: str | None = None
    """Coarse action label: lie, promise, refuse, harm, protect, ..."""
    purpose: str | None = None
    """What the agent is trying to achieve via the action."""
    treats_persons_as: dict[str, str] = Field(default_factory=dict)
    """stakeholder_id -> 'end' | 'means' | 'mere_means'. 'mere_means'
    is the Kantian failure mode (treating a rational agent ONLY as
    an instrument with no regard for them as a self-determining end)."""


class ConsentState(BaseModel):
    """Per-stakeholder explicit consent record."""

    model_config = ConfigDict(frozen=True)

    stakeholder_id: str
    given: bool
    """True iff this stakeholder has affirmatively consented to bearing
    the effect at stake. Default-false for non-consenting third
    parties; explicit-true requires affirmative evidence."""
    under_duress: bool = False
    """True iff the consent was given under coercive conditions
    (rendering it Kantianly void)."""
    informed: bool = True
    """False iff the stakeholder consented without knowing material
    facts (also voids consent under most ethical frameworks)."""


class AuthorityLegitimacy(BaseModel):
    """Per-authority procedural-standing record."""

    model_config = ConfigDict(frozen=True)

    authority_id: str
    """Stakeholder id of the entity issuing demands / claiming standing."""
    legitimate: bool
    """True iff the authority has procedural standing under the
    relevant institutional norms. Mirrors what `legitimacy_trust`
    would tag at high confidence."""
    reason: str = ""
    """Brief explanation of why this authority is or isn't legitimate
    in this context."""


class MoralSubstrate(BaseModel):
    """The descriptive IR layer projections read from.

    Contains: who is involved (stakeholders + relations), what
    happened (events), what's been promised (commitments), what was
    found (ethical_facts), what rules apply (norms), under what
    consent / authority structure, and the maxim under which the
    action is being evaluated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    document: Document
    segments: list[Segment] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    commitments: list[Commitment] = Field(default_factory=list)
    norms: list[Norm] = Field(default_factory=list)
    ethical_facts: list[EthicalFact] = Field(default_factory=list)
    canonical_form: str | None = None

    maxim: Maxim | None = None
    consent_states: list[ConsentState] = Field(default_factory=list)
    authority_legitimacies: list[AuthorityLegitimacy] = Field(default_factory=list)


def substrate_from_ir(ir) -> MoralSubstrate:
    """Project a CompilerIR into the substrate view.

    DAG-native path (preferred): if `ir.graph` is populated, derive
    the substrate from the graph (graph is primary).

    Legacy path: if no graph, fall back to deriving from the flat
    fields directly. Used in tests + during incremental migration.
    """
    if getattr(ir, "graph", None) is not None:
        return substrate_from_graph(ir.graph, ir=ir)

    maxim = _derive_maxim(ir)
    consent_states = _derive_consent_states(ir)
    auth_leg = _derive_authority_legitimacies(ir)

    return MoralSubstrate(
        document=ir.document,
        segments=list(ir.segments),
        stakeholders=list(ir.stakeholders),
        relations=list(ir.relations),
        events=list(ir.events),
        commitments=list(ir.commitments),
        norms=list(ir.norms),
        ethical_facts=list(ir.ethical_facts),
        canonical_form=ir.canonical_form,
        maxim=maxim,
        consent_states=consent_states,
        authority_legitimacies=auth_leg,
    )


def substrate_from_graph(graph, *, ir=None) -> MoralSubstrate:
    """Substrate-as-graph-view: derive Maxim, ConsentState, and
    AuthorityLegitimacy from graph queries instead of flat-field
    heuristics.

    The flat fields (stakeholders/events/commitments/...) come from
    `ir` when supplied, since the rule extractor still emits those;
    when `ir` is None we attempt to recover them from the graph's
    payloads.
    """
    from erisml_compiler.ir.graph import EdgeKind, NodeKind

    # Derive the maxim from the graph: pick the (unique-ish) maxim node.
    maxim_nodes = graph.nodes_of_kind(NodeKind.MAXIM)
    maxim: Maxim | None = None
    if maxim_nodes:
        m = maxim_nodes[0]
        # Find the act this maxim's under, then any treats_as=mere_means
        # edges from that act.
        treats: dict[str, str] = {}
        for edge in graph.in_edges(m.id, kind=EdgeKind.UNDER_MAXIM):
            for ta in graph.out_edges(edge.src, kind=EdgeKind.TREATS_AS):
                role = (ta.payload or {}).get("role")
                if role:
                    sid = ta.dst.removeprefix("stakeholder:")
                    treats[sid] = role
        ak = m.payload.get("action_kind")
        purpose = m.payload.get("purpose")
        agent_id = None
        for n in graph.nodes_of_kind(NodeKind.STAKEHOLDER):
            if "agent" in n.labels:
                agent_id = n.payload.get("id") or n.id.removeprefix("stakeholder:")
                break
        title = ""
        if ir is not None and ir.document and ir.document.title:
            title = ir.document.title
        desc = f"{ak}" + (f" to {purpose}" if purpose else "")
        if title:
            desc = f"{desc} (case: {title[:80]})"
        maxim = Maxim(
            description=desc,
            agent_id=agent_id,
            action_kind=ak,
            purpose=purpose,
            treats_persons_as=treats,
        )

    # Derive consent states: for every stakeholder that is the target
    # of an `imposes_on` edge and lacks a corresponding `consents_to`
    # edge, emit a no-consent state. Coercion edges flag duress.
    consent_states: list[ConsentState] = []
    seen: set[str] = set()
    for edge in graph.edges_of_kind(EdgeKind.IMPOSES_ON):
        sid = edge.dst.removeprefix("stakeholder:")
        if sid in seen:
            continue
        seen.add(sid)
        has_consent = graph.has_edge(edge.dst, edge.src, kind=EdgeKind.CONSENTS_TO)
        if not has_consent:
            # Check if this stakeholder is coerced.
            coerced = any(e.dst == edge.dst for e in graph.edges_of_kind(EdgeKind.COERCES))
            consent_states.append(
                ConsentState(stakeholder_id=sid, given=False, under_duress=coerced)
            )
    # Also include the act's primary actor if they're being coerced.
    for edge in graph.edges_of_kind(EdgeKind.COERCES):
        sid = edge.dst.removeprefix("stakeholder:")
        if sid in seen:
            continue
        seen.add(sid)
        consent_states.append(ConsentState(stakeholder_id=sid, given=False, under_duress=True))

    # Derive authority legitimacies: a stakeholder with role=authority
    # or role=coercer is legitimate unless tagged otherwise via a
    # legitimacy fact node. The extractor's current legitimacy fact
    # is a generic "Authority issuing threats has defeasible legitimacy",
    # so any authority/coercer in the same graph is suspect.
    auth_leg: list[AuthorityLegitimacy] = []
    has_legit_fact = any(
        "legitimacy" in (n.labels or []) for n in graph.nodes_of_kind(NodeKind.FACT)
    )
    for n in graph.nodes_of_kind(NodeKind.STAKEHOLDER):
        labels = n.labels or []
        if "authority" not in labels and "coercer" not in labels:
            continue
        sid = n.payload.get("id") or n.id.removeprefix("stakeholder:")
        auth_leg.append(
            AuthorityLegitimacy(
                authority_id=sid,
                legitimate=not has_legit_fact,
                reason=(
                    "graph fact(legitimacy) flagged this authority"
                    if has_legit_fact
                    else "no legitimacy concern surfaced"
                ),
            )
        )

    # Flat-field views come from `ir` (still the extractor's output);
    # future versions derive these from graph payloads exclusively.
    if ir is not None:
        flat_stakeholders = list(ir.stakeholders)
        flat_events = list(ir.events)
        flat_commitments = list(ir.commitments)
        flat_norms = list(ir.norms)
        flat_facts = list(ir.ethical_facts)
        flat_relations = list(ir.relations)
        document = ir.document
        segments = list(ir.segments)
        canonical_form = ir.canonical_form
    else:
        flat_stakeholders = []
        flat_events = []
        flat_commitments = []
        flat_norms = []
        flat_facts = []
        flat_relations = []
        from erisml_compiler.ir.schemas import Document

        document = Document(doc_id="graph", title="", raw_text="")
        segments = []
        canonical_form = None

    return MoralSubstrate(
        document=document,
        segments=segments,
        stakeholders=flat_stakeholders,
        relations=flat_relations,
        events=flat_events,
        commitments=flat_commitments,
        norms=flat_norms,
        ethical_facts=flat_facts,
        canonical_form=canonical_form,
        maxim=maxim,
        consent_states=consent_states,
        authority_legitimacies=auth_leg,
    )


# --------------------------------------------------- derivation heuristics


_DECEIT_KINDS = {"deception", "lie", "concealment"}
_COERCION_KINDS = {"coercion", "threat", "duress"}
_EXTERNALITY_KINDS = {"externality", "imposed_harm", "third_party_risk"}
_INSTRUMENTAL_KINDS = {"instrumental_use", "exploitation"}


def _kind_str(fact) -> str:
    k = getattr(fact, "kind", None) or getattr(fact, "type", "")
    return (k.value if hasattr(k, "value") else str(k)).lower()


def _heuristic_action_kind(ir) -> str | None:
    """Pre-v1 rule-list: dominant ethical_fact.kind → action_kind.
    Used as fallback when the prose extractor finds no verb pattern."""
    kinds = [_kind_str(f) for f in ir.ethical_facts]
    if any(k in _DECEIT_KINDS for k in kinds):
        return "deceive"
    if any(k in _COERCION_KINDS for k in kinds):
        return "coerce_or_be_coerced"
    if any(k in _EXTERNALITY_KINDS for k in kinds):
        return "impose_externality"
    if ir.commitments:
        return "make_or_keep_commitment"
    return None


def _derive_maxim(ir) -> Maxim | None:
    """Derive the Maxim for an IR.

    v1 (v0.8.x): pattern-based prose extraction via
    `annotation/maxim_extractor.py`. Falls back to the previous
    kind-based heuristic when the prose extractor finds no verb
    pattern in the document's raw text.
    """
    # Prose-first extraction (v1).
    if ir.document and ir.document.raw_text:
        from erisml_compiler.annotation.maxim_extractor import extract_maxim

        # Determine the heuristic action_kind to use as fallback.
        heuristic_action_kind = _heuristic_action_kind(ir)
        maxim, _ev = extract_maxim(
            ir.document.raw_text,
            stakeholders=list(ir.stakeholders),
            fallback_action_kind=heuristic_action_kind,
        )
        if maxim is not None:
            # Augment treats_persons_as with kind-derived signals the
            # prose extractor may have missed (instrumental facts on
            # specific stakeholders).
            treats = dict(maxim.treats_persons_as)
            for f in ir.ethical_facts:
                k = _kind_str(f)
                if k in _INSTRUMENTAL_KINDS:
                    for sid in getattr(f, "subjects", []) or []:
                        treats[sid] = "mere_means"
                elif k in _EXTERNALITY_KINDS:
                    for sid in getattr(f, "subjects", []) or []:
                        treats.setdefault(sid, "means")
            if treats != maxim.treats_persons_as:
                maxim = maxim.model_copy(update={"treats_persons_as": treats})
            return maxim

    # Heuristic fallback (legacy path).
    if not ir.ethical_facts and not ir.commitments:
        return None
    action_kind = _heuristic_action_kind(ir) or "act_under_norm"
    purpose = None
    kinds = [_kind_str(f) for f in ir.ethical_facts]

    if "care" in kinds:
        purpose = "protect_vulnerable"
    elif "legitimacy" in kinds:
        purpose = "act_under_authority"

    agent_id = next(
        (s.id for s in ir.stakeholders if "agent" in (getattr(s, "roles", None) or [])),
        None,
    )

    treats: dict[str, str] = {}
    for f in ir.ethical_facts:
        k = _kind_str(f)
        if k in _INSTRUMENTAL_KINDS:
            for sid in getattr(f, "subjects", []) or []:
                treats[sid] = "mere_means"
        elif k in _EXTERNALITY_KINDS:
            for sid in getattr(f, "subjects", []) or []:
                treats.setdefault(sid, "means")

    title = (ir.document.title or "").strip() if ir.document else ""
    desc = f"{action_kind}" + (f" to {purpose}" if purpose else "")
    if title:
        desc = f"{desc} (case: {title[:80]})"

    return Maxim(
        description=desc,
        agent_id=agent_id,
        action_kind=action_kind,
        purpose=purpose,
        treats_persons_as=treats,
    )


def _derive_consent_states(ir) -> list[ConsentState]:
    """For every stakeholder marked as nonconsenting_third_party or
    bearing externality, emit a ConsentState(given=False)."""
    out: list[ConsentState] = []
    seen: set[str] = set()

    nonconsenting_subjects: set[str] = set()
    coerced_subjects: set[str] = set()
    for f in ir.ethical_facts:
        k = _kind_str(f)
        subs = list(getattr(f, "subjects", []) or [])
        if k in _EXTERNALITY_KINDS:
            nonconsenting_subjects.update(subs)
        if k in _COERCION_KINDS:
            coerced_subjects.update(subs)

    for s in ir.stakeholders:
        if s.id in seen:
            continue
        seen.add(s.id)
        roles = [r.lower() for r in (getattr(s, "roles", None) or [])]
        if "nonconsenting_third_party" in roles or s.id in nonconsenting_subjects:
            out.append(ConsentState(stakeholder_id=s.id, given=False))
        elif s.id in coerced_subjects:
            out.append(ConsentState(stakeholder_id=s.id, given=False, under_duress=True))
    return out


def _derive_authority_legitimacies(ir) -> list[AuthorityLegitimacy]:
    """If any ethical_fact of kind=legitimacy fires with negative
    framing, the named authority is marked illegitimate. Otherwise
    authorities default to legitimate=True."""
    out: list[AuthorityLegitimacy] = []
    illegit_subjects: set[str] = set()
    for f in ir.ethical_facts:
        if _kind_str(f) == "legitimacy":
            # The default rule extractor emits "Authority issuing
            # threats has defeasible or void legitimacy" with empty
            # subjects; treat any role=coercer or authority stakeholder
            # in the same IR as the suspect.
            for s in ir.stakeholders:
                roles = [r.lower() for r in (getattr(s, "roles", None) or [])]
                if "coercer" in roles or "authority" in roles:
                    illegit_subjects.add(s.id)

    for s in ir.stakeholders:
        roles = [r.lower() for r in (getattr(s, "roles", None) or [])]
        if "authority" not in roles and "coercer" not in roles:
            continue
        out.append(
            AuthorityLegitimacy(
                authority_id=s.id,
                legitimate=s.id not in illegit_subjects,
                reason=(
                    "ethical_fact(kind=legitimacy) flagged this authority"
                    if s.id in illegit_subjects
                    else "no legitimacy concern surfaced"
                ),
            )
        )
    return out
