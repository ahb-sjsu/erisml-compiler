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
    """Project a CompilerIR into the substrate view it implicitly contains.

    v0 derives `maxim`, `consent_states`, `authority_legitimacies` from
    rule-based heuristics over ethical_facts. Future versions extract
    these directly.
    """
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


# --------------------------------------------------- derivation heuristics


_DECEIT_KINDS = {"deception", "lie", "concealment"}
_COERCION_KINDS = {"coercion", "threat", "duress"}
_EXTERNALITY_KINDS = {"externality", "imposed_harm", "third_party_risk"}
_INSTRUMENTAL_KINDS = {"instrumental_use", "exploitation"}


def _kind_str(fact) -> str:
    k = getattr(fact, "kind", None) or getattr(fact, "type", "")
    return (k.value if hasattr(k, "value") else str(k)).lower()


def _derive_maxim(ir) -> Maxim | None:
    """Coarse heuristic. v0 uses the dominant action_kind from
    ethical_facts; future versions extract the maxim directly from text.
    """
    if not ir.ethical_facts and not ir.commitments:
        return None
    action_kind = None
    purpose = None
    kinds = [_kind_str(f) for f in ir.ethical_facts]
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
            out.append(
                ConsentState(stakeholder_id=s.id, given=False, under_duress=True)
            )
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
