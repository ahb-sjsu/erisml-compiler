"""Care-ethics projection (Gilligan, Noddings, Held, Tronto).

Care ethics is *not* a third way between consequentialism and
deontology — it operates on different primitives entirely:

  - **Relations, not parties.** Moral life is constituted by webs of
    care relations (parent-child, friend-friend, citizen-stranger),
    not by isolated agents weighing isolated acts.
  - **Particularised attentiveness.** What is owed depends on who
    *this specific* relational counterpart is, what their situation
    is, what they need — not on universalizable principles.
  - **Asymmetric responsibility.** Those in a position of
    competence/power toward dependents owe more than universal duty
    requires.
  - **Two phases**: caring-about (recognising the need) and
    caring-for (acting to meet it). Failure on either phase is a
    care-ethical failure.

The projection emits findings on:
  - `relational_attentiveness` — was the relational web recognised
    in the act, or did the agent treat the situation abstractly?
  - `asymmetric_responsibility` — does the agent bear special
    obligations toward dependents present in the graph?
  - `dependency_response` — does the act respond to dependence /
    vulnerability, or ignore it?

Like VirtueProjection, this is v0 heuristic. Real care-ethics would
build the full relational topology over time.
"""
from __future__ import annotations

from typing import Any

from erisml_compiler.projections.base import GateFinding, Projection, ProjectionResult
from erisml_compiler.projections.substrate import MoralSubstrate


class CareEthicsProjection(Projection):
    """Gilligan/Noddings/Tronto-style relational-care reading."""

    framework = "care_ethics_relational"

    def project(
        self,
        substrate: MoralSubstrate,
        *,
        graph: Any = None,
        **kwargs: Any,
    ) -> ProjectionResult:
        findings: list[GateFinding] = []

        findings.append(self._gate_relational_attentiveness(substrate, graph))
        findings.append(self._gate_asymmetric_responsibility(substrate, graph))
        findings.append(self._gate_dependency_response(substrate, graph))

        failed = [f for f in findings if not f.passed]
        n_grave_failed = sum(1 for f in failed if f.severity == "grave")

        if n_grave_failed >= 2:
            verdict = "uncaring"
        elif failed:
            verdict = "requires_caring_attention"
        else:
            verdict = "caring"

        return ProjectionResult(
            framework=self.framework,
            verdict=verdict,
            confidence=0.6,
            findings=findings,
            framework_specific={
                "n_relations": len(substrate.relations),
                "n_dependents": sum(
                    1 for s in substrate.stakeholders
                    if any(r in ("patient", "dependent") for r in getattr(s, "roles", []) or [])
                ),
            },
            metadata={"projection_version": "v0_heuristic"},
        )

    # --------------------------------------------------- attentiveness

    def _gate_relational_attentiveness(
        self, substrate: MoralSubstrate, graph: Any = None
    ) -> GateFinding:
        """Care ethics asks whether the agent SAW the relational
        context. Proxy: did the extractor surface relations + were
        any of them between the agent and a dependent?"""
        agent_id = (
            substrate.maxim.agent_id if substrate.maxim else None
        ) or next(
            (s.id for s in substrate.stakeholders
             if "agent" in (getattr(s, "roles", None) or [])),
            None,
        )
        if not substrate.relations:
            return GateFinding(
                name="relational_attentiveness",
                passed=False,
                reason=(
                    "No relations extracted from the substrate. Either "
                    "the situation truly has none, or the agent / "
                    "extractor missed the relational context — care "
                    "ethics treats both as concerning."
                ),
                severity="moderate",
                subjects=[agent_id] if agent_id else [],
            )
        return GateFinding(
            name="relational_attentiveness",
            passed=True,
            reason=f"{len(substrate.relations)} relation(s) recognised",
            severity="moderate",
            detail={"n_relations": len(substrate.relations)},
        )

    # --------------------------------------------------- asymmetric responsibility

    def _gate_asymmetric_responsibility(
        self, substrate: MoralSubstrate, graph: Any = None
    ) -> GateFinding:
        """If dependents are present, the agent bears asymmetric
        responsibility toward them. Whether that responsibility is
        being honoured is a separate gate; this one just flags the
        asymmetry."""
        dependents: list[str] = []
        for s in substrate.stakeholders:
            roles = [r.lower() for r in getattr(s, "roles", None) or []]
            if "dependent" in roles or "patient" in roles:
                dependents.append(s.id)
            elif (getattr(s, "vulnerability", None) or "").lower() in ("high", "extreme"):
                dependents.append(s.id)

        if not dependents:
            return GateFinding(
                name="asymmetric_responsibility",
                passed=True,
                reason="No dependents flagged in the substrate",
                severity="moderate",
            )
        return GateFinding(
            name="asymmetric_responsibility",
            passed=False,
            reason=(
                f"Substrate flags {len(dependents)} dependent / vulnerable "
                f"party(ies): {', '.join(dependents[:3])}"
                + (" ..." if len(dependents) > 3 else "")
                + ". Care ethics assigns asymmetric duty to whoever bears "
                "competence toward them."
            ),
            severity="moderate",
            subjects=dependents,
        )

    # --------------------------------------------------- dependency response

    def _gate_dependency_response(
        self, substrate: MoralSubstrate, graph: Any = None
    ) -> GateFinding:
        """Does the act actually MEET the recognised dependence, or
        does it bypass it? Proxy: dependents + an imposes_on edge
        targeting them = the act is imposing on the dependent rather
        than meeting their need. Dependents without imposition + with
        a care fact = the act is responsive."""
        from erisml_compiler.ir.graph import EdgeKind

        dependents_at_risk: list[str] = []
        if graph is not None:
            for e in graph.edges_of_kind(EdgeKind.IMPOSES_ON):
                target = graph.get_node(e.dst)
                if target is None:
                    continue
                labels = target.labels or []
                if any(l in ("dependent", "patient", "vulnerable") for l in labels):
                    dependents_at_risk.append(target.id.removeprefix("stakeholder:"))

        if dependents_at_risk:
            return GateFinding(
                name="dependency_response",
                passed=False,
                reason=(
                    f"Act imposes on {len(dependents_at_risk)} dependent "
                    f"party(ies): {', '.join(dependents_at_risk[:3])}. "
                    f"Care ethics asks whether the act MEETS the dependence "
                    f"rather than adding to it."
                ),
                severity="grave",
                subjects=dependents_at_risk,
            )

        # Substrate fallback
        care_kinds = {"care", "non_maleficence"}
        has_care_act = any(
            (getattr(f, "kind", None).value if hasattr(getattr(f, "kind", None), "value")
             else str(getattr(f, "kind", ""))).lower() in care_kinds
            for f in substrate.ethical_facts
        )
        if has_care_act:
            return GateFinding(
                name="dependency_response",
                passed=True,
                reason="Substrate surfaces care-acting toward dependents",
                severity="grave",
            )
        return GateFinding(
            name="dependency_response",
            passed=True,
            reason="No dependent-targeting impositions in the substrate",
            severity="grave",
        )
