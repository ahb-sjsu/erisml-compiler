"""Deontic (Kantian) projection.

Reads the substrate and emits categorical gate findings:

  - **universalizability** — can the maxim of the action be willed as
    a universal law? Failure: the maxim contradicts itself when
    universalized (e.g. "lie when convenient" — if everyone lied,
    promises become meaningless, the action itself becomes impossible).
  - **mere_means** — does the action treat any rational agent merely
    as a means? Failure: a stakeholder is used instrumentally without
    being respected as a self-determining end.
  - **valid_consent** — was the consent of affected parties valid?
    Failure: consent was absent, given under duress, or uninformed.
  - **legitimate_authority** — does any acting authority have
    procedural standing? Failure: the action invokes authority whose
    legitimacy has been defeated (e.g. illegitimate coercer).

These are *gates*, not aggregates. The verdict is `forbidden` if any
grave-or-catastrophic gate fails; `requires_review` if any moderate
gate fails; `permissible` otherwise.

This is a v0 implementation using rule-based heuristics over the
substrate's `maxim`, `consent_states`, and `authority_legitimacies`
(which are themselves derived by heuristics in `substrate.py`).
Production Kantian analysis would need a far richer maxim model
(natural-deduction over maxim contradictions, etc.) — out of scope.
"""

from __future__ import annotations

from typing import Any

from erisml_compiler.projections.base import GateFinding, Projection, ProjectionResult
from erisml_compiler.projections.substrate import MoralSubstrate

# Action-kinds whose maxim universalised typically contradicts itself.
# This is a coarse rule list; a richer Kantian analyser would build
# the universalised-world model and run the contradiction test.
_NON_UNIVERSALISABLE_KINDS = {
    "deceive",
    "impose_externality",
}


class DeonticProjection(Projection):
    """Kantian categorical-imperative analysis as gates."""

    framework = "deontic_kantian"

    def project(
        self,
        substrate: MoralSubstrate,
        *,
        graph: Any = None,
        **kwargs: Any,
    ) -> ProjectionResult:
        """Run the four Kantian gates over the substrate.

        When `graph` is supplied (a `MoralGraph`), gates that benefit
        from direct subgraph pattern-matching use it; otherwise they
        fall back to substrate fields (which are themselves derived
        from the graph when one was attached upstream).
        """
        findings: list[GateFinding] = []

        findings.append(self._gate_universalizability(substrate, graph))
        findings.append(self._gate_mere_means(substrate, graph))
        findings.append(self._gate_valid_consent(substrate, graph))
        findings.append(self._gate_legitimate_authority(substrate, graph))

        grave = [f for f in findings if not f.passed and f.severity in ("grave", "catastrophic")]
        moderate = [f for f in findings if not f.passed and f.severity == "moderate"]

        if grave:
            verdict = "forbidden"
        elif moderate:
            verdict = "requires_review"
        else:
            verdict = "permissible"

        return ProjectionResult(
            framework=self.framework,
            verdict=verdict,
            confidence=1.0,
            findings=findings,
            framework_specific={
                "n_gates_failed": sum(1 for f in findings if not f.passed),
                "n_gates_passed": sum(1 for f in findings if f.passed),
            },
            metadata={"projection_version": "v0_heuristic"},
        )

    # ----------------------------------------------------- universalizability

    def _gate_universalizability(self, substrate: MoralSubstrate, graph: Any = None) -> GateFinding:
        from erisml_compiler.delta.universalizability import test_universalizability

        if substrate.maxim is None:
            return GateFinding(
                name="universalizability",
                passed=True,
                reason="No maxim extracted; cannot test universalizability",
                severity="moderate",
                detail={"result": "undetermined"},
            )
        kind = substrate.maxim.action_kind
        dep = test_universalizability(kind)

        # Build detail block including the contradiction type, the
        # institution(s) the act presupposes, and the Kantian
        # justification — so the gate firing is auditable, not just a
        # boolean.
        detail: dict[str, Any] = {
            "action_kind": kind,
            "contradiction_type": dep.contradiction_type,
            "presupposes": list(dep.presupposes),
            "justification": dep.justification,
        }
        if dep.contested_reading:
            detail["contested_reading"] = dep.contested_reading

        if not dep.passes:
            return GateFinding(
                name="universalizability",
                passed=False,
                reason=(
                    f"{dep.contradiction_type.replace('_', ' ').upper()}: " f"{dep.justification}"
                ),
                severity="grave",
                detail=detail,
            )
        return GateFinding(
            name="universalizability",
            passed=True,
            reason=(
                f"Maxim's action kind '{kind}' passes universalizability: " f"{dep.justification}"
            ),
            severity="grave",
            detail=detail,
        )

    # ----------------------------------------------------- mere means

    def _gate_mere_means(self, substrate: MoralSubstrate, graph: Any = None) -> GateFinding:
        # Graph-native path: pattern-match `treats_as` edges directly.
        if graph is not None:
            from erisml_compiler.ir.graph import EdgeKind

            mere_means_edges = [
                e
                for e in graph.edges_of_kind(EdgeKind.TREATS_AS)
                if (e.payload or {}).get("role") == "mere_means"
            ]
            if mere_means_edges:
                subs = [e.dst.removeprefix("stakeholder:") for e in mere_means_edges]
                return GateFinding(
                    name="mere_means",
                    passed=False,
                    reason=(
                        f"Graph has {len(mere_means_edges)} treats_as[role=mere_means] "
                        f"edge(s): {', '.join(subs[:3])}" + (" ..." if len(subs) > 3 else "")
                    ),
                    severity="grave",
                    subjects=subs,
                )

        if substrate.maxim is None or not substrate.maxim.treats_persons_as:
            # Fall back to: any non-consenting third party in the
            # substrate is, by default, being treated as means.
            non_consenting = [c for c in substrate.consent_states if not c.given]
            if non_consenting:
                return GateFinding(
                    name="mere_means",
                    passed=False,
                    reason=(
                        f"{len(non_consenting)} stakeholder(s) bear the action's "
                        f"effects without having consented: "
                        f"{', '.join(c.stakeholder_id for c in non_consenting[:3])}"
                        + (" ..." if len(non_consenting) > 3 else "")
                    ),
                    severity="grave",
                    subjects=[c.stakeholder_id for c in non_consenting],
                )
            return GateFinding(
                name="mere_means",
                passed=True,
                reason="No stakeholder treated as mere means in the extracted substrate",
                severity="grave",
            )
        mere_means_subs = [
            sid for sid, role in substrate.maxim.treats_persons_as.items() if role == "mere_means"
        ]
        if mere_means_subs:
            return GateFinding(
                name="mere_means",
                passed=False,
                reason=(
                    f"Maxim treats {', '.join(mere_means_subs[:3])} as mere means"
                    + (" ..." if len(mere_means_subs) > 3 else "")
                ),
                severity="grave",
                subjects=mere_means_subs,
            )
        return GateFinding(
            name="mere_means",
            passed=True,
            reason="Maxim treats all stakeholders at least as means-AND-ends, not mere means",
            severity="grave",
        )

    # ----------------------------------------------------- valid consent

    def _gate_valid_consent(self, substrate: MoralSubstrate, graph: Any = None) -> GateFinding:
        invalid = [
            c for c in substrate.consent_states if not c.given or c.under_duress or not c.informed
        ]
        if invalid:
            duress = [c.stakeholder_id for c in invalid if c.under_duress]
            uninformed = [c.stakeholder_id for c in invalid if not c.informed]
            absent = [
                c.stakeholder_id
                for c in invalid
                if not c.given and not c.under_duress and c.informed
            ]
            parts = []
            if absent:
                parts.append(f"consent absent for {', '.join(absent[:3])}")
            if duress:
                parts.append(f"consent under duress for {', '.join(duress[:3])}")
            if uninformed:
                parts.append(f"consent uninformed for {', '.join(uninformed[:3])}")
            return GateFinding(
                name="valid_consent",
                passed=False,
                reason="; ".join(parts),
                severity="grave",
                subjects=[c.stakeholder_id for c in invalid],
            )
        return GateFinding(
            name="valid_consent",
            passed=True,
            reason="No invalid-consent finding in the substrate",
            severity="grave",
        )

    # ----------------------------------------------------- legitimate authority

    def _gate_legitimate_authority(
        self, substrate: MoralSubstrate, graph: Any = None
    ) -> GateFinding:
        illegit = [a for a in substrate.authority_legitimacies if not a.legitimate]
        if illegit:
            return GateFinding(
                name="legitimate_authority",
                passed=False,
                reason=(
                    f"Illegitimate authority/authorities: "
                    f"{', '.join(a.authority_id for a in illegit[:3])}"
                ),
                severity="moderate",
                subjects=[a.authority_id for a in illegit],
            )
        return GateFinding(
            name="legitimate_authority",
            passed=True,
            reason="No illegitimate authority surfaced in the substrate",
            severity="moderate",
        )
