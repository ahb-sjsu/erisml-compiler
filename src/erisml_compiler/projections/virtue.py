"""Virtue-ethics projection.

Reads the graph and emits character-trait findings rather than
verdict-on-the-act. The virtue tradition (Aristotle, Foot, Hursthouse,
MacIntyre) evaluates moral situations by asking *what kind of agent
the doer is being / becoming*, not whether the act is permitted in
isolation. Verdicts here are framework-relative:

  - `virtuous`     — the agent's traits surfaced by this act are
                     consistent with the central virtues (courage,
                     honesty, justice, prudence, temperance, fidelity).
  - `vicious`      — the act expresses a corresponding vice
                     (cowardice, deception, injustice, etc.) on
                     evidence visible in the graph.
  - `requires_practical_wisdom` — competing virtues, no clear-cut
                     reading; the situation calls for phronesis
                     rather than rule-application.

Implementation is v0 heuristic: map graph-visible action_kinds and
edge patterns to virtue / vice findings. A richer implementation
would model habituation across multiple acts (a single act doesn't
make someone vicious; a pattern does), agent-stakeholder relational
history, and the doctrine of the mean.

This projection is *not* trying to be a complete virtue ethics. It's
trying to be honest about what the framework would surface from the
substrate we currently extract.
"""

from __future__ import annotations

from typing import Any

from erisml_compiler.projections.base import GateFinding, Projection, ProjectionResult
from erisml_compiler.projections.substrate import MoralSubstrate

# Action-kind -> (virtue-expressed-when-done-well, vice-expressed-when-done-poorly)
_ACTION_VIRTUE_AXES: dict[str, tuple[str, str]] = {
    "deceive": ("honesty", "deception"),
    "coerce_or_be_coerced": ("courage", "cowardice"),
    "impose_externality": ("justice", "injustice"),
    "make_or_keep_commitment": ("fidelity", "perfidy"),
    "protect": ("care", "callousness"),
    "act_under_norm": ("prudence", "imprudence"),
}


class VirtueProjection(Projection):
    """Aristotelian-virtue reading of the graph."""

    framework = "virtue_aristotelian"

    def project(
        self,
        substrate: MoralSubstrate,
        *,
        graph: Any = None,
        **kwargs: Any,
    ) -> ProjectionResult:
        findings: list[GateFinding] = []

        findings.append(self._character_finding(substrate))
        findings.append(self._fidelity_finding(substrate))
        findings.append(self._asymmetry_finding(substrate, graph))

        failed = [f for f in findings if not f.passed]
        n_severe_failed = sum(1 for f in failed if f.severity in ("grave", "catastrophic"))

        if n_severe_failed >= 2:
            verdict = "vicious"
        elif failed:
            verdict = "requires_practical_wisdom"
        else:
            verdict = "virtuous"

        return ProjectionResult(
            framework=self.framework,
            verdict=verdict,
            confidence=0.6,  # virtue verdicts are coarse at v0
            findings=findings,
            framework_specific={
                "n_virtue_concerns": len(failed),
                "central_virtues_surfaced": sorted(
                    {f.detail.get("virtue") for f in findings if f.detail.get("virtue")} - {None}
                ),
            },
            metadata={"projection_version": "v0_heuristic"},
        )

    # ---------------------------------------------------- character

    def _character_finding(self, substrate: MoralSubstrate) -> GateFinding:
        """Read the maxim's action_kind and ask whether it expresses
        the virtue or vice on that axis. v0 treats single acts as
        weak evidence — a true virtue ethics requires patterns of
        acts, which we don't have at compile time."""
        if substrate.maxim is None or substrate.maxim.action_kind is None:
            return GateFinding(
                name="character_consistency",
                passed=True,
                reason="No maxim extracted; cannot assess character",
                severity="moderate",
            )
        kind = substrate.maxim.action_kind
        polarity = substrate.maxim.polarity
        if kind not in _ACTION_VIRTUE_AXES:
            return GateFinding(
                name="character_consistency",
                passed=True,
                reason=f"Action kind {kind!r} not in v0 virtue axis library",
                severity="moderate",
            )
        virtue, vice = _ACTION_VIRTUE_AXES[kind]
        detail = {"virtue": virtue, "vice": vice, "action_kind": kind, "polarity": polarity}
        # v0: a vice-evidence kind normally fires as a concern. Negation flips
        # the valence: refraining from a vice ("did not deceive") expresses the
        # virtue, while omitting a virtue ("did not protect") becomes the concern.
        expresses_vice = kind in ("deceive", "impose_externality")
        if polarity == "negated":
            expresses_vice = not expresses_vice
        if expresses_vice:
            refrain = " (by refraining)" if polarity == "negated" else ""
            return GateFinding(
                name="character_consistency",
                passed=False,
                reason=(
                    f"This act{refrain} expresses {vice!r} on the {virtue!r}/{vice!r} "
                    f"axis. A single act is weak evidence; virtue ethics "
                    f"reads patterns. Flag for habit-level review."
                ),
                severity="moderate",
                detail=detail,
            )
        refrain = " (by refraining)" if polarity == "negated" else ""
        return GateFinding(
            name="character_consistency",
            passed=True,
            reason=f"Act{refrain} expresses {virtue!r} on the {virtue!r}/{vice!r} axis",
            severity="moderate",
            detail=detail,
        )

    # ---------------------------------------------------- fidelity

    def _fidelity_finding(self, substrate: MoralSubstrate) -> GateFinding:
        """A standing commitment is virtue-relevant context. Honouring
        commitments is fidelity; breaking them is perfidy. The substrate
        has commitments but no breach-detection yet; we surface the
        commitment count as context."""
        n = len(substrate.commitments)
        if n == 0:
            return GateFinding(
                name="commitment_context",
                passed=True,
                reason="No standing commitments in substrate",
                severity="minor",
            )
        return GateFinding(
            name="commitment_context",
            passed=True,
            reason=(
                f"{n} standing commitment(s); virtue assessment should " f"weigh fidelity to them"
            ),
            severity="minor",
            detail={"n_commitments": n, "virtue": "fidelity"},
        )

    # ---------------------------------------------------- asymmetry

    def _asymmetry_finding(self, substrate: MoralSubstrate, graph: Any = None) -> GateFinding:
        """Virtue ethics is sensitive to *who is doing what to whom* —
        asymmetric power, asymmetric dependence, asymmetric stakes. If
        the act imposes on parties whose vulnerability/agency the
        agent's status doesn't match, the situation calls for
        practical wisdom rather than rule-application."""
        if graph is not None:
            from erisml_compiler.ir.graph import EdgeKind

            imposes = graph.edges_of_kind(EdgeKind.IMPOSES_ON)
            vulnerable_targets = []
            for e in imposes:
                target = graph.get_node(e.dst)
                if target is None:
                    continue
                # vulnerability label or 'vulnerable' / 'patient' / 'dependent' role label
                if any(
                    label in ("vulnerable", "patient", "dependent")
                    for label in (target.labels or [])
                ):
                    vulnerable_targets.append(target.id.removeprefix("stakeholder:"))
            if vulnerable_targets:
                return GateFinding(
                    name="power_asymmetry",
                    passed=False,
                    reason=(
                        f"Act imposes on {len(vulnerable_targets)} vulnerable / "
                        f"dependent stakeholder(s): "
                        f"{', '.join(vulnerable_targets[:3])}. Calls for "
                        f"phronesis."
                    ),
                    severity="moderate",
                    subjects=vulnerable_targets,
                )

        # Substrate fallback: look at consent_states + stakeholder roles.
        vuln_sids = [
            s.id
            for s in substrate.stakeholders
            if "vulnerable" in (getattr(s, "vulnerability", "") or "")
            or any(r in ("patient", "dependent") for r in (getattr(s, "roles", []) or []))
        ]
        if vuln_sids and any(not c.given for c in substrate.consent_states):
            return GateFinding(
                name="power_asymmetry",
                passed=False,
                reason=(
                    f"Substrate flags vulnerable parties {vuln_sids[:3]} + "
                    f"missing consent. Asymmetry warrants practical wisdom."
                ),
                severity="moderate",
                subjects=vuln_sids,
            )
        return GateFinding(
            name="power_asymmetry",
            passed=True,
            reason="No significant power-asymmetry signals in the substrate",
            severity="moderate",
        )
