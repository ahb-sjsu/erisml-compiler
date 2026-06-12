"""Detect conflicts between dimensions of the MoralVector.

A conflict is flagged when two dimensions are simultaneously active with
opposite signs of significant magnitude. The MVP implementation looks at
the final MoralVector and at the extractor-supplied conflicts; it leaves
the conflict objects mostly intact (the extractor knows the most).
"""

from __future__ import annotations

from erisml_compiler.ir.schemas import CompilerIR, Conflict, MoralVector

THRESHOLD = 0.5


def detect_conflicts(ir: CompilerIR, vector: MoralVector) -> list[Conflict]:
    """Augment extractor-supplied conflicts with vector-based detection."""
    dims = {
        "physical_harm": vector.physical_harm,
        "rights_respect": vector.rights_respect,
        "fairness_equity": vector.fairness_equity,
        "autonomy_consent": vector.autonomy_consent,
        "legitimacy_trust": vector.legitimacy_trust,
        "epistemic_quality": vector.epistemic_quality,
        "care_protection": vector.care_protection,
        "vow_fidelity": vector.vow_fidelity,
        "third_party_externality": vector.third_party_externality,
        "repair_residue": vector.repair_residue,
    }
    positives = [(d, s) for d, s in dims.items() if s.value > THRESHOLD]
    negatives = [(d, s) for d, s in dims.items() if s.value < -THRESHOLD]

    extra: list[Conflict] = []
    counter = len(ir.conflicts) + 1
    for pd, ps in positives:
        for nd, ns in negatives:
            if pd == nd:
                continue
            extra.append(
                Conflict(
                    id=f"vector_conflict_{counter:03d}",
                    name=f"{pd} (+{ps.value:.2f}) vs {nd} ({ns.value:.2f})",
                    stakeholders=[],
                    dimensions=[pd, nd],
                    severity="grave" if abs(ns.value) > 0.85 else "moderate",
                    resolution_status="unresolved",
                    requires_escalation=abs(ns.value) > 0.85,
                    description=(
                        f"Vector-level conflict between positive {pd} and " f"negative {nd}."
                    ),
                )
            )
            counter += 1

    return ir.conflicts + extra
