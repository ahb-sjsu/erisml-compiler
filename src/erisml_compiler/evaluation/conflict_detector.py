"""Detect conflicts between dimensions of DEME's moral vector.

A conflict is flagged when two dimensions are simultaneously active with opposite signs of
significant magnitude. Each of DEME's nine dimensions (erisml-lib, values in [0, 1], higher
better except physical harm, where higher is worse) is read on the signed scale the earlier
EM-DAG vector used, s = 2v - 1 (physical harm inverted first), so the threshold keeps its
meaning: a dimension above +0.5 is clearly respected, one below -0.5 clearly violated. The
extractor-supplied conflicts are kept as they are (the extractor knows the most).
"""

from __future__ import annotations

from erisml.ethics.moral_vector import MoralVector

from erisml_compiler.ir.schemas import CompilerIR, Conflict

THRESHOLD = 0.5


def signed(vector: MoralVector) -> dict[str, float]:
    """Each DEME dimension on the signed scale: +1 fully respected, -1 fully violated."""
    out: dict[str, float] = {}
    for dim in MoralVector.core_dimension_names():
        v = getattr(vector, dim)
        good = 1.0 - v if dim == "physical_harm" else v
        out[dim] = 2.0 * good - 1.0
    return out


def detect_conflicts(ir: CompilerIR, vector: MoralVector) -> list[Conflict]:
    """Augment extractor-supplied conflicts with vector-based detection."""
    dims = signed(vector)
    positives = [(d, s) for d, s in dims.items() if s > THRESHOLD]
    negatives = [(d, s) for d, s in dims.items() if s < -THRESHOLD]

    extra: list[Conflict] = []
    counter = len(ir.conflicts) + 1
    for pd, ps in positives:
        for nd, ns in negatives:
            if pd == nd:
                continue
            extra.append(
                Conflict(
                    id=f"vector_conflict_{counter:03d}",
                    name=f"{pd} (+{ps:.2f}) vs {nd} ({ns:.2f})",
                    stakeholders=[],
                    dimensions=[pd, nd],
                    severity="grave" if abs(ns) > 0.85 else "moderate",
                    resolution_status="unresolved",
                    requires_escalation=abs(ns) > 0.85,
                    description=f"Vector-level conflict between positive {pd} and negative {nd}.",
                )
            )
            counter += 1

    return ir.conflicts + extra
