"""ErisML Compiler Intermediate Representation.

V2 surface (10-dim MoralVector / per-stakeholder MoralTensor) is exported
at module top level and remains the producer-side default through Phase
1 of the DEME V3 alignment.

V3 surface (9-dim, rank-1..6 MoralTensorV3) is available from
`erisml_compiler.ir.v3` and is the consumer-side default for the
DEME bridge from Phase 2 onward. See `docs/migration/deme_v3_alignment.md`.
"""
from erisml_compiler.ir.schemas import (
    AuditRecord,
    Commitment,
    CompilerIR,
    Conflict,
    DEMEVerdict,
    DimensionScore,
    Document,
    EthicalFact,
    Event,
    MoralTensor,
    MoralVector,
    Norm,
    Relation,
    Segment,
    SourceSpan,
    Stakeholder,
    TimelineEntry,
)
from erisml_compiler.ir.v3 import (
    DIMENSION_MATRIX_3X3,
    MORAL_DIMENSIONS_V3,
    DimensionMetadata,
    MoralTensorV3,
    migrate_v2_tensor_to_v3,
    migrate_v2_vector_to_v3,
)

__all__ = [
    # V2
    "AuditRecord",
    "Commitment",
    "CompilerIR",
    "Conflict",
    "DEMEVerdict",
    "DimensionScore",
    "Document",
    "EthicalFact",
    "Event",
    "MoralTensor",
    "MoralVector",
    "Norm",
    "Relation",
    "Segment",
    "SourceSpan",
    "Stakeholder",
    "TimelineEntry",
    # V3
    "DIMENSION_MATRIX_3X3",
    "DimensionMetadata",
    "MORAL_DIMENSIONS_V3",
    "MoralTensorV3",
    "migrate_v2_tensor_to_v3",
    "migrate_v2_vector_to_v3",
]
