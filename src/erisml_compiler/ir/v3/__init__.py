"""DEME V3 alignment — IR schema for ranks 1-6 moral tensors.

This subpackage adds DEME V3-compatible IR structures alongside the
existing V2 (10-dimension, rank-1 / rank-2-dict) IR in
`erisml_compiler.ir.schemas`. The V2 surface is preserved while
producers migrate (Phases 2-3 of the alignment plan).

See `docs/migration/deme_v3_alignment.md` for the full plan.

Public surface:

    MORAL_DIMENSIONS_V3      tuple of 9 dimension names in canonical order
    DIMENSION_MATRIX_3X3     mapping of dim name to (row, col) in the
                             "Nine Dimensions of Ethical Assessment" matrix
    DimensionMetadata        per-dimension provenance fields (carried by the
                             compiler IR, not by DEME's raw numpy tensor)
    MoralTensorV3            Pydantic model: rank, shape, axis_names,
                             axis_labels, values, dimension_metadata,
                             veto_flags, veto_locations
    migrate_v2_vector_to_v3  V2 MoralVector -> rank-1 MoralTensorV3
    migrate_v2_tensor_to_v3  V2 MoralTensor -> rank-2 MoralTensorV3
"""

from erisml_compiler.ir.v3.dimensions import (
    DIMENSION_MATRIX_3X3,
    MORAL_DIMENSIONS_V3,
    V2_TO_V3_DIMENSION_MAP,
    DimensionAxis,
)
from erisml_compiler.ir.v3.tensor import (
    DimensionMetadata,
    MoralTensorV3,
)
from erisml_compiler.ir.v3.migration import (
    migrate_v2_tensor_to_v3,
    migrate_v2_vector_to_v3,
)

__all__ = [
    "DIMENSION_MATRIX_3X3",
    "DimensionAxis",
    "DimensionMetadata",
    "MORAL_DIMENSIONS_V3",
    "MoralTensorV3",
    "V2_TO_V3_DIMENSION_MAP",
    "migrate_v2_tensor_to_v3",
    "migrate_v2_vector_to_v3",
]
