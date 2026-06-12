"""MoralTensorV3 — JSON-serialisable Pydantic mirror of DEME V3's tensor.

DEME V3's `erisml.ethics.moral_tensor.MoralTensor` stores its values
as `numpy.ndarray` (with a `SparseCOO` fallback). That's fine for
runtime computation but not for the compiler's IR, which is canonical
JSON over the wire.

`MoralTensorV3` keeps the same logical shape — ranks 1-6, first axis
length 9, named axes — but the values live as nested Python lists for
Pydantic serialisation. We also extend the V3 surface with
`DimensionMetadata` per (k, n, ...) cell so the compiler can carry
the same `confidence`, `uncertainty`, `direction`, `source_spans`
provenance that V2 had.

A `to_deme_tensor()` helper converts to the native numpy-backed
`erisml.ethics.moral_tensor.MoralTensor` for DEME consumption. The
import is lazy so this module remains usable without erisml-lib
installed (e.g. during early-phase test runs).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from erisml_compiler.ir.v3.dimensions import DimensionAxis

# Maximum rank supported by DEME V3.
MAX_RANK: int = 6

# First-axis length, by V3 contract.
K_DIM: int = 9


# ===========================================================================
# Per-cell metadata
# ===========================================================================


class DimensionMetadata(BaseModel):
    """Provenance + confidence carried per tensor cell.

    Parallel to the tensor's `values` array but keyed by a flat
    cell-coordinate string `"i0,i1,...,iN"`. Sparse — only cells with
    non-default metadata need an entry.
    """

    model_config = ConfigDict(frozen=False)

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    direction: Literal["positive", "negative", "neutral"] = "neutral"
    source_spans: list[str] = Field(default_factory=list)
    explanation: str | None = None


# ===========================================================================
# Tensor proper
# ===========================================================================


# Nested list type alias for tensor values. Pydantic doesn't accept
# parametric numpy types directly; the validator below enforces shape
# at construction time.
TensorValues = list  # nested floats, depth = rank


class MoralTensorV3(BaseModel):
    """A DEME-V3 compatible moral tensor at ranks 1-6.

    Attributes:
        rank: 1..6 inclusive. Equal to `len(shape)`.
        shape: per-axis lengths. `shape[0]` must equal 9.
        axis_names: per-axis name. `axis_names[0]` must equal `"k"`.
        axis_labels: optional mapping `axis_name -> list[str]` giving
            human-readable labels for the indices along that axis
            (e.g. axis `"n"` -> stakeholder ids).
        values: nested list of floats, shape `shape`. Floats in
            [-1, 1] by V3 convention (signed, like the V2
            `DimensionScore.value`).
        dimension_metadata: sparse mapping from cell-coordinate string
            `"i0,i1,...,iN"` to `DimensionMetadata`.
        veto_flags: hard-constraint violations triggered during DEME
            evaluation. List of human-readable strings.
        veto_locations: coordinates `(i0, i1, ..., iN)` where the vetoes
            apply. Empty tuple `()` means "global".
        metadata: free-form metadata; reserved keys include
            `"repair_residue"` (a [-1, 1] scalar carried from V2
            migration), `"source"`, `"timestamp"`.

    Construction is strict: rank, shape, first-axis length, axis names,
    and values shape are all validated. Use the `zeros` / `from_v2_*`
    constructors when building from existing IR.
    """

    model_config = ConfigDict(frozen=False)

    rank: int = Field(ge=1, le=MAX_RANK)
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    axis_labels: dict[str, list[str]] = Field(default_factory=dict)
    values: TensorValues = Field(default_factory=list)
    dimension_metadata: dict[str, DimensionMetadata] = Field(default_factory=dict)
    veto_flags: list[str] = Field(default_factory=list)
    veto_locations: list[tuple[int, ...]] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ---------- validators ----------

    @field_validator("shape")
    @classmethod
    def _shape_first_axis_is_k(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(v) == 0:
            raise ValueError("shape cannot be empty")
        if v[0] != K_DIM:
            raise ValueError(f"shape[0] must equal {K_DIM} (moral dimensions); got {v[0]}")
        for i, d in enumerate(v):
            if d <= 0:
                raise ValueError(f"shape[{i}] must be positive; got {d}")
        return v

    @field_validator("axis_names")
    @classmethod
    def _axis_names_start_with_k(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(v) == 0:
            raise ValueError("axis_names cannot be empty")
        if v[0] != DimensionAxis.K.value:
            raise ValueError(f"axis_names[0] must be 'k'; got {v[0]!r}")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "MoralTensorV3":
        if self.rank != len(self.shape):
            raise ValueError(f"rank ({self.rank}) must equal len(shape) ({len(self.shape)})")
        if len(self.axis_names) != self.rank:
            raise ValueError(
                f"axis_names length ({len(self.axis_names)}) must equal rank ({self.rank})"
            )
        # Lazily validate values shape only if values are provided.
        if self.values:
            actual_shape = _detect_shape(self.values)
            if actual_shape != self.shape:
                raise ValueError(
                    f"values shape {actual_shape} does not match declared shape {self.shape}"
                )
        # Validate veto-location arity. DEME V3 uses two conventions:
        #   () global veto (applies to whole tensor)
        #   (party_idx,) single-axis veto — length 1, interpreted as the
        #       *party* axis (n) on rank-2+ tensors, or the dimension
        #       axis (k) on rank-1 tensors. Bounds check against axis 1
        #       for rank >= 2, axis 0 for rank 1.
        #   (i0, ..., iN) full coordinate, length = rank
        for loc in self.veto_locations:
            length = len(loc)
            if length == 0:
                continue  # global, no bounds check
            if length == 1:
                axis = 0 if self.rank == 1 else 1
                if loc[0] < 0 or loc[0] >= self.shape[axis]:
                    raise ValueError(
                        f"veto_location {loc} (single-axis) index out of range "
                        f"for axis {axis} (length {self.shape[axis]})"
                    )
                continue
            if length != self.rank:
                raise ValueError(
                    f"veto_location {loc} must have length 0 (global), "
                    f"1 (single-axis), or {self.rank} (full)"
                )
            for i, idx in enumerate(loc):
                if idx < 0 or idx >= self.shape[i]:
                    raise ValueError(f"veto_location {loc} index out of range at axis {i}")
        return self

    # ---------- constructors ----------

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        axis_names: tuple[str, ...] | None = None,
        axis_labels: dict[str, list[str]] | None = None,
    ) -> "MoralTensorV3":
        """Construct a tensor of the given shape, all-zero values."""
        rank = len(shape)
        names = axis_names if axis_names is not None else _default_axis_names(rank)
        return cls(
            rank=rank,
            shape=shape,
            axis_names=names,
            axis_labels=axis_labels or {},
            values=_nested_zeros(shape),
            dimension_metadata={},
        )

    # ---------- accessors ----------

    def get_cell(self, *indices: int) -> float:
        """Return the float value at `indices`. Length must equal rank."""
        if len(indices) != self.rank:
            raise ValueError(f"get_cell needs {self.rank} indices; got {len(indices)}")
        node: Any = self.values
        for i in indices:
            node = node[i]
        return float(node)

    def set_cell(self, *indices_and_value) -> None:
        """`set_cell(i0, i1, ..., iN, value)` — set the value at indices."""
        if len(indices_and_value) != self.rank + 1:
            raise ValueError(
                f"set_cell needs {self.rank} indices + value; got {len(indices_and_value)}"
            )
        *indices, value = indices_and_value
        node: Any = self.values
        for i in indices[:-1]:
            node = node[i]
        node[indices[-1]] = float(value)

    def get_metadata(self, *indices: int) -> DimensionMetadata | None:
        """Return per-cell metadata if any."""
        return self.dimension_metadata.get(_cell_key(indices))

    def set_metadata(self, indices: tuple[int, ...], md: DimensionMetadata) -> None:
        self.dimension_metadata[_cell_key(indices)] = md

    # ---------- DEME interop ----------

    def to_deme_tensor(self):
        """Convert to `erisml.ethics.moral_tensor.MoralTensor` (numpy-backed).

        Raises ImportError when erisml-lib is not installed.
        """
        import numpy as np  # noqa: PLC0415

        try:
            from erisml.ethics.moral_tensor import MoralTensor as DemeMoralTensor  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "to_deme_tensor requires erisml-lib (the DEME V3 reference "
                "implementation) to be installed."
            ) from e

        arr = np.array(self.values, dtype=float)
        if arr.shape != self.shape:
            raise ValueError(
                f"internal: nested values produced shape {arr.shape}, expected {self.shape}"
            )
        return DemeMoralTensor(
            _data=arr,
            shape=self.shape,
            rank=self.rank,
            axis_names=self.axis_names,
            axis_labels=self.axis_labels,
            veto_flags=list(self.veto_flags),
            veto_locations=list(self.veto_locations),
            reason_codes=list(self.reason_codes),
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_deme_tensor(cls, deme_tensor) -> "MoralTensorV3":
        """Construct from a `erisml.ethics.moral_tensor.MoralTensor`.

        Loses metadata that the compiler IR carries but the DEME tensor
        does not (dimension_metadata is set to empty).
        """
        import numpy as np  # noqa: PLC0415

        arr = (
            np.asarray(deme_tensor._data)
            if not isinstance(deme_tensor._data, np.ndarray)
            else deme_tensor._data
        )
        return cls(
            rank=int(deme_tensor.rank),
            shape=tuple(int(x) for x in deme_tensor.shape),
            axis_names=tuple(str(x) for x in deme_tensor.axis_names),
            axis_labels=dict(deme_tensor.axis_labels),
            values=arr.tolist(),
            veto_flags=list(deme_tensor.veto_flags),
            veto_locations=[tuple(int(x) for x in v) for v in deme_tensor.veto_locations],
            reason_codes=list(getattr(deme_tensor, "reason_codes", [])),
            metadata=dict(getattr(deme_tensor, "metadata", {})),
        )


# ===========================================================================
# helpers
# ===========================================================================


def _default_axis_names(rank: int) -> tuple[str, ...]:
    canonical = (
        DimensionAxis.K.value,
        DimensionAxis.N.value,
        DimensionAxis.TAU.value,
        DimensionAxis.A.value,
        DimensionAxis.C.value,
        DimensionAxis.S.value,
    )
    if rank > len(canonical):
        raise ValueError(f"rank {rank} exceeds maximum supported ({len(canonical)})")
    return canonical[:rank]


def _nested_zeros(shape: tuple[int, ...]) -> Any:
    if not shape:
        return 0.0
    return [_nested_zeros(shape[1:]) for _ in range(shape[0])]


def _detect_shape(values: Any) -> tuple[int, ...]:
    shape: list[int] = []
    node = values
    while isinstance(node, list):
        shape.append(len(node))
        node = node[0] if node else None
    return tuple(shape)


def _cell_key(indices: tuple[int, ...]) -> str:
    return ",".join(str(i) for i in indices)
