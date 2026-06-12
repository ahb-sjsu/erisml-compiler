"""DEME V3 tensor builder.

Produces a `MoralTensorV3` at the requested rank from the EM-DAG
outputs and the IR's stakeholder list. Rank semantics:

  rank 1 (k,)               — global moral state vector (V3 equivalent of
                              the V2 MoralVector)
  rank 2 (k, n)             — per-stakeholder moral state (default for
                              Phase 2; aggregate score fanned across all
                              stakeholders until Phase 5 wires per-party
                              evaluation)
  rank 3 (k, n, τ)          — per-stakeholder × per-time (Phase 5)
  rank 4 (k, n, a, c)       — coalition actions (Phase 5)
  rank 5 (k, n, τ, s)       — temporal × uncertainty samples (Phase 5)
  rank 6 (k, n, τ, a, c, s) — full multi-agent context (Phase 5)

This file currently only supports ranks 1 and 2. Higher ranks raise
`NotImplementedError` so the contract is honest about what's wired.
"""

from __future__ import annotations

from erisml_compiler.em_dag import EMDAG
from erisml_compiler.evaluation.moral_vector import build_moral_vector_from_em_outputs
from erisml_compiler.ir.schemas import CompilerIR, EMOutput
from erisml_compiler.ir.v3 import (
    MORAL_DIMENSIONS_V3,
    DimensionMetadata,
    MoralTensorV3,
    migrate_v2_vector_to_v3,
)


def build_moral_tensor_v3(
    ir: CompilerIR,
    em_outputs: dict[str, EMOutput],
    dag: EMDAG,
    *,
    rank: int = 2,
) -> MoralTensorV3:
    """Build a rank-1 or rank-2 V3 tensor from EM outputs + IR stakeholders.

    Args:
        ir: the full compiler IR (we need its `stakeholders` list).
        em_outputs: the EM-DAG output map for the final IR state.
        dag: the EM-DAG used to evaluate; needed to project EM outputs
            onto V2 dimensions before migrating to V3.
        rank: 1 (global vector) or 2 (per-stakeholder fanout). Higher
            ranks are not yet wired; they raise NotImplementedError.

    Returns:
        A `MoralTensorV3` whose JSON-serialisable shape matches DEME V3.
    """
    # The Phase 2 fanout builder is the *fallback* path for when
    # erisml-lib is unavailable. It supports rank 1 and 2 only — the
    # higher-rank builder requires erisml-lib (V3 modules). Higher
    # ranks land via `v3_higher_rank.build_moral_tensor_v3_rank3plus`
    # in the bridge path; the orchestrator dispatches.
    if rank not in (1, 2):
        raise NotImplementedError(
            f"Phase 2 fallback supports rank 1 and 2 only; rank {rank} requires "
            f"erisml-lib (V3 modules). Install with `pip install erisml-lib` and "
            f"the higher-rank path in v3_higher_rank will activate. See "
            f"docs/migration/deme_v3_alignment.md."
        )

    # Step 1: build the V2 vector via the existing projector (no change
    # to legacy code; we'll deprecate this in Phase 4).
    v2_vector = build_moral_vector_from_em_outputs(em_outputs, dag)

    # Step 2: migrate V2 -> V3 rank-1.
    rank1 = migrate_v2_vector_to_v3(v2_vector)

    if rank == 1:
        return rank1

    # Step 3 (rank 2): fan the rank-1 across stakeholders.
    stakeholders = list(ir.stakeholders) if ir.stakeholders else []
    n = len(stakeholders) if stakeholders else 1
    stakeholder_ids = [s.id for s in stakeholders] if stakeholders else ["aggregate"]

    rank2 = MoralTensorV3.zeros(
        shape=(9, n),
        axis_names=("k", "n"),
        axis_labels={
            "k": list(MORAL_DIMENSIONS_V3),
            "n": stakeholder_ids,
        },
    )

    # Copy the rank-1 values across every stakeholder column. Phase 5
    # replaces this fanout with per-party EM evaluation.
    for k in range(9):
        v = rank1.get_cell(k)
        for n_idx in range(n):
            rank2.set_cell(k, n_idx, v)

    # Carry the rank-1 per-dimension metadata onto every cell of each
    # column. Same fanout reasoning applies.
    for k in range(9):
        md = rank1.get_metadata(k)
        if md is None:
            continue
        for n_idx in range(n):
            rank2.set_metadata(
                (k, n_idx),
                DimensionMetadata(
                    confidence=md.confidence,
                    uncertainty=md.uncertainty,
                    direction=md.direction,
                    source_spans=list(md.source_spans),
                    explanation=md.explanation,
                ),
            )

    # Pass tensor-level metadata through.
    rank2.metadata.update(rank1.metadata)
    rank2.metadata["build_strategy"] = "phase2_fanout_from_rank1"
    return rank2
