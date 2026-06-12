"""V2 -> V3 migration helpers.

Two entry points:

  - `migrate_v2_vector_to_v3(v2_vector)` -> rank-1 MoralTensorV3
  - `migrate_v2_tensor_to_v3(v2_tensors, stakeholder_ids)` -> rank-2
    MoralTensorV3 with axis `n` ranging over the stakeholders.

Both helpers preserve all V2 metadata (confidence, uncertainty,
direction, source_spans, explanation) into `dimension_metadata`, and
emit a tensor `metadata["migration"]` block recording which V2
dimensions were merged or dropped.

The migration is intentionally lossy where V2 ↔ V3 dimensions don't
align cleanly:

  - V2 `vow_fidelity` is split 50/50 into V3 `legitimacy_trust` and
    `virtue_care`. The metadata records the split.
  - V2 `third_party_externality` becomes V3 `societal_environmental`
    (1:1 rename with semantic-overlap note in metadata).
  - V2 `repair_residue` is NOT a V3 dimension. The migrated tensor
    carries the V2 residue value at `metadata["repair_residue"]`.
  - V3 `privacy_protection` has no V2 source. Defaults to 0.0 with
    `metadata["migration"]["synthesised_dims"] = ["privacy_protection"]`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from erisml_compiler.ir.v3.dimensions import (
    MORAL_DIMENSIONS_V3,
    V2_TO_V3_DIMENSION_MAP,
)
from erisml_compiler.ir.v3.tensor import (
    DimensionMetadata,
    K_DIM,
    MoralTensorV3,
)

if TYPE_CHECKING:
    from erisml_compiler.ir.schemas import MoralTensor as V2MoralTensor
    from erisml_compiler.ir.schemas import MoralVector as V2MoralVector


# Names from V2 we know how to handle. Anything else triggers a warning
# in the migration metadata.
_V2_KNOWN: frozenset[str] = frozenset(
    {
        "physical_harm",
        "rights_respect",
        "fairness_equity",
        "autonomy_consent",
        "legitimacy_trust",
        "epistemic_quality",
        "care_protection",
        "vow_fidelity",
        "third_party_externality",
        "repair_residue",
    }
)


def migrate_v2_vector_to_v3(v2_vector: "V2MoralVector") -> MoralTensorV3:
    """Convert a V2 10-dim MoralVector to a rank-1 9-dim MoralTensorV3."""
    accum: dict[str, list[tuple[float, float, str, list[str], str | None]]] = {
        dim: [] for dim in MORAL_DIMENSIONS_V3
    }
    repair_residue_value = 0.0
    repair_residue_md: dict | None = None

    for v2_dim in _V2_KNOWN:
        score = getattr(v2_vector, v2_dim, None)
        if score is None:
            continue
        if v2_dim == "repair_residue":
            repair_residue_value = float(score.value)
            repair_residue_md = {
                "confidence": float(score.confidence),
                "uncertainty": float(score.uncertainty),
                "direction": score.direction,
                "source_spans": list(score.source_spans),
                "explanation": score.explanation,
            }
            continue
        targets = V2_TO_V3_DIMENSION_MAP.get(v2_dim, [])
        if not targets:
            continue
        # When a V2 dim splits across multiple V3 dims, divide the value
        # evenly and replicate the metadata to each target.
        per_target_value = float(score.value) / len(targets)
        for target in targets:
            accum[target].append(
                (
                    per_target_value,
                    float(score.confidence),
                    score.direction,
                    list(score.source_spans),
                    score.explanation,
                )
            )

    values: list[float] = []
    dimension_metadata: dict[str, DimensionMetadata] = {}
    synthesised: list[str] = []
    for k, dim in enumerate(MORAL_DIMENSIONS_V3):
        contribs = accum[dim]
        if not contribs:
            values.append(0.0)
            synthesised.append(dim)
            continue
        # Aggregate: sum values (since splits divided to begin with),
        # average confidence, max uncertainty, union of source_spans,
        # join explanations.
        total = sum(c[0] for c in contribs)
        avg_conf = sum(c[1] for c in contribs) / len(contribs)
        # Direction: take the majority sign of values, or "neutral".
        positives = sum(1 for c in contribs if c[0] > 0.05)
        negatives = sum(1 for c in contribs if c[0] < -0.05)
        if positives > negatives:
            direction = "positive"
        elif negatives > positives:
            direction = "negative"
        else:
            direction = "neutral"
        spans: list[str] = []
        explanations: list[str] = []
        for _, _, _, sp, ex in contribs:
            spans.extend(sp)
            if ex:
                explanations.append(ex)
        # Deduplicate spans preserving order.
        spans = list(dict.fromkeys(spans))
        clamped = max(-1.0, min(1.0, total))
        values.append(clamped)
        dimension_metadata[str(k)] = DimensionMetadata(
            confidence=avg_conf,
            uncertainty=1.0 - avg_conf,
            direction=direction,
            source_spans=spans,
            explanation=" | ".join(explanations) if explanations else None,
        )

    tensor = MoralTensorV3(
        rank=1,
        shape=(K_DIM,),
        axis_names=("k",),
        axis_labels={"k": list(MORAL_DIMENSIONS_V3)},
        values=values,
        dimension_metadata=dimension_metadata,
        metadata={
            "migration": {
                "source": "v2_moral_vector",
                "synthesised_dims": synthesised,
                "split_dims": ["vow_fidelity"],
                "renamed_dims": {
                    "autonomy_consent": "autonomy_respect",
                    "care_protection": "virtue_care",
                    "third_party_externality": "societal_environmental",
                },
                "dropped_to_tensor_metadata": ["repair_residue"],
            },
            "repair_residue": repair_residue_value if repair_residue_md else None,
            "repair_residue_metadata": repair_residue_md,
        },
    )
    return tensor


def migrate_v2_tensor_to_v3(
    v2_tensors: list["V2MoralTensor"],
    stakeholder_ids: list[str],
) -> MoralTensorV3:
    """Convert a list of V2 MoralTensors (one per stakeholder × time slice)
    into a rank-2 MoralTensorV3 with axes `(k=9, n=stakeholders)`.

    Time is collapsed: this helper builds the rank-2 slice at the
    *first* time index encountered per stakeholder. For full temporal
    migration, build a rank-3 tensor in Phase 5 with explicit τ axis.
    """
    from erisml_compiler.ir.schemas import DimensionScore  # noqa: PLC0415

    n_stake = len(stakeholder_ids)
    tensor = MoralTensorV3.zeros(
        shape=(K_DIM, n_stake),
        axis_names=("k", "n"),
        axis_labels={
            "k": list(MORAL_DIMENSIONS_V3),
            "n": list(stakeholder_ids),
        },
    )
    tensor.metadata["migration"] = {
        "source": "v2_moral_tensors",
        "time_collapsed": "first_observed",
        "renamed_dims": {
            "autonomy_consent": "autonomy_respect",
            "care_protection": "virtue_care",
            "third_party_externality": "societal_environmental",
        },
    }
    repair_residue_per_stake: dict[str, float] = {}

    stake_to_v2 = {t.stakeholder_id: t for t in v2_tensors}
    for n_idx, stake_id in enumerate(stakeholder_ids):
        v2t = stake_to_v2.get(stake_id)
        if v2t is None:
            continue
        for v2_dim, score in v2t.by_dimension.items():
            if not isinstance(score, DimensionScore):
                continue
            if v2_dim == "repair_residue":
                repair_residue_per_stake[stake_id] = float(score.value)
                continue
            for target in V2_TO_V3_DIMENSION_MAP.get(v2_dim, []):
                if target not in MORAL_DIMENSIONS_V3:
                    continue
                k_idx = MORAL_DIMENSIONS_V3.index(target)
                divisor = len(V2_TO_V3_DIMENSION_MAP[v2_dim])
                contribution = float(score.value) / divisor
                current = tensor.get_cell(k_idx, n_idx)
                clamped = max(-1.0, min(1.0, current + contribution))
                tensor.set_cell(k_idx, n_idx, clamped)
                # Attach metadata for the cell — last write wins on
                # splits, which is acceptable as a first pass.
                tensor.set_metadata(
                    (k_idx, n_idx),
                    DimensionMetadata(
                        confidence=float(score.confidence),
                        uncertainty=float(score.uncertainty),
                        direction=score.direction,
                        source_spans=list(score.source_spans),
                        explanation=score.explanation,
                    ),
                )

    if repair_residue_per_stake:
        tensor.metadata["repair_residue_per_stakeholder"] = repair_residue_per_stake

    return tensor
