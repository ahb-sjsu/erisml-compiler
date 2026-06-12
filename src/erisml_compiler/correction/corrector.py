"""Apply human corrections to an IR.

A "corrections file" is a small JSON document of the form:

    {
        "corrector_id": "alice@example.org",
        "rationale": "LLM mislabelled village as 'individual'; fixing to 'community'.",
        "patches": [
            {"op": "set",    "path": "stakeholders.village.type", "value": "community"},
            {"op": "remove", "path": "stakeholders.bogus_id"},
            {"op": "add",    "path": "ethical_facts.fact_new",
                "value": {"id": "fact_new", "kind": "consent", "description": "..."}}
        ]
    }

The corrector validates each patch, applies it to the IR, re-runs the
schema validator, and records the correction in the IR's audit trail.

Supported ops: set, add, remove. Paths are dotted (`stakeholders.<id>.<field>`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from erisml_compiler.ir.schemas import CompilerIR

# Collections that can be patched by id.
_KEYED_COLLECTIONS = frozenset(
    {
        "stakeholders",
        "commitments",
        "events",
        "ethical_facts",
        "conflicts",
        "norms",
        "relations",
    }
)


@dataclass
class CorrectionRecord:
    """One record in the audit trail of human corrections."""

    corrector_id: str
    rationale: str
    applied_at_utc: str
    n_patches_applied: int
    n_patches_failed: int
    pre_correction_ir_hash: str
    post_correction_ir_hash: str
    diff_summary: list[str]


def _split_path(path: str) -> list[str]:
    return path.split(".")


def _index_collection_by_id(items: list) -> dict[str, Any]:
    """Return a map from id to (index_in_list, item)."""
    out = {}
    for i, item in enumerate(items):
        item_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if item_id is not None:
            out[item_id] = (i, item)
    return out


def _apply_op(ir_dict: dict, op: dict) -> None:
    """Mutate `ir_dict` in place per a single patch op.

    Path grammar:
        <scalar_field>                                e.g. "canonical_form"
        <collection>.<entity_id>                       e.g. "stakeholders.village"
        <collection>.<entity_id>.<field>               e.g. "stakeholders.village.type"
    """
    op_kind = op.get("op")
    path = op.get("path", "")
    parts = _split_path(path)
    if not parts:
        raise ValueError(f"Empty path in op: {op}")

    head, *rest = parts

    # ----- scalar field -----
    if not rest:
        if op_kind == "set":
            ir_dict[head] = op.get("value")
            return
        if op_kind == "remove":
            ir_dict.pop(head, None)
            return
        if op_kind == "add":
            if head in ir_dict:
                raise ValueError(f"'add' on existing key {head!r}")
            ir_dict[head] = op.get("value")
            return
        raise ValueError(f"Unknown op {op_kind!r}")

    # ----- collection.<id>[.<field>] -----
    if head not in _KEYED_COLLECTIONS:
        raise ValueError(f"Unsupported collection in path: {head!r}")
    if head not in ir_dict or ir_dict[head] is None:
        ir_dict[head] = []

    collection = ir_dict[head]
    entity_id = rest[0]
    indexed = _index_collection_by_id(collection)

    if len(rest) == 1:
        # Whole-entity ops.
        if op_kind == "add":
            if entity_id in indexed:
                raise ValueError(f"'add' on existing entity {head}.{entity_id}")
            value = dict(op.get("value", {}))
            value["id"] = entity_id
            collection.append(value)
            return
        if op_kind == "set":
            value = dict(op.get("value", {}))
            value["id"] = entity_id
            if entity_id in indexed:
                idx, _ = indexed[entity_id]
                collection[idx] = value
            else:
                collection.append(value)
            return
        if op_kind == "remove":
            if entity_id not in indexed:
                return
            idx, _ = indexed[entity_id]
            collection.pop(idx)
            return
        raise ValueError(f"Unknown op {op_kind!r}")

    # ----- collection.<id>.<field>[.<nested>] -----
    if entity_id not in indexed:
        raise ValueError(f"No such entity {head}.{entity_id}")
    idx, item = indexed[entity_id]
    target = item
    field_chain = rest[1:]
    for f in field_chain[:-1]:
        if f not in target:
            target[f] = {}
        target = target[f]
    final_field = field_chain[-1]

    if op_kind == "set":
        target[final_field] = op.get("value")
    elif op_kind == "add":
        if final_field in target:
            raise ValueError(f"'add' on existing field {path}")
        target[final_field] = op.get("value")
    elif op_kind == "remove":
        target.pop(final_field, None)
    else:
        raise ValueError(f"Unknown op {op_kind!r}")

    collection[idx] = item  # write back


def apply_corrections(
    ir: CompilerIR,
    corrections: dict,
    corrector_id: str | None = None,
) -> tuple[CompilerIR, CorrectionRecord, list[str]]:
    """Apply a corrections document to an IR.

    Returns (new_ir, correction_record, applied_op_summaries).

    Patches that fail (e.g., 'add' on existing key) are skipped and logged
    in the record's n_patches_failed counter; valid patches still apply.
    """
    from erisml_compiler.audit.hash_chain import compute_ir_hash
    from erisml_compiler.correction.diff import diff_irs

    pre_hash = compute_ir_hash(ir)

    ir_dict = ir.model_dump(mode="json")
    # Drop audit since the corrector will re-finalize.
    ir_dict.pop("audit", None)

    summaries: list[str] = []
    failed = 0
    applied = 0
    for op in corrections.get("patches", []):
        try:
            _apply_op(ir_dict, op)
            applied += 1
            summaries.append(f"OK     {op.get('op')} {op.get('path')}")
        except Exception as exc:
            failed += 1
            summaries.append(f"FAILED {op.get('op')} {op.get('path')}: {exc}")

    # Re-validate by reconstructing through Pydantic.
    new_ir = CompilerIR.model_validate(ir_dict)

    post_hash = compute_ir_hash(new_ir)
    diff = diff_irs(ir, new_ir)

    record = CorrectionRecord(
        corrector_id=corrector_id or corrections.get("corrector_id", "anonymous"),
        rationale=corrections.get("rationale", ""),
        applied_at_utc=datetime.now(timezone.utc).isoformat(),
        n_patches_applied=applied,
        n_patches_failed=failed,
        pre_correction_ir_hash=pre_hash,
        post_correction_ir_hash=post_hash,
        diff_summary=diff.summary_lines(),
    )

    # Stamp the record into the IR's `extra` block under "corrections".
    new_ir.extra = dict(new_ir.extra)
    history = new_ir.extra.setdefault("corrections", [])
    history.append(
        {
            "corrector_id": record.corrector_id,
            "rationale": record.rationale,
            "applied_at_utc": record.applied_at_utc,
            "n_patches_applied": record.n_patches_applied,
            "n_patches_failed": record.n_patches_failed,
            "pre_correction_ir_hash": record.pre_correction_ir_hash,
            "post_correction_ir_hash": record.post_correction_ir_hash,
            "diff_summary": record.diff_summary,
        }
    )

    return new_ir, record, summaries


class Corrector:
    """Functional facade: load corrections from a file, apply, optionally
    re-run the EM-DAG/canonicalizer to verify the corrected IR is still
    coherent."""

    def __init__(self, ir: CompilerIR):
        self.ir = ir

    def apply(self, corrections_path: str | Path, corrector_id: str | None = None):
        path = Path(corrections_path)
        corrections = json.loads(path.read_text(encoding="utf-8"))
        return apply_corrections(self.ir, corrections, corrector_id=corrector_id)
