"""Structural diff between two CompilerIR objects.

The diff is computed on the canonical JSON dump, organised by collection
(stakeholders, commitments, events, ethical_facts, conflicts), so a
reviewer sees per-entity additions, removals, and per-field changes
rather than a raw text diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from erisml_compiler.ir.schemas import CompilerIR

# Top-level collections we diff entity-by-entity (keyed by `id`).
_KEYED_COLLECTIONS = (
    "stakeholders",
    "commitments",
    "events",
    "ethical_facts",
    "conflicts",
    "norms",
    "relations",
)

# Top-level scalar fields we diff directly.
_SCALAR_FIELDS = (
    "canonical_form",
    "schema_version",
)


@dataclass
class FieldChange:
    """One field value changed between two versions of the same entity."""

    field_path: str
    old: Any
    new: Any


@dataclass
class EntityDiff:
    """All changes that apply to one entity (one stakeholder, commitment, ...).

    Exactly one of {added=True, removed=True, changes=non-empty} is true.
    """

    collection: str
    entity_id: str
    added: bool = False
    removed: bool = False
    changes: list[FieldChange] = field(default_factory=list)

    @property
    def is_modified(self) -> bool:
        return bool(self.changes)


@dataclass
class IRDiff:
    """Total diff between two IRs."""

    scalar_changes: list[FieldChange] = field(default_factory=list)
    entity_diffs: list[EntityDiff] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scalar_changes and not self.entity_diffs

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for fc in self.scalar_changes:
            lines.append(f"  [scalar] {fc.field_path}: {fc.old!r} -> {fc.new!r}")
        for ed in self.entity_diffs:
            if ed.added:
                lines.append(f"  [+ added] {ed.collection}.{ed.entity_id}")
            elif ed.removed:
                lines.append(f"  [- removed] {ed.collection}.{ed.entity_id}")
            else:
                for fc in ed.changes:
                    lines.append(
                        f"  [~ changed] {ed.collection}.{ed.entity_id}.{fc.field_path}: "
                        f"{fc.old!r} -> {fc.new!r}"
                    )
        return lines

    def as_dict(self) -> dict:
        return {
            "scalar_changes": [
                {"field_path": fc.field_path, "old": fc.old, "new": fc.new}
                for fc in self.scalar_changes
            ],
            "entity_diffs": [
                {
                    "collection": ed.collection,
                    "entity_id": ed.entity_id,
                    "added": ed.added,
                    "removed": ed.removed,
                    "changes": [
                        {"field_path": fc.field_path, "old": fc.old, "new": fc.new}
                        for fc in ed.changes
                    ],
                }
                for ed in self.entity_diffs
            ],
        }


def _index_by_id(items: list[dict]) -> dict[str, dict]:
    return {item["id"]: item for item in items if "id" in item}


def _field_diffs(a: dict, b: dict, skip: tuple = ("id",)) -> list[FieldChange]:
    """All keys where a and b differ. Recurses one level into nested dicts."""
    out: list[FieldChange] = []
    keys = set(a.keys()) | set(b.keys())
    for k in sorted(keys):
        if k in skip:
            continue
        av = a.get(k)
        bv = b.get(k)
        if av == bv:
            continue
        out.append(FieldChange(field_path=k, old=av, new=bv))
    return out


def diff_irs(old: CompilerIR, new: CompilerIR) -> IRDiff:
    """Compute a structured diff between two CompilerIR objects.

    Skips audit and timestamps (those change every run by design).
    """
    diff = IRDiff()
    old_d = old.model_dump(mode="json", exclude={"audit"})
    new_d = new.model_dump(mode="json", exclude={"audit"})

    # ----- scalar fields -----
    for field_name in _SCALAR_FIELDS:
        ov = old_d.get(field_name)
        nv = new_d.get(field_name)
        if ov != nv:
            diff.scalar_changes.append(FieldChange(field_path=field_name, old=ov, new=nv))

    # ----- keyed collections -----
    for col in _KEYED_COLLECTIONS:
        old_items = _index_by_id(old_d.get(col, []) or [])
        new_items = _index_by_id(new_d.get(col, []) or [])
        all_ids = set(old_items) | set(new_items)
        for eid in sorted(all_ids):
            if eid not in old_items:
                diff.entity_diffs.append(EntityDiff(collection=col, entity_id=eid, added=True))
            elif eid not in new_items:
                diff.entity_diffs.append(EntityDiff(collection=col, entity_id=eid, removed=True))
            else:
                changes = _field_diffs(old_items[eid], new_items[eid])
                if changes:
                    diff.entity_diffs.append(
                        EntityDiff(collection=col, entity_id=eid, changes=changes)
                    )
    return diff
