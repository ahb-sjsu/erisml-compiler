"""RLEF training-record export.

A single RLEF record bundles:
    - source text
    - structured annotation (stakeholders, commitments, ethical facts,
      conflicts, canonical form)
    - reference moral vector / timeline
    - DEME verdict
    - any human corrections (the human-corrected IR replaces the raw IR
      when present)

The format is intentionally simple JSON so downstream RL-trainer code can
ingest it without a Pydantic dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from erisml_compiler.ir.schemas import CompilerIR


def to_rlef_record(ir: CompilerIR, human_corrections: dict | None = None) -> dict:
    """Build the RLEF record from a compiled IR.

    Includes the DAG-native `moral_graph` (nodes + edges + canonical
    hash) and the per-framework `projections` block so trainer code
    can learn against the typed graph and/or any specific framework's
    verdict, not just the (consequentialist) flat-tensor timeline.

    Bumped to `rlef_v0.2` to record this richer shape.
    """
    record = {
        "schema": "rlef_v0.2",
        "source_text": ir.document.raw_text,
        "document_id": ir.document.doc_id,
        "canonical_form": ir.canonical_form,
        "stakeholders": [s.model_dump(mode="json") for s in ir.stakeholders],
        "commitments": [c.model_dump(mode="json") for c in ir.commitments],
        "events": [e.model_dump(mode="json") for e in ir.events],
        "ethical_facts": [f.model_dump(mode="json") for f in ir.ethical_facts],
        "conflicts": [cf.model_dump(mode="json") for cf in ir.conflicts],
        "moral_vector_timeline": [t.model_dump(mode="json") for t in ir.timeline],
        "deme_verdict": ir.deme_verdict.model_dump(mode="json") if ir.deme_verdict else None,
        "em_outputs": {k: v.model_dump(mode="json") for k, v in ir.em_outputs.items()},
        "audit": ir.audit.model_dump(mode="json") if ir.audit else None,
        "human_corrections": human_corrections,
        "moral_graph": _serialise_graph(ir),
        "projections": dict(ir.projections),
        "cross_projection_disagreement": ir.cross_projection_disagreement,
    }
    return record


def _serialise_graph(ir: CompilerIR) -> dict | None:
    """Serialise the MoralGraph (if present) into RLEF-friendly JSON."""
    if ir.graph is None:
        return None
    from erisml_compiler.ir.graph import canonical_graph_json, graph_hash

    return {
        "graph_hash": graph_hash(ir.graph),
        "canonical_json": canonical_graph_json(ir.graph),
        "nodes": [n.model_dump(mode="json") for n in ir.graph.nodes],
        "edges": [e.model_dump(mode="json") for e in ir.graph.edges],
        "node_counts": ir.graph.node_count_by_kind(),
        "edge_counts": ir.graph.edge_count_by_kind(),
    }


def export_rlef(
    ir: CompilerIR,
    path: str | Path,
    human_corrections: dict | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = to_rlef_record(ir, human_corrections)
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return p
