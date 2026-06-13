"""Canonical JSON + SHA-256 hash for a MoralGraph.

Same node+edge content → same hash regardless of insertion order,
so two compiles of the same input produce bit-identical audit
chains. Edges that differ only in payload are distinct multi-edges
and contribute distinct hash bytes.

Canonical form:

    {
      "nodes": [
        {"id": ..., "kind": ..., "labels": <sorted>, "payload": <sorted-key json>},
        ...                                 (sorted by id)
      ],
      "edges": [
        {"src": ..., "dst": ..., "kind": ..., "payload": <sorted-key json>},
        ...                                 (sorted by (src, dst, kind, payload-json))
      ]
    }
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from erisml_compiler.ir.graph.container import MoralGraph


def _canon_payload(p: dict[str, Any]) -> str:
    """JSON-encode a payload dict with sorted keys and stable separators."""
    return json.dumps(_normalise(p), sort_keys=True, separators=(",", ":"))


def _normalise(v: Any) -> Any:
    """Recursively normalise a payload value so floats/None/bool/etc.
    serialise predictably."""
    if isinstance(v, dict):
        return {k: _normalise(v[k]) for k in sorted(v.keys())}
    if isinstance(v, list):
        return [_normalise(x) for x in v]
    if isinstance(v, tuple):
        return [_normalise(x) for x in v]
    return v


def canonical_graph_json(graph: MoralGraph) -> str:
    """Canonical JSON string for `graph`. Deterministic w.r.t. node /
    edge insertion order."""
    nodes_sorted = sorted(graph.nodes, key=lambda n: n.id)
    nodes_canon = [
        {
            "id": n.id,
            "kind": n.kind.value,
            "labels": sorted(n.labels),
            "payload": _normalise(n.payload),
        }
        for n in nodes_sorted
    ]
    edges_canon = sorted(
        (
            {
                "src": e.src,
                "dst": e.dst,
                "kind": e.kind.value,
                "payload": _normalise(e.payload),
            }
            for e in graph.edges
        ),
        key=lambda d: (d["src"], d["dst"], d["kind"], json.dumps(d["payload"], sort_keys=True)),
    )
    return json.dumps(
        {"nodes": nodes_canon, "edges": edges_canon},
        sort_keys=True,
        separators=(",", ":"),
    )


def graph_hash(graph: MoralGraph) -> str:
    """SHA-256 over the canonical JSON. Returns a 64-char hex string."""
    return hashlib.sha256(canonical_graph_json(graph).encode("utf-8")).hexdigest()
