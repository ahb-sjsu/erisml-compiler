# Release planning 07 — Graph-aware consumer migration

**Status:** design + deferral note.
**Owner:** TBD.
**Predecessor:** [release-planning-06](./release-planning-06-framework-pluralist-architecture.md).

## Context

The DAG-native refactor (release-planning-06) lands the `MoralGraph` as
the primary descriptive substrate. Several consumers of the IR still
read flat fields (`ir.stakeholders`, `ir.commitments`, ...). The flat
fields remain populated for backward compat (graph -> flat
back-derivation via `flat_from_graph`), so nothing is broken — but
those consumers also don't benefit from the typed-edge structure of
the graph.

This note documents the migration plan for each consumer with explicit
*why-deferred* reasoning where appropriate.

## Migration matrix

| Consumer | Current state | Migration plan | Deferred because |
|---|---|---|---|
| **`ConsequentialistProjection`** | reads flat fields via EM-DAG | metadata records graph-derived stats; EM-DAG is now graph-native at the helper layer (see below) | n/a — landed |
| **EM-DAG (10 modules)** | now graph-native via `_helpers.py` | helpers (`facts_of_kind`, `active_commitments`, `vulnerable_stakeholders`, `nonconsenting_third_party_ids`, ...) read MoralGraph nodes/edges when `ir.graph` is set; flat fallback otherwise. Verdicts byte-identical against golden baseline on all 3 bundled scenarios. | n/a — landed |
| **`DeonticProjection`** | graph-native (pattern-matches `treats_as`, `imposes_on`) | ✅ done | n/a |
| **`VirtueProjection`** | graph-native for power-asymmetry; substrate fallback elsewhere | ✅ done | n/a |
| **`CareEthicsProjection`** | graph-native for dependency-response; substrate fallback elsewhere | ✅ done | n/a |
| **`bench/runner.py`** | reads flat fields | extend to score *per-projection*; needs framework-specific gold | v0.2 milestone — requires curating gold answers per framework, orders of magnitude more curation work |
| **`monitor/`** (I-EIP) | operates on activation traces, not facts | minimal graph use; could read `graph_hash` for trace identity | low ROI — monitor is about activation-lens vs text-lens disagreement, not about how facts are represented |
| **`export/rlef.py`** | flat + timeline + verdict | ✅ now includes `moral_graph`, `projections`, `cross_projection_disagreement` (schema bump rlef_v0.2) | n/a |
| **`silicon/hls_emit.py`** | operates on the EM-DAG (not the IR's facts) | no graph consumption needed | silicon emits the *evaluator* (EM-DAG) as HLS code, independent of which facts feed it |

## EM-DAG graph-native port — landed

The port took a different shape than originally planned. Rather than
rewriting all 10 module evaluators to take a `MoralGraph` parameter
(which would have required changing every module's signature and
risked subtle behaviour changes), we made the *helpers* graph-native:

```python
# In em_dag/modules/_helpers.py
def facts_of_kind(ir, kind):
    if getattr(ir, "graph", None) is not None:
        return _facts_from_graph(ir.graph, kind)
    return [f for f in ir.ethical_facts if f.kind == kind]

def active_commitments(ir):
    if getattr(ir, "graph", None) is not None:
        comms = _commitments_from_graph(ir.graph)
    else:
        comms = list(ir.commitments)
    return [c for c in comms if c.status in ("active", "active_but_defeasible", "fulfilled")]
```

The 10 modules continue to call the same helpers; the helpers now
prefer the graph when present and fall back to flat fields otherwise.
Two modules (`CareEM`, `ExternalityEM`) that previously read
`ir.stakeholders` directly were updated to use new graph-native
helpers (`vulnerable_stakeholders`, `stakeholders_with_role`,
`nonconsenting_third_party_ids`). The non-consenting-third-party
helper specifically reads the typed-edge way:
**`IMPOSES_ON` targets without a paired `CONSENTS_TO` edge** — the
actual semantic relationship rather than role-label matching.

**Golden-test verification:** EM outputs (10 modules × value +
confidence + direction across 3 bundled scenarios = 90 numbers
per scenario) are byte-identical to the pre-port flat-field baseline
captured in `tests/golden_em_dag_flat.json`. The
`tests/test_em_dag_graph_native.py` suite checks the golden as well
as graph-read behaviour (helpers genuinely query the graph: blanking
the flat lists doesn't break them) and the flat fallback (IRs without
a graph still work).

Why this shape rather than per-module rewrites:

  - Lower regression risk (the data flowing into the aggregators
    `aggregate_negative` / `aggregate_positive` is the same)
  - Smaller diff (~150 LOC in `_helpers.py` + 2-line edits in
    `care.py` and `externality.py`)
  - The "graph is the source of truth" property holds: the helpers
    can run with the flat lists explicitly cleared and still return
    correct data, proving the graph path is load-bearing
  - Future modules that genuinely want typed-edge access (`IMPOSES_ON`
    with severity payloads, `TREATS_AS[role=...]` filters, etc.) can
    bypass the helpers entirely — that path is now open

## Bench-per-projection (deferred to v0.2)

`MoralTensorBench v0.1` scored against the legacy single-verdict
path. With four projections live (consequentialist + deontic +
virtue + care-ethics), the natural extension is per-projection
scoring:

```yaml
expected:
  consequentialist:
    canonical_form: ...
    overall_verdict: permitted
    per_party_verdicts: {...}
  deontic:
    universalizability: pass
    mere_means: pass
    valid_consent: pass
    overall_verdict: permissible
  virtue:
    overall_verdict: virtuous
  care:
    overall_verdict: caring
```

Plus aggregate metrics like `cross_projection_disagreement_rate`
that capture which scenarios reveal real framework tension. This is
v0.2 work because it multiplies the gold-curation burden by N.

## Monitor + silicon: explicit non-migration

Two consumers will likely never become graph-aware:

  - **I-EIP Monitor.** Operates on (text-lens MoralVector,
    activation-lens MoralVector) deltas. The lens-disagreement
    failure modes are about *whether* the model represents moral
    state consistently across text and activations, not about
    *how* the substrate is shaped. The monitor reads MoralVector
    timelines (which are the consequentialist projection's
    output); no graph access needed.
  - **Silicon emit.** Emits the EM-DAG-as-HLS-code. The EM-DAG is
    the evaluator pipeline, not the facts being evaluated.
    HLS gen is framework-evaluator code, not data, and is
    independent of whether the data is flat or graph-shaped.

If either consumer ever needs graph context (e.g. monitoring
substrate consistency across paraphrase via `graph_hash`
comparison), that's a future extension with a clear path.

## Done in v0 (release-planning-07 scope)

- `ConsequentialistProjection.metadata` records graph_summary
  (`n_stakeholders`, `n_acts`, `n_imposes_on_edges`, ...) and
  `graph_aware: True` so audit consumers can confirm graph
  participation even though EM-DAG hasn't been ported yet.
- `export/rlef.py` schema bumped to `rlef_v0.2`: every record now
  includes the `moral_graph` block (nodes + edges + canonical_json +
  graph_hash) and the full `projections` dict + cross-projection
  disagreement.

These are small but they make the graph load-bearing in the audit
chain + trainer-data pipeline, which is where it matters first.
