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
| **`ConsequentialistProjection`** | reads flat fields via EM-DAG | metadata records graph-derived stats; full migration depends on EM-DAG porting | EM-DAG is its own subsystem (10 modules, each with its own read patterns); porting touches ~1500 lines and risks regressions in DEME bridge + V3 dispatcher |
| **`DeonticProjection`** | graph-native (pattern-matches `treats_as`, `imposes_on`) | ✅ done | n/a |
| **`VirtueProjection`** | graph-native for power-asymmetry; substrate fallback elsewhere | ✅ done | n/a |
| **`CareEthicsProjection`** | graph-native for dependency-response; substrate fallback elsewhere | ✅ done | n/a |
| **`bench/runner.py`** | reads flat fields | extend to score *per-projection*; needs framework-specific gold | v0.2 milestone — requires curating gold answers per framework, orders of magnitude more curation work |
| **`monitor/`** (I-EIP) | operates on activation traces, not facts | minimal graph use; could read `graph_hash` for trace identity | low ROI — monitor is about activation-lens vs text-lens disagreement, not about how facts are represented |
| **`export/rlef.py`** | flat + timeline + verdict | ✅ now includes `moral_graph`, `projections`, `cross_projection_disagreement` (schema bump rlef_v0.2) | n/a |
| **`silicon/hls_emit.py`** | operates on the EM-DAG (not the IR's facts) | no graph consumption needed | silicon emits the *evaluator* (EM-DAG) as HLS code, independent of which facts feed it |

## EM-DAG graph-native port (deferred)

The biggest remaining item is making the EM-DAG modules read graph
queries instead of flat fields. Today every module looks like:

```python
class HarmEM(EthicalModule):
    def evaluate(self, ir: CompilerIR, ...) -> EMOutput:
        for fact in ir.ethical_facts:                   # <-- flat read
            if fact.kind == "harm":
                ...
        for stakeholder in ir.stakeholders:             # <-- flat read
            ...
```

A graph-native EM module would look like:

```python
class HarmEM(EthicalModule):
    def evaluate(self, graph: MoralGraph, ...) -> EMOutput:
        for edge in graph.edges_of_kind(EdgeKind.IMPOSES_ON):  # <-- typed
            severity = edge.payload.get("severity")
            subject = graph.get_node(edge.dst)
            ...
```

The port is mechanical but tedious (10 modules × ~50 LOC each =
~500 LOC). The risk is subtle behaviour changes — the rule extractor
emits facts with `subjects` field that don't always map cleanly to
graph edges, and edge-case behaviour may shift in ways that ripple
through the DEME bridge and downstream verdicts.

Plan: do this port in a dedicated commit cycle with golden-test
coverage on every bundled scenario showing identical verdicts
before/after. Estimate ~2-3 days.

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
