# Architecture

## Three tiers, one spine

The compiler offers three tiers of operation; the **evaluator core** is shared
across all three. Tiers differ in how structured input arrives at the
evaluator.

| Tier | Input | Path |
|---|---|---|
| 1 Geometric | Pre-parsed JSON event stream | structured loader → FSM + EM-DAG + DEME |
| 2 Rules | Natural-language text | text loader → segmenter → RuleExtractor → FSM + EM-DAG + DEME |
| 3 LLM | Natural-language text | text loader → segmenter → LLMExtractor → FSM + EM-DAG + DEME |

The evaluator core is FSM + EM-DAG + DEME bridge. It is deterministic,
bounded-memory, fixed-point-friendly, and silicon-castable. The three tiers
differ only in their extraction frontend.

## The 12-pass pipeline

Per spec section 12, in tier-agnostic form:

| Pass | Stage | Implementation |
|---|---|---|
| 0 | Ingestion | `ingestion/text_loader.py` or `ingestion/structured_loader.py` |
| 1 | Segmentation | `segmentation/segmenter.py` (text tiers only) |
| 2–7 | Extraction (entity, stakeholder, event, norm, ethical-fact, canonical form) | `annotation/{mock,rule,llm}_extractor.py` |
| 8 | Tensorisation | `evaluation/tensor_builder.py` (walks the EM-DAG at each time step) |
| 9 | ErisML IR | `erisml_backend/codegen.py` (renders the in-memory IR) |
| 10 | DEME evaluation | `erisml_backend/deme_bridge.py` |
| 11 | Contraction & residue | Embedded in `DEMEVerdict` |
| 12 | Audit artifact | `audit/hash_chain.py`, `audit/artifact.py` |

## EM-DAG

The DEME evaluator is a directed acyclic graph of **Ethical Modules**
(see `em_dag/`). Each module owns one MoralVector dimension (per spec §14)
and declares its dependencies on other modules. The DAG is topologically
sorted at construction time and evaluated in that order.

The default Phase-1 DAG (`em_dag/profiles/default.yaml`) has 10 modules:

- **Roots** (no deps): harm, rights, fairness, legitimacy, epistemic
- **Second tier**: autonomy (← legitimacy), fidelity (← legitimacy), externality (← harm), care (← harm)
- **Composition tier**: repair (← harm, externality, fidelity)

A different deployment can swap in a different DAG by writing a new YAML
profile and pointing the CLI at it via `--em-profile`. The topology is the
DEME profile.

## FSM layer

For Tier 1 (silicon target), moral state is tracked by deterministic
finite-state machines (`fsm/`):

- `CommitmentFSM`: vow lifecycle (active → defeasible → defeated/violated/fulfilled/void/expired)
- `LegitimacyFSM`: authority legitimacy (fully_legitimate → defeasible/coercive/fraudulent → tyrannical/void)
- `ConsentFSM`: consent (not_obtained → obtained/coerced → withdrawn)

Each FSM is a small state register (3 bits is enough for any of them) with a
deterministic transition table. Terminal states are absorbing. In silicon,
the entire moral-state vector for a scenario is bounded by
`O(n_commitments + n_authorities + n_consenters)` state registers.

## Audit

Every compilation produces an `AuditRecord` (`audit/hash_chain.py`)
containing:

- SHA-256 of the canonical-JSON IR (excluding the audit field)
- SHA-256 of the source text
- Compiler version, schema version
- Tier and extractor names
- EM-DAG profile name
- Timestamp (UTC)
- Per-pass duration records

The `bundle` subcommand bundles the IR, the source text, the ErisML
rendering, and a human-readable Markdown report into a single directory.
The `report` subcommand produces a self-contained HTML artifact with
embedded CSS and (if matplotlib is available) an embedded base64 timeline
PNG.

## Streaming layer

`streaming/` exposes a `MoralStreamer` that yields `StreamEvent` objects as
the pipeline progresses. The `TerminalCaptioner` renders the stream as
live text on stdout (`compile --stream`). The same stream feeds the HTML
report's timeline section.

In Phase 1 the streamer emits events post-hoc from a fully-compiled IR.
Phase 2 will switch to incremental emission during pipeline execution so
the captioner becomes real real-time.

## Phase boundaries

| Phase | Adds |
|---|---|
| 1 (this MVP) | Schema, pipeline, mock + rule extractor, EM-DAG, FSMs, DEME stub, audit, streaming, HTML, RLEF, CLI, tests |
| 2 | LLM adapters (NRP + vLLM), critic pass, learned canonicaliser, real DEME |
| 3 | Critic pass + human correction loop with web UI |
| 4 | React frontend, FastAPI HTTP layer, three-pane editor |
| 5 | Batch corpus mode, PostgreSQL + Celery, RLEF dataset generation |
| 5+ | Internal / Activation / Delta lenses (I-EIP Monitor architecture) |
| 5+ | Silicon-target compilation (PyMTL3 / MyHDL / HLS) of the FSM + EM-DAG spine |

See `SCOPE.md` for what is built versus stubbed versus deferred at the file level.
