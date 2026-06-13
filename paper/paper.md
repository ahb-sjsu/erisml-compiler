---
title: 'ErisML Compiler: a structure-preserving moral intermediate representation with pluggable framework analysers'
tags:
  - Python
  - AI safety
  - alignment
  - moral reasoning
  - intermediate representation
  - graph IR
authors:
  - name: Andrew H. Bond
    orcid: 0009-0009-1769-5099
    affiliation: 1
affiliations:
  - name: San José State University, San José, CA, USA
    index: 1
date: 13 June 2026
bibliography: paper.bib
---

# Summary

`erisml-compiler` is a Python compiler that takes natural-language
moral material and emits a typed directed acyclic graph — the
**MoralGraph** — anchored by a canonical SHA-256 hash. Multiple
framework analysers (consequentialist, Kantian, Aristotelian-virtue,
and care-ethical) read the same substrate and emit framework-relative
verdicts. When their normalised polarities disagree, the compiler
**refuses to aggregate** and surfaces all verdicts via
`ir.cross_projection_disagreement`. The Kantian analyser uses a Z3
SMT solver to test maxim universalizability against an explicit
institutional-fact universe; the virtue analyser optionally consults
a SQLite-backed longitudinal habit store of an agent's prior acts.
Everything is auditable: every IR carries hashed provenance for the
input text, the graph, the applied ethos profile, and the projections.

# Statement of need

Production alignment tooling almost always collapses moral evaluation
to a scalar: a reward score, a safety classifier output, a guardrail
pass/fail [@christiano2017deep; @rafailov2023direct]. This is
defensible engineering — a scalar is easy to compose with downstream
systems. But it discards the very structure that ethics is *about*.
A medical professional choosing whether to break confidentiality to
warn a third party of imminent harm is balancing care, fidelity,
externality, autonomy, and legitimacy simultaneously, and the weights
follow from her institutional role. A scalar "0.74 unsafe" cannot
represent that. Worse, scoring across cases pre-commits to a specific
(typically distributive-consequentialist) framework whose choices are
invisible to the user: the system *looks* framework-neutral while
quietly making first-order ethical commitments at every aggregation
step.

`erisml-compiler` addresses both problems architecturally. **Structure
preservation**: the IR is a typed graph (nodes are stakeholders, acts,
maxims, commitments, ethical facts, norms; edges are typed moral
relations such as `imposes_on`, `consents_to`, `treats_as`,
`under_maxim`, `coerces`), not a scalar. The graph plays the same
role for ethical reasoning that an SSA-form IR plays for code
generation [@cytron1991efficiently]: a structured intermediate that
later passes can analyse, transform, and emit from. **Framework
pluralism**: each framework gets a first-class `Projection` of its
own primitives — Kantian gates emit *categorical* findings
(universalizability fails by Contradiction in Conception; this
stakeholder is treated as a mere means), virtue findings emit
character-axis assessments, care-ethical findings emit relational
attentiveness markers. When projections disagree, the compiler does
not pick a winner — that choice is itself a metaethical move, and
it is deferred to the caller explicitly.

To our knowledge no other open-source artifact combines (a) a typed
moral graph IR, (b) multiple framework analysers over a shared
substrate with honest categorical disagreement, and (c) an SMT-based
test of the Kantian categorical imperative. Adjacent work falls into
three buckets: RLHF/DPO toolkits that reduce to scalar reward
[@christiano2017deep; @rafailov2023direct]; interpretability tools
that surface internal state but don't connect it to a structured
evaluator [@belrose2023eliciting]; and constitutional-AI frameworks
that constrain behaviour via natural-language rules without
producing a verifiable intermediate object [@bai2022constitutional].
`erisml-compiler` occupies the structural-compositional gap between
these.

# Software description

The compiler implements a 13-pass pipeline. Passes 0–7 ingest text,
segment it, and extract stakeholders, events, commitments, ethical
facts, and norms through one of four tiered extractors (deterministic
mock; regex-based rule extractor; calibrated `LaBSE`-backed probe;
or an LLM extractor with a critic-based consensus check). Pass 7.5
promotes the flat extractor output into a typed `MoralGraph` (or
accepts a graph emitted directly by a graph-native extractor). Pass 8
runs every enabled framework projection over the substrate. Pass 12
finalises the audit record with `ir_hash`, `graph_hash`, and (when
applied) the fitted-ethos profile's `ethos_profile_sha256`.

Five capabilities are first-class:

- **Typed graph IR** (`ir/graph/`). Six node kinds and eleven edge
  kinds; canonical SHA-256 hashing; bidirectional derivation from
  the flat-list extractor output.
- **Four framework projections** (`projections/`).
  `ConsequentialistProjection` emits the per-stakeholder rank-N
  DEME V3 tensor with Gini/Shapley aggregates.
  `DeonticProjection` emits gate findings (universalizability,
  mere_means, valid_consent, legitimate_authority).
  `VirtueProjection` and `CareEthicsProjection` emit
  character-axis and relational-attentiveness findings respectively.
- **SMT universalizability** (`delta/universalizability_smt.py`).
  Encodes institutional facts as Z3 Bool variables. The CIC check
  asks whether the universalised maxim is consistent with what the
  act presupposes; the CIW check asks whether the universalised
  world preserves the agent's own rational ends. The solver returns
  a satisfying assignment when SAT, recorded in the gate's audit
  detail block.
- **SRL maxim extraction** (`annotation/maxim_extractor_srl.py`).
  spaCy dependency parsing identifies (subject, predicate, dobj,
  purpose) per sentence; score-based selection across candidates
  with context-sensitive disambiguation of polysemous verbs.
- **Longitudinal virtue tracking** (`history/`). SQLite-backed
  habit store in WAL mode with UPSERT semantics, plus a
  temporally-weighted assessment with exponential decay and
  per-case severity multipliers. Virtue verdicts can now reflect
  patterns of acts across time, as Aristotelian habituation
  requires, rather than judging from a single observation.

The compiler also ships an out-of-band **I-EIP Monitor**
(`monitor/`, `delta/`) that hooks per-layer activations of a
deployed model and compares per-layer projection outputs against
the text-side IR via cosine similarity, layerwise drift, and BIP
equivariance under semantics-preserving rewrites. When lenses
disagree the Monitor flags `requires_human_review`; it never
overrules the projections.

# Worked example

On the bundled `nazi_attic` scenario (a Constant–Kant variant of
the murderer-at-the-door problem), the consequentialist projection
returns `tragic_conflict_escalate` while the deontic projection
returns `forbidden`. The Z3 solver classifies the maxim
`deceive` as a Contradiction in Conception: universalising it
destroys `truth_telling_default`, which the act of deceiving itself
presupposes. The compiler emits both verdicts via
`ir.cross_projection_disagreement` and records the SMT satisfying
assignment for audit. The full result is reproducible with
`eris-compile compile examples/nazi_attic.txt`.

# Limitations

The MoralGraph's extraction categories (we extract maxims,
stakeholders, commitments, ethical facts; we do not yet extract
relational webs or virtue-disposition records as primitives) are
themselves choices. The metaethical commitment is smaller after the
v0.8.0 refactor but not zero. The SMT solver's institutional-fact
universe is hand-curated. Probe checkpoints for the activation-lens
work remain uncalibrated against a real moral-language corpus, and
on-FPGA silicon bring-up is gated by the NRP Coder pipeline. Each
limitation is documented in `SCOPE.md` and the design notes under
`docs/plans/`.

# Acknowledgements

Development was supported by computational resources from the
National Research Platform (NRP) Nautilus cluster. The
framework-pluralist refactor that became v0.8.0 was prompted by a
substantive r/Compiler review that correctly identified the
pre-refactor IR as encoding metaethical commitments below the
profile layer; the v0.9.0 production-grade analysers were the
follow-on response.

# References
