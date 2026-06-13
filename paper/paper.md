---
title: 'ErisML Compiler: A framework-pluralist DAG-native compiler from natural language to a moral intermediate representation'
tags:
  - Python
  - AI safety
  - alignment
  - moral reasoning
  - natural language processing
  - intermediate representation
  - graph IR
  - silicon-castable
  - FPGA
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

`erisml-compiler` is a Python compiler that maps natural-language
moral material into a typed `MoralGraph` — a directed acyclic graph
with stakeholder, act, maxim, commitment, fact, and norm nodes and
typed moral-relation edges (performs, imposes_on, consents_to,
treats_as, under_maxim, coerces, …). The graph carries a canonical
SHA-256 hash in every audit record. Multiple framework `Projection`
strategies then read the same substrate via typed queries and emit
framework-relative verdicts: a consequentialist projection produces
a per-stakeholder harm tensor with Gini/Shapley aggregates and a
DEME (Deterministic Ethical Modular Evaluator) verdict; a Kantian
projection emits *categorical gate findings* (universalizability,
mere_means, valid_consent, legitimate_authority) rather than channel
contributions; a virtue projection reads character-trait axes; a
care-ethics projection reads relational primitives (attentiveness,
asymmetric responsibility, dependency response).

When projection verdicts disagree by normalised polarity, the
compiler *refuses to aggregate*. The disagreement is surfaced
explicitly via `ir.cross_projection_disagreement`; choosing across
frameworks is left to the caller as an explicit metaethical move.
This is the core architectural commitment: structure-preservation
against premature scalar contraction, plus honest first-class
representation of the framework pluralism that scalar safety
classifiers conceal.

A complementary contribution — the I-EIP Monitor — closes the loop
between the model's **text output** and the model's **internal
state**. Forward hooks on chosen transformer layers feed a calibrated
per-layer probe; the resulting per-layer moral vectors are compared
against the text-side projection. Disagreement (direction flips,
monotone layerwise drift, equivariance failure under
semantics-preserving rewrites with optional learned ρ_ℓ, joint
uncertainty spikes, audit-chain corruption) flags
`requires_human_review`. The Monitor never overrules a projection's
verdict.

# Statement of need

Existing alignment tooling typically collapses to a scalar: a reward
score, a safety classifier output, a guardrail pass/fail
[@christiano2017deep; @rafailov2023direct]. This is defensible as an
engineering interface but discards the structure that ethics is
*about*. Worse, it conceals a metaethical commitment: a single
scalar that compares "permitted" across cases pre-commits to a
specific (typically consequentialist-distributive) framework whose
choices are invisible to the user.

`erisml-compiler` answers both concerns architecturally. Structure
preservation: the IR is a typed graph, not a flat score, and the
graph's typed edges (`imposes_on`, `consents_to`, `treats_as`,
`under_maxim`, …) carry the relational structure that justifies or
defeats candidate actions. Framework pluralism: each framework gets
a first-class `Projection` of its own primitives — Kantian gates
are categorical pass/fail, virtue findings are character-axis
readings, care ethics tracks relational webs — and when frameworks
disagree, the compiler surfaces all verdicts rather than picking
one silently.

Two practical examples:

- A medical professional choosing whether to break confidentiality
  to warn a third party of imminent harm balances care, fidelity,
  externality, autonomy, and legitimacy simultaneously, and the
  weights follow from her institutional role. The consequentialist
  projection returns `permitted` on this case (the warned party
  benefits more than the patient loses); the care-ethics projection
  flags `requires_caring_attention` because explicit relational
  ties to the dependents are missing in the substrate. The compiler
  reports both, and the divergence is visible to the practitioner.

- A model that produces innocuous text while its internal
  representations encode something the head suppresses is the case
  where a scalar safety classifier fails by construction. The I-EIP
  Monitor is built so that *disagreement between text and
  activations is the safety signal*, not agreement.

To our knowledge no other open-source tool currently provides this
combination: a typed-graph moral IR, framework-pluralist projections
emitting honest categorical disagreement, a silicon-castable
evaluation kernel, and a three-lens monitor over a deployed model's
internal state. Adjacent work occupies one of three categories:
RLHF/DPO toolkits that ultimately reduce to scalar reward modelling
[@christiano2017deep; @rafailov2023direct]; interpretability tools
that surface internal state without connecting it to a structured
ethical evaluator [@belrose2023eliciting]; and constitutional-AI
frameworks that constrain model behaviour via natural-language rules
but do not produce a verifiable intermediate object
[@bai2022constitutional]. `erisml-compiler` occupies the
structural-compositional gap between these — the moral IR plays the
role for ethical reasoning that an SSA-form intermediate
representation plays for code generation [@cytron1991efficiently].

Intended users: AI-safety researchers who need structured failure
reports rather than scalar scores, ethics review boards that need
auditable provenance for AI decisions, philosophy and applied-ethics
researchers studying framework comparison, and hardware-software
co-design teams investigating real-time ethical interlocks for
safety-critical agents.

# Software description

The compiler implements a 13-pass pipeline (see
`docs/architecture.md`). Passes 0–7 ingest text, segment it, extract
stakeholders/events/commitments/facts/norms through one of four
tiered extractors (Mock, Rule, Probe, LLM), and canonicalise the
case. Pass 7.5 promotes the flat extractor output to a typed
`MoralGraph` (or accepts a graph emitted directly by a graph-native
extractor). Pass 8 runs every enabled framework projection over the
substrate; pass 12 finalises the audit record with `ir_hash`,
`graph_hash`, and (if applied) the fitted-ethos profile's
`ethos_profile_sha256`.

Five subpackages compose the architecture:

- **`ir/graph/`** — Typed graph schema (`MoralNode`, `MoralEdge`,
  `MoralGraph`), canonical SHA-256 hash, `graph_from_flat`
  promotion, and `flat_from_graph` back-derivation (bit-stable
  round-trip).
- **`projections/`** — `MoralSubstrate` view over the graph; four
  framework projections (`ConsequentialistProjection`,
  `DeonticProjection`, `VirtueProjection`, `CareEthicsProjection`).
  Each emits a `ProjectionResult` with a normalised verdict
  polarity (`permit` / `forbid` / `escalate` / `neutral`) used by
  the orchestrator to detect genuine cross-framework disagreement.
- **`em_dag/`** — 10 ethical modules (harm, rights, fairness,
  legitimacy, epistemic, autonomy, fidelity, externality, care,
  repair) that the consequentialist projection composes into a
  rank-1 through rank-6 DEME V3 tensor. Helpers read the
  `MoralGraph` directly (typed-edge queries) when one is attached,
  with a flat-field fallback. EM-DAG output values on the bundled
  examples are byte-identical to the pre-DAG-port baseline.
- **`monitor/`** + **`delta/`** — The I-EIP Monitor and the
  three-lens delta comparator, including the BIP equivariance test
  `h_ℓ(g·x) ≈ ρ_ℓ(g)·h_ℓ(x)` from sqnd-probe. v0.8.0 ships
  closed-form ρ_ℓ estimators (orthogonal Procrustes and
  unconstrained least squares) plus per-pair residual computation;
  the CLI `fit-rho` subcommand is deferred to a follow-up release.
- **`silicon/`** — Vitis HLS C++ emitter for the deterministic EM-DAG
  + FSM core targeting the Xilinx Alveo U55C. Bit-exact verified
  through hardware emulation (70/70 PASS); on-FPGA bring-up gated by
  the NRP Coder pipeline.

The package is distributed on PyPI as `erisml-compiler` (v0.8.0,
MIT-licensed) and on GitHub. The CLI exposes 15 subcommands
including `compile`, `bench run`, `monitor`, `delta`,
`silicon-emit`, and `fit-profile`.

# End-to-end demonstration

To verify framework disagreement is visible on real cases, we
compile the three bundled scenarios (`nazi_attic`,
`medical_confidentiality`, `whistleblower`) with all four
projections enabled. On the `nazi_attic` case (Constant–Kant variant
of the murderer-at-the-door problem), the consequentialist
projection returns `tragic_conflict_escalate` while the deontic
projection returns `forbidden` because the maxim's action_kind
`deceive` is non-universalisable and the village (a non-consenting
third party identified by `imposes_on`-without-paired-`consents_to`
edges) fails the `valid_consent` and `mere_means` gates. The
compiler emits both verdicts via
`ir.cross_projection_disagreement` and refuses to aggregate.

To verify the activation-side pipeline runs against a real
production model, we instantiate the I-EIP Monitor with a
`HuggingFaceActivationSource` over `Qwen/Qwen2.5-7B-Instruct`
(28 transformer layers, hidden dimension 3584) hosted on a
dual-Quadro-GV100 workstation reachable via Tailscale + paramiko.
We hook every fourth layer plus the final layer and run the monitor
+ delta + equivariance pipeline. Activation norms climb
monotonically through the residual stream; trace hashes are
deterministic; the BIP equivariance check (`ρ_ℓ = identity`) under
a lowercase rewrite fails specifically at the final layer on two
of three scenarios — consistent with the final layer being the
locus of output-distribution commitment. With random probes the
divergence and direction-break counts are noise; calibrated probes
against a real moral-language corpus are deferred to a separate
empirical paper.

`MoralTensor-Bench` (`bench/v0.1/`) ships with three seed scenarios
recast from the bundled examples and seven per-metric scorers
(stakeholder recall, role F1, commitment F1, canonical-form match,
ethical-fact-kind recall, per-party verdict accuracy, overall
verdict match) plus a premature-contraction penalty. Baseline score
on the seed corpus with the rule extractor is 0.136 — an honest
finding about extractor coverage rather than a victory lap: the
rule extractor uses generic stakeholder IDs (`self`,
`collective_*_seg_*`) while the bench gold uses semantic IDs
(`speaker`, `gestapo`). Improvements to the LLM extractor should
move this number upward against the bench's stable corpus_hash.

The repository has 330+ tests passing on Ubuntu Python 3.10/3.11/3.12
in GitHub Actions CI; ruff lint and black format checks both clean.

# Architectural commitments

The compiler does not claim metaethical neutrality. The
`MoralSubstrate`'s extraction categories (we extract stakeholders,
maxims, commitments, ethical facts; we don't yet extract — for
example — virtue dispositions or care-ethical relational webs as
primitive objects) remain a real commitment, smaller than the
previous tensor-level commitment but not zero. The choice to make
this commitment explicit and load-bearing — rather than hiding it
in a scalar — is itself a methodological stance documented in
`docs/plans/release-planning-06-framework-pluralist-architecture.md`.

# Ongoing and future work

Calibrated probe checkpoints against a real moral-language corpus
(beyond the synthetic dataset shipping with the calibration stack)
are in preparation, along with a separate methodological paper on
the I-EIP Monitor's empirical behaviour on `Qwen2.5-7B-Instruct`.
Per-projection bench scoring with framework-specific gold answers,
a richer Kantian universalizability gate that builds a
universalised-world model rather than the v0 rule-list-based test,
and on-FPGA silicon bring-up are tracked in `docs/plans/`.

# Acknowledgements

Development of `erisml-compiler` was supported by computational
resources provided by the National Research Platform (NRP) Nautilus
cluster. The framework-pluralist refactor that became v0.8.0 was
prompted by a substantive critique on the r/Compiler community
forum (2026-06-12) that correctly identified the pre-refactor IR as
encoding metaethical commitments below the profile layer.

# References
