---
title: 'ErisML Compiler: A structure-preserving compiler from natural language to a moral intermediate representation'
tags:
  - Python
  - AI safety
  - alignment
  - moral reasoning
  - natural language processing
  - intermediate representation
  - silicon-castable
  - FPGA
authors:
  - name: Andrew H. Bond
    orcid: 0009-0009-1769-5099
    affiliation: 1
affiliations:
  - name: San José State University, San José, CA, USA
    index: 1
date: 12 June 2026
bibliography: paper.bib
---

# Summary

`erisml-compiler` is a Python compiler that maps natural-language moral
material into a canonical, structure-preserving intermediate
representation called ErisML. The compiler exists to operationalise
a single thesis: that moral reasoning requires a structured
representation **before** decision contraction. A scalar
"good / bad / safe / unsafe" label discards the very dimensions that
justify or defeat a candidate action — who the stakeholders are, what
commitments bind them, which authorities are legitimate, who bears
imposed risk. The compiler preserves this tensorial structure as a
first-class object, evaluates it through a deterministic core
(`DEME` — the Deterministic Ethical Modular Evaluator), records an
SHA-256-anchored audit trace, and exports both training data for
human-feedback loops (`RLEF`) and synthesisable Vitis HLS C++ for an
FPGA silicon target.

A second contribution closes the loop between the model's **text
output** and the model's **internal state**. The I-EIP Monitor
(Internal / Activation / Delta lenses) registers forward hooks on a
chosen subset of transformer layers, runs a calibrated probe per
layer, and compares the resulting per-layer moral vectors against the
text-side IR. When the lenses disagree — by direction flips, monotone
layerwise drift, equivariance failure under semantics-preserving
rewrites, joint-uncertainty spikes, or audit-chain corruption — the
Monitor flags the input for human review. It never overrules `DEME`.

# Statement of need

Most existing alignment tooling collapses to a scalar: a reward
score, a safety classifier output, a guardrail pass/fail. This is
defensible as an engineering interface but it discards the structure
that ethics is *about*. Two cases:

- A medical professional choosing whether to break confidentiality to
  warn a third party of imminent harm is not navigating a
  one-dimensional good–bad axis; she is balancing care, fidelity,
  externality, autonomy, and legitimacy simultaneously, and the
  weights are not free parameters of preference but consequences of
  her institutional role. A scalar score that returns "0.74 unsafe"
  cannot represent this; an IR that decomposes the situation into a
  10-dimensional moral state, a stakeholder graph, a commitment
  registry, and a verdict with structured residue can.

- A model that produces innocuous text while its internal
  representations encode something the head was trained to suppress
  is exactly the case where a scalar safety classifier fails by
  construction. The I-EIP Monitor is built so that the *disagreement
  between text and activations is the safety signal*, not the
  agreement.

To our knowledge no other open-source tool currently provides this
combination: a structured moral IR, a silicon-castable evaluation
kernel, and a three-lens monitor over the internal state of a
deployed model. Adjacent work falls into one of three categories:
toolkits for *value alignment* via RLHF / DPO that ultimately reduce
to scalar reward modelling [@christiano2017deep; @rafailov2023direct];
probing tools for *interpretability* that surface internal state but
do not connect it to a structured ethical evaluator [@belrose2023eliciting];
and *constitutional AI* style frameworks that constrain model
behaviour via natural-language rules but do not produce a verifiable
intermediate object [@bai2022constitutional]. `erisml-compiler`
occupies the structural compositional gap between these — the moral IR
plays the same role for ethical reasoning that an SSA-form
intermediate representation plays for code generation
[@cytron1991efficiently], and the I-EIP Monitor plays the same role
for a deployed model that an architecture-level performance counter
plays for a deployed binary.

The intended users are AI-safety researchers (especially those who
need structured failure reports rather than scalar scores), ethics
review boards that need auditable provenance for AI decisions, and
hardware-software co-design teams investigating real-time ethical
interlocks for safety-critical agents (autonomous vehicles, surgical
robots, lethal-autonomy systems).

# Software description

The compiler implements a 12-pass pipeline (see
`docs/architecture.md`) with a tiered extractor stack:

- **Tier 1 (Geometric):** pre-parsed JSON event stream into the
  deterministic core. Used by the silicon target.
- **Tier 2 (Rules):** regex/grammar-driven `RuleExtractor` over
  natural language.
- **Tier 2.5 (Probe):** calibrated LaBSE-backed classifier head using
  the sqnd-probe v10.16.9 invariance methods: spectral decoupling,
  variational information bottleneck, multi-head GRL adversarial,
  confusion loss [@alemi2017deep; @ganin2016domain].
- **Tier 3 (LLM):** OpenAI-compatible chat-completion adapter (NRP
  Nautilus `gpt-oss`, `qwen3`, `glm-5`; local vLLM for self-hosted
  models), with a critic pass that compares the LLM's canonical-form
  choice against a deterministic second-opinion extractor and flags
  disagreements for human review.

The deterministic evaluator core is shared across all tiers and
consists of (i) three small finite-state machines — `CommitmentFSM`,
`LegitimacyFSM`, `ConsentFSM` — each implementable in three bits of
state register, and (ii) an EM-DAG (Ethical Module Directed Acyclic
Graph) of 10 modules: harm, rights, fairness, autonomy, legitimacy,
epistemic, care, fidelity, externality, repair. The DAG is
topologically sorted at compile time and evaluated in pipeline order.
The `silicon/` package emits Vitis HLS C++ for this core targeting
the Xilinx Alveo U55C in the NRP Coder environment.

The I-EIP Monitor (`monitor/`, `delta/`) is the activation-side
complement. `ActivationSource` is an ABC with three concrete
implementations: `MockActivationSource` (deterministic synthetic
hidden states for CI), `HuggingFaceActivationSource` (forward hooks
on `model.model.layers` for Qwen2/Qwen3/LLaMA/Mistral, `transformer.h`
for GPT-2, `encoder.layer` for BERT/RoBERTa), and
`RemoteAtlasActivationSource` (paramiko-driven inference on a remote
GPU host with the harness baked in as a literal string for trust
control). Per-layer `ActivationProbe` instances reuse the Phase-3
`ProbeHead` shape and accept Phase-3 checkpoints directly. The
`IEIPMonitor` orchestrator produces a `MonitorTrace` with a
SHA-256 anchor (`trace_hash()`) that extends the existing audit chain.

The Delta lens (`delta/`) supplies `compare_morals(text_mv, activation_mv)`,
the BIP equivariance check `h_ℓ(g·x) ≈ ρ_ℓ(g)·h_ℓ(x)` from
sqnd-probe with `ρ_ℓ = identity` (invariance under
semantics-preserving rewrites), and five named failure-mode detectors
(`text_internal_mismatch`, `layerwise_drift`,
`group_symmetry_break`, `probe_uncertainty_spike`,
`audit_chain_break`). The Monitor's only authorised output when any
of these fires is `requires_human_review`; verdicts remain `DEME`'s
job. The threat model and trust-boundary diagram are documented in
`docs/i_eip_monitor.md`.

The package is distributed on PyPI as `erisml-compiler` and on GitHub
under MIT license. The CLI exposes 12 subcommands including
`compile`, `validate`, `rlef`, `report`, `bundle`, `calibrate`,
`correct`, `diff`, `silicon-emit`, `monitor`, `delta`, and `version`. The project ships
three bundled examples — `nazi_attic.txt`,
`medical_confidentiality.txt`, `whistleblower.txt` — that cover the
hardest cases in classical normative ethics (the
trolley/inquirer/Kant exception structure) and that the compiler is
known to produce structurally faithful IR for end-to-end. The
repository has 142 tests passing on Ubuntu Python 3.10/3.11/3.12 in
GitHub Actions CI and is MIT-licensed.

# Ongoing and future work

Calibrated probe checkpoints against a real moral-language corpus
(rather than the synthetic dataset shipped with the calibration
stack) are in preparation, as is a separate methodological paper on
the I-EIP Monitor's empirical behaviour on Qwen2.5-7B-Instruct
running on NRP Nautilus. Silicon hardware bring-up on the U55C
target is gated by the NRP Coder bitstream pipeline; the Vitis HLS
emit is exercised in CI on every push and is bit-exact verified
through hardware emulation (70/70 PASS), but on-FPGA validation
remains the next milestone.

# Acknowledgements

Development of `erisml-compiler` was supported by computational
resources provided by the National Research Platform (NRP) Nautilus
cluster.

# References
