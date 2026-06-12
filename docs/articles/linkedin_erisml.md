# Why "Is This Safe?" Is the Wrong Question — and What to Replace It With

*A compiler for moral reasoning, a three-lens monitor, and what they tell us about AI alignment.*

---

It is 1943. Soldiers are at your door. They ask whether you are hiding
Jews in your attic. You are. You can lie, you can be silent, you can
tell the truth.

Now imagine you ask a state-of-the-art language model what to do.

Most safety classifiers in production today would give you back a
single number. Maybe `0.74 — unsafe`. Maybe a refusal. Maybe a
hand-wringing essay about respecting all perspectives. None of these
is a useful answer, because none of them encodes **the structure of
the dilemma**: that lying is a violation of one commitment (honesty)
in service of a much stronger commitment (preventing murder); that
the asking authority is not legitimate; that the third party (the
family in the attic) has not and cannot consent to being disclosed;
that the cost of refusal — silence interpreted as confirmation — is
death.

A scalar score throws all of that away.

I have spent the last several months building software that doesn't.
It's called **ErisML Compiler** ([github.com/ahb-sjsu/erisml-compiler](https://github.com/ahb-sjsu/erisml-compiler))
and it just shipped its fourth major release. I want to use this
article to make the case for what I think is a structural error in
how the field currently approaches AI safety, and to show what a
different approach looks like in code.

## The scalar problem

The dominant pattern in modern alignment tooling — RLHF, DPO,
constitutional AI, safety classifiers, content filters — is to
collapse moral evaluation to a scalar. A reward score, a probability
of harm, a pass/fail. This is engineerable. You can backprop through
it, you can compare deployments, you can put it on a dashboard.

But it discards the dimensions that ethics is *about*.

A scalar can tell you that the nazi-attic case is "unsafe." It
cannot tell you *which axes of the situation are loaded* and *which
trade-off is being made*. A doctor breaking confidentiality to warn a
third party of imminent harm is not navigating a one-dimensional
good–bad axis; she is balancing **care**, **fidelity to her
institutional vow**, **the externality borne by the threatened
party**, **the autonomy of her patient**, and the **legitimacy of the
authority she would be reporting to**. The weights between these are
not free parameters of personal preference. They are constrained by
her role, by case law, by professional ethics.

If your safety system reduces this to "0.62 unsafe," you have not
helped her. You have told her what your classifier thinks of the
prompt, not what the situation actually contains.

## What a compiler can do that a classifier cannot

The thing that has always struck me about ethical reasoning is that
it is *compositional*. Cases share structure. The nazi-attic case and
the "do I lie to the murderer asking where my friend is" case are
the same case — they share a commitment topology, a stakeholder
graph, a verdict shape. Modern programming-language theory has
extremely good tools for representing compositional structure: type
systems, intermediate representations, static analyses. These are not
mysterious; they are forty years of computer science.

**ErisML Compiler** applies that machinery to moral material. Given a
natural-language scenario, it produces a structured intermediate
representation containing:

- a **stakeholder graph** (who is involved, what role they play),
- a **commitment registry** (what vows bind whom, in what state — active,
  defeasible, fulfilled, violated),
- a **moral state tensor** at rank 1–6, indexed by axes for *moral
  dimension* (9 dims from the "Nine Dimensions of Ethical Assessment"
  3×3 matrix), *stakeholder*, *time*, *action*, *coalition*, and
  *uncertainty sample*; at rank 2 the rows tell you what each
  stakeholder is actually bearing,
- a **verdict** produced by a deterministic evaluator (DEME) that walks
  a DAG of *ethical modules* in topological order, and
- a **deterministic audit trace**, SHA-256 anchored, that records every
  pass that produced the verdict.

The IR is the contract. Once you have it, you can do things you
cannot do with a scalar: you can compare two cases at the structural
level; you can re-evaluate after a human correction without
re-running an LLM; you can transform the case under symmetry
operations and check that the verdict commutes; you can — and this
matters — **cast the evaluator into silicon**.

Concretely: feed the compiler the "nazi at the door" scenario and the
rank-2 tensor that comes out splits cleanly across stakeholders.
Speaker bears expected harm 0.76 (verdict: forbid). Village bears
0.83 (forbid). The hiding refugees bear 0.00 (prefer the action).
The nazis themselves bear 0.18 (neutral). The Gini coefficient over
that harm distribution is 0.43 — a real, quantitative measure of
how unequally the cost lands. None of that survives a scalar
collapse.

That last point is not rhetorical. The deterministic core of ErisML
— three small finite-state machines (Commitment, Legitimacy, Consent)
plus the 10-module ethical DAG — has been carefully designed to be
silicon-castable. It uses no floating-point in its decision path; it
has bounded state; it does no dynamic dispatch. The compiler emits
Vitis HLS C++ that synthesises to a Xilinx Alveo U55C FPGA. This is
not science fiction: hardware emulation passes 70 of 70 reference
test vectors today; the on-FPGA bring-up is gated only by the cluster
bitstream pipeline. The reason to do this is exactly the reason
hardware kill-switches exist: at some point, the moral interlock on
an autonomous system has to be in a place the model cannot influence.

## The thing nobody is talking about: when the text and the activations disagree

Here is the part I think is most consequential.

Suppose you build a safety classifier on top of a 7B-parameter
language model. You train it well, you evaluate it well, the
benchmarks look good. The model outputs text that says all the right
things. But the model's *internal representations* — its activations
at the residual stream — have been trained, by gradient descent,
toward a structure that the final layer learned to suppress at the
output. The text says one thing. The internals say another. **Your
safety classifier reads the text.** It does not see the disagreement.

This is not a hypothetical. It is one of the open problems in
mechanistic interpretability, and it gets harder as models get
bigger. The standard response is "we'll train it out" — but training
something out at the output without addressing it in the
representation is exactly the kind of fix that fails when conditions
shift.

ErisML's Phase 4 release, which shipped this week, addresses this
directly. It is called the **I-EIP Monitor**, and it has three
lenses.

The **text lens** is Phases 1–3 of the compiler: the IR extracted
from what the model says. This is the surface.

The **activation lens** is a set of probes on the model's hidden
states. We register forward hooks on a subset of transformer layers
(by default, every fourth layer plus the final), pool the per-token
hidden states, and run a per-layer probe that maps each pooled
activation to the same 10-dimensional moral vector that the text
lens produces. We now have two parallel readings of the same input:
one from what the model says, one from what the model internally
represents.

The **delta lens** is what makes them speak to each other. It
computes a per-dimension delta, an overall divergence score, a count
of direction breaks (where text says positive and activations say
negative, or vice versa), and — crucially — five named failure-mode
detectors:

- **text_internal_mismatch**: the lenses disagree on direction enough
  to matter.
- **layerwise_drift**: some moral dimension drifts monotonically
  across enough layers, in a way suggesting a representation present
  mid-stack that the final-layer head is suppressing.
- **group_symmetry_break**: the BIP equivariance test fails for at
  least one layer, meaning the probe is responding to surface form
  rather than moral content.
- **probe_uncertainty_spike**: joint uncertainty exceeds a hard
  ceiling on at least one dimension — the monitor admits it does not
  know.
- **audit_chain_break**: the SHA-256 hash of the captured trace does
  not match the expected chain — provenance failure, replay attack,
  or storage corruption.

If any of these fires, the monitor's only authorised output is to
raise `requires_human_review`. **The Monitor never overrules DEME and
never produces a verdict.** Its job is exactly the job a fire alarm
has: to make a thing visible that the rest of the system cannot see
on its own.

## What it looks like running on a real 7B model

This week I ran the full Phase-4 pipeline against **Qwen2.5-7B-Instruct**
on a dual-Quadro-GV100 workstation, with paramiko transporting hidden
states back to the host. Three scenarios — the nazi-attic case,
medical-confidentiality, and whistleblower — across eight transformer
layers each.

The structural findings reproduced cleanly across runs:

- Activation norms climb monotonically through the residual stream
  on every scenario (e.g., nazi-attic: 8.8 → 398 → … → 571 at layer
  24, dropping to 402 at the final layer). This is the model's
  representational magnitude growing through depth.
- Trace hashes were deterministic — the audit anchor is reliable.
- The BIP equivariance check, under a surface-form rewrite
  (lowercasing the input), failed *specifically at the final layer*
  on two of three scenarios and passed throughout on the third. The
  final layer is where the model commits to its output distribution,
  and is therefore exactly the layer most sensitive to surface form.
  This is the kind of structural sensitivity you cannot see by
  staring at outputs.

The probes themselves are currently uncalibrated (random
initialisation; calibrated probes against a real moral-language
corpus is the next paper). So the divergence numbers right now are
noise. But the *infrastructure* — hook resolution, audit chaining,
equivariance localisation, failure-mode escalation — works on a
production-class model. That was the engineering milestone.

## Why I think this matters

There is going to be — there already is — a fight about how to
regulate AI. Some of the proposals on the table reduce to "give us
the scalar safety score." That fight will go badly if the only
artifact we hand to regulators, ethics boards, and courts is a
number, because numbers do not preserve the dimensions that justify
or defeat a decision. A regulator who needs to know *why* an AI
system denied a loan, refused a medical recommendation, or escalated
a security incident is going to want the *structure*. Compilers give
you that structure; classifiers do not.

There is also a particular kind of safety claim that I think we
should be very cautious about: the claim that a model is safe
because we trained it to be safe and the benchmarks agree. That
claim survives only as long as the training distribution survives.
The minute the inputs shift, the internal representations the model
has actually learned start to matter more than the outputs it has
been trained to produce. **The disagreement between the two is the
safety signal.** Build systems that surface it.

## Where to look

The compiler and its full toolchain are available now:

- **GitHub**: [github.com/ahb-sjsu/erisml-compiler](https://github.com/ahb-sjsu/erisml-compiler)
- **PyPI**: `pip install erisml-compiler`
- **DOI** (concept, always latest): [10.5281/zenodo.20659432](https://doi.org/10.5281/zenodo.20659432)
- **Paper draft**: in the repository under `paper/paper.md` (JOSS submission imminent)

It is MIT licensed. 194 tests pass on Ubuntu × Python 3.10/3.11/3.12,
with ruff lint and black format both clean. The bundled examples
include the three scenarios I named in this article, with
hand-curated reference IR you can compare your own extractions
against.

If you build AI systems for production — especially safety-critical
ones — I would love to hear what would have to be true for ErisML's
IR (or something like it) to slot into your stack. If you are a
researcher working on mechanistic interpretability, the I-EIP
Monitor's activation lens is designed to take your probes; I would
love to compare notes on calibration. If you are in policy or ethics
review and you find yourself frustrated by scalar safety scores, the
audit artifact may be the thing you have been wishing for.

DMs open. Issues open. Pull requests open.

---

*Andrew H. Bond is a researcher at San José State University. He
publishes the Geometric Series, a multi-volume project on the
mathematical structure of normative reasoning across domains. He can
be reached at andrew.bond@sjsu.edu or via GitHub at*
[github.com/ahb-sjsu](https://github.com/ahb-sjsu)*.*
