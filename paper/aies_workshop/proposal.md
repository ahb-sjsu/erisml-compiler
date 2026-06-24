# Workshop Proposal — Philosophy Engineering: Executable, Falsifiable, and Auditable Machine Ethics

**Proposed venue:** AIES 2027 (AAAI/ACM Conference on AI, Ethics, and Society)
**Format:** Full-day workshop (a half-day version is available on request)
**Status of this document:** DRAFT for the AIES 2027 workshop call. Items marked *[TBC]* are
to be confirmed (co-organizers, invited speakers, exact dates) once the call opens.

---

## 1. Abstract

AI safety and governance increasingly turn on *normative* judgments — what is harmful, fair,
permitted, or owed — yet our tools for making those judgments are split between two
unsatisfying poles. On one side, deployed systems collapse moral evaluation into a **scalar**
(a reward score, a safety probability, a guardrail pass/fail) that discards moral structure
and silently pre-commits to a single ethical framework while appearing neutral. On the other,
rigorous **formal and philosophical** accounts of ethics rarely execute, evaluate, or ship.
Between them sits a discipline that does not yet have a name or a home: the practice of
building **executable, falsifiable, and auditable** implementations of normative frameworks —
with modeling assumptions *declared*, invariances *stated*, predictions *tested*, and the
result *versioned and revised* when it fails.

This workshop convenes researchers across machine ethics, value alignment, formal and
computational ethics, computational social choice, and AI governance to define and advance
this discipline, which we call **Philosophy Engineering**. We solicit systems, methods,
benchmarks, critiques, and position papers, and we aim to leave AIES 2027 with a shared
vocabulary, a public research agenda, and a community.

## 2. Motivation and relevance to AIES

AIES 2026 invited work that "not only analyzes ethical and societal challenges, but helps
reimagine the institutions, practices, and values needed to govern AI responsibly." Philosophy
Engineering is precisely such a *practice*: a methodology for turning normative commitments
into artifacts that can be inspected, tested, and held accountable.

The need is acute and the relevant work already exists — but scattered, under different
banners, talking past one another:

- **Machine ethics** has produced learned moral judges (e.g., Delphi; the ETHICS benchmark),
  but their outputs are descriptive scalars whose reasoning is not recoverable and whose
  ethical commitments are undeclared.
- **Value alignment** embeds values implicitly in model weights (RLHF) or semi-explicitly in a
  natural-language constitution; the *application* of those values remains an opaque forward
  pass.
- **Formal and deontic ethics** offers rigor (deontic logic, Hohfeldian analysis, Kantian
  formalizations) but is largely disconnected from deployment and evaluation.
- **Computational social choice and pluralistic alignment** study how to aggregate diverse
  values, yet aggregation is often an implicit social-welfare function hidden inside a reward.
- **AI governance** (model cards, NIST AI RMF, the IEEE 7000 series, the EU AI Act ecosystem)
  specifies *what* to disclose, but not a mechanism that *produces* inspectable moral reasoning.

No venue today focuses on the **engineering methodology** that would connect these: how to make
a normative model explicit enough to build, honest enough to be falsified, and auditable enough
to be governed. That is the gap this workshop fills. **Why now:** large language models have
made normative reasoning deployable at scale and at consequence, which raises — rather than
settles — the stakes of getting the methodology right.

## 3. The four commitments (the workshop's conceptual spine)

We propose Philosophy Engineering rests on four commitments, each of which is a call for
contributions:

1. **Declared modeling choices** — every bridge from a moral claim to a computational object
   carries its epistemic status (definition, assumption, conditional theorem, empirical result),
   so what is *proven* is not confused with what is *posited*.
2. **Stated invariances** — the transformations under which an evaluation must be stable (e.g.,
   meaning-preserving re-description) are written down and *tested*, not assumed.
3. **Extracted predictions** — a normative model earns its keep by entailing measurable claims
   (reliability, robustness, human-grounded validity) that can fail.
4. **Versioned revision** — when a prediction fails, the model is amended and the change
   recorded, like any engineered system.

## 4. Topics of interest

We invite contributions on, but not limited to:

- Executable representations of normative frameworks (consequentialist, deontic/Kantian, virtue,
  care, contractualist, pluralist) and their composition.
- **Structure-preserving** moral representations and the critique of scalar reward / safety
  collapse; multi-dimensional value spaces.
- Framework **pluralism**, aggregation, and *honest disagreement*; value incommensurability;
  computational social choice for normative systems.
- **Falsifiability and evaluation**: benchmarks, invariance/robustness and manipulation testing,
  human-grounded validation of moral representations.
- **Auditability and provenance** for moral reasoning; conformance to standards (IEEE 7000-series,
  NIST AI RMF, EU AI Act).
- **Formal verification** of ethical constraints (SMT, model checking, deontic logic in
  deployment).
- **Reproducibility, versioning, and epistemic-status discipline** for normative models.
- **Applications and case studies**: content moderation, clinical/triage ethics, autonomous
  systems, governance and compliance tooling.
- **Critiques and limits**: the reification objection ("is the formalism doing the work?"),
  cultural and pluralistic validity, the limits of formalizing the normative, and where
  Philosophy Engineering should *not* be applied.

Critical and adversarial submissions are explicitly welcome; a discipline is defined as much by
its boundaries as by its successes.

## 5. Format and tentative schedule (full day)

A working, building-oriented day rather than a mini-conference of talks.

| Time | Session |
|---|---|
| 09:00 | Opening + framing: what is (and isn't) Philosophy Engineering |
| 09:20 | Invited keynote I *[TBC]* — machine ethics / value alignment |
| 10:00 | Contributed talks (3 × 20 min) |
| 11:00 | Break |
| 11:15 | Invited keynote II *[TBC]* — formal ethics / governance / standards |
| 11:55 | Contributed talks (3 × 20 min) |
| 12:55 | Lunch |
| 14:00 | Hands-on / tooling session: building and falsifying a normative model end-to-end |
| 15:15 | Lightning talks + posters (incl. critiques and negative results) |
| 16:00 | Break |
| 16:15 | Panel: *Can ethics be engineered without being reduced?* |
| 17:15 | Community agenda: drafting a shared research roadmap |
| 17:45 | Close |

**Non-archival** (in line with AIES practice): accepted papers may be presented from a 1–2 page
abstract with a link to a full version, so the workshop does not preclude later journal or
conference publication.

## 6. Target invited speakers *[aspirational — not yet invited]*

We will invite speakers spanning the relevant communities so the workshop is a *bridge*, not a
single group's meeting: machine ethics, value alignment, computational social choice,
philosophy of AI, formal/deontic ethics, and AI governance/standards. The organizing committee
will prioritize diversity of subfield, methodology, career stage, gender, and geography. Specific
invitations will be issued on acceptance.

## 7. Submissions and review

- **Submission types:** full papers (up to 8 pp), short/position papers (up to 4 pp), and
  demos/artifacts. Critiques, negative results, and reproducibility reports are first-class.
- **Review:** each submission reviewed by ≥2 PC members; lightly double-blind; criteria emphasize
  *declared assumptions, falsifiability, and reproducibility* alongside novelty.
- **Program committee** *[TBC]*: recruited across the listed communities (~15–20 members).
- **Artifact encouragement:** submissions with runnable code / reproducible notebooks are
  highlighted; we will provide an artifact-evaluation track if volume warrants.

## 8. Expected outcomes

1. A shared **vocabulary and research agenda** for Philosophy Engineering, published as a
   community report after the workshop.
2. **Cross-pollination** between communities that rarely co-locate (formal ethics ↔ ML ↔
   governance/standards).
3. A nucleus for a recurring venue and a possible **special issue** (e.g., in a machine-ethics
   or AI-and-society journal).

## 9. Expected attendance and audience

We anticipate **30–60 attendees**: AIES regulars in machine ethics and governance, plus
ML researchers working on alignment/safety and philosophers of technology. The hands-on session
and the explicitly welcomed critiques are designed to draw both builders and skeptics.

## 10. Diversity and inclusion

The organizing committee, PC, and invited speakers will be assembled for diversity of
discipline, methodology, geography, institution type, gender, and career stage. We will offer
remote participation, ensure accessibility of materials, and (budget permitting) seek student
travel/registration support. The CFP will explicitly invite non-Western and pluralistic
perspectives on normativity, which the field's predominantly English-language, Western corpora
under-represent.

## 11. Organizers

**Lead organizer:** Andrew H. Bond (San José State University) — author of the ErisML compiler
and the DEME (Democratically-Governed Ethics Modules) engine, an open, shipped framework for
executable, framework-pluralist, auditable machine ethics; of the keystone paper on the
framework; and of related accepted/under-review work on manipulation detection in LLM moral
judgment and per-stakeholder content moderation. Contact: agi.hpc@gmail.com.

**Co-organizers *[TBC — actively being recruited]*.** *A competitive AIES workshop is
co-organized by 3–5 people spanning multiple institutions and subfields; recruiting co-organizers
from machine ethics, formal ethics, and AI governance is the top priority before submission and
will materially strengthen the proposal.*

## 12. Related workshops and differentiation

AIES itself, the AAAI/IJCAI AI-ethics tracks, WOAH (Workshop on Online Abuse and Harms), and the
NeurIPS/ICML safety, alignment, and *pluralistic alignment* workshops all touch adjacent ground.
They focus, respectively, on harms, alignment/safety broadly, online abuse, or value pluralism.
**None centers the engineering *methodology*** — making normative systems explicit, falsifiable,
versioned, and auditable — as a discipline in its own right. That methodological focus, and the
deliberate bridging of formal ethics, ML, and governance, is this workshop's distinguishing
contribution.

## 13. Logistics

Standard room with projector; one full day; breakout space for the hands-on session; remote/
hybrid option. No special equipment required.

---

### Open items for the lead organizer (not part of the submitted proposal)
- **Recruit 2–4 co-organizers** across machine ethics / formal ethics / governance — the single
  most important thing for acceptance. AIES workshop committees weight a diverse, multi-institution
  organizing team heavily; a solo proposal is a red flag.
- **Confirm AIES 2027 workshop CFP** (separate deadline + format from the main track) and adapt
  length/sections to its template.
- **Line up 2 invited speakers** with at least informal agreement before submission.
- Decide whether to anchor the workshop on the keystone paper (cite it once, lightly) or keep the
  framing fully community-neutral — recommend the latter to avoid a "vendor workshop" perception.
