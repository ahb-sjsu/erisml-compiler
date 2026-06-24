# Keystone paper — ErisML + DEME

The canonical, citable framework paper for ErisML + DEME as an auditable machine-ethics
system. Anchors the broader "acceptance" campaign (the workshop and the Philosophy
Engineering manifesto depend on this existing first).

## Target & strategy
- **Primary venue:** AIES 2027 (AAAI/ACM Conf. on AI, Ethics & Society). AIES 2026 closed
  21 May 2026; AIES 2027 CFP not yet out (expect ~May 2027 deadline on the annual cycle).
- **AIES is non-archival** (one-page abstract + URL to full paper), so the *same* full
  paper can also go to arXiv (citable anchor) and, later, a journal.
- **Journal leg:** NOT Minds and Machines (desk-rejected the foundational/field-theory
  papers — too physics-forward). If a journal is wanted later: Ethics & Information
  Technology, AI & Society, or ACM TIST. Nearer conference alt: FAccT 2027 (~Jan).
- **Format:** skeleton uses `article`; PORT to AAAI style (`aaai2X.sty`) before submission.

## Hard constraints (learned the hard way)
- **No field-theory / gauge-theory / GR framing.** That framing got the foundational
  papers desk-rejected at M&M. Geometry appears as exactly ONE hedged citation
  (`bond_geometric_ethics`), framed as optional background we do not depend on.
- **Lead with artifact + evidence**, grounded in real moral philosophy (pluralism,
  incommensurability, prima facie duties) — that is what earns an AIES accept.
- **No self-overlap:** the evaluation cites the companion TCSS (moderation) and
  BigDataService (SecureAI) papers for full detail rather than re-running them.

## Status
- [x] Abstract
- [x] §1 Introduction
- [x] §2 Related Work (moral-philosophy-grounded)
- [x] §3 ErisML (from code)
- [x] §4 DEME (from code) + dimension derivation (3x3 scope x mode + subsumption, NOT arbitrary)
      + formalization (instruments / moral core / invariance, tolerance not group action)
- [x] §5 Evaluation (VERIFIED numbers: ICC 0.969, euphemism flip 13.7%, worst-off/escalate 33.8%)
- [x] §6 Philosophy Engineering (scoped tight) · §7 Limitations · §8 Conclusion
- [x] Appendix A: instrument-to-core projection map (PROPOSED — confirm axis assignments)
- [x] Subsumption-framework citations added (EU HLEG, Beauchamp-Childress, Markkula)
- [x] Port to AAAI 2026 format (keystone_aaai.tex; needs aaai2026.sty from the kit)
- [~] Citations: high-risk ones web-VERIFIED (see below); rest are high-confidence from
      knowledge and still warrant a final human proof
- [ ] Confirm Appendix A core mapping; then it becomes exact, not proposed
- [ ] Confirm author block / anonymity policy for AIES (wrapper defaults to [submission])

## Bibliography verification status
Web-verified + corrected: Lindner et al. (AAAI 2019, was wrong), Conitzer et al. (ICML 2024
"Position:" paper), Mitchell et al. (FAT* 2019, pp. 220-229), EU HLEG (2019, 7 reqs),
Beauchamp-Childress (8th ed. 2019), Markkula (now SIX lenses, was "five" in text — fixed).
NOT individually re-verified (high-confidence canonical): Ross, Williams, Berlin, Chang,
Hohfeld, Horty, von Wright, Powers, Dancy, Wallach-Allen, Anderson-Anderson, Gabriel, Bai,
Christiano, Sorensen, Hendrycks, Jiang, Raji, NIST. Do a final proof before submission.

## Dimension justification (settled)
9-D is canonical and DERIVED (3x3 scope x mode; subsumes EU HLEG / Beauchamp-Childress /
Markkula without residue; falsifiable completeness). 7-D harm space + 10-module = task
instruments (projections/refinements). Gauge theory / Noether / quantum stay in the book
(one hedged citation). Per author: the quantum CHSH null is MODEL-MEDIATED — "awaiting
human-subjects trials," not falsified — but it is not load-bearing here and stays out.

## Peer-review responses folded in (§7 "Limitations and Open Problems")
Three reviewer critiques addressed in the paper, not just acknowledged:
1. **Ingestion bottleneck** ("formally verified garbage out") — named the three partial
   safeguards (critic pass, probe-vs-LLM disagreement, monitor) and flagged *auditing the
   parser itself* as the #1 open problem.
2. **Escalation trap** — AITA is a worst-case all-dilemmas corpus (clear cases escalate
   90.7% vs contested 91.2% — escalation tracks *framework* disagreement, near-absent on
   benign streams); 91% is an upper bound. Added the tunable integrity-throughput family
   (decision-relevant conflicts, severity gating, escalation budget) as future work.
3. **Core-map validation** — elevated to a concrete high-priority item; added an empirical
   down-payment: PCA shows PC1 explains only 35% of moral variance, 5 comps for 90% (in the
   notebook), so the representation is far from one-dimensional.

## Stronger-eval datasets (next — AITA labels are coarse)
AITA gives one coarse label (NTA/YTA/ESH/NAH). Richer human-labeled corpora that would
strengthen §5, ranked by fit:
1. **MFRC** (Moral Foundations Reddit Corpus) — 16k comments, ≥3 annotators, **8 moral
   dimensions** (Care, Proportionality, Equality, Purity, Authority, Loyalty, Thin,
   Implicit/Explicit). Best for *validating the multi-dimensional representation/core against
   human labels* (HF: USC-MOLA-Lab/MFRC).
2. **ValuePrism / Kaleido** (Sorensen, AAAI 2024) — 218k values/rights/duties over 31k
   situations, Support/Oppose + rationale, split into Rights/Values/Duties. Best for the
   *pluralism + Hohfeld rights/duties* angle.
3. **ETHICS** (Hendrycks) — per-framework labels (justice/deontology/virtue/util/commonsense).
   Best for *validating the four projections* against framework-specific human judgments.
4. **Scruples** (32k anecdotes, 625k judgments) — AITA-style but with the full *judgment
   distribution* (captures disagreement directly). Easy richer drop-in for the AITA eval.
Also: Social Chemistry 101 (already used for ethos profiles), MFTC, Moral Stories.

## External validation (MFRC) — human-grounded, in the paper (§5)
- `mfrc_validation.ipynb` (+ `deme_mfrc_scores.jsonl`, `score_mfrc_nrp.py`) — does DEME's
  representation track *independent human* moral labels? Scored 280 Moral Foundations Reddit
  Corpus comments (human-annotated) on DEME's dims via the NRP LLM panel. **All four
  pre-registered alignments hold and are the argmax**: care↔Care ρ=0.49, fairness↔Equality
  ρ=0.47, legitimacy↔Authority ρ=0.48, fidelity↔Loyalty ρ=0.47 (all p<1e-16, n=280); Purity
  (no DEME analog) is the quiet negative control. This is the strongest single answer to the
  "are the dimensions real / validate the core" critique — against human ground truth, not
  model-mediated self-reference. `deme_mfrc_scores.jsonl` is derived (scores + human label
  fractions, no raw comment text; source HF: USC-MOLA-Lab/MFRC). Notebook verified to run.

## Companion notebook
- `keystone_aita_demo.ipynb` — reproduces the four thesis claims on 240 AITA dilemmas from the
  precomputed DEME evaluations in `aita_deme_results.jsonl`. Self-contained: needs only
  `numpy`/`scipy`/`matplotlib` (no LLM/GPU/network). Verified to run clean; reproduces §5
  (divergence 34.2%, L2 0.140, harm ρ≈0 vs fairness/care significant, escalation ~91%). The
  results jsonl carries vectors/verdicts only (no raw Reddit post text). Ships un-executed
  (run it to populate figures; nbconvert was unavailable in the authoring env).

## Files & build
- `body.tex` — shared content (abstract + sections + appendix). Single source of truth.
- `keystone.tex` — article wrapper for LOCAL PREVIEW. `pdflatex keystone && bibtex keystone
  && pdflatex keystone && pdflatex keystone`. Compiles anywhere (10 pages).
- `keystone_aaai.tex` — AAAI 2026 / AIES SUBMISSION wrapper. Needs `aaai2026.sty` +
  `aaai2026.bst` from the AAAI 2026 author kit (not on CTAN — get from the Overleaf
  template). `[submission]` = anonymous build; remove it + fill author for camera-ready.
- `keystone.bib` — references (see verification status above).
