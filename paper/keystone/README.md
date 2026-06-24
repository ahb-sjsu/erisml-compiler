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
