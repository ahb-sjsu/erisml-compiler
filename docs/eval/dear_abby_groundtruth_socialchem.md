# Dear Abby + bundled examples ethos eval

Two evaluation slices:

1. **Dear Abby dating-advice ground-truth** — 12 hand-curated question+answer pairs. Casual relationship/dating advice ("how long should I wait to date after my spouse died?"). The rule extractor surfaces few or no ethical facts in this register, so per-dimension shifts are typically zero — that is itself an honest finding about extractor coverage, not a wiring bug.
2. **Bundled morally-loaded examples** — 3 high-stakes scenarios (`nazi_attic.txt`, `medical_confidentiality.txt`, `whistleblower.txt`) where the rule extractor fires densely and ethos shifts are observable.

## Part 1: Dear Abby dating-advice ground-truth

Each of 12 cases was compiled with `eris-compile compile --rank 2 --canonicalizer registry` under three settings:

- `baseline` — no `--ethos-profile`
- `dear_abby_socialchem_v0.1`
- `aita_socialchem_v0.1`

### Aggregate effect of each profile vs baseline

### dear_abby_socialchem_v0.1

- n = 12
- principal-axis flips: 0/12
- verdict flips: 0/12

Per-dimension mean abs-delta from baseline:

| Dimension | mean abs-delta | mean signed delta |
|---|---:|---:|
| `physical_harm` | 0.0000 | +0.0000 |
| `rights_respect` | 0.0000 | +0.0000 |
| `fairness_equity` | 0.0000 | +0.0000 |
| `legitimacy_trust` | 0.0000 | +0.0000 |
| `epistemic_quality` | 0.0000 | +0.0000 |
| `autonomy_consent` | 0.0000 | +0.0000 |
| `vow_fidelity` | 0.0000 | +0.0000 |
| `third_party_externality` | 0.0000 | +0.0000 |
| `care_protection` | 0.0000 | +0.0000 |
| `repair_residue` | 0.0000 | +0.0000 |

### aita_socialchem_v0.1

- n = 12
- principal-axis flips: 0/12
- verdict flips: 0/12

Per-dimension mean abs-delta from baseline:

| Dimension | mean abs-delta | mean signed delta |
|---|---:|---:|
| `physical_harm` | 0.0000 | +0.0000 |
| `rights_respect` | 0.0000 | +0.0000 |
| `fairness_equity` | 0.0000 | +0.0000 |
| `legitimacy_trust` | 0.0000 | +0.0000 |
| `epistemic_quality` | 0.0000 | +0.0000 |
| `autonomy_consent` | 0.0000 | +0.0000 |
| `vow_fidelity` | 0.0000 | +0.0000 |
| `third_party_externality` | 0.0000 | +0.0000 |
| `care_protection` | 0.0000 | +0.0000 |
| `repair_residue` | 0.0000 | +0.0000 |

## Per-question audit table

| q | baseline verdict | baseline axis | dearabby axis (delta-dim, abs-delta) | aita axis (delta-dim, abs-delta) |
|---:|---|---|---|---|
| 0 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 1 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 2 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 3 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 4 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 5 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 6 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 7 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 8 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 9 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 10 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |
| 11 | requires_human_review | - | - (physical_harm, 0.000) | - (physical_harm, 0.000) |

---

## Part 2: Bundled morally-loaded examples

Each of 3 cases was compiled with `eris-compile compile --rank 2 --canonicalizer registry` under three settings:

- `baseline` — no `--ethos-profile`
- `dear_abby_socialchem_v0.1`
- `aita_socialchem_v0.1`

### Aggregate effect of each profile vs baseline

### dear_abby_socialchem_v0.1

- n = 3
- principal-axis flips: 0/3
- verdict flips: 2/3

Per-dimension mean abs-delta from baseline:

| Dimension | mean abs-delta | mean signed delta |
|---|---:|---:|
| `physical_harm` | 0.0000 | +0.0000 |
| `rights_respect` | 0.0000 | +0.0000 |
| `fairness_equity` | 0.0000 | +0.0000 |
| `legitimacy_trust` | 0.0168 | +0.0168 |
| `epistemic_quality` | 0.0000 | +0.0000 |
| `autonomy_consent` | 0.0000 | +0.0000 |
| `vow_fidelity` | 0.1658 | -0.1658 |
| `third_party_externality` | 0.0000 | +0.0000 |
| `care_protection` | 0.0500 | +0.0500 |
| `repair_residue` | 0.0000 | +0.0000 |

### aita_socialchem_v0.1

- n = 3
- principal-axis flips: 0/3
- verdict flips: 0/3

Per-dimension mean abs-delta from baseline:

| Dimension | mean abs-delta | mean signed delta |
|---|---:|---:|
| `physical_harm` | 0.0000 | +0.0000 |
| `rights_respect` | 0.0000 | +0.0000 |
| `fairness_equity` | 0.0000 | +0.0000 |
| `legitimacy_trust` | 0.0614 | -0.0614 |
| `epistemic_quality` | 0.0000 | +0.0000 |
| `autonomy_consent` | 0.0000 | +0.0000 |
| `vow_fidelity` | 0.0873 | -0.0873 |
| `third_party_externality` | 0.0000 | +0.0000 |
| `care_protection` | 0.0500 | +0.0500 |
| `repair_residue` | 0.0000 | +0.0000 |

## Per-question audit table

| q | baseline verdict | baseline axis | dearabby axis (delta-dim, abs-delta) | aita axis (delta-dim, abs-delta) |
|---:|---|---|---|---|
| 0 | tragic_conflict_escalate | - | - (vow_fidelity, 0.166) | - (legitimacy_trust, 0.092) |
| 1 | permitted | - | - (vow_fidelity, 0.166) | - (care_protection, 0.150) |
| 2 | permitted | - | - (vow_fidelity, 0.166) | - (vow_fidelity, 0.087) |

## Notes

- Abby's voice in the dating-advice sample is strongly **autonomy-respecting** ("no one can presume to make rules", "you're mature enough to know"). The SocialChem MFT-to-EM mapping has NO link to `autonomy_consent` (no MFT channel covers it), so the fitted dear_abby profile does NOT amplify the channel Abby herself emphasises most. Documented coverage gap, surfaced by this eval.
- On the morally-loaded examples, the dear_abby profile flips the DEME verdict in 2/3 cases, with the largest shift on `vow_fidelity` (mean -0.166) reflecting that channel's fitted downweighting (~0.72).
- n=12 / n=3 are descriptive, not inferential — wiring smoke test + coverage-gap demonstration, not an empirical claim.
