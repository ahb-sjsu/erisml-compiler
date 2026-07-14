# Phase-5 §6 probe calibration — activation lens (FIXED readout layer)

Teacher: qwen3+glm-5 signed-valence consensus (independent of the rule-based text lens).
Corpus: Social Chemistry 101, N=5000 (CC-BY). Metric: held-out sign-agreement vs teacher
on dims the teacher took a stance on (|v|>0.05). Included iff >= 0.70 (prereg C3).

**Readout layer = 8** (frozen before the Phase-5 scenarios). Chosen on calibration data:
early-mid layers calibrate best and the final layer (27) is empirically WORST — corroborating
C1 (the readout layer is a next-token-statistics artifact). Per-layer included counts:
0:7  4:7  **8:7**  12:6  16:6  20:6  24:6  27:6.

| dimension | held-out sign-acc @L8 | n | included? |
|---|---|---|---|
| physical_harm | 0.665 | 155 | NO (excluded) |
| rights_respect | 0.677 | 829 | NO (excluded) |
| fairness_equity | 0.710 | 896 | yes |
| autonomy_consent | 0.714 | 975 | yes |
| legitimacy_trust | 0.792 | 725 | yes |
| epistemic_quality | 0.633 | 493 | NO (excluded) |
| care_protection | 0.769 | 1057 | yes |
| vow_fidelity | 0.862 | 897 | yes |
| third_party_externality | 0.881 | 596 | yes |
| repair_residue | 0.838 | 314 | yes |

**7/10 dimensions calibrated at layer 8.** Excluded (below gate): physical_harm (0.66), rights_respect (0.68), epistemic_quality (0.63). The Phase-5 delta lens uses only the included set; excluded dims are reported uncalibrated.
