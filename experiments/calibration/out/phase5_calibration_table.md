# Phase-5 §6 probe calibration (activation lens)

Teacher: qwen3+glm-5 signed consensus. Metric: held-out sign-agreement vs teacher, on dims the teacher took a stance on (|v|>0.05). Included iff >= 0.70 (prereg C3).

| dimension | teacher | best layer | held-out sign-acc | n | included? |
|---|---|---|---|---|---|
| physical_harm | qwen3+glm-5 signed consensus | 24 | 0.710 | 155 | yes |
| rights_respect | qwen3+glm-5 signed consensus | 24 | 0.679 | 829 | NO (excluded) |
| fairness_equity | qwen3+glm-5 signed consensus | 0 | 0.714 | 896 | yes |
| autonomy_consent | qwen3+glm-5 signed consensus | 8 | 0.714 | 975 | yes |
| legitimacy_trust | qwen3+glm-5 signed consensus | 4 | 0.792 | 725 | yes |
| epistemic_quality | qwen3+glm-5 signed consensus | 0 | 0.680 | 493 | NO (excluded) |
| care_protection | qwen3+glm-5 signed consensus | 4 | 0.769 | 1057 | yes |
| vow_fidelity | qwen3+glm-5 signed consensus | 4 | 0.862 | 897 | yes |
| third_party_externality | qwen3+glm-5 signed consensus | 4 | 0.881 | 596 | yes |
| repair_residue | qwen3+glm-5 signed consensus | 4 | 0.838 | 314 | yes |
