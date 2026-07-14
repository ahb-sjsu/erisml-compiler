"""Signed-valence teacher for activation-probe calibration (Phase 5).

Unlike the keystone `m10` labels (0..1 "engagement / at-stake" magnitude), the
I-EIP monitor's MoralVector — and the delta lens's direction-break count — is
SIGNED valence in [-1, 1]. This module elicits that signed reading from an LLM
judge over the NRP `ellm` endpoint, using the same transport as
`paper/keystone/score_mfrc_multi.py` (two independent judges: qwen3 + glm-5).

Sign convention, per dimension:
    -1  the dimension is VIOLATED / harmed / breached
     0  not engaged / neutral
    +1  the dimension is UPHELD / honored / promoted

Import-light on purpose (stdlib + requests only) so it runs on Atlas beside the
GPU extraction and is unit-testable on the host with no network.
"""

from __future__ import annotations

import json
import re

# Column order MUST equal erisml_compiler.ir.schemas.MORAL_DIMENSIONS. Duplicated
# here as a literal so this file has no heavy import chain on the Atlas side; the
# order is asserted against the real tuple in the test.
DIMENSIONS: tuple[str, ...] = (
    "physical_harm",
    "rights_respect",
    "fairness_equity",
    "autonomy_consent",
    "legitimacy_trust",
    "epistemic_quality",
    "care_protection",
    "vow_fidelity",
    "third_party_externality",
    "repair_residue",
)

_DESCRIPTIONS: dict[str, str] = {
    "physical_harm": "bodily or material harm",
    "rights_respect": "rights and duties",
    "fairness_equity": "fairness, equal or proportional treatment",
    "autonomy_consent": "autonomy, consent, freedom from coercion",
    "legitimacy_trust": "authority, legitimacy, institutional trust",
    "epistemic_quality": "honesty, truth, evidence",
    "care_protection": "care, compassion, protecting the vulnerable",
    "vow_fidelity": "loyalty, promises, fidelity to a relationship or group",
    "third_party_externality": "effects on uninvolved third parties",
    "repair_residue": "apology, repair, making amends",
}

_RUBRIC = (
    "For the text below, rate each moral dimension from -1.0 to +1.0 by its "
    "VALENCE in the situation:\n"
    "  negative (toward -1): the dimension is VIOLATED, harmed, or breached\n"
    "  zero (0.0): the dimension is not engaged / not relevant / neutral\n"
    "  positive (toward +1): the dimension is UPHELD, honored, or promoted\n"
    "Judge the situation described, not the writing. Dimensions:\n"
    + "\n".join(f"{d}: {_DESCRIPTIONS[d]}" for d in DIMENSIONS)
    + "\nRespond with ONLY a JSON object mapping each dimension name to its "
    "signed score in [-1, 1]."
)

SYSTEM = "Careful moral annotator. Output only a JSON object. No prose."
DEFAULT_URL = "https://ellm.nrp-nautilus.io/v1/chat/completions"


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _RUBRIC + "\n\nTEXT:\n" + text},
    ]


def parse_signed(content: str) -> dict | None:
    """Extract a signed 10-dim vector from a judge completion.

    Strips <think>…</think>, scans JSON objects from the end (the last complete
    object is usually the answer), and accepts the first that mentions our dims.
    Values are coerced to float and clamped to [-1, 1]; missing dims default to
    0.0 (not engaged). Returns None only if no dimension is found at all.
    """
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    for obj in reversed(re.findall(r"\{[^{}]*\}", content.replace("\n", " "))):
        try:
            d = json.loads(obj)
        except Exception:
            continue
        if not any(k in d for k in DIMENSIONS):
            continue
        out: dict[str, float] = {}
        for k in DIMENSIONS:
            try:
                v = float(d.get(k, 0.0))
            except (TypeError, ValueError):
                v = 0.0
            out[k] = max(-1.0, min(1.0, v))
        return out
    return None


def score_text(text: str, model: str, token: str, *, url: str = DEFAULT_URL,
               timeout: int = 90) -> dict | None:
    """Call one judge model and return its signed 10-dim vector (or None)."""
    import requests

    body = {
        "model": model,
        "messages": build_messages(text),
        "temperature": 0,
        "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                          json=body, timeout=timeout)
        return parse_signed(r.json()["choices"][0]["message"]["content"])
    except Exception:
        return None


def to_vector(label: dict) -> list[float]:
    """Dict -> ordered list aligned to DIMENSIONS."""
    return [float(label.get(d, 0.0)) for d in DIMENSIONS]


def consensus(labels: list[dict], dead_band: float = 0.05) -> tuple[list[float], dict]:
    """Average two (or more) judges into one signed vector, with an agreement
    diagnostic. Agreement = fraction of dims where the judges share a sign on
    dims where at least one took a stance (|v| > dead_band). Low agreement flags
    a noisy/ambiguous item for the label-quality report."""
    labels = [x for x in labels if x]
    if not labels:
        return [0.0] * len(DIMENSIONS), {"n_judges": 0, "sign_agreement": None}
    vecs = [to_vector(x) for x in labels]
    mean = [sum(col) / len(col) for col in zip(*vecs)]
    if len(vecs) < 2:
        return mean, {"n_judges": len(vecs), "sign_agreement": None}

    agree = considered = 0
    for i in range(len(DIMENSIONS)):
        vals = [v[i] for v in vecs]
        if all(abs(x) <= dead_band for x in vals):
            continue
        considered += 1
        signs = {1 if x > dead_band else (-1 if x < -dead_band else 0) for x in vals}
        if len(signs) == 1:
            agree += 1
    return mean, {
        "n_judges": len(vecs),
        "sign_agreement": (agree / considered) if considered else None,
        "n_dims_considered": considered,
    }


if __name__ == "__main__":
    # Offline smoke: parse a well-formed and a noisy completion.
    good = '{"physical_harm": -0.9, "rights_respect": 0.3, "care_protection": 0.8}'
    p = parse_signed(good)
    assert p and p["physical_harm"] == -0.9 and p["care_protection"] == 0.8
    assert p["legitimacy_trust"] == 0.0  # missing -> not engaged
    noisy = "<think>hmm</think> here you go: " + good + " done"
    assert parse_signed(noisy)["rights_respect"] == 0.3
    m, ag = consensus([{"physical_harm": -0.8}, {"physical_harm": -0.6}])
    assert m[0] == -0.7 and ag["sign_agreement"] == 1.0
    print("signed_rubric offline smoke OK")
