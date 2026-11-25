"""Grader of the precedent experiment. The bars live here and nowhere else.

Input: per-item squared errors of each arm on each held-out family, with the sampling unit of
each item. Output: one verdict per hypothesis and a diagnosis.

A hypothesis "X beats Y on family F" passes when the mean paired difference err_X - err_Y is
below -Z_BAR standard errors, the standard error taken from resampling whole sampling units.
The bar is in units of the noise of the difference it judges (geometric-methods 21.3).
"""

from __future__ import annotations

import numpy as np

Z_BAR = 2.5
N_BOOT = 1000
BOOT_SEED = 20261125

# (name, better arm, worse arm, family)
HYPOTHESES = (
    ("H1_gates", "B", "A", "flips"),  # an explicit gate check beats similarity on gate flips
    ("H2_metric", "Cp", "B", "tradeoff"),  # a learned metric adds prediction beyond gates
    ("H3_local", "Cl", "Cp", "tradeoff"),  # a metric per stratum beats one pooled metric
    ("H4_retrieval", "B", "G", "tradeoff"),  # precedents add prediction beyond the gate rule
)

DIAGNOSES = {
    # (H1, H2, H3): reading
    (False, False, False): "no_structure_detected",
    (False, True, False): "smooth_only",
    (True, False, False): "gates_only",
    (True, True, False): "gates_plus_pooled_metric",
    (True, True, True): "gates_plus_local_metric",
}


def paired(err_x: np.ndarray, err_y: np.ndarray, cluster: np.ndarray) -> dict:
    d = err_x - err_y
    units, inv = np.unique(cluster, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(units))
    counts = np.bincount(inv, minlength=len(units)).astype(float)
    rng = np.random.default_rng(BOOT_SEED)
    pick = rng.integers(0, len(units), size=(N_BOOT, len(units)))
    boot = sums[pick].sum(1) / counts[pick].sum(1)
    se = float(boot.std(ddof=1))
    mean = float(d.mean())
    return {
        "mean_diff": mean,
        "se": se,
        "z": mean / se if se > 0 else 0.0,
        "n_items": int(d.size),
        "n_units": int(len(units)),
    }


def grade(errors: dict, clusters: dict, source: str) -> dict:
    """errors[family][arm] -> per-item squared error; clusters[family] -> unit ids.

    `source` must be "simulated" or "held_out". A simulated input gets a rehearsal reading and
    never a grade; anything else is refused (feedback: provenance on measured inputs).
    """
    if source not in ("simulated", "held_out"):
        raise ValueError(f"REFUSED: source must be 'simulated' or 'held_out', got {source!r}")
    verdicts = {}
    for name, x, y, fam in HYPOTHESES:
        if fam not in errors:  # a family with no data is reported as not run, never as a miss
            verdicts[name] = {"not_run": True, "passes": False, "compares": f"{x} vs {y} on {fam}"}
            continue
        r = paired(np.asarray(errors[fam][x]), np.asarray(errors[fam][y]), np.asarray(clusters[fam]))
        r["passes"] = bool(r["z"] < -Z_BAR)
        r["compares"] = f"{x} vs {y} on {fam}"
        verdicts[name] = r
    if verdicts["H1_gates"].get("not_run"):
        # prereg-precedent-v1 as sealed: no flip family, so the reading rests on H2 alone
        reading = "metric_adds_beyond_gates_and_precedents" if verdicts["H2_metric"]["passes"] else "metric_unsupported_on_this_data"
    else:
        key = tuple(verdicts[h]["passes"] for h in ("H1_gates", "H2_metric", "H3_local"))
        reading = DIAGNOSES.get(key, "inconsistent_pattern")
    out = {
        "source": source,
        "z_bar": Z_BAR,
        "verdicts": verdicts,
        "mean_sq_error": {f: {a: float(np.mean(e)) for a, e in arms.items()} for f, arms in errors.items()},
    }
    # V5: the instrument must read some signal. At least one arm has to beat the grand-mean
    # predictor ("Mean") on ordinary held-out cases by the same bar, or the run is void.
    if "Mean" in errors.get("tradeoff", {}):
        zs = {
            a: paired(np.asarray(e), np.asarray(errors["tradeoff"]["Mean"]), np.asarray(clusters["tradeoff"]))["z"]
            for a, e in errors["tradeoff"].items()
            if a != "Mean"
        }
        out["V5_z_against_grand_mean"] = {a: round(z, 2) for a, z in zs.items()}
        out["V5_signal"] = bool(min(zs.values()) < -Z_BAR)
        if source == "held_out" and not out["V5_signal"]:
            reading = "VOID V5: no arm beats the grand-mean predictor"
    out["rehearsal_reading" if source == "simulated" else "diagnosis"] = reading
    return out
