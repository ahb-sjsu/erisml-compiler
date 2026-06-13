"""Smoke test for the Dear Abby ground-truth eval harness.

Runs the harness on a 1-row synthetic NDJSON to make sure the
compile-and-aggregate plumbing doesn't regress. Not an empirical
test of the profiles — just shape verification.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_dear_abby_groundtruth import (  # type: ignore[import-not-found]
    DIM_KEYS,
    evaluate,
    summarise,
)


def test_evaluate_returns_three_runs_per_pair() -> None:
    pairs = [{"question": "I had to choose between telling the truth and keeping my friend's secret.", "answer": "—"}]
    results = evaluate(pairs)
    assert set(results.keys()) == {
        "baseline",
        "dear_abby_socialchem_v0.1",
        "aita_socialchem_v0.1",
    }
    for runs in results.values():
        assert len(runs) == 1
        r = runs[0]
        assert "values" in r and set(r["values"].keys()) == set(DIM_KEYS)
        assert "verdict" in r


def test_summarise_emits_per_dim_stats() -> None:
    pairs = [{"question": "test situation involving a promise broken.", "answer": "—"}]
    results = evaluate(pairs)
    summary = summarise(results)
    assert "profiles" in summary
    for label, stats in summary["profiles"].items():
        assert stats["n"] == 1
        assert set(stats["mean_abs_dim_delta"].keys()) == set(DIM_KEYS)
