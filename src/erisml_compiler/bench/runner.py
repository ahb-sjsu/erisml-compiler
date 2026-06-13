"""Iterate scenarios under a bench corpus, compile each, score."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from erisml_compiler import __version__
from erisml_compiler.bench.schema import (
    BenchAggregate,
    ScenarioGold,
    ScenarioScore,
)
from erisml_compiler.bench.scoring import aggregate_score, load_weights, score_scenario
from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier


@dataclass
class BenchRun:
    bench_dir: Path
    scenarios: list[ScenarioGold] = field(default_factory=list)
    scores: list[ScenarioScore] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class BenchReport:
    bench_version: str
    compiler_version: str
    corpus_hash: str
    weights_used: dict[str, float]
    aggregate: BenchAggregate
    per_scenario: list[ScenarioScore]
    failed: list[tuple[str, str]]
    extractor: str
    tier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bench_version": self.bench_version,
            "compiler_version": self.compiler_version,
            "corpus_hash": self.corpus_hash,
            "weights_used": self.weights_used,
            "aggregate": self.aggregate.model_dump(),
            "per_scenario": [s.model_dump() for s in self.per_scenario],
            "failed": list(self.failed),
            "extractor": self.extractor,
            "tier": self.tier,
        }


def load_scenarios(bench_dir: str | Path) -> list[ScenarioGold]:
    """Load every *.yaml under bench_dir/scenarios/ into ScenarioGold."""
    d = Path(bench_dir) / "scenarios"
    if not d.exists():
        raise FileNotFoundError(f"No scenarios directory at {d}")
    out: list[ScenarioGold] = []
    for f in sorted(d.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        out.append(ScenarioGold.model_validate(data))
    return out


def corpus_hash(scenarios: list[ScenarioGold]) -> str:
    """Deterministic hash over (id, sha256(canonical-yaml)) per scenario."""
    parts = []
    for s in sorted(scenarios, key=lambda x: x.scenario_id):
        canonical = s.model_dump_json()
        h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        parts.append({"id": s.scenario_id, "h": h})
    canonical_all = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_all.encode("utf-8")).hexdigest()


def _compile_scenario(scenario: ScenarioGold, *, extractor: str) -> Any:
    """Compile one scenario; returns the IR or raises."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(scenario.raw_text)
        tmp = Path(f.name)
    try:
        return compile_document(
            tmp,
            CompileOptions(
                tier=CompilerTier.RULES,
                extractor=extractor,
                canonicalizer=RegistryCanonicalizer(),
                tensor_rank=2,
            ),
        )
    finally:
        tmp.unlink(missing_ok=True)


def run_bench(
    bench_dir: str | Path,
    *,
    extractor: str = "rule",
    weights_file: str | Path | None = None,
) -> BenchReport:
    """Iterate every scenario, compile, score, aggregate."""
    bench_dir = Path(bench_dir)
    scenarios = load_scenarios(bench_dir)
    weights = load_weights(weights_file)

    per_scenario: list[ScenarioScore] = []
    failed: list[tuple[str, str]] = []
    for s in scenarios:
        try:
            ir = _compile_scenario(s, extractor=extractor)
        except Exception as e:  # noqa: BLE001
            failed.append((s.scenario_id, f"{type(e).__name__}: {e}"))
            continue
        per_scenario.append(score_scenario(s, ir))

    agg = aggregate_score(per_scenario, weights)
    object.__setattr__(agg, "n_failed_compile", len(failed)) if False else None  # frozen model
    agg_d = agg.model_dump()
    agg_d["n_failed_compile"] = len(failed)
    agg = BenchAggregate.model_validate(agg_d)

    manifest_path = bench_dir / "manifest.yaml"
    bench_version = "unknown"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        bench_version = manifest.get("version", "unknown")

    return BenchReport(
        bench_version=bench_version,
        compiler_version=__version__,
        corpus_hash=corpus_hash(scenarios),
        weights_used=weights,
        aggregate=agg,
        per_scenario=per_scenario,
        failed=failed,
        extractor=extractor,
        tier="rules",
    )


def render_report_markdown(report: BenchReport) -> str:
    lines: list[str] = []
    lines.append(f"# MoralTensor-Bench {report.bench_version} report\n\n")
    lines.append(f"- compiler version: `{report.compiler_version}`\n")
    lines.append(f"- corpus hash: `{report.corpus_hash[:16]}...`\n")
    lines.append(f"- extractor: `{report.extractor}` (tier: {report.tier})\n")
    lines.append(f"- n scenarios: {report.aggregate.n_scenarios}\n")
    lines.append(f"- n failed compile: {report.aggregate.n_failed_compile}\n\n")

    lines.append("## Aggregate\n\n")
    a = report.aggregate
    lines.append("| Metric | Value |\n|---|---:|\n")
    lines.append(f"| mean stakeholder recall | {a.mean_stakeholder_recall:.3f} |\n")
    lines.append(f"| mean stakeholder role F1 | {a.mean_stakeholder_role_f1:.3f} |\n")
    lines.append(f"| mean commitment F1 | {a.mean_commitment_f1:.3f} |\n")
    lines.append(f"| mean canonical-form match | {a.mean_canonical_form_match:.3f} |\n")
    lines.append(f"| mean ethical-fact-kind recall | {a.mean_ethical_fact_kind_recall:.3f} |\n")
    lines.append(f"| mean per-party verdict accuracy | {a.mean_per_party_verdict_accuracy:.3f} |\n")
    lines.append(f"| mean overall verdict match | {a.mean_overall_verdict_match:.3f} |\n")
    lines.append(f"| premature-contraction rate | {a.premature_contraction_rate:.3f} |\n")
    lines.append(f"| **MoralTensor-Bench score** | **{a.moral_tensor_bench_score:.3f}** |\n\n")

    lines.append("## Per-scenario\n\n")
    lines.append("| id | category | stake recall | role F1 | comm F1 | canon | verdict |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for s in report.per_scenario:
        cf = (
            "-"
            if (s.canonical_form_match != s.canonical_form_match)
            else f"{s.canonical_form_match:.0f}"
        )
        ov = (
            "-"
            if (s.overall_verdict_match != s.overall_verdict_match)
            else f"{s.overall_verdict_match:.0f}"
        )
        lines.append(
            f"| `{s.scenario_id}` | {s.category} | {s.stakeholder_recall:.2f} "
            f"| {s.stakeholder_role_f1:.2f} | {s.commitment_f1:.2f} "
            f"| {cf} | {ov} |\n"
        )

    if report.failed:
        lines.append("\n## Compile failures\n\n")
        for sid, err in report.failed:
            lines.append(f"- `{sid}`: {err}\n")
    return "".join(lines)
