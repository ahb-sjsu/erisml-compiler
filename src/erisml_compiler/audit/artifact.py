"""Bundle a self-contained audit artifact (a folder of files)."""
from __future__ import annotations

import json
from pathlib import Path

from erisml_compiler.erisml_backend.codegen import render_erisml
from erisml_compiler.ir.schemas import CompilerIR


def bundle_artifact(ir: CompilerIR, out_dir: str | Path) -> Path:
    """Write the IR, source text, ErisML source, and a human-readable
    report into a directory. Returns the path to the directory.

    The folder is the smallest self-contained unit you can hand to a
    reviewer or auditor: it contains everything needed to evaluate or
    contest the verdict, with no external dependencies.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. The IR itself (canonical JSON).
    (out / "ir.json").write_text(
        json.dumps(ir.model_dump(mode="json"), indent=2, sort_keys=False),
        encoding="utf-8",
    )

    # 2. The source text.
    (out / "source.txt").write_text(ir.document.raw_text, encoding="utf-8")

    # 3. The ErisML rendering.
    (out / "compiled.erisml.yaml").write_text(render_erisml(ir), encoding="utf-8")

    # 4. A human-readable report.
    (out / "report.md").write_text(_render_report(ir), encoding="utf-8")

    return out


def _render_report(ir: CompilerIR) -> str:
    lines: list[str] = []
    lines.append(f"# Audit Report — {ir.document.doc_id}")
    lines.append("")
    if ir.audit:
        lines.append(f"- Compiler version: {ir.audit.compiler_version}")
        lines.append(f"- Tier: {ir.audit.tier}")
        lines.append(f"- Extractor: {ir.audit.extractor}")
        lines.append(f"- EM-DAG profile: {ir.audit.em_profile}")
        lines.append(f"- Timestamp (UTC): {ir.audit.timestamp_utc}")
        lines.append(f"- IR hash: `{ir.audit.ir_hash}`")
        lines.append(f"- Source text hash: `{ir.audit.source_text_hash}`")
        lines.append("")
    lines.append("## Source text")
    lines.append("```")
    lines.append(ir.document.raw_text)
    lines.append("```")
    lines.append("")
    lines.append(f"## Canonical form: `{ir.canonical_form or '(none)'}`")
    lines.append("")
    lines.append(f"## Stakeholders ({len(ir.stakeholders)})")
    for s in ir.stakeholders:
        lines.append(f"- **{s.label}** ({s.id}) — type={s.type}, roles={s.roles}, vulnerability={s.vulnerability}")
    lines.append("")
    lines.append(f"## Commitments ({len(ir.commitments)})")
    for c in ir.commitments:
        lines.append(f"- {c.id}: {c.type} by {c.holder} → {c.beneficiary or '(no beneficiary)'} | status={c.status}, legitimacy={c.legitimacy}")
    lines.append("")
    lines.append(f"## Ethical facts ({len(ir.ethical_facts)})")
    for f in ir.ethical_facts:
        lines.append(f"- {f.id} [{f.kind}, severity={f.severity}]: {f.description}")
    lines.append("")
    lines.append(f"## Conflicts ({len(ir.conflicts)})")
    for cf in ir.conflicts:
        lines.append(f"- **{cf.name}** (severity={cf.severity}, status={cf.resolution_status}, escalation={cf.requires_escalation})")
    lines.append("")
    lines.append("## EM-DAG outputs")
    for name, out in ir.em_outputs.items():
        score = out.score
        lines.append(f"- **{name}** (dim={out.module_name}): value={score.value:+.3f}, conf={score.confidence:.2f} — {score.explanation}")
    lines.append("")
    if ir.deme_verdict:
        v = ir.deme_verdict
        lines.append("## DEME verdict")
        lines.append(f"- **Verdict**: `{v.verdict}` (confidence {v.confidence:.2f})")
        lines.append(f"- Rationale: {v.rationale}")
        if v.allowed_actions:
            lines.append(f"- Allowed: {v.allowed_actions}")
        if v.prohibited_actions:
            lines.append(f"- Prohibited: {v.prohibited_actions}")
        if v.moral_residue:
            lines.append(f"- Moral residue: {v.moral_residue}")
        if v.escalation_required:
            lines.append("- **Escalation required.**")
        lines.append("")
    return "\n".join(lines)
