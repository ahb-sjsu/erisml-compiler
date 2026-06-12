"""Self-contained HTML report.

Bundles everything needed to evaluate or contest the verdict into a single
HTML file with embedded CSS and (optionally) a base64-embedded timeline
plot. No external assets: a reviewer can email it, save it, archive it.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

from erisml_compiler.ir.schemas import CompilerIR


def _embed_image(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="MoralVector timeline">'


def render_html_report(
    ir: CompilerIR,
    timeline_png: Path | None = None,
) -> str:
    """Return a complete, self-contained HTML document."""
    e = html.escape

    rows_stakeholders = "".join(
        f"<tr><td>{e(s.id)}</td><td>{e(s.label)}</td><td>{e(s.type)}</td>"
        f"<td>{e(', '.join(s.roles))}</td><td>{e(s.vulnerability or '')}</td></tr>"
        for s in ir.stakeholders
    )
    rows_commitments = "".join(
        f"<tr><td>{e(c.id)}</td><td>{e(c.type)}</td><td>{e(c.holder)}</td>"
        f"<td>{e(c.beneficiary or '')}</td><td>{e(c.status)}</td>"
        f"<td>{e(c.legitimacy)}</td><td>{e(c.content)}</td></tr>"
        for c in ir.commitments
    )
    rows_facts = "".join(
        f"<tr><td>{e(f.id)}</td><td>{e(f.kind)}</td>"
        f"<td>{e(f.severity or '')}</td><td>{f.confidence:.2f}</td>"
        f"<td>{e(f.description)}</td></tr>"
        for f in ir.ethical_facts
    )
    rows_em = "".join(
        f"<tr><td>{e(name)}</td><td>{o.score.value:+.3f}</td>"
        f"<td>{o.score.confidence:.2f}</td><td>{e(o.score.explanation or '')}</td></tr>"
        for name, o in sorted(ir.em_outputs.items())
    )
    rows_timeline = "".join(
        f"<tr><td>{te.time_index}</td><td>{e(te.event_label)}</td>"
        f"<td>{te.vector.physical_harm.value:+.2f}</td>"
        f"<td>{te.vector.third_party_externality.value:+.2f}</td>"
        f"<td>{te.vector.vow_fidelity.value:+.2f}</td>"
        f"<td>{te.vector.care_protection.value:+.2f}</td>"
        f"<td>{te.vector.legitimacy_trust.value:+.2f}</td></tr>"
        for te in ir.timeline
    )

    verdict_html = ""
    if ir.deme_verdict:
        v = ir.deme_verdict
        verdict_html = (
            f"<div class='verdict'><h2>DEME Verdict</h2>"
            f"<div class='verdict-value'>{e(v.verdict)} "
            f"<span class='conf'>(confidence {v.confidence:.2f})</span></div>"
            f"<p><b>Rationale.</b> {e(v.rationale)}</p>"
        )
        if v.allowed_actions:
            verdict_html += f"<p><b>Allowed.</b> {e(', '.join(v.allowed_actions))}</p>"
        if v.prohibited_actions:
            verdict_html += f"<p><b>Prohibited.</b> {e(', '.join(v.prohibited_actions))}</p>"
        if v.moral_residue:
            verdict_html += f"<p><b>Moral residue.</b> {e(', '.join(v.moral_residue))}</p>"
        if v.escalation_required:
            verdict_html += "<p class='escalate'><b>Escalation required.</b></p>"
        verdict_html += "</div>"

    audit_html = ""
    if ir.audit:
        a = ir.audit
        audit_html = (
            "<div class='audit'><h2>Audit Trail</h2><ul>"
            f"<li>Compiler version: {e(a.compiler_version)}</li>"
            f"<li>Tier: {e(a.tier)}</li>"
            f"<li>Extractor: {e(a.extractor)}</li>"
            f"<li>EM-DAG profile: {e(a.em_profile or '')}</li>"
            f"<li>Timestamp: {e(a.timestamp_utc)}</li>"
            f"<li>IR hash: <code>{e(a.ir_hash)}</code></li>"
            f"<li>Source text hash: <code>{e(a.source_text_hash)}</code></li>"
            "</ul></div>"
        )

    plot_html = _embed_image(timeline_png) if timeline_png else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ErisML Compiler Report — {e(ir.document.doc_id)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }}
  h1, h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 13px; }}
  th, td {{ text-align: left; padding: 4px 8px; border: 1px solid #ddd; }}
  th {{ background: #f4f4f4; }}
  pre {{ background: #f4f4f4; padding: 1em; overflow-x: auto; }}
  .verdict {{ background: #fffbe6; padding: 1em; border-left: 4px solid #d39e00; margin: 1em 0; }}
  .verdict-value {{ font-size: 1.2em; font-weight: bold; }}
  .conf {{ font-weight: normal; color: #666; }}
  .audit {{ background: #f0f7ff; padding: 1em; border-left: 4px solid #0066cc; margin: 1em 0; font-size: 13px; }}
  .escalate {{ color: #d04444; font-weight: bold; }}
  code {{ font-family: ui-monospace, Consolas, monospace; }}
</style>
</head><body>
<h1>ErisML Compiler Report</h1>
<p><b>Document:</b> {e(ir.document.doc_id)} &middot; <b>Language:</b> {e(ir.document.language)} &middot; <b>Canonical form:</b> <code>{e(ir.canonical_form or '(none)')}</code></p>
<h2>Source text</h2>
<pre>{e(ir.document.raw_text)}</pre>
{verdict_html}
<h2>Stakeholders ({len(ir.stakeholders)})</h2>
<table><thead><tr><th>id</th><th>label</th><th>type</th><th>roles</th><th>vulnerability</th></tr></thead>
<tbody>{rows_stakeholders}</tbody></table>
<h2>Commitments ({len(ir.commitments)})</h2>
<table><thead><tr><th>id</th><th>type</th><th>holder</th><th>beneficiary</th><th>status</th><th>legitimacy</th><th>content</th></tr></thead>
<tbody>{rows_commitments}</tbody></table>
<h2>Ethical facts ({len(ir.ethical_facts)})</h2>
<table><thead><tr><th>id</th><th>kind</th><th>severity</th><th>conf</th><th>description</th></tr></thead>
<tbody>{rows_facts}</tbody></table>
<h2>EM-DAG outputs</h2>
<table><thead><tr><th>module</th><th>value</th><th>conf</th><th>explanation</th></tr></thead>
<tbody>{rows_em}</tbody></table>
{plot_html}
<h2>MoralVector timeline</h2>
<table><thead><tr><th>t</th><th>event</th><th>harm</th><th>externality</th><th>fidelity</th><th>care</th><th>legitimacy</th></tr></thead>
<tbody>{rows_timeline}</tbody></table>
{audit_html}
</body></html>
"""
