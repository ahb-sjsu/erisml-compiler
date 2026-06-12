"""Build the quickstart Jupyter notebook from cell definitions."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("""# ErisML Compiler — Quickstart

This notebook walks through compiling each of the three example texts end-to-end
through the Phase 1 MVP pipeline.

Pipeline: text → segmentation → MockExtractor → EM-DAG → MoralVector timeline → DEME stub → audit.

For an honest, deterministic, offline run, we use the **MockExtractor** here. The
**RuleExtractor** (Tier 2) works on arbitrary text but with the limits of pattern-based
extraction; the **LLMExtractor** (Tier 3) is a Phase-2 skeleton.
""")

code("""import sys
from pathlib import Path

from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier
from erisml_compiler.streaming.captioner import TerminalCaptioner
from erisml_compiler.streaming.streamer import MoralStreamer

EXAMPLES = Path("..") / "examples" if Path("..").joinpath("examples").exists() else Path("examples")
list(EXAMPLES.glob("*.txt"))""")

md("## 1. Nazi attic (canonical example from spec §28)")

code("""ir_nazi = compile_document(
    EXAMPLES / "nazi_attic.txt",
    CompileOptions(tier=CompilerTier.RULES, extractor="mock"),
)
print(f"Verdict: {ir_nazi.deme_verdict.verdict}")
print(f"Confidence: {ir_nazi.deme_verdict.confidence}")
print(f"Canonical form: {ir_nazi.canonical_form}")
print(f"IR hash: {ir_nazi.audit.ir_hash}")""")

code("""# Stream the compiled IR as real-time captions.
TerminalCaptioner().render(MoralStreamer(ir_nazi))""")

md("## 2. Medical confidentiality vs duty to warn")

code("""ir_med = compile_document(
    EXAMPLES / "medical_confidentiality.txt",
    CompileOptions(tier=CompilerTier.RULES, extractor="mock"),
)
print(f"Verdict: {ir_med.deme_verdict.verdict}")
print(f"Canonical form: {ir_med.canonical_form}")
for c in ir_med.commitments:
    print(f"  - {c.id}: {c.type} ({c.status})")""")

md("## 3. Whistleblower")

code("""ir_whistle = compile_document(
    EXAMPLES / "whistleblower.txt",
    CompileOptions(tier=CompilerTier.RULES, extractor="mock"),
)
print(f"Verdict: {ir_whistle.deme_verdict.verdict}")
print(f"Canonical form: {ir_whistle.canonical_form}")""")

md("## 4. The same nazi attic text with the rule extractor (no fixtures)")

code("""ir_nazi_rule = compile_document(
    EXAMPLES / "nazi_attic.txt",
    CompileOptions(tier=CompilerTier.RULES, extractor="rule"),
)
print(f"Rule-extractor verdict: {ir_nazi_rule.deme_verdict.verdict}")
print(f"Stakeholders detected: {[s.label for s in ir_nazi_rule.stakeholders]}")
print(f"Ethical facts detected: {[(f.kind, f.severity) for f in ir_nazi_rule.ethical_facts]}")""")

md("""## 5. MoralVector timeline visualization""")

code("""from erisml_compiler.viz.timeline_plot import save_timeline_plot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

png_path = Path("out_quickstart") / "nazi_attic.png"
png_path.parent.mkdir(exist_ok=True)
save_timeline_plot(ir_nazi, png_path)

img = mpimg.imread(png_path)
plt.figure(figsize=(10, 5))
plt.imshow(img)
plt.axis('off')
plt.show()""")

md("""## 6. EM-DAG outputs (per-module evaluations)""")

code("""for name in sorted(ir_nazi.em_outputs):
    out = ir_nazi.em_outputs[name]
    deps = f"  (deps: {out.upstream_dependencies})" if out.upstream_dependencies else ""
    print(f"  {name:12} = {out.score.value:+.3f}  conf={out.score.confidence:.2f}{deps}")
    print(f"               {out.score.explanation}")""")

md("""## 7. Audit record

The audit record contains everything needed to reproduce or contest the verdict.""")

code("""print(f"Compiler version : {ir_nazi.audit.compiler_version}")
print(f"Schema version   : {ir_nazi.audit.schema_version}")
print(f"Tier             : {ir_nazi.audit.tier}")
print(f"Extractor        : {ir_nazi.audit.extractor}")
print(f"EM-DAG profile   : {ir_nazi.audit.em_profile}")
print(f"Timestamp        : {ir_nazi.audit.timestamp_utc}")
print(f"IR hash          : {ir_nazi.audit.ir_hash}")
print(f"Source text hash : {ir_nazi.audit.source_text_hash}")
print()
print("Passes:")
for p in ir_nazi.audit.passes:
    print(f"  {p.pass_index:2d} {p.pass_name:35s} {p.duration_ms:.2f} ms")""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
out = Path(__file__).parent / "quickstart.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote: {out}")
