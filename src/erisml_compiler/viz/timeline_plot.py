"""Render the MoralVector timeline as a matplotlib figure.

Matplotlib is an optional dependency (in the `notebook` extra). The function
imports it lazily so the core CLI does not require it.
"""

from __future__ import annotations

from pathlib import Path

from erisml_compiler.ir.schemas import CompilerIR, MORAL_DIMENSIONS


def save_timeline_plot(ir: CompilerIR, out_path: str | Path) -> Path:
    """Save a 10-dimensional timeline plot. Returns the path to the file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if not ir.timeline:
        ax.text(0.5, 0.5, "No timeline data", ha="center", va="center", transform=ax.transAxes)
        out = Path(out_path)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    xs = [e.time_index for e in ir.timeline]
    cmap = plt.get_cmap("tab10")
    for i, dim in enumerate(MORAL_DIMENSIONS):
        ys = [getattr(e.vector, dim).value for e in ir.timeline]
        ax.plot(xs, ys, marker="o", color=cmap(i % 10), label=dim, alpha=0.85)

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Dimension value")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"MoralVector timeline — {ir.document.doc_id}")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
