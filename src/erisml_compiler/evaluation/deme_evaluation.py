"""The compiler's ethics evaluation, by erisml-lib's DEME.

The IR's ``moral_vectors`` and ``timeline`` are DEME's aggregated moral vector (erisml-lib's
nine dimensions) for the situation the document describes, as ``erisml_backend.deme_bridge``
judges it. The timeline replays the events as the earlier EM-DAG timeline did: one entry per
event, judged on the IR as it stood after that event, and one entry at time 0 when there are no
events. The bridge builds DEME's facts from the IR's ethical facts, which carry no time, so the
entries agree with each other until facts are time-stamped; the EM-DAG's modules read the events
directly, which DEME's modules, judging facts, do not.

Until 0.10 these were the compiler's own EM-DAG (ten one-dimension modules) and its ten-channel
vector. That graph is retired (erisml-lib's DEME is the evaluator); the compiler's ten-channel
``MoralVector`` remains only as the output space of the I-EIP activation probes.
"""

from __future__ import annotations

from erisml.ethics.moral_vector import MoralVector

from erisml_compiler.erisml_backend.deme_bridge import DEMEBridge
from erisml_compiler.ir.schemas import CompilerIR, DEMEVerdict, TimelineEntry


def evaluate(ir: CompilerIR, bridge: DEMEBridge) -> tuple[DEMEVerdict, MoralVector | None]:
    """DEME's verdict and moral vector for the IR."""
    return bridge.judge(ir)


def build_timeline(ir: CompilerIR, bridge: DEMEBridge) -> list[TimelineEntry]:
    """DEME's moral vector after each event (see the module docstring); entries for which DEME
    has no facts to judge are left out."""
    if not ir.events:
        _, vector = bridge.judge(ir)
        return (
            []
            if vector is None
            else [TimelineEntry(time_index=0, event_label="(no events)", vector=vector)]
        )
    timeline: list[TimelineEntry] = []
    events = sorted(ir.events, key=lambda e: e.time_index)
    for cutoff in events:
        snapshot = ir.model_copy(
            update={"events": [e for e in events if e.time_index <= cutoff.time_index]}
        )
        _, vector = bridge.judge(snapshot)
        if vector is not None:
            timeline.append(
                TimelineEntry(
                    time_index=cutoff.time_index,
                    event_id=cutoff.id,
                    event_label=cutoff.type,
                    vector=vector,
                )
            )
    return timeline


__all__ = ["build_timeline", "evaluate"]
