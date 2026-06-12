"""Build the MoralVector timeline by replaying events through the FSM layer
and re-running the EM-DAG at each time step.

For the MVP, the timeline is computed as: for each event in temporal order,
take a snapshot of the IR with all events up to and including this one,
re-evaluate the EM-DAG, and project to a MoralVector.

This is intentionally simple: it gives us a deterministic, reproducible
timeline. Phase 2 will replace it with incremental FSM-driven updates
(only re-evaluate EMs whose dependencies changed) for performance.
"""
from __future__ import annotations

from erisml_compiler.em_dag import EMDAG
from erisml_compiler.evaluation.moral_vector import build_moral_vector_from_em_outputs
from erisml_compiler.ir.schemas import CompilerIR, TimelineEntry


def build_timeline(ir: CompilerIR, dag: EMDAG) -> list[TimelineEntry]:
    """Replay events through the EM-DAG and produce a TimelineEntry per
    time step.

    Special case: if the IR has no events, emit a single entry at t=0 with
    the static MoralVector computed from the full IR.
    """
    if not ir.events:
        em_outputs = dag.evaluate(ir)
        vector = build_moral_vector_from_em_outputs(em_outputs, dag)
        return [TimelineEntry(time_index=0, event_label="(no events)", vector=vector)]

    timeline: list[TimelineEntry] = []
    events_sorted = sorted(ir.events, key=lambda e: e.time_index)
    for cutoff_event in events_sorted:
        # Snapshot IR with events up to and including cutoff_event.
        snapshot_events = [e for e in events_sorted if e.time_index <= cutoff_event.time_index]
        # Commitments: include all (status reflects FSM at this time).
        # Ethical facts: include all (the extractor places them at appropriate spans).
        snapshot_ir = ir.model_copy(update={"events": snapshot_events})
        em_outputs = dag.evaluate(snapshot_ir)
        vector = build_moral_vector_from_em_outputs(em_outputs, dag)
        timeline.append(
            TimelineEntry(
                time_index=cutoff_event.time_index,
                event_id=cutoff_event.id,
                event_label=cutoff_event.type,
                vector=vector,
            )
        )
    return timeline
