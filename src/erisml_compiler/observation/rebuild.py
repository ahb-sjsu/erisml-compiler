"""A running scene rebuilt on an updated IR, with one causal history.

SceneRuntime builds its state machines once, from the IR it is given, and then accepts events only.
An observation patch can add stakeholders (who need consent and legitimacy machines) as well as
events, so the runtime is rebuilt on the patched IR and its whole history replayed.

Every event lives in the IR and carries a vector clock (erisml_compiler.observation.clock): the
observation compiler's events are stamped by the `observer`, the agent's own events (rulings,
measurements, performed actions) by their producer through `append_event`, and another party's
ledger is merged with `merge_ledger`. A runtime is never stepped with an event the IR does not hold.
`rebuild_runtime` refuses unless the new history extends the old one causally: every event the
running scene had stepped is still there, every happened-before pair keeps its order, and the new
history is consistent. (An earlier version demanded an exact prefix; a patch compiled while other
events were appended then could not be placed, though its events were merely concurrent with them.)
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from erisml_compiler.ir.schemas import CompilerIR, Event
from erisml_compiler.observation import clock as vc
from erisml_compiler.observation.compiler import history_clock
from erisml_compiler.runtime.scene import SceneRuntime


class SceneHistoryError(ValueError):
    """The updated IR does not extend the running scene's history causally."""


def _ordered(ir: CompilerIR) -> list:
    return sorted(ir.events, key=lambda e: e.time_index)


def append_event(ir: CompilerIR, event: Mapping[str, Any], producer: str = "system") -> CompilerIR:
    """A copy of `ir` with `event` appended at the next time index, its clock the producer's tick on
    top of everything the history has seen. The id is assigned here."""
    t = max((e.time_index for e in ir.events), default=-1) + 1
    eid = f"sys{len(ir.events)}"
    taken = {e.id for e in ir.events}
    while eid in taken:
        eid += "_"
    body = {k: v for k, v in dict(event).items() if k not in ("id", "time_index", "clock")}
    new = ir.model_copy(deep=True)
    new.events = list(new.events) + [
        Event(id=eid, time_index=t, clock=vc.tick(history_clock(ir), producer), **body)
    ]
    return new


def merge_ledger(ir: CompilerIR, events: Sequence[Mapping[str, Any]]) -> CompilerIR:
    """Merge another party's ledger: its events (each with id and clock) that the IR does not hold
    are appended in their ledger order. Refused if the result would not be causally consistent, that
    is, if a merged event happened before an event already in the history."""
    have = {e.id for e in ir.events}
    fresh = [dict(e) for e in events if e["id"] not in have]
    if not fresh:
        return ir
    t = max((e.time_index for e in ir.events), default=-1) + 1
    new = ir.model_copy(deep=True)
    new.events = list(new.events) + [
        Event(**{**e, "time_index": t + i}) for i, e in enumerate(fresh)
    ]
    if not vc.consistent([e.clock for e in _ordered(new)]):
        raise SceneHistoryError(
            "the merged ledger has an event that happened before one already in the history"
        )
    return new


def rebuild_runtime(old: SceneRuntime, new_ir: CompilerIR) -> SceneRuntime:
    rt = SceneRuntime(new_ir, getattr(old, "agent", None))
    if not vc.extends([(e.id, e.clock) for e in old.events], [(e.id, e.clock) for e in rt.events]):
        raise SceneHistoryError(
            "the updated IR does not extend the running scene's history causally "
            "(an event was dropped or reordered against happened-before, or the runtime "
            "was stepped with an event the IR does not hold)"
        )
    return rt
