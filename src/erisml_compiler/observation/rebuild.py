"""A running scene rebuilt on an updated IR, with one source of truth for history.

SceneRuntime builds its state machines once, from the IR it is given, and then accepts events only.
An observation patch can add stakeholders (who need consent and legitimacy machines) as well as
events, so the runtime is rebuilt on the patched IR and its whole history replayed.

Replay is exact only if the IR holds every event, in order. So every event lives in the IR: the
observation compiler appends observed events through its patches, and `append_event` appends the
system's own events (rulings, performed actions, measurements) with the next time index. A runtime
is never stepped with an event the IR does not hold. `rebuild_runtime` then checks that the old
runtime's history is a prefix of the new one and refuses otherwise.
"""

from __future__ import annotations

from typing import Any, Mapping

from erisml_compiler.ir.schemas import CompilerIR, Event
from erisml_compiler.runtime.scene import SceneRuntime


class SceneHistoryError(ValueError):
    """The updated IR does not extend the running scene's history."""


def _sig(e: Event) -> tuple:
    return (e.id, e.type, e.actor, e.target, e.content)


def append_event(ir: CompilerIR, event: Mapping[str, Any]) -> CompilerIR:
    """A copy of `ir` with `event` appended at the next time index. The id is assigned here."""
    t = max((e.time_index for e in ir.events), default=-1) + 1
    eid = f"sys{len(ir.events)}"
    taken = {e.id for e in ir.events}
    while eid in taken:
        eid += "_"
    body = {k: v for k, v in dict(event).items() if k not in ("id", "time_index")}
    new = ir.model_copy(deep=True)
    new.events = list(new.events) + [Event(id=eid, time_index=t, **body)]
    return new


def rebuild_runtime(old: SceneRuntime, new_ir: CompilerIR) -> SceneRuntime:
    rt = SceneRuntime(new_ir, getattr(old, "agent", None))
    before = [_sig(e) for e in old.events]
    after = [_sig(e) for e in rt.events]
    if after[: len(before)] != before:
        raise SceneHistoryError(
            "the updated IR does not extend the running scene's history "
            "(an event was rewritten or dropped, or the runtime was stepped "
            "with an event the IR does not hold)"
        )
    return rt
