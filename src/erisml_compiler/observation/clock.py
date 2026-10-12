"""Vector clocks for a scene's history: causal order across producers.

A scene's events come from several producers: the observation compiler, the agent's evidence plane
(rulings, measurements, performed actions), and other parties that keep ledgers of their own (a
monitoring centre, a dispatcher). Each event carries a vector clock, one counter per producer. The
happened-before relation is the clock order: a precedes b when a's clock is at most b's in every
component and differs in one. Two events neither of which precedes the other are concurrent.

The scene's linear order (time_index) is arrival order, and it must be a linear extension of
happened-before: no event may stand before an event it depends on. A history that grows may only
extend causally: every happened-before pair of the old history keeps its order, and the new history
is itself consistent. Concurrent events may be ordered either way; nothing depends on which.

An event without a clock (a scene's standing events, compiled before clocks) has the empty clock,
which precedes every clocked event; such events keep their written order among themselves.

`consistent` and `extends` compare pairs, so they cost the square of the history's length: well under
a millisecond for a scene's few hundred events. A history of many thousands would want an index by
producer instead.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

Clock = Dict[str, int]


def tick(clock: Mapping[str, int], producer: str) -> Clock:
    out = dict(clock)
    out[producer] = out.get(producer, 0) + 1
    return out


def merge(*clocks: Mapping[str, int]) -> Clock:
    out: Clock = {}
    for c in clocks:
        for k, v in c.items():
            out[k] = max(out.get(k, 0), v)
    return out


def leq(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    return all(v <= b.get(k, 0) for k, v in a.items())


def before(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    """a happened before b."""
    return leq(a, b) and dict(a) != dict(b)


def concurrent(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    return not before(a, b) and not before(b, a) and dict(a) != dict(b)


def frontier(clocks: Iterable[Mapping[str, int]]) -> Clock:
    """The clock that has seen every given clock."""
    return merge(*list(clocks))


def consistent(clocks: Sequence[Mapping[str, int]]) -> bool:
    """No later entry happened before an earlier one."""
    return not any(
        before(clocks[j], clocks[i]) for i in range(len(clocks)) for j in range(i + 1, len(clocks))
    )


def extends(old: Sequence[tuple], new: Sequence[tuple]) -> bool:
    """`old` and `new` are sequences of (event_id, clock). True when every old event is in `new`, every
    happened-before pair of the old history keeps its order in the new one, events without clocks keep
    their old relative order, and the new history is consistent."""
    pos = {eid: i for i, (eid, _) in enumerate(new)}
    if any(eid not in pos for eid, _ in old):
        return False
    for i, (a, ca) in enumerate(old):
        for b, cb in old[i + 1 :]:
            ordered = before(ca, cb) or (not ca and not cb)
            if ordered and pos[a] > pos[b]:
                return False
    return consistent([c for _, c in new])
