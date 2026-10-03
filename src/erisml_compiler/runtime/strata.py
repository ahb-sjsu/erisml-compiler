"""Strata: regions of the moral space and the semantic gates between them.

The discrete realization of Geometric Ethics chapter 8 ("Stratification: Boundaries, Thresholds,
and Phase Transitions"). A scene's moral space is stratified: within a stratum the scene's
allowed, obliged and prohibited sets are constant, and they change only when a boundary is
crossed. A scene declares one Stratification per axis (who a visitor is to the household,
what an animal is to it, ...) in ``extra["strata"]``:

    visitor_standing:
      states: [none, stranger, welcomed, arranged, hostile]
      initial: none
      authority: [arranged]          # states that grant something
      absorbing: [hostile]           # Type III (Def. 8.6): left only by human oversight
      gates:                         # semantic gates (Def. 8.8), first match wins
        - {id: g1, trigger: person_entered, from: [none], to: stranger, boundary: threshold}
        - {id: g2, trigger: attack_by_person, to: hostile, boundary: phase}

A gate G: F x S_alpha -> S_beta fires discretely when its triggering feature is the event just
stepped (`trigger`: "type" or "type=content"; the key is not `on`, which
YAML 1.1 reads as the boolean True; edge-triggered, so history never re-fires it) and the
stratification is in one of its source strata (`from`; any stratum when omitted). Its boundary
is Type I (`threshold`, a measured quantity crossing a value) or Type II (`phase`, a change of
regime); Type IV constraint surfaces are not states but the scene's non-defeasible prohibitions.
Each crossing is recorded as boundary crossing data (Def. 8.11): the stratification, the gate,
its boundary type, and the source and target strata.

Two rules are checked when the scene loads, so a scene that breaks them never runs:

- containment: an authority state grants something, so only a system event (declared
  `source: system` in ``extra["event_types"]``) may lead into it, never a model's reading of the
  world (erisml-lib docs/papers/foundations/no_escape.tex);
- absorption: no gate leaves an absorbing state except on a human-oversight event
  (``extra["oversight"]``), the only thing that restores a defeated commitment as well.

Norms read the current stratum as ``stratum:<name>=<state>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BOUNDARIES = ("threshold", "phase")


@dataclass(frozen=True)
class Gate:
    id: str
    on_type: str
    on_content: str | None
    to: str
    sources: tuple[str, ...]
    boundary: str

    def fires(self, event_type: str, content: str | None, current: str) -> bool:
        return (
            event_type == self.on_type
            and (self.on_content is None or (content or "") == self.on_content)
            and (not self.sources or current in self.sources)
        )


class Stratification:
    """One axis of the scene's moral space: its strata, gates and current stratum."""

    def __init__(
        self, name: str, decl: dict[str, Any], event_types: dict[str, Any], oversight: set[str]
    ):
        self.name = name
        self.states: list[str] = list(decl.get("states") or [])
        self.initial: str = decl.get("initial", self.states[0] if self.states else "")
        self.authority = set(decl.get("authority") or [])
        self.absorbing = set(decl.get("absorbing") or [])
        self.entered_at = 0  # the run index of the event that last established the stratum
        if not self.states or self.initial not in self.states:
            raise ValueError(
                f"stratum {name!r}: initial state {self.initial!r} not among {self.states}"
            )
        for label, group in (("authority", self.authority), ("absorbing", self.absorbing)):
            if group - set(self.states):
                raise ValueError(
                    f"stratum {name!r}: {label} states {sorted(group - set(self.states))} undeclared"
                )
        if self.initial in self.authority:
            raise ValueError(f"stratum {name!r}: an authority state cannot be where it starts")
        self.gates: list[Gate] = []
        for i, g in enumerate(decl.get("gates") or []):
            if "trigger" not in g:
                raise ValueError(
                    f"stratum {name!r}: gate {g.get('id', i)!r} has no trigger (a YAML key `on:` is read as True)"
                )
            typ, _, content = str(g["trigger"]).partition("=")
            gate = Gate(
                id=str(g.get("id") or f"{name}.{i}"),
                on_type=typ,
                on_content=content or None,
                to=g.get("to"),
                sources=tuple(g.get("from") or ()),
                boundary=g.get("boundary", "threshold"),
            )
            self._check(gate, event_types, oversight)
            self.gates.append(gate)
        self.state = self.initial

    def _check(self, g: Gate, event_types: dict[str, Any], oversight: set[str]) -> None:
        where = f"stratum {self.name!r}, gate {g.id!r}"
        if g.to not in self.states or any(s not in self.states for s in g.sources):
            raise ValueError(f"{where}: names an undeclared state")
        if g.boundary not in BOUNDARIES:
            raise ValueError(f"{where}: boundary {g.boundary!r} is not one of {BOUNDARIES}")
        if g.on_type not in event_types:
            raise ValueError(f"{where}: fires on the undeclared event type {g.on_type!r}")
        if g.to in self.authority and (event_types[g.on_type] or {}).get("source") != "system":
            raise ValueError(
                f"{where}: {g.on_type!r} is not a system event, so it cannot lead into the authority state {g.to!r}"
            )
        leaves = set(g.sources or self.states) - {g.to}
        if leaves & self.absorbing and g.on_type not in oversight:
            raise ValueError(
                f"{where}: leaves the absorbing state(s) {sorted(leaves & self.absorbing)} on {g.on_type!r}, "
                "which is not a human-oversight event"
            )

    def cross(self, event_type: str, content: str | None, index: int = 0) -> dict[str, Any] | None:
        """Step one event; the boundary crossing data if a gate fired and moved the stratum.

        `index` is the event's position in the run: a gate that fires records it as the moment the
        stratum was (re-)established, even when it leaves the stratum where it was (fresh evidence
        for the same stratum), so an obligation discharged earlier is owed again."""
        for g in self.gates:
            if g.fires(event_type, content, self.state):
                self.entered_at = index
                if g.to == self.state:
                    return None
                crossing = {
                    "stratum": self.name,
                    "gate": g.id,
                    "boundary": g.boundary,
                    "from": self.state,
                    "to": g.to,
                }
                self.state = g.to
                return crossing
        return None


def load(extra: dict[str, Any]) -> dict[str, Stratification]:
    """Every stratification a scene declares, checked."""
    types = extra.get("event_types") or {}
    oversight = set(extra.get("oversight") or [])
    return {
        name: Stratification(name, d, types, oversight)
        for name, d in (extra.get("strata") or {}).items()
    }
