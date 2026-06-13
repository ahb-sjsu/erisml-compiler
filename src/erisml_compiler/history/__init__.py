"""Longitudinal habit tracking for the virtue projection.

Virtue ethics (Aristotle, Foot, Hursthouse, MacIntyre) fundamentally
requires *patterns* of acts, not single observations. "One swallow
does not make a summer; one fine day, neither" — a single courageous
act doesn't establish courage; only the disposition to act
courageously across many situations does.

The single-case projection in `projections/virtue.py` is honest
about this and assigns weak evidence: a v0 reading of "this act
expresses courage" is recorded as one data point. This subpackage
adds the longitudinal layer:

  - `HabitStore` — append-only filesystem-backed record per agent
  - `VirtueAssessment` — aggregated trait reading over the history
  - `LongitudinalVirtueProjection` — drop-in alternative to the
    single-case `VirtueProjection`. Reads the history, writes the
    current observation, returns a `ProjectionResult` informed by
    the pattern.

The store is **experimental**: it introduces *external state* into
what was previously a stateless deterministic compiler. The audit
chain captures the store's content hash at compile time so two
runs against the same history produce identical results, but the
history itself is mutable across runs.
"""

from erisml_compiler.history.habit_store import (
    ActRecord,
    HabitStore,
    VirtueAssessment,
)
from erisml_compiler.history.longitudinal_virtue import (
    LongitudinalVirtueProjection,
)
from erisml_compiler.history.sqlite_store import (
    SqliteHabitStore,
    migrate_from_jsonl,
)
from erisml_compiler.history.temporal_weighting import (
    TemporallyWeightedAssessment,
    temporally_weighted_assessment,
)

__all__ = [
    "ActRecord",
    "HabitStore",
    "LongitudinalVirtueProjection",
    "SqliteHabitStore",
    "TemporallyWeightedAssessment",
    "VirtueAssessment",
    "migrate_from_jsonl",
    "temporally_weighted_assessment",
]
