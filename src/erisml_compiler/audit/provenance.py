"""Per-pass provenance records."""
from __future__ import annotations

import time
from contextlib import contextmanager

from erisml_compiler.ir.schemas import PassRecord


@contextmanager
def record_pass(passes: list[PassRecord], pass_index: int, pass_name: str, tier: str):
    """Context manager: time a pass and append a record."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        passes.append(
            PassRecord(
                pass_index=pass_index,
                pass_name=pass_name,
                tier=tier,
                duration_ms=duration_ms,
            )
        )
