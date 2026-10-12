"""Kantian universalizability for the deontic projection: erisml-lib's knowledge base.

The table of action kinds (the institution each presupposes, the form of contradiction its
universalised maxim meets, a justification) and the test that honours a maxim's polarity
live in erisml-lib (``erisml.ethics.universalizability``), where DEME's deontic gate derives
its sets from them. They were written here (v0.8) and moved to the library so that one copy
serves both the compiler's deontic projection and DEME. This module keeps the compiler's
import path.
"""

from __future__ import annotations

from erisml.ethics.universalizability import (
    DEFAULT_INSTITUTION_DEPENDENCIES,
    IMPERFECT_DUTIES,
    ContradictionType,
    InstitutionDependency,
    test_universalizability,
)

__all__ = [
    "ContradictionType",
    "DEFAULT_INSTITUTION_DEPENDENCIES",
    "IMPERFECT_DUTIES",
    "InstitutionDependency",
    "test_universalizability",
]
