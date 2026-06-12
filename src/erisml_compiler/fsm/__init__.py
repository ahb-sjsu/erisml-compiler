"""Finite State Machines for moral state lifecycle tracking.

These FSMs are the silicon-castable core of Tier 1. Each FSM is deterministic,
has a bounded state space, and processes events one at a time. They are used
to track:

  - CommitmentFSM: a vow/promise/contract lifecycle (active -> defeasible ->
    defeated/violated/fulfilled).
  - LegitimacyFSM: an authority's legitimacy state across events.
  - ConsentFSM: consent acquisition, coercion, withdrawal.

When this layer is cast to silicon (FPGA/ASIC), each FSM becomes a small
state register with a transition function implemented as combinational
logic. Bounded memory: O(n_commitments) state registers. Bounded computation
per event: O(1) per FSM, summed across active FSMs.
"""

from erisml_compiler.fsm.commitment_fsm import CommitmentFSM
from erisml_compiler.fsm.consent_fsm import ConsentFSM
from erisml_compiler.fsm.legitimacy_fsm import LegitimacyFSM

__all__ = ["CommitmentFSM", "ConsentFSM", "LegitimacyFSM"]
