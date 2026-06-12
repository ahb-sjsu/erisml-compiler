"""FSM tests: commitment, legitimacy, consent state machines."""
from erisml_compiler.fsm import CommitmentFSM, ConsentFSM, LegitimacyFSM
from erisml_compiler.ir.schemas import Commitment


def test_commitment_active_to_defeasible_to_defeated():
    c = Commitment(id="c1", type="vow", holder="self", content="x", status="active")
    fsm = CommitmentFSM(c)
    assert fsm.state == "active"
    fsm.step("defeasibility_condition_triggered", time_index=1)
    assert fsm.state == "active_but_defeasible"
    fsm.step("defeasibility_overrides", time_index=2)
    assert fsm.state == "defeated"
    assert fsm.is_terminal()


def test_commitment_terminal_states_are_absorbing():
    c = Commitment(id="c1", type="vow", holder="self", content="x", status="fulfilled")
    fsm = CommitmentFSM(c)
    fsm.step("violating_event", time_index=1)
    # Terminal states absorb further events.
    assert fsm.state == "fulfilled"


def test_legitimacy_full_to_coercive_to_void():
    fsm = LegitimacyFSM(authority_id="auth_x")
    assert fsm.state == "fully_legitimate"
    fsm.step("coercion_detected", time_index=1)
    assert fsm.state == "coercive"
    fsm.step("catastrophic_intent", time_index=2)
    assert fsm.state == "void"


def test_legitimacy_void_is_absorbing():
    fsm = LegitimacyFSM(authority_id="auth_x", initial="void")
    fsm.step("restored", time_index=1)
    assert fsm.state == "void"


def test_consent_obtained_then_withdrawn():
    fsm = ConsentFSM(subject_id="s1")
    fsm.step("consent_given", time_index=1)
    assert fsm.state == "obtained"
    fsm.step("withdrawn", time_index=2)
    assert fsm.state == "withdrawn"
