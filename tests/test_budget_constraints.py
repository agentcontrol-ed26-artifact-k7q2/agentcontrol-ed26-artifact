from agentcontrol.deliberation_graph import Action, State, apply_budget, feasible_actions


def test_budget_constraints_enforced():
    state = State(task_id='t', family='math', remaining_budget=0.5)
    assert Action.STRONG_ANSWER not in feasible_actions(state)
    new_state = apply_budget(state, Action.CALL_VERIFIER, cost=0.1)
    assert new_state.remaining_budget == 0.4
