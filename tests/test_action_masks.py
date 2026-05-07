from agentcontrol.deliberation_graph import Action, State, feasible_actions


def test_done_state_has_no_actions():
    assert feasible_actions(State(task_id='t', family='math', done=True)) == []


def test_read_document_masked_without_retrieval():
    actions = feasible_actions(State(task_id='t', family='evidence', remaining_budget=10.0))
    assert Action.READ_DOCUMENT not in actions
