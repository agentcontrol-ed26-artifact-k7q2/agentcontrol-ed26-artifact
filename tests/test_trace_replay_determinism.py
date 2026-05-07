from agentcontrol.trace_store import TraceEvent, TraceStore, ReplayTraceStore, state_hash


def test_trace_replay_determinism(tmp_path):
    path = tmp_path / 'trace.jsonl'
    store = TraceStore(path)
    store.append(TraceEvent(task_id='t', family='math', step=0, action='cheap_answer', state_hash=state_hash({'x':1}), observation='5'))
    assert len(store.read()) == 1
    row = ReplayTraceStore(path).next()
    assert row['task_id'] == 't'
    assert row['observation'] == '5'
