from agentcontrol.oracle import oracle_gap_summary


def test_oracle_graph_beats_query_router_on_hint_case():
    outcomes = {'t1': {'cheap_answer': {'success': False, 'cost': 1.0, 'latency_ms': 1, 'unsupported_risk': 0.0}, 'cheap_repair': {'success': False, 'cost': 1.0, 'latency_ms': 1, 'unsupported_risk': 0.0}, 'strong_hint': {'success': False, 'cost': 2.0, 'latency_ms': 1, 'unsupported_risk': 0.0}, 'cheap_repair_after_hint': {'success': True, 'cost': 1.0, 'latency_ms': 1, 'unsupported_risk': 0.0}, 'strong_answer': {'success': True, 'cost': 10.0, 'latency_ms': 1, 'unsupported_risk': 0.0}}}
    summary = oracle_gap_summary(outcomes, budget=20.0, cost_penalty=0.01)
    assert summary['deliberation_graph']['success_rate'] == 1.0
    assert summary['query_router']['success_rate'] == 1.0
    assert summary['deliberation_graph']['avg_cost'] < summary['query_router']['avg_cost']
