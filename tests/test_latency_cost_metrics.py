from agentcontrol.metrics import mean_cost, mean_latency_ms, pareto_auc, cost_at_target_success


def test_latency_cost_metrics():
    rows = [{'cost':1.0,'latency_ms':10}, {'cost':3.0,'latency_ms':30}]
    assert mean_cost(rows) == 2.0
    assert mean_latency_ms(rows) == 20.0


def test_pareto_auc_and_cost_at_target():
    points = [(1.0,0.5), (2.0,0.8), (3.0,0.7)]
    assert pareto_auc(points) > 0
    assert cost_at_target_success(points, 0.75) == 2.0
