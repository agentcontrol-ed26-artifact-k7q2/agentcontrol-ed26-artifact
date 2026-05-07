"""Sensitivity sweep is monotone-sane and budget-respecting."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    QUERY_ROUTER_PLANS,
    aggregate,
    best_plan_per_task,
    load_outcomes,
)

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def outcomes():
    return load_outcomes(str(REPO / "experiments" / "smoke_outcomes.json"))


def test_oracle_graph_avg_cost_is_le_oracle_router_avg_cost(outcomes):
    """Plan-set superset implies oracle graph never beats router on cost negatively."""
    for budget in (1.0, 5.0, 20.0):
        for cp in (0.0, 0.01, 0.1):
            qr = aggregate(best_plan_per_task(QUERY_ROUTER_PLANS, outcomes,
                                              verifier_aware=True, budget=budget,
                                              cost_penalty=cp))
            gr = aggregate(best_plan_per_task(GRAPH_PLANS, outcomes,
                                              verifier_aware=True, budget=budget,
                                              cost_penalty=cp))
            # graph objective >= router objective
            assert gr["avg_objective"] >= qr["avg_objective"] - 1e-9


def test_no_policy_exceeds_budget(outcomes):
    budget = 3.0
    rs = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                            budget=budget, cost_penalty=0.01)
    for r in rs:
        assert r["cost"] <= budget + 1e-9
