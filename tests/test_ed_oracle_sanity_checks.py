"""Oracle sanity invariants on cached outcomes."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    QUERY_ROUTER_PLANS,
    best_plan_per_task,
    evaluate_plan,
    load_outcomes,
)

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def outcomes():
    return load_outcomes(str(REPO / "experiments" / "smoke_outcomes.json"))


def test_graph_oracle_dominates_router_per_task(outcomes):
    qr = {r["task_id"]: r for r in best_plan_per_task(QUERY_ROUTER_PLANS, outcomes,
                                                      verifier_aware=True, budget=20.0,
                                                      cost_penalty=0.01)}
    gr = {r["task_id"]: r for r in best_plan_per_task(GRAPH_PLANS, outcomes,
                                                      verifier_aware=True, budget=20.0,
                                                      cost_penalty=0.01)}
    for tid in qr:
        assert gr[tid]["objective"] >= qr[tid]["objective"] - 1e-9


def test_router_plan_set_is_subset_of_graph_plan_set():
    assert set(QUERY_ROUTER_PLANS).issubset(set(GRAPH_PLANS))


def test_missing_action_handled_safely(outcomes):
    tid = next(iter(outcomes))
    r = evaluate_plan(tid, "missing", ["nonexistent_xyz", "cheap_answer"],
                     outcomes, verifier_aware=True, budget=20.0,
                     cost_penalty=0.01)
    assert "nonexistent_xyz" not in r["actions_run"]


def test_replay_determinism(outcomes):
    a = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                           budget=20.0, cost_penalty=0.01)
    b = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                           budget=20.0, cost_penalty=0.01)
    assert [(r["task_id"], r["plan_name"], r["cost"]) for r in a] == \
           [(r["task_id"], r["plan_name"], r["cost"]) for r in b]


def test_conditional_skipped_after_success(outcomes):
    if not outcomes.get("m001", {}).get("cheap_answer", {}).get("success"):
        pytest.skip("m001 cheap_answer not successful in this fixture")
    r = evaluate_plan("m001", "automix",
                      ["cheap_answer", "cheap_repair_if_needed",
                       "strong_answer_if_needed"],
                     outcomes, verifier_aware=True, budget=20.0,
                     cost_penalty=0.01)
    assert r["actions_run"] == ["cheap_answer"]
