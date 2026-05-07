"""Tests for cached-only ablations: verifier toggle, action-set masking,
no provider import."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    PLAN_HEURISTIC_BDELG,
    aggregate,
    best_plan_per_task,
    evaluate_plan,
    load_outcomes,
    run_plan_over_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
OUTCOMES = REPO / "experiments" / "smoke_outcomes.json"


@pytest.fixture(scope="module")
def outcomes() -> dict:
    return json.loads(OUTCOMES.read_text(encoding="utf-8"))


def test_no_verifier_costs_at_least_as_much_as_verifier_aware(outcomes):
    va = aggregate(run_plan_over_outcomes("h_va", PLAN_HEURISTIC_BDELG, outcomes,
                                          verifier_aware=True, budget=20.0,
                                          cost_penalty=0.01))
    nv = aggregate(run_plan_over_outcomes("h_nv", PLAN_HEURISTIC_BDELG, outcomes,
                                          verifier_aware=False, budget=20.0,
                                          cost_penalty=0.01))
    # No-verifier executes more actions, so cost cannot be lower.
    assert nv["avg_cost"] >= va["avg_cost"] - 1e-9


def test_action_set_masking_drops_strong_hint(outcomes):
    no_hint = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                                 budget=20.0, cost_penalty=0.01,
                                 allowed_actions={"cheap_answer", "cheap_repair",
                                                  "strong_answer"})
    for r in no_hint:
        assert "strong_hint" not in r["actions_run"]
        assert "cheap_repair_after_hint" not in r["actions_run"]


def test_verifier_short_circuits_after_success(outcomes):
    # Pick a task where cheap_answer succeeds (m001).
    r = evaluate_plan("m001", "automix",
                      ["cheap_answer", "cheap_repair_if_needed",
                       "strong_answer_if_needed"],
                      outcomes, verifier_aware=True, budget=20.0,
                      cost_penalty=0.01)
    assert r["actions_run"] == ["cheap_answer"]
    assert r["success"] is True


def test_no_verifier_executes_full_plan_when_no_truncation(outcomes):
    r = evaluate_plan("m001", "automix",
                      ["cheap_answer", "cheap_repair_if_needed",
                       "strong_answer_if_needed"],
                      outcomes, verifier_aware=False, budget=20.0,
                      cost_penalty=0.01)
    assert "cheap_repair" in r["actions_run"]
    assert "strong_answer" in r["actions_run"]


def test_simulator_does_not_import_provider_layer():
    import agentcontrol.ed_sim as sim
    forbidden = ("CachedProvider", "DummyProvider")
    for sym in forbidden:
        assert not hasattr(sim, sym)
