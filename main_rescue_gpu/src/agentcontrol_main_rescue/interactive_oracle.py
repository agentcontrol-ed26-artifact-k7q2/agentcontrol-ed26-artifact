"""Shared oracle / plan-evaluation logic for interactive analyses."""
from __future__ import annotations

from collections import defaultdict


QUERY_ROUTER_PLANS = {
    "query_cheap_only": ["cheap_answer"],
    "query_strong_only": ["strong_answer"],
    "query_cascade": ["cheap_answer", "strong_answer_if_needed"],
    "query_automix": ["cheap_answer", "cheap_repair_if_needed", "strong_answer_if_needed"],
}

GRAPH_PLANS = {
    **QUERY_ROUTER_PLANS,
    "graph_observe_repair": ["cheap_answer", "OBS",
                              "cheap_repair_after_observation_if_needed",
                              "strong_answer_if_needed"],
    "graph_strong_hint": ["cheap_answer", "strong_hint_if_needed",
                           "cheap_repair_after_strong_partial_if_needed",
                           "strong_answer_if_needed"],
    "graph_observe_then_hint": ["cheap_answer", "OBS",
                                  "cheap_repair_after_observation_if_needed",
                                  "strong_hint_if_needed",
                                  "cheap_repair_after_strong_partial_if_needed",
                                  "strong_answer_if_needed"],
    "graph_full_bdelg": ["cheap_answer", "cheap_repair_if_needed", "OBS",
                          "cheap_repair_after_observation_if_needed",
                          "strong_hint_if_needed",
                          "cheap_repair_after_strong_partial_if_needed",
                          "strong_answer_if_needed"],
}

FAMILY_OBS = {
    "code_debug_interactive": "run_tests",
    "data_analysis_code": "run_code",
    "evidence_multihop_local": "citation_check",
    "tool_planning_deterministic": "tool_observation",
    "math_checkpoint": "checkpoint_check",
}

ANSWER_ACTIONS = {"cheap_answer", "cheap_repair", "cheap_repair_after_observation",
                  "cheap_repair_after_strong_partial", "strong_answer"}

BUDGET = 25.0
COST_PENALTY = 0.01


def family_of(tid: str) -> str:
    if tid.startswith("ic"): return "code_debug_interactive"
    if tid.startswith("id"): return "data_analysis_code"
    if tid.startswith("ie"): return "evidence_multihop_local"
    if tid.startswith("it"): return "tool_planning_deterministic"
    if tid.startswith("im"): return "math_checkpoint"
    return "other"


def resolve_plan(actions: list[str], family: str) -> list[str]:
    obs = FAMILY_OBS.get(family, "run_tests")
    return [obs if a == "OBS" else a for a in actions]


def evaluate_plan(plan_name: str, actions: list[str], outcomes: dict, task_id: str,
                  budget: float = BUDGET, cost_penalty: float = COST_PENALTY) -> dict:
    fam = family_of(task_id)
    actions_resolved = resolve_plan(actions, fam)
    out = outcomes[task_id]
    success = False
    cost = 0.0
    lat = 0
    risk = 0.0
    actions_run = []
    for a in actions_resolved:
        base = a[:-len("_if_needed")] if a.endswith("_if_needed") else a
        conditional = a.endswith("_if_needed")
        if conditional and success:
            continue
        oa = out.get(base)
        if oa is None:
            continue
        if cost + oa["cost"] > budget:
            break
        cost += oa["cost"]
        lat += oa["latency_ms"]
        risk = max(risk, oa.get("unsupported_risk", 0.0))
        if base in ANSWER_ACTIONS and oa.get("success"):
            success = True
        actions_run.append(base)
    objective = (1.0 if success else 0.0) - cost_penalty * cost
    return {"task_id": task_id, "plan_name": plan_name, "family": fam,
            "success": success, "cost": cost, "latency_ms": lat,
            "unsupported_risk": risk, "objective": objective,
            "actions_run": actions_run}


def best_plan(plans: dict, outcomes: dict, task_id: str,
              budget: float = BUDGET, cost_penalty: float = COST_PENALTY) -> dict:
    cands = [evaluate_plan(name, actions, outcomes, task_id, budget, cost_penalty)
             for name, actions in plans.items()]
    return max(cands, key=lambda r: (r["objective"], r["success"], -r["cost"]))


def aggregate(rs):
    n = max(1, len(rs))
    return {"n": len(rs),
            "success_rate": sum(r["success"] for r in rs) / n,
            "avg_cost": sum(r["cost"] for r in rs) / n,
            "avg_latency_ms": sum(r["latency_ms"] for r in rs) / n,
            "avg_unsupported_risk": sum(r["unsupported_risk"] for r in rs) / n,
            "avg_objective": sum(r["objective"] for r in rs) / n}


def per_family(rs):
    by = defaultdict(list)
    for r in rs:
        by[r["family"]].append(r)
    return {f: aggregate(v) for f, v in by.items()}


def saving_pct(b: float, c: float) -> float:
    return 100.0 * (b - c) / b if b > 0 else 0.0
