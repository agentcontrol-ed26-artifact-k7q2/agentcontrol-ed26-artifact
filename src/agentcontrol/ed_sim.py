"""Cached-only simulator helpers for E&D ablations / sensitivity / bootstrap.

These utilities never call providers. They consume cached outcomes
(`experiments/smoke_outcomes.json` schema) and replay deterministic plans.
"""
from __future__ import annotations

from typing import Any, Iterable

ANSWER_ACTIONS = {
    "cheap_answer",
    "cheap_repair",
    "cheap_repair_after_hint",
    "strong_answer",
}
HINT_ACTIONS = {"strong_hint"}
IF_NEEDED = "_if_needed"


def family_of(task_id: str) -> str:
    if task_id.startswith("m"):
        return "math"
    if task_id.startswith("code"):
        return "code"
    if task_id.startswith("e") and not task_id.startswith("code"):
        return "evidence"
    return "other"


def _base_action(raw: str) -> tuple[str, bool]:
    if raw.endswith(IF_NEEDED):
        return raw[: -len(IF_NEEDED)], True
    return raw, False


def evaluate_plan(
    task_id: str,
    plan_name: str,
    actions: list[str],
    outcomes: dict[str, dict[str, dict[str, Any]]],
    *,
    verifier_aware: bool = True,
    budget: float = 20.0,
    cost_penalty: float = 0.0,
    allowed_actions: set[str] | None = None,
) -> dict[str, Any]:
    """Replay a plan against cached outcomes for one task.

    - verifier_aware=True : skip ``_if_needed`` actions after a prior success.
    - verifier_aware=False: ignore ``_if_needed``; run every action in the plan
      until budget exhausts.
    - allowed_actions: optional whitelist; actions not in the set are skipped.
    """
    task_outcomes = outcomes[task_id]
    success = False
    total_cost = 0.0
    total_lat = 0
    risk = 0.0
    actions_run: list[str] = []
    for raw in actions:
        action, conditional = _base_action(raw)
        if allowed_actions is not None and action not in allowed_actions:
            continue
        if verifier_aware and conditional and success:
            continue
        obs = task_outcomes.get(action)
        if obs is None:
            continue
        next_cost = float(obs.get("cost", 0.0))
        if total_cost + next_cost > budget:
            break
        total_cost += next_cost
        total_lat += int(obs.get("latency_ms", 0))
        risk = max(risk, float(obs.get("unsupported_risk", 0.0)))
        if bool(obs.get("success", False)):
            success = True
        actions_run.append(action)
    objective = (1.0 if success else 0.0) - cost_penalty * total_cost
    return {
        "task_id": task_id,
        "plan_name": plan_name,
        "success": success,
        "cost": total_cost,
        "latency_ms": total_lat,
        "unsupported_risk": risk,
        "objective": objective,
        "actions_run": actions_run,
        "verifier_aware": verifier_aware,
    }


def aggregate(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rs = list(results)
    n = max(1, len(rs))
    return {
        "n": len(rs),
        "success_rate": sum(r["success"] for r in rs) / n,
        "avg_cost": sum(r["cost"] for r in rs) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in rs) / n,
        "avg_unsupported_risk": sum(r["unsupported_risk"] for r in rs) / n,
        "avg_objective": sum(r["objective"] for r in rs) / n,
    }


def per_family(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by.setdefault(family_of(r["task_id"]), []).append(r)
    return {f: aggregate(rs) for f, rs in by.items()}


def run_plan_over_outcomes(
    plan_name: str,
    actions: list[str],
    outcomes: dict[str, Any],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return [
        evaluate_plan(tid, plan_name, actions, outcomes, **kwargs)
        for tid in sorted(outcomes)
    ]


def best_plan_per_task(
    plans: dict[str, list[str]],
    outcomes: dict[str, Any],
    *,
    verifier_aware: bool = True,
    budget: float = 20.0,
    cost_penalty: float = 0.0,
    allowed_actions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Oracle: pick best plan per task by (objective, success, -cost)."""
    chosen: list[dict[str, Any]] = []
    for tid in sorted(outcomes):
        candidates = [
            evaluate_plan(
                tid,
                name,
                actions,
                outcomes,
                verifier_aware=verifier_aware,
                budget=budget,
                cost_penalty=cost_penalty,
                allowed_actions=allowed_actions,
            )
            for name, actions in plans.items()
        ]
        chosen.append(max(candidates, key=lambda r: (r["objective"], r["success"], -r["cost"])))
    return chosen


# Standard plan catalogue used across analyses.
PLANS_CHEAP_ONLY = ["cheap_answer"]
PLANS_STRONG_ONLY = ["strong_answer"]
PLAN_FRUGALGPT = ["cheap_answer", "strong_answer_if_needed"]
PLAN_AUTOMIX = [
    "cheap_answer",
    "cheap_repair_if_needed",
    "strong_answer_if_needed",
]
PLAN_SHEPHERDING = [
    "cheap_answer",
    "strong_hint_if_needed",
    "cheap_repair_after_hint_if_needed",
]
PLAN_HEURISTIC_BDELG = [
    "cheap_answer",
    "cheap_repair_if_needed",
    "strong_hint_if_needed",
    "cheap_repair_after_hint_if_needed",
    "strong_answer_if_needed",
]

QUERY_ROUTER_PLANS = {
    "query_cheap_only": PLANS_CHEAP_ONLY,
    "query_strong_only": PLANS_STRONG_ONLY,
    "query_cascade": PLAN_FRUGALGPT,
    "query_automix": PLAN_AUTOMIX,
}

GRAPH_PLANS = {
    **QUERY_ROUTER_PLANS,
    "graph_shepherding_hint": PLAN_SHEPHERDING,
    "graph_bdelg": PLAN_HEURISTIC_BDELG,
}


def load_outcomes(path: str = "experiments/smoke_outcomes.json") -> dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
