from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .policies import BASELINE_PLANS, heuristic_bdelg_plan


@dataclass
class PlanResult:
    task_id: str
    plan_name: str
    success: bool
    cost: float
    latency_ms: int
    unsupported_risk: float
    objective: float
    actions_run: list[str]


QUERY_ROUTER_PLANS = {
    'query_cheap_only': BASELINE_PLANS['always_cheapest'],
    'query_strong_only': BASELINE_PLANS['always_strongest'],
    'query_cascade': BASELINE_PLANS['frugalgpt_cascade'],
    'query_automix': BASELINE_PLANS['automix_self_verification_cascade'],
}
GRAPH_PLANS = {**QUERY_ROUTER_PLANS, 'graph_shepherding_hint': BASELINE_PLANS['shepherding_hint'], 'graph_bdelg': heuristic_bdelg_plan()}


def _base_action(action: str) -> tuple[str, bool]:
    suffix = '_if_needed'
    return (action[:-len(suffix)], True) if action.endswith(suffix) else (action, False)


def evaluate_plan(task_id: str, plan_name: str, actions: list[str], outcomes: dict[str, dict[str, dict[str, Any]]], budget: float = 20.0, cost_penalty: float = 0.0, latency_penalty: float = 0.0, risk_penalty: float = 0.0) -> PlanResult:
    task_outcomes = outcomes[task_id]
    success = False
    total_cost = 0.0
    total_latency = 0
    unsupported_risk = 0.0
    actions_run: list[str] = []
    for raw_action in actions:
        action, conditional = _base_action(raw_action)
        if conditional and success:
            continue
        obs = task_outcomes.get(action, {'success': False, 'cost': 999.0, 'latency_ms': 0, 'unsupported_risk': 1.0})
        next_cost = float(obs.get('cost', 0.0))
        if total_cost + next_cost > budget:
            break
        total_cost += next_cost
        total_latency += int(obs.get('latency_ms', 0))
        unsupported_risk = max(unsupported_risk, float(obs.get('unsupported_risk', 0.0)))
        success = success or bool(obs.get('success', False))
        actions_run.append(action)
    objective = (1.0 if success else 0.0) - cost_penalty * total_cost - latency_penalty * total_latency - risk_penalty * unsupported_risk
    return PlanResult(task_id, plan_name, success, total_cost, total_latency, unsupported_risk, objective, actions_run)


def _best_for_task(task_id: str, plans: dict[str, list[str]], outcomes: dict[str, Any], budget: float, cost_penalty: float) -> PlanResult:
    results = [evaluate_plan(task_id, name, actions, outcomes, budget=budget, cost_penalty=cost_penalty) for name, actions in plans.items()]
    return max(results, key=lambda r: (r.objective, r.success, -r.cost))


def summarize_results(results: list[PlanResult]) -> dict[str, Any]:
    n = max(1, len(results))
    return {'n': len(results), 'success_rate': sum(r.success for r in results) / n, 'avg_cost': sum(r.cost for r in results) / n, 'avg_latency_ms': sum(r.latency_ms for r in results) / n, 'avg_unsupported_risk': sum(r.unsupported_risk for r in results) / n, 'avg_objective': sum(r.objective for r in results) / n, 'task_results': [r.__dict__ for r in results]}


def oracle_query_router(outcomes: dict[str, Any], budget: float = 20.0, cost_penalty: float = 0.0) -> dict[str, Any]:
    out = summarize_results([_best_for_task(tid, QUERY_ROUTER_PLANS, outcomes, budget, cost_penalty) for tid in sorted(outcomes)])
    out['oracle_type'] = 'query_router'
    return out


def oracle_deliberation_graph(outcomes: dict[str, Any], budget: float = 20.0, cost_penalty: float = 0.0) -> dict[str, Any]:
    out = summarize_results([_best_for_task(tid, GRAPH_PLANS, outcomes, budget, cost_penalty) for tid in sorted(outcomes)])
    out['oracle_type'] = 'deliberation_graph'
    return out


def oracle_gap_summary(outcomes: dict[str, Any], budget: float = 20.0, cost_penalty: float = 0.0) -> dict[str, Any]:
    q = oracle_query_router(outcomes, budget=budget, cost_penalty=cost_penalty)
    g = oracle_deliberation_graph(outcomes, budget=budget, cost_penalty=cost_penalty)
    return {'budget': budget, 'cost_penalty': cost_penalty, 'query_router': q, 'deliberation_graph': g, 'success_delta_pp': 100.0 * (g['success_rate'] - q['success_rate']), 'avg_cost_delta': g['avg_cost'] - q['avg_cost'], 'avg_objective_delta': g['avg_objective'] - q['avg_objective'], 'cost_saving_pct_at_observed': 100.0 * (q['avg_cost'] - g['avg_cost']) / q['avg_cost'] if q['avg_cost'] > 0 else 0.0}
