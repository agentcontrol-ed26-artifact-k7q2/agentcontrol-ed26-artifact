"""Cached-only oracle sanity checks.

Verifies invariants on the existing oracle DP and replay path. No provider
calls; no dataset changes. Reports pass/fail counts.
"""
from __future__ import annotations

import json
from pathlib import Path

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    QUERY_ROUTER_PLANS,
    best_plan_per_task,
    evaluate_plan,
    load_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
BUDGET = 20.0
COST_PENALTY = 0.01


def main() -> int:
    outcomes = load_outcomes(str(REPO / "experiments" / "smoke_outcomes.json"))
    checks: list[dict] = []

    qr_results = {r["task_id"]: r for r in best_plan_per_task(
        QUERY_ROUTER_PLANS, outcomes, verifier_aware=True,
        budget=BUDGET, cost_penalty=COST_PENALTY)}
    gr_results = {r["task_id"]: r for r in best_plan_per_task(
        GRAPH_PLANS, outcomes, verifier_aware=True,
        budget=BUDGET, cost_penalty=COST_PENALTY)}

    # Check 1: graph oracle objective >= router oracle objective per task.
    objections = []
    for tid in qr_results:
        if gr_results[tid]["objective"] + 1e-9 < qr_results[tid]["objective"]:
            objections.append({
                "task_id": tid,
                "router_objective": qr_results[tid]["objective"],
                "graph_objective": gr_results[tid]["objective"],
            })
    checks.append({
        "name": "graph_objective_ge_router_objective_per_task",
        "status": "PASS" if not objections else "FAIL",
        "violations": objections,
        "note": "Graph plan set is a superset of router plan set, so the graph "
                "oracle should never lose to the router oracle on objective.",
    })

    # Check 2: graph plan set is a superset of router plan set.
    qr_keys, gr_keys = set(QUERY_ROUTER_PLANS), set(GRAPH_PLANS)
    checks.append({
        "name": "graph_plan_set_is_superset_of_router_plan_set",
        "status": "PASS" if qr_keys.issubset(gr_keys) else "FAIL",
        "router_plans": sorted(qr_keys),
        "graph_plans": sorted(gr_keys),
    })

    # Check 3: budget respected by every executed plan (router + graph).
    overspend = []
    for d in (qr_results, gr_results):
        for tid, r in d.items():
            if r["cost"] > BUDGET + 1e-9:
                overspend.append({"task_id": tid, "plan_name": r["plan_name"],
                                  "cost": r["cost"]})
    checks.append({
        "name": "budget_respected",
        "budget": BUDGET,
        "status": "PASS" if not overspend else "FAIL",
        "violations": overspend,
    })

    # Check 4: conditional `_if_needed` actions are skipped after success.
    # We re-run a known plan deliberately: cheap_answer succeeds on m001 →
    # the _if_needed steps after must NOT appear in actions_run.
    skip_violations = []
    for tid, oc in outcomes.items():
        if oc.get("cheap_answer", {}).get("success"):
            r = evaluate_plan(tid, "automix_va",
                              ["cheap_answer", "cheap_repair_if_needed",
                               "strong_answer_if_needed"],
                              outcomes, verifier_aware=True,
                              budget=BUDGET, cost_penalty=COST_PENALTY)
            if any(a in r["actions_run"] for a in ("cheap_repair", "strong_answer")):
                skip_violations.append({"task_id": tid,
                                        "actions_run": r["actions_run"]})
    checks.append({
        "name": "conditional_actions_skipped_after_success",
        "status": "PASS" if not skip_violations else "FAIL",
        "violations": skip_violations,
        "note": "When verifier-aware and an answer action has succeeded, "
                "subsequent `_if_needed` actions must not be executed.",
    })

    # Check 5: missing outcomes handled safely (synthetic absent action).
    safety = []
    for tid in list(outcomes)[:1]:
        r = evaluate_plan(tid, "missing_action_test",
                          ["nonexistent_action_xyz", "cheap_answer"],
                          outcomes, verifier_aware=True, budget=BUDGET,
                          cost_penalty=COST_PENALTY)
        if r["actions_run"] != ["cheap_answer"]:
            safety.append({"task_id": tid, "actions_run": r["actions_run"]})
    checks.append({
        "name": "missing_outcomes_handled_safely",
        "status": "PASS" if not safety else "FAIL",
        "violations": safety,
    })

    # Check 6: no real provider calls required (replay path).
    # Simulator imports nothing from providers and only reads cached JSON.
    import importlib

    sim_mod = importlib.import_module("agentcontrol.ed_sim")
    forbidden = []
    for sym in ("CachedProvider", "DummyProvider"):
        if hasattr(sim_mod, sym):
            forbidden.append(sym)
    checks.append({
        "name": "simulator_does_not_import_provider_layer",
        "status": "PASS" if not forbidden else "FAIL",
        "leaked_symbols": forbidden,
    })

    # Check 7: no policy spends above max possible budget tier.
    max_observed_cost = max(r["cost"] for r in qr_results.values())
    max_observed_cost = max(max_observed_cost, max(r["cost"] for r in gr_results.values()))
    checks.append({
        "name": "max_observed_cost_within_budget",
        "max_observed_cost": max_observed_cost,
        "budget": BUDGET,
        "status": "PASS" if max_observed_cost <= BUDGET + 1e-9 else "FAIL",
    })

    # Check 8: replay determinism — same inputs produce same outputs.
    qr2 = {r["task_id"]: r for r in best_plan_per_task(
        QUERY_ROUTER_PLANS, outcomes, verifier_aware=True,
        budget=BUDGET, cost_penalty=COST_PENALTY)}
    deterministic = all(qr2[t]["cost"] == qr_results[t]["cost"]
                        and qr2[t]["plan_name"] == qr_results[t]["plan_name"]
                        for t in qr_results)
    checks.append({
        "name": "replay_determinism",
        "status": "PASS" if deterministic else "FAIL",
    })

    summary = {
        "budget": BUDGET,
        "cost_penalty": COST_PENALTY,
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["status"] == "PASS"),
        "n_fail": sum(1 for c in checks if c["status"] == "FAIL"),
        "checks": checks,
    }
    out_json = REPO / "experiments" / "ed_oracle_sanity_checks.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# ED_ORACLE_SANITY_CHECKS\n",
          "\nDECISION: BACKUP / E&D-only (unchanged). Sanity checks on cached oracle.\n\n",
          f"- {summary['n_pass']}/{summary['n_checks']} checks pass\n",
          "\n## Checks\n\n",
          "| check | status | notes |\n|---|---|---|\n"]
    for c in checks:
        notes = c.get("note", "")
        if c.get("violations"):
            notes = (notes + " — VIOLATIONS: " + str(len(c["violations"]))).strip()
        md.append(f"| {c['name']} | {c['status']} | {notes} |\n")

    out_md = REPO / "reports" / "ED_ORACLE_SANITY_CHECKS.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO)} and {out_md.relative_to(REPO)}")
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
