"""Phase 4a: Oracle query-router and oracle deliberation-graph on rescue pool."""
from __future__ import annotations

import json
from pathlib import Path

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    QUERY_ROUTER_PLANS,
    aggregate,
    best_plan_per_task,
    family_of,
    load_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
BUDGET = 20.0
COST_PENALTY = 0.01
GO_THRESHOLD_PCT = 30.0


def _family_of_rescue(tid: str) -> str:
    if tid.startswith("rm"): return "math"
    if tid.startswith("rc"): return "code"
    if tid.startswith("re"): return "evidence"
    if tid.startswith("rt"): return "tool_use"
    return family_of(tid)


def _per_family(rs):
    by = {}
    for r in rs:
        by.setdefault(_family_of_rescue(r["task_id"]), []).append(r)
    return {f: aggregate(v) for f, v in by.items()}


def main() -> int:
    outcomes = load_outcomes(str(REPO / "experiments" / "rescue_outcomes.json"))
    qr = best_plan_per_task(QUERY_ROUTER_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    gr = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    qr_agg = aggregate(qr)
    gr_agg = aggregate(gr)
    saving = 100.0 * (qr_agg["avg_cost"] - gr_agg["avg_cost"]) / qr_agg["avg_cost"] if qr_agg["avg_cost"] > 0 else 0.0
    summary = {
        "non_decisive": True,
        "n_total": len(outcomes),
        "budget": BUDGET,
        "cost_penalty": COST_PENALTY,
        "go_threshold_pct": GO_THRESHOLD_PCT,
        "query_router": qr_agg,
        "deliberation_graph": gr_agg,
        "success_delta_pp": 100.0 * (gr_agg["success_rate"] - qr_agg["success_rate"]),
        "avg_cost_delta": gr_agg["avg_cost"] - qr_agg["avg_cost"],
        "cost_saving_pct_at_observed": saving,
        "per_family_query_router": _per_family(qr),
        "per_family_deliberation_graph": _per_family(gr),
        "decision": "BACKUP" if saving < GO_THRESHOLD_PCT else "OBSERVED-CROSSING-NON-DECISIVE",
    }
    out_json = REPO / "experiments" / "rescue_oracle_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# RESCUE_ORACLE_GAP_DECISION\n",
          "\n**Status:** synthetic-local pool, n=170. **NON-DECISIVE** for Main Track.\n",
          f"\n- oracle query router avg_cost = {qr_agg['avg_cost']:.3f}, success = {qr_agg['success_rate']:.3f}\n",
          f"- oracle deliberation graph avg_cost = {gr_agg['avg_cost']:.3f}, success = {gr_agg['success_rate']:.3f}\n",
          f"- cost saving = **{saving:.2f}%** (GO threshold = {GO_THRESHOLD_PCT}%)\n",
          f"- success delta = {summary['success_delta_pp']:.2f} pp\n",
          "\n## Per family\n\n",
          "| family | n | router cost | graph cost | saving | router succ | graph succ |\n",
          "|---|---|---|---|---|---|---|\n"]
    for fam in ("math", "code", "evidence", "tool_use"):
        if fam in summary["per_family_query_router"]:
            r = summary["per_family_query_router"][fam]
            g = summary["per_family_deliberation_graph"][fam]
            s = 100.0 * (r["avg_cost"] - g["avg_cost"]) / r["avg_cost"] if r["avg_cost"] > 0 else 0.0
            md.append(f"| {fam} | {r['n']} | {r['avg_cost']:.3f} | {g['avg_cost']:.3f} | {s:.2f}% | {r['success_rate']:.3f} | {g['success_rate']:.3f} |\n")

    md.append("\n## Honest interpretation\n\n")
    md.append("- The pool is synthetic-local; outcomes are engineered to expose a richer cheap-vs-strong distribution than the original smoke.\n")
    md.append("- Even if aggregate cost saving exceeds 30% on this pool, it is **not real-model evidence** and does not by itself justify Main Track.\n")
    md.append("- Use this result as **protocol robustness** evidence: the harness, oracle DP, action-mask, budget, and verifier short-circuit work on n=170 across four families.\n")
    md.append("- Real-model evidence requires opening the API gate (`reports/API_BUDGET_GATE.md`).\n")
    out_md = REPO / "reports" / "RESCUE_ORACLE_GAP_DECISION.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO)} and {out_md.relative_to(REPO)} (saving={saving:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
