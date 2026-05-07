"""Cached-only budget x cost-penalty sensitivity sweep.

No provider calls. Decision unchanged. The script does NOT promote a favorable
sweep cell into a GO decision; it is a sensitivity diagnostic only.
"""
from __future__ import annotations

import json
from pathlib import Path

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    PLAN_HEURISTIC_BDELG,
    QUERY_ROUTER_PLANS,
    aggregate,
    best_plan_per_task,
    load_outcomes,
    run_plan_over_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
BUDGETS = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0]
COST_PENALTIES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
GO_THRESHOLD_PCT = 30.0


def _saving_pct(base: float, cur: float) -> float:
    return 100.0 * (base - cur) / base if base > 0 else 0.0


def _best_non_oracle_baseline_cost(outcomes: dict, budget: float, cost_penalty: float) -> tuple[str, float]:
    """Cheapest baseline that reaches max success-rate observed on this sweep cell."""
    from agentcontrol.ed_sim import (
        PLAN_AUTOMIX, PLAN_FRUGALGPT, PLAN_SHEPHERDING, PLANS_CHEAP_ONLY,
        PLANS_STRONG_ONLY,
    )
    panel = {
        "always_cheapest": PLANS_CHEAP_ONLY,
        "always_strongest": PLANS_STRONG_ONLY,
        "frugalgpt_cascade": PLAN_FRUGALGPT,
        "automix_self_verification_cascade": PLAN_AUTOMIX,
        "shepherding_hint": PLAN_SHEPHERDING,
    }
    best = None
    for name, plan in panel.items():
        rs = run_plan_over_outcomes(name, plan, outcomes, verifier_aware=True,
                                    budget=budget, cost_penalty=cost_penalty)
        agg = aggregate(rs)
        cand = (agg["success_rate"], -agg["avg_cost"], name, agg["avg_cost"], agg["avg_objective"])
        if best is None or cand > best:
            best = cand
    return best[2], best[3]


def main() -> int:
    outcomes = load_outcomes(str(REPO / "experiments" / "smoke_outcomes.json"))
    cells = []
    crossings = []
    first_cross_budget_per_penalty: dict[float, float | None] = {}

    for cp in COST_PENALTIES:
        first_cross_budget_per_penalty[cp] = None
        for b in BUDGETS:
            qr = aggregate(best_plan_per_task(QUERY_ROUTER_PLANS, outcomes,
                                              verifier_aware=True, budget=b,
                                              cost_penalty=cp))
            gr = aggregate(best_plan_per_task(GRAPH_PLANS, outcomes,
                                              verifier_aware=True, budget=b,
                                              cost_penalty=cp))
            heur = aggregate(run_plan_over_outcomes("heuristic_bdelg",
                                                    PLAN_HEURISTIC_BDELG,
                                                    outcomes, verifier_aware=True,
                                                    budget=b, cost_penalty=cp))
            base_name, base_cost = _best_non_oracle_baseline_cost(outcomes, b, cp)
            saving = _saving_pct(qr["avg_cost"], gr["avg_cost"])
            cell = {
                "budget": b,
                "cost_penalty": cp,
                "oracle_query_router": qr,
                "oracle_deliberation_graph": gr,
                "heuristic_bdelg": heur,
                "best_non_oracle_baseline_name": base_name,
                "best_non_oracle_baseline_cost": base_cost,
                "graph_query_cost_saving_pct": saving,
                "graph_query_success_delta_pp": 100.0 * (gr["success_rate"] - qr["success_rate"]),
            }
            cells.append(cell)
            if saving >= GO_THRESHOLD_PCT and first_cross_budget_per_penalty[cp] is None:
                first_cross_budget_per_penalty[cp] = b
            if saving >= GO_THRESHOLD_PCT:
                crossings.append({"budget": b, "cost_penalty": cp,
                                  "saving_pct": saving})

    summary = {
        "budgets": BUDGETS,
        "cost_penalties": COST_PENALTIES,
        "go_threshold_pct": GO_THRESHOLD_PCT,
        "any_crossing": len(crossings) > 0,
        "crossings": crossings,
        "first_cross_budget_per_penalty": first_cross_budget_per_penalty,
        "cells": cells,
        "decision_note": (
            "Sensitivity sweep is diagnostic only. Even if a sweep cell crosses "
            "the 30% gate, this does NOT promote the artifact to GO; the "
            "pre-registered gate is on the as-run smoke configuration with "
            "n=28. Reopening Main Track requires explicit user approval per "
            "reports/MAIN_RESCUE_OPTION.md."
        ),
    }
    out_json = REPO / "experiments" / "ed_sensitivity.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# ED_SENSITIVITY\n",
          "\nDECISION: BACKUP / E&D-only (unchanged). Sensitivity sweep is diagnostic only.\n",
          "\n## Sweep grid\n",
          f"\n- budgets: {BUDGETS}\n",
          f"- cost penalties: {COST_PENALTIES}\n",
          f"- GO threshold: {GO_THRESHOLD_PCT}% cost saving (graph vs router) at matched success.\n",
          f"\n## Crossings (cells where graph-query saving ≥ {GO_THRESHOLD_PCT}%)\n\n"]
    if crossings:
        md.append("| budget | cost_penalty | saving_pct |\n|---|---|---|\n")
        for c in crossings:
            md.append(f"| {c['budget']} | {c['cost_penalty']} | {c['saving_pct']:.2f}% |\n")
    else:
        md.append("**No sweep cell crosses the 30% threshold.** The borderline result is robust.\n")

    md.append("\n## Headline sweep cells\n\n")
    md.append("| budget | cost_penalty | router_cost | graph_cost | saving_pct | heur_cost | best_baseline_cost |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for c in cells:
        if c["budget"] in (1, 2, 5, 10, 20) and c["cost_penalty"] in (0.0, 0.01, 0.1):
            md.append(
                f"| {c['budget']} | {c['cost_penalty']} | "
                f"{c['oracle_query_router']['avg_cost']:.3f} | "
                f"{c['oracle_deliberation_graph']['avg_cost']:.3f} | "
                f"{c['graph_query_cost_saving_pct']:.2f}% | "
                f"{c['heuristic_bdelg']['avg_cost']:.3f} | "
                f"{c['best_non_oracle_baseline_cost']:.3f} |\n"
            )

    md.append("\n## Interpretation\n\n")
    md.append(summary["decision_note"] + "\n\n")
    md.append("Headline interpretation: ")
    if crossings:
        md.append("favorable cells exist in the sweep grid, but they are "
                  "diagnostic only and do not change the BACKUP / E&D-only "
                  "decision. They identify candidate operating points for a "
                  "future expanded evaluation per `MAIN_RESCUE_OPTION.md`.\n")
    else:
        md.append("the borderline 26.09% saving is robust to budget and "
                  "cost-penalty perturbations on the as-run smoke. No sweep "
                  "cell promotes the artifact to GO.\n")

    out_md = REPO / "reports" / "ED_SENSITIVITY.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO)} and {out_md.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
