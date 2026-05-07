"""Phase 5b: budget x cost-penalty x model-price sensitivity sweep on hard regime."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    QUERY_ROUTER_PLANS,
    aggregate,
    best_plan_per_task,
)

REPO = Path(__file__).resolve().parent.parent
GO = 30.0
PROVIDERS = {
    "deepseek": "experiments/hard_regime_outcomes_deepseek.json",
    "together": "experiments/hard_regime_outcomes_together.json",
}


def _saving_pct(b, c):
    return 100.0 * (b - c) / b if b > 0 else 0.0


def _perturb_outcomes(outcomes, strong_cost_mult=1.0):
    """Return a copy of outcomes with strong_answer / strong_hint costs scaled."""
    out = deepcopy(outcomes)
    for tid in out:
        for action in out[tid]:
            if action.startswith("strong_"):
                out[tid][action]["cost"] = out[tid][action]["cost"] * strong_cost_mult
    return out


def _sweep(outcomes):
    cells = []
    crossings = []
    for b in (3.0, 5.0, 8.0, 12.0, 20.0):
        for cp in (0.0, 0.01, 0.1):
            for sp in (1.0, 0.5, 2.0):
                oc = _perturb_outcomes(outcomes, strong_cost_mult=sp)
                qr = aggregate(best_plan_per_task(QUERY_ROUTER_PLANS, oc,
                                                  verifier_aware=True, budget=b,
                                                  cost_penalty=cp))
                gr = aggregate(best_plan_per_task(GRAPH_PLANS, oc,
                                                  verifier_aware=True, budget=b,
                                                  cost_penalty=cp))
                saving = _saving_pct(qr["avg_cost"], gr["avg_cost"])
                cells.append({"budget": b, "cost_penalty": cp,
                              "strong_cost_mult": sp,
                              "saving_pct": saving,
                              "router_succ": qr["success_rate"],
                              "graph_succ": gr["success_rate"]})
                if saving >= GO:
                    crossings.append(cells[-1])
    return {"cells": cells, "crossings": crossings, "any_crossing": bool(crossings)}


def main() -> int:
    out = {}
    for p, path in PROVIDERS.items():
        f = REPO / path
        if not f.exists():
            continue
        outcomes = json.loads(f.read_text(encoding="utf-8"))
        out[p] = _sweep(outcomes)
    (REPO / "experiments" / "hard_regime_sensitivity.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    md = ["# HARD_REGIME_SENSITIVITY\n",
          f"\nbudget × cost_penalty × strong_cost_multiplier sweep. GO threshold {GO}%.\n\n",
          "## Per-provider crossings\n\n"]
    for p, s in out.items():
        md.append(f"### {p}\n\n")
        md.append(f"any sweep cell crosses {GO}%? **{s['any_crossing']}**\n")
        if s["crossings"]:
            md.append("\n| budget | cost_penalty | strong_cost_mult | saving_pct |\n|---|---|---|---|\n")
            for c in s["crossings"]:
                md.append(f"| {c['budget']} | {c['cost_penalty']} | {c['strong_cost_mult']} | {c['saving_pct']:.2f}% |\n")
        else:
            md.append("\n*No sweep cell crosses 30%.*\n")
    (REPO / "reports" / "HARD_REGIME_SENSITIVITY.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote experiments/hard_regime_sensitivity.json and reports/HARD_REGIME_SENSITIVITY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
