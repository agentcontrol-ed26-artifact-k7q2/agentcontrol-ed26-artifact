"""Phase 6b: budget x cost-penalty sensitivity sweep on interactive outcomes."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "src"))

from agentcontrol_main_rescue.interactive_oracle import (  # noqa: E402
    GRAPH_PLANS, QUERY_ROUTER_PLANS, aggregate, best_plan, saving_pct,
)

PROVS = ("deepseek", "together")
GO = 30.0


def _perturb_strong_cost(outcomes: dict, mult: float) -> dict:
    out = deepcopy(outcomes)
    for tid in out:
        for action in out[tid]:
            if action.startswith("strong_"):
                out[tid][action]["cost"] = out[tid][action]["cost"] * mult
    return out


def main() -> int:
    out_all = {}
    for prov in PROVS:
        path = HERE / "experiments" / f"local_interactive_outcomes_{prov}.json"
        if not path.exists():
            continue
        outcomes = json.loads(path.read_text(encoding="utf-8"))
        cells = []
        crossings = []
        for b in (5.0, 10.0, 15.0, 25.0):
            for cp in (0.0, 0.01, 0.1):
                for sm in (0.5, 1.0, 2.0):
                    oc = _perturb_strong_cost(outcomes, sm)
                    qr = aggregate([best_plan(QUERY_ROUTER_PLANS, oc, tid, budget=b, cost_penalty=cp)
                                    for tid in oc])
                    gr = aggregate([best_plan(GRAPH_PLANS, oc, tid, budget=b, cost_penalty=cp)
                                    for tid in oc])
                    saving = saving_pct(qr["avg_cost"], gr["avg_cost"])
                    cell = {"budget": b, "cost_penalty": cp, "strong_mult": sm,
                            "saving_pct": saving,
                            "router_succ": qr["success_rate"],
                            "graph_succ": gr["success_rate"]}
                    cells.append(cell)
                    if saving >= GO:
                        crossings.append(cell)
        out_all[prov] = {"cells": cells, "crossings": crossings,
                         "any_crossing": bool(crossings)}

    (HERE / "experiments" / "local_interactive_sensitivity.json").write_text(
        json.dumps(out_all, indent=2, default=str), encoding="utf-8")
    md = ["# LOCAL_INTERACTIVE_SENSITIVITY\n",
          f"\nbudget × cost_penalty × strong_cost_multiplier sweep. GO threshold {GO}%.\n\n"]
    for prov, s in out_all.items():
        md.append(f"## {prov}\n\n")
        md.append(f"any cell crosses {GO}%? **{s['any_crossing']}**\n")
        if s["crossings"]:
            md.append("\n| budget | cost_penalty | strong_mult | saving_pct |\n|---|---|---|---|\n")
            for c in s["crossings"][:20]:
                md.append(f"| {c['budget']} | {c['cost_penalty']} | {c['strong_mult']} | {c['saving_pct']:.2f}% |\n")
            if len(s["crossings"]) > 20:
                md.append(f"\n*… {len(s['crossings']) - 20} more crossings.*\n")
        md.append("\n")
    (HERE / "reports" / "LOCAL_INTERACTIVE_SENSITIVITY.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote experiments/local_interactive_sensitivity.json and reports/LOCAL_INTERACTIVE_SENSITIVITY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
