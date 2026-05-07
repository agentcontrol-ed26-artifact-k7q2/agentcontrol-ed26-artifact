"""Cached-only family-reweighting diagnostic.

Reweights the per-family aggregate costs already computed on the n=28 smoke
to ask: at what code/tool-use family weight would the oracle-graph cost saving
exceed 30%? This is a *diagnostic*, not new evidence. Decision unchanged.
"""
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
    per_family,
)

REPO = Path(__file__).resolve().parent.parent
BUDGET = 20.0
COST_PENALTY = 0.01
GO_THRESHOLD_PCT = 30.0
WEIGHT_GRID = [round(x * 0.01, 2) for x in range(0, 101, 5)]  # 0%, 5%, ..., 100%


def _saving_pct(base: float, cur: float) -> float:
    return 100.0 * (base - cur) / base if base > 0 else 0.0


def main() -> int:
    outcomes = load_outcomes(str(REPO / "experiments" / "smoke_outcomes.json"))
    qr = best_plan_per_task(QUERY_ROUTER_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    gr = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    qr_fam = per_family(qr)
    gr_fam = per_family(gr)

    sweeps = []
    first_cross_w = None
    for w_code in WEIGHT_GRID:
        w_other = (1.0 - w_code) / 2.0  # split equally between math and evidence
        weights = {"code": w_code, "math": w_other, "evidence": w_other}
        # Skip families absent from the smoke (none here, but be safe).
        weights = {k: v for k, v in weights.items() if k in qr_fam}
        # Renormalize in case a family is missing.
        wsum = sum(weights.values())
        if wsum == 0:
            continue
        weights = {k: v / wsum for k, v in weights.items()}
        router_cost = sum(weights[f] * qr_fam[f]["avg_cost"] for f in weights)
        graph_cost = sum(weights[f] * gr_fam[f]["avg_cost"] for f in weights)
        saving = _saving_pct(router_cost, graph_cost)
        sweeps.append({
            "code_weight": w_code,
            "math_weight": weights.get("math", 0.0),
            "evidence_weight": weights.get("evidence", 0.0),
            "router_avg_cost": router_cost,
            "graph_avg_cost": graph_cost,
            "graph_query_cost_saving_pct": saving,
            "crosses_30pct": saving >= GO_THRESHOLD_PCT,
        })
        if first_cross_w is None and saving >= GO_THRESHOLD_PCT:
            first_cross_w = w_code

    summary = {
        "n_total": len(outcomes),
        "per_family_n": {f: sum(1 for t in outcomes if family_of(t) == f)
                         for f in {family_of(t) for t in outcomes}},
        "per_family_router_avg_cost": {f: qr_fam[f]["avg_cost"] for f in qr_fam},
        "per_family_graph_avg_cost": {f: gr_fam[f]["avg_cost"] for f in gr_fam},
        "weight_grid": WEIGHT_GRID,
        "go_threshold_pct": GO_THRESHOLD_PCT,
        "first_code_weight_crossing_threshold": first_cross_w,
        "sweeps": sweeps,
        "interpretation_note": (
            "This is a reweighting DIAGNOSTIC over the existing per-family "
            "averages, not new evidence. Treating per-family averages as "
            "stable across mixtures is an assumption that breaks down for "
            "small per-family n (especially code n=4). Cannot reopen Main "
            "Track. See reports/MAIN_RESCUE_OPTION.md for the actual rescue "
            "path (which requires user approval and dataset expansion)."
        ),
    }

    out_json = REPO / "experiments" / "ed_family_reweighting.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# ED_FAMILY_REWEIGHTING\n",
          "\nDECISION: BACKUP / E&D-only (unchanged). Reweighting is a diagnostic only.\n\n",
          "## Per-family aggregate (from cached smoke)\n\n",
          "| family | n | router_avg_cost | graph_avg_cost | saving_pct |\n",
          "|---|---|---|---|---|\n"]
    for f in ("math", "code", "evidence"):
        if f in qr_fam:
            r = qr_fam[f]["avg_cost"]
            g = gr_fam[f]["avg_cost"]
            md.append(f"| {f} | {summary['per_family_n'].get(f, '?')} | {r:.3f} | {g:.3f} | {_saving_pct(r, g):.2f}% |\n")

    md.append("\n## Weighted aggregate as code-family weight varies\n\n")
    md.append("| code_weight | math_weight | evidence_weight | router_cost | graph_cost | saving_pct | crosses_30%? |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for s in sweeps:
        md.append(
            f"| {s['code_weight']:.2f} | {s['math_weight']:.2f} | "
            f"{s['evidence_weight']:.2f} | {s['router_avg_cost']:.3f} | "
            f"{s['graph_avg_cost']:.3f} | {s['graph_query_cost_saving_pct']:.2f}% | "
            f"{'yes' if s['crosses_30pct'] else 'no'} |\n"
        )

    md.append("\n## Headline\n\n")
    if first_cross_w is None:
        md.append("No code-weight in [0, 1] reaches the 30% saving threshold under "
                  "the current per-family averages. (Code is the only family with "
                  "headroom; if its weight is large enough, the threshold is met.)\n")
    else:
        md.append(f"At a code-family weight of **{first_cross_w:.2f}** (with the "
                  "remaining mass split equally between math and evidence), the "
                  "weighted aggregate first crosses 30% saving. **This is a "
                  "diagnostic about task-mix sensitivity, not new evidence.**\n")
    md.append("\n## Caveats\n\n")
    md.append(summary["interpretation_note"] + "\n")

    out_md = REPO / "reports" / "ED_FAMILY_REWEIGHTING.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO)} and {out_md.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
