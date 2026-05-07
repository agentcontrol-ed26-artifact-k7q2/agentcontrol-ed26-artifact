"""Generate paper/table_*.md from existing experiment JSONs.

Read-only over experiments/. No provider calls. No new evidence is created.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
EXP = REPO / "experiments"


def _load(name: str) -> dict | None:
    p = EXP / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |\n"


def smoke_table() -> str:
    agg = _load("aggregate_summary.json") or {}
    base = (agg.get("experiments") or {}).get("baselines") or {}
    heur = (agg.get("experiments") or {}).get("heuristic_bdelg") or {}
    og = (agg.get("experiments") or {}).get("oracle_gap") or {}
    md = ["# Table — Smoke results (n=28: math=20, code=4, evidence=4)\n\n",
          "| policy | success | avg_cost | avg_objective | avg_unsupported_risk |\n",
          "|---|---|---|---|---|\n"]
    for name, d in sorted(base.items()):
        md.append(_row(name, f"{d['success_rate']:.3f}", f"{d['avg_cost']:.3f}",
                       f"{d['avg_objective']:.4f}", f"{d['avg_unsupported_risk']:.3f}"))
    if heur:
        md.append(_row("**heuristic_bdelg (reference)**",
                       f"**{heur['success_rate']:.3f}**", f"**{heur['avg_cost']:.3f}**",
                       f"**{heur['avg_objective']:.4f}**", f"**{heur['avg_unsupported_risk']:.3f}**"))
    if og:
        qr = og.get("query_router", {})
        gr = og.get("deliberation_graph", {})
        md.append(_row("oracle_query_router", f"{qr['success_rate']:.3f}",
                       f"{qr['avg_cost']:.3f}", f"{qr['avg_objective']:.4f}",
                       f"{qr['avg_unsupported_risk']:.3f}"))
        md.append(_row("**oracle_deliberation_graph**", f"**{gr['success_rate']:.3f}**",
                       f"**{gr['avg_cost']:.3f}**", f"**{gr['avg_objective']:.4f}**",
                       f"**{gr['avg_unsupported_risk']:.3f}**"))
    md.append("\nDecision: **BACKUP / E&D-only**.\n")
    return "".join(md)


def oracle_gap_table() -> str:
    og = _load("oracle_gap_summary.json") or {}
    md = ["# Table — Oracle deliberation graph vs oracle query router\n\n",
          "| metric | router | graph | delta |\n|---|---|---|---|\n"]
    qr = og.get("query_router", {})
    gr = og.get("deliberation_graph", {})
    md.append(_row("success_rate", f"{qr.get('success_rate', 0):.3f}",
                   f"{gr.get('success_rate', 0):.3f}",
                   f"{og.get('success_delta_pp', 0):.2f} pp"))
    md.append(_row("avg_cost", f"{qr.get('avg_cost', 0):.3f}",
                   f"{gr.get('avg_cost', 0):.3f}",
                   f"{og.get('avg_cost_delta', 0):+.3f}"))
    md.append(_row("avg_objective", f"{qr.get('avg_objective', 0):.4f}",
                   f"{gr.get('avg_objective', 0):.4f}",
                   f"{og.get('avg_objective_delta', 0):+.4f}"))
    md.append(f"\ncost_saving_pct_at_observed = **{og.get('cost_saving_pct_at_observed', 0):.2f}%** (GO threshold = 30%)\n")
    md.append("\nGO criterion 1 result: **FAIL** (below threshold). Decision: BACKUP.\n")
    return "".join(md)


def ablation_table() -> str:
    abl = _load("ed_cached_ablations.json") or {}
    if not abl:
        return "# Table — Ablation results\n\n*(experiments/ed_cached_ablations.json missing)*\n"
    md = ["# Table — Ablation results\n\n",
          "## A. Verifier-aware vs no-verifier (cost saving due to verifier)\n\n",
          "| policy | verifier_aware avg_cost | no_verifier avg_cost | saving_pct |\n",
          "|---|---|---|---|\n"]
    for name, block in abl.get("A_verifier", {}).items():
        va = block["verifier_aware"]["avg_cost"]
        nv = block["no_verifier"]["avg_cost"]
        s = block["cost_saving_due_to_verifier_pct"]
        md.append(_row(name, f"{va:.3f}", f"{nv:.3f}", f"{s:.2f}%"))

    md.append("\n## B. Partial-strong (strong_hint) ablation\n\n")
    md.append("| variant | success | avg_cost |\n|---|---|---|\n")
    for k in ["oracle_graph_with_strong_hint", "oracle_graph_without_strong_hint",
              "oracle_graph_full_strong_only", "heuristic_bdelg_with_hint",
              "heuristic_bdelg_no_hint"]:
        if k in abl.get("B_partial_strong", {}):
            agg = abl["B_partial_strong"][k]["aggregate"]
            md.append(_row(k, f"{agg['success_rate']:.3f}", f"{agg['avg_cost']:.3f}"))
    md.append("\n*Note: code n=4. Partial-strong evidence remains underpowered for any method claim.*\n")

    md.append("\n## C. Repair-action ablation\n\n")
    md.append("| variant | success | avg_cost |\n|---|---|---|\n")
    for k, v in abl.get("C_repair", {}).items():
        agg = v["aggregate"]
        md.append(_row(k, f"{agg['success_rate']:.3f}", f"{agg['avg_cost']:.3f}"))

    md.append("\n## D. Action-set ablation\n\n")
    qr_cost = abl.get("D_action_set", {}).get("_oracle_query_router_avg_cost")
    if qr_cost is not None:
        md.append(f"Reference oracle-query-router avg_cost = {qr_cost:.3f}\n\n")
    md.append("| action_set | success | avg_cost | saving_vs_router_pct |\n|---|---|---|---|\n")
    for k, v in abl.get("D_action_set", {}).items():
        if k.startswith("_"):
            continue
        agg = v["aggregate"]
        md.append(_row(k, f"{agg['success_rate']:.3f}", f"{agg['avg_cost']:.3f}",
                       f"{v['cost_saving_vs_oracle_query_router_pct']:.2f}%"))
    return "".join(md)


def sensitivity_table() -> str:
    s = _load("ed_sensitivity.json") or {}
    if not s:
        return "# Table — Sensitivity sweep\n\n*(experiments/ed_sensitivity.json missing)*\n"
    md = ["# Table — Sensitivity sweep summary\n\n",
          f"- budgets: {s['budgets']}\n",
          f"- cost penalties: {s['cost_penalties']}\n",
          f"- GO threshold: {s['go_threshold_pct']}%\n",
          f"- any sweep cell crosses 30%? **{'yes' if s['any_crossing'] else 'no'}**\n",
          "\n## Crossings\n\n"]
    if s["crossings"]:
        md.append("| budget | cost_penalty | saving_pct |\n|---|---|---|\n")
        for c in s["crossings"]:
            md.append(_row(c["budget"], c["cost_penalty"], f"{c['saving_pct']:.2f}%"))
    else:
        md.append("None. The borderline 26.09% saving is robust on the as-run smoke.\n")
    md.append("\n*This is a sensitivity diagnostic; no GO promotion.*\n")
    return "".join(md)


def bootstrap_table() -> str:
    b = _load("ed_bootstrap_ci.json") or {}
    if not b:
        return "# Table — Bootstrap CI\n\n*(experiments/ed_bootstrap_ci.json missing)*\n"

    def row_for(label: str, ci: dict) -> str:
        return _row(label, f"{ci['mean']:.3f}", f"{ci['ci_lo']:.3f}", f"{ci['ci_hi']:.3f}")

    md = ["# Table — Bootstrap CI (95%)\n\n",
          f"seed={b['seed']}, n_resamples={b['n_resamples']}, n_tasks={b['n_tasks']}\n\n"]
    md.append("## Family-stratified\n\n| metric | mean | ci_lo | ci_hi |\n|---|---|---|---|\n")
    s = b["family_stratified"]
    md.append(row_for("oracle_query_router_avg_cost", s["oracle_query_router_avg_cost"]))
    md.append(row_for("oracle_deliberation_graph_avg_cost", s["oracle_deliberation_graph_avg_cost"]))
    md.append(row_for("graph_query_cost_saving_pct", s["graph_query_cost_saving_pct"]))
    md.append(row_for("heuristic_bdelg_avg_cost", s["heuristic_bdelg_avg_cost"]))
    md.append(row_for("best_non_oracle_baseline_avg_cost", s["best_non_oracle_baseline_avg_cost"]))
    if s.get("code_family_graph_query_cost_saving_pct"):
        md.append(row_for("code_family_graph_query_cost_saving_pct",
                          s["code_family_graph_query_cost_saving_pct"]))
    md.append("\n*Caveats (mirroring `reports/ED_BOOTSTRAP_CI.md`): code n=4 makes the "
              "code-family CI **wide and nearly degenerate**; CIs reflect "
              "**within-pool sampling uncertainty only** and do not capture model-, "
              "prompt-, or distribution-shift uncertainty. Decision: BACKUP / E&D-only.*\n")
    return "".join(md)


def reweighting_table() -> str:
    r = _load("ed_family_reweighting.json") or {}
    if not r:
        return "# Table — Family reweighting\n\n*(experiments/ed_family_reweighting.json missing)*\n"
    md = ["# Table — Family-mixture reweighting (diagnostic)\n\n",
          f"- first code-weight crossing 30%: **{r.get('first_code_weight_crossing_threshold')}**\n",
          "\n| code_weight | router_cost | graph_cost | saving_pct | crosses_30%? |\n",
          "|---|---|---|---|---|\n"]
    for s in r.get("sweeps", []):
        md.append(_row(f"{s['code_weight']:.2f}", f"{s['router_avg_cost']:.3f}",
                       f"{s['graph_avg_cost']:.3f}",
                       f"{s['graph_query_cost_saving_pct']:.2f}%",
                       "yes" if s["crosses_30pct"] else "no"))
    md.append("\n*Diagnostic only; treats per-family averages as stable across mixtures, "
              "an assumption that breaks down for small per-family n. Decision: BACKUP / E&D-only.*\n")
    return "".join(md)


def update_figures_plan() -> None:
    p = PAPER / "figures_plan.md"
    if not p.exists():
        return
    note = ("\n\n## Active vs future-gated (auto-refreshed by `scripts/make_ed_tables.py`)\n\n"
            "**Active in the artifact:**\n"
            "- `figures/pareto.png` — aggregate cost-vs-success on n=28 (math/code/evidence).\n\n"
            "**Future-gated (NOT in the E&D submission):**\n"
            "- Partial-strong-only headline figure on code (n=4 too small).\n"
            "- Verifier-state ablation figure (covered in `reports/ED_CACHED_ABLATIONS.md` as a cached-only ablation; would need real-API or expanded-task evidence to be a headline figure).\n"
            "- Unsupported-risk frontier (no spread on the current evidence pool).\n")
    text = p.read_text(encoding="utf-8")
    marker = "## Active vs future-gated (auto-refreshed by"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    p.write_text(text + note, encoding="utf-8")


def main() -> int:
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "table_smoke_results.md").write_text(smoke_table(), encoding="utf-8")
    (PAPER / "table_oracle_gap.md").write_text(oracle_gap_table(), encoding="utf-8")
    (PAPER / "table_ablation_results.md").write_text(ablation_table(), encoding="utf-8")
    (PAPER / "table_sensitivity_summary.md").write_text(sensitivity_table(), encoding="utf-8")
    (PAPER / "table_bootstrap_ci.md").write_text(bootstrap_table(), encoding="utf-8")
    (PAPER / "table_family_reweighting.md").write_text(reweighting_table(), encoding="utf-8")
    update_figures_plan()
    print("wrote paper/table_smoke_results.md, paper/table_oracle_gap.md, "
          "paper/table_ablation_results.md, paper/table_sensitivity_summary.md, "
          "paper/table_bootstrap_ci.md, paper/table_family_reweighting.md")
    print("updated paper/figures_plan.md (active vs future-gated section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
