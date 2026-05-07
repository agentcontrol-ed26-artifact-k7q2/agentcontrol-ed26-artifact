"""Cached-only bootstrap confidence intervals on n=28 smoke outcomes.

Uses task-level resampling with replacement, plus a family-stratified variant.
Deterministic seed. No provider calls. Decision unchanged.
"""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    PLAN_AUTOMIX,
    PLAN_FRUGALGPT,
    PLAN_HEURISTIC_BDELG,
    PLAN_SHEPHERDING,
    PLANS_CHEAP_ONLY,
    PLANS_STRONG_ONLY,
    QUERY_ROUTER_PLANS,
    aggregate,
    best_plan_per_task,
    family_of,
    load_outcomes,
    run_plan_over_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
SEED = 20260427
N_BOOT = 2000
BUDGET = 20.0
COST_PENALTY = 0.01


def _per_task_results(outcomes: dict) -> dict[str, dict[str, dict]]:
    """Return per-task per-policy result dicts so we can resample at the task level."""
    pol_results: dict[str, dict[str, dict]] = {}

    qr = best_plan_per_task(QUERY_ROUTER_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    gr = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    pol_results["oracle_query_router"] = {r["task_id"]: r for r in qr}
    pol_results["oracle_deliberation_graph"] = {r["task_id"]: r for r in gr}

    for name, plan in {
        "always_cheapest": PLANS_CHEAP_ONLY,
        "always_strongest": PLANS_STRONG_ONLY,
        "frugalgpt_cascade": PLAN_FRUGALGPT,
        "automix_self_verification_cascade": PLAN_AUTOMIX,
        "shepherding_hint": PLAN_SHEPHERDING,
        "heuristic_bdelg": PLAN_HEURISTIC_BDELG,
    }.items():
        rs = run_plan_over_outcomes(name, plan, outcomes, verifier_aware=True,
                                    budget=BUDGET, cost_penalty=COST_PENALTY)
        pol_results[name] = {r["task_id"]: r for r in rs}
    return pol_results


def _ci(samples: list[float], lo: float = 2.5, hi: float = 97.5) -> dict[str, float]:
    s = sorted(samples)
    n = len(s)
    return {
        "mean": statistics.fmean(samples),
        "ci_lo": s[max(0, int(n * lo / 100.0) - 1)],
        "ci_hi": s[min(n - 1, int(n * hi / 100.0) - 1)],
        "n_resamples": n,
    }


def _avg(metric: str, ids: list[str], pol: dict) -> float:
    return statistics.fmean(pol[t][metric] if metric != "success" else float(pol[t]["success"])
                            for t in ids)


def _saving_pct(base: float, cur: float) -> float:
    return 100.0 * (base - cur) / base if base > 0 else 0.0


def _best_baseline_cost(ids: list[str], pol_results: dict) -> float:
    """Per-bootstrap "best non-oracle baseline" by lowest avg_cost at full success."""
    best = None
    for name in ["always_cheapest", "always_strongest", "frugalgpt_cascade",
                 "automix_self_verification_cascade", "shepherding_hint"]:
        succ = _avg("success", ids, pol_results[name])
        cost = _avg("cost", ids, pol_results[name])
        cand = (succ, -cost, name, cost)
        if best is None or cand > best:
            best = cand
    return best[3]


def _bootstrap(rng: random.Random, ids: list[str], pol_results: dict,
               family_groups: dict[str, list[str]] | None) -> tuple[list[float], ...]:
    graph_costs, router_costs, savings = [], [], []
    heur_costs, best_baseline_costs = [], []
    code_savings: list[float] = []

    has_code = "code" in (family_groups or {})

    for _ in range(N_BOOT):
        if family_groups is None:
            sample = [rng.choice(ids) for _ in ids]
        else:
            sample = []
            for fam, fam_ids in family_groups.items():
                sample.extend(rng.choice(fam_ids) for _ in fam_ids)
        gc = _avg("cost", sample, pol_results["oracle_deliberation_graph"])
        rc = _avg("cost", sample, pol_results["oracle_query_router"])
        graph_costs.append(gc)
        router_costs.append(rc)
        savings.append(_saving_pct(rc, gc))
        heur_costs.append(_avg("cost", sample, pol_results["heuristic_bdelg"]))
        best_baseline_costs.append(_best_baseline_cost(sample, pol_results))

        if has_code:
            code_sample = [t for t in sample if family_of(t) == "code"]
            if code_sample:
                gc_code = _avg("cost", code_sample, pol_results["oracle_deliberation_graph"])
                rc_code = _avg("cost", code_sample, pol_results["oracle_query_router"])
                code_savings.append(_saving_pct(rc_code, gc_code))

    return graph_costs, router_costs, savings, heur_costs, best_baseline_costs, code_savings


def main() -> int:
    outcomes = load_outcomes(str(REPO / "experiments" / "smoke_outcomes.json"))
    ids = sorted(outcomes.keys())
    pol_results = _per_task_results(outcomes)

    family_groups: dict[str, list[str]] = defaultdict(list)
    for t in ids:
        family_groups[family_of(t)].append(t)
    family_groups = dict(family_groups)

    rng_unstrat = random.Random(SEED)
    g_u, r_u, s_u, h_u, b_u, code_u = _bootstrap(rng_unstrat, ids, pol_results, None)

    rng_strat = random.Random(SEED + 1)
    g_s, r_s, s_s, h_s, b_s, code_s = _bootstrap(rng_strat, ids, pol_results, family_groups)

    summary = {
        "seed": SEED,
        "n_resamples": N_BOOT,
        "n_tasks": len(ids),
        "family_counts": {k: len(v) for k, v in family_groups.items()},
        "unstratified": {
            "oracle_deliberation_graph_avg_cost": _ci(g_u),
            "oracle_query_router_avg_cost": _ci(r_u),
            "graph_query_cost_saving_pct": _ci(s_u),
            "heuristic_bdelg_avg_cost": _ci(h_u),
            "best_non_oracle_baseline_avg_cost": _ci(b_u),
            "code_family_graph_query_cost_saving_pct": _ci(code_u) if code_u else None,
        },
        "family_stratified": {
            "oracle_deliberation_graph_avg_cost": _ci(g_s),
            "oracle_query_router_avg_cost": _ci(r_s),
            "graph_query_cost_saving_pct": _ci(s_s),
            "heuristic_bdelg_avg_cost": _ci(h_s),
            "best_non_oracle_baseline_avg_cost": _ci(b_s),
            "code_family_graph_query_cost_saving_pct": _ci(code_s) if code_s else None,
        },
        "caveats": (
            "Code n=4 → code-family CI is wide and nearly degenerate. "
            "Bootstrap CIs reflect within-pool sampling uncertainty only; they "
            "do not capture model-, prompt-, or distribution-shift uncertainty. "
            "Decision remains BACKUP / E&D-only."
        ),
    }
    out_json = REPO / "experiments" / "ed_bootstrap_ci.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _row(label: str, ci: dict) -> str:
        return f"| {label} | {ci['mean']:.3f} | {ci['ci_lo']:.3f} | {ci['ci_hi']:.3f} |\n"

    md = ["# ED_BOOTSTRAP_CI\n",
          "\nDECISION: BACKUP / E&D-only (unchanged). Bootstrap is diagnostic only.\n\n",
          f"- seed: {SEED}\n",
          f"- resamples per row: {N_BOOT}\n",
          f"- tasks: {len(ids)} (family counts {summary['family_counts']})\n",
          "\n## Unstratified bootstrap (95% CI)\n\n",
          "| metric | mean | ci_lo | ci_hi |\n|---|---|---|---|\n"]
    u = summary["unstratified"]
    md.append(_row("oracle_deliberation_graph_avg_cost", u["oracle_deliberation_graph_avg_cost"]))
    md.append(_row("oracle_query_router_avg_cost", u["oracle_query_router_avg_cost"]))
    md.append(_row("graph_query_cost_saving_pct", u["graph_query_cost_saving_pct"]))
    md.append(_row("heuristic_bdelg_avg_cost", u["heuristic_bdelg_avg_cost"]))
    md.append(_row("best_non_oracle_baseline_avg_cost", u["best_non_oracle_baseline_avg_cost"]))
    if u["code_family_graph_query_cost_saving_pct"]:
        md.append(_row("code_family_graph_query_cost_saving_pct",
                       u["code_family_graph_query_cost_saving_pct"]))

    md.append("\n## Family-stratified bootstrap (preserves math/code/evidence counts)\n\n")
    md.append("| metric | mean | ci_lo | ci_hi |\n|---|---|---|---|\n")
    s = summary["family_stratified"]
    md.append(_row("oracle_deliberation_graph_avg_cost", s["oracle_deliberation_graph_avg_cost"]))
    md.append(_row("oracle_query_router_avg_cost", s["oracle_query_router_avg_cost"]))
    md.append(_row("graph_query_cost_saving_pct", s["graph_query_cost_saving_pct"]))
    md.append(_row("heuristic_bdelg_avg_cost", s["heuristic_bdelg_avg_cost"]))
    md.append(_row("best_non_oracle_baseline_avg_cost", s["best_non_oracle_baseline_avg_cost"]))
    if s["code_family_graph_query_cost_saving_pct"]:
        md.append(_row("code_family_graph_query_cost_saving_pct",
                       s["code_family_graph_query_cost_saving_pct"]))

    md.append("\n## Caveats\n\n")
    md.append(summary["caveats"] + "\n")

    out_md = REPO / "reports" / "ED_BOOTSTRAP_CI.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO)} and {out_md.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
