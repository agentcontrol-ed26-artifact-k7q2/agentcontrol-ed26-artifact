"""Phase 4-7: comprehensive rescue analyses on real-model outcomes.

Runs oracle/baselines/heuristic/verifier-ablation/bootstrap/sensitivity on
each real-model provider snapshot, plus a joint comparison and final track
decision. No new API calls.
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
    run_plan_over_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
BUDGET = 20.0
COST_PENALTY = 0.01
GO_THRESHOLD_PCT = 30.0
SEED = 20260427
N_BOOT = 1000

PROVIDER_SNAPSHOTS = {
    "deepseek": {
        "outcomes": "experiments/rescue_outcomes_deepseek.json",
        "cheap_model": "deepseek-chat (V4 non-thinking)",
        "strong_model": "deepseek-reasoner (V4 thinking / R1)",
    },
    "together": {
        "outcomes": "experiments/rescue_outcomes_together.json",
        "cheap_model": "Qwen2.5-7B-Instruct-Turbo",
        "strong_model": "Llama-3.3-70B-Instruct-Turbo",
    },
}


def _fam(tid: str) -> str:
    if tid.startswith("rm"): return "math"
    if tid.startswith("rc"): return "code"
    if tid.startswith("re"): return "evidence"
    return "other"


def _per_family(rs):
    by = defaultdict(list)
    for r in rs:
        by[_fam(r["task_id"])].append(r)
    return {f: aggregate(v) for f, v in by.items()}


def _saving_pct(base: float, cur: float) -> float:
    return 100.0 * (base - cur) / base if base > 0 else 0.0


def _baselines(outcomes: dict) -> dict:
    panel = {
        "always_cheapest": PLANS_CHEAP_ONLY,
        "always_strongest": PLANS_STRONG_ONLY,
        "frugalgpt_cascade": PLAN_FRUGALGPT,
        "automix_self_verification_cascade": PLAN_AUTOMIX,
        "shepherding_hint": PLAN_SHEPHERDING,
        "fixed_react": ["cheap_answer", "strong_critique", "cheap_repair_if_needed"],
        "fixed_retrieve_answer_verify": ["cheap_answer", "strong_checklist",
                                          "cheap_repair_after_hint_if_needed"],
    }
    out = {}
    for name, plan in panel.items():
        rs = run_plan_over_outcomes(name, plan, outcomes, verifier_aware=True,
                                     budget=BUDGET, cost_penalty=COST_PENALTY)
        out[name] = {"aggregate": aggregate(rs), "per_family": _per_family(rs)}
    return out


def _heuristic(outcomes: dict) -> dict:
    rs = run_plan_over_outcomes("heuristic_bdelg", PLAN_HEURISTIC_BDELG, outcomes,
                                 verifier_aware=True, budget=BUDGET,
                                 cost_penalty=COST_PENALTY)
    return {"aggregate": aggregate(rs), "per_family": _per_family(rs),
            "task_results": rs}


def _oracles(outcomes: dict) -> dict:
    qr = best_plan_per_task(QUERY_ROUTER_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    gr = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    qra, gra = aggregate(qr), aggregate(gr)
    return {
        "query_router": {"aggregate": qra, "per_family": _per_family(qr)},
        "deliberation_graph": {"aggregate": gra, "per_family": _per_family(gr)},
        "success_delta_pp": 100.0 * (gra["success_rate"] - qra["success_rate"]),
        "cost_saving_pct_at_observed": _saving_pct(qra["avg_cost"], gra["avg_cost"]),
        "qr_results": qr,
        "gr_results": gr,
    }


def _verifier_ablation(outcomes: dict) -> dict:
    """Pareto comparison: verifier-aware vs no-verifier across budget tiers."""
    budgets = [3.0, 5.0, 8.0, 12.0, 20.0]
    rows = []
    for b in budgets:
        gr_va = aggregate(best_plan_per_task(GRAPH_PLANS, outcomes,
                                              verifier_aware=True, budget=b,
                                              cost_penalty=COST_PENALTY))
        gr_nv = aggregate(best_plan_per_task(GRAPH_PLANS, outcomes,
                                              verifier_aware=False, budget=b,
                                              cost_penalty=COST_PENALTY))
        rows.append({
            "budget": b,
            "verifier_aware_success": gr_va["success_rate"],
            "verifier_aware_cost": gr_va["avg_cost"],
            "no_verifier_success": gr_nv["success_rate"],
            "no_verifier_cost": gr_nv["avg_cost"],
            "no_verifier_cost_premium_pct": _saving_pct(gr_va["avg_cost"], gr_nv["avg_cost"]),
        })
    # Pareto-frontier check: does no-verifier dominate verifier-aware on any budget?
    dominated_by_no_verifier = []
    for r in rows:
        if (r["no_verifier_success"] >= r["verifier_aware_success"] - 1e-9
                and r["no_verifier_cost"] < r["verifier_aware_cost"] - 1e-9):
            dominated_by_no_verifier.append(r["budget"])
    return {"budget_tiers": rows,
            "no_verifier_dominates_at_budgets": dominated_by_no_verifier,
            "go_criterion_4_passed": not dominated_by_no_verifier and any(
                r["no_verifier_cost_premium_pct"] > 0 for r in rows
            )}


def _bootstrap(outcomes: dict) -> dict:
    ids = sorted(outcomes.keys())
    qr = {r["task_id"]: r for r in best_plan_per_task(
        QUERY_ROUTER_PLANS, outcomes, verifier_aware=True, budget=BUDGET,
        cost_penalty=COST_PENALTY)}
    gr = {r["task_id"]: r for r in best_plan_per_task(
        GRAPH_PLANS, outcomes, verifier_aware=True, budget=BUDGET,
        cost_penalty=COST_PENALTY)}
    h = {r["task_id"]: r for r in run_plan_over_outcomes(
        "heur", PLAN_HEURISTIC_BDELG, outcomes, verifier_aware=True,
        budget=BUDGET, cost_penalty=COST_PENALTY)}

    fam_groups = defaultdict(list)
    for t in ids:
        fam_groups[_fam(t)].append(t)

    rng = random.Random(SEED)
    savings, qr_costs, gr_costs, h_costs = [], [], [], []
    for _ in range(N_BOOT):
        sample = []
        for fam, fids in fam_groups.items():
            sample.extend(rng.choice(fids) for _ in fids)
        gc = statistics.fmean(gr[t]["cost"] for t in sample)
        rc = statistics.fmean(qr[t]["cost"] for t in sample)
        hc = statistics.fmean(h[t]["cost"] for t in sample)
        savings.append(_saving_pct(rc, gc))
        qr_costs.append(rc)
        gr_costs.append(gc)
        h_costs.append(hc)

    def ci(s):
        sl = sorted(s)
        return {"mean": statistics.fmean(s), "ci_lo": sl[int(N_BOOT * 0.025)],
                "ci_hi": sl[int(N_BOOT * 0.975)]}

    return {"n_resamples": N_BOOT, "seed": SEED,
            "graph_query_cost_saving_pct": ci(savings),
            "oracle_query_router_avg_cost": ci(qr_costs),
            "oracle_deliberation_graph_avg_cost": ci(gr_costs),
            "heuristic_bdelg_avg_cost": ci(h_costs)}


def _sensitivity(outcomes: dict) -> dict:
    cells = []
    crossings = []
    for b in (3.0, 5.0, 8.0, 12.0, 20.0):
        for cp in (0.0, 0.01, 0.1):
            qr = aggregate(best_plan_per_task(QUERY_ROUTER_PLANS, outcomes,
                                              verifier_aware=True, budget=b,
                                              cost_penalty=cp))
            gr = aggregate(best_plan_per_task(GRAPH_PLANS, outcomes,
                                              verifier_aware=True, budget=b,
                                              cost_penalty=cp))
            saving = _saving_pct(qr["avg_cost"], gr["avg_cost"])
            cells.append({"budget": b, "cost_penalty": cp, "saving_pct": saving})
            if saving >= GO_THRESHOLD_PCT:
                crossings.append({"budget": b, "cost_penalty": cp, "saving_pct": saving})
    return {"cells": cells, "crossings": crossings, "any_crossing": bool(crossings)}


def analyze_provider(provider_key: str, cfg: dict) -> dict:
    path = REPO / cfg["outcomes"]
    outcomes = json.loads(path.read_text(encoding="utf-8"))
    return {
        "provider": provider_key,
        "cheap_model": cfg["cheap_model"],
        "strong_model": cfg["strong_model"],
        "n_tasks": len(outcomes),
        "oracles": _oracles(outcomes),
        "baselines": _baselines(outcomes),
        "heuristic": _heuristic(outcomes),
        "verifier_ablation": _verifier_ablation(outcomes),
        "bootstrap": _bootstrap(outcomes),
        "sensitivity": _sensitivity(outcomes),
    }


def write_provider_md(provider_key: str, a: dict, out_md: Path) -> None:
    o = a["oracles"]
    qr = o["query_router"]["aggregate"]
    gr = o["deliberation_graph"]["aggregate"]
    md = [f"# RESCUE — {provider_key}\n",
          f"\n- cheap: {a['cheap_model']}\n",
          f"- strong: {a['strong_model']}\n",
          f"- n_tasks: {a['n_tasks']}\n",
          f"\n## Oracle gap\n",
          f"\n| metric | router | graph | delta |\n|---|---|---|---|\n",
          f"| success | {qr['success_rate']:.3f} | {gr['success_rate']:.3f} | {o['success_delta_pp']:.2f} pp |\n",
          f"| avg_cost | {qr['avg_cost']:.3f} | {gr['avg_cost']:.3f} | {(gr['avg_cost']-qr['avg_cost']):+.3f} |\n",
          f"\ncost_saving_pct = **{o['cost_saving_pct_at_observed']:.2f}%** (GO threshold = 30%)\n",
          "\n## Per family\n\n",
          "| family | router cost | graph cost | saving | router succ | graph succ |\n",
          "|---|---|---|---|---|---|\n"]
    for fam in ("math", "code", "evidence"):
        if fam in o["query_router"]["per_family"]:
            r = o["query_router"]["per_family"][fam]
            g = o["deliberation_graph"]["per_family"][fam]
            s = _saving_pct(r["avg_cost"], g["avg_cost"])
            md.append(f"| {fam} | {r['avg_cost']:.3f} | {g['avg_cost']:.3f} | {s:.2f}% | {r['success_rate']:.3f} | {g['success_rate']:.3f} |\n")

    md.append("\n## Baselines\n\n")
    md.append("| baseline | success | avg_cost |\n|---|---|---|\n")
    for name, b in a["baselines"].items():
        ag = b["aggregate"]
        md.append(f"| {name} | {ag['success_rate']:.3f} | {ag['avg_cost']:.3f} |\n")
    h = a["heuristic"]["aggregate"]
    md.append(f"| **heuristic_bdelg** | **{h['success_rate']:.3f}** | **{h['avg_cost']:.3f}** |\n")

    md.append("\n## Verifier Pareto ablation (GO criterion 4)\n\n")
    md.append("| budget | va_success | va_cost | nv_success | nv_cost | nv cost premium |\n|---|---|---|---|---|---|\n")
    for r in a["verifier_ablation"]["budget_tiers"]:
        md.append(f"| {r['budget']} | {r['verifier_aware_success']:.3f} | {r['verifier_aware_cost']:.3f} | {r['no_verifier_success']:.3f} | {r['no_verifier_cost']:.3f} | {r['no_verifier_cost_premium_pct']:.2f}% |\n")
    md.append(f"\nGO criterion 4 (verifier hurts Pareto on no-verifier side): **{'PASS' if a['verifier_ablation']['go_criterion_4_passed'] else 'FAIL/MIXED'}**\n")

    md.append("\n## Bootstrap (family-stratified, 95% CI, n_resamples="
              f"{a['bootstrap']['n_resamples']}, seed={a['bootstrap']['seed']})\n\n")
    md.append("| metric | mean | ci_lo | ci_hi |\n|---|---|---|---|\n")
    for k in ("oracle_query_router_avg_cost", "oracle_deliberation_graph_avg_cost",
              "graph_query_cost_saving_pct", "heuristic_bdelg_avg_cost"):
        c = a["bootstrap"][k]
        md.append(f"| {k} | {c['mean']:.3f} | {c['ci_lo']:.3f} | {c['ci_hi']:.3f} |\n")

    md.append("\n## Sensitivity\n\n")
    md.append(f"any sweep cell crosses 30%? **{a['sensitivity']['any_crossing']}**\n")
    if a["sensitivity"]["crossings"]:
        md.append("\n| budget | cost_penalty | saving_pct |\n|---|---|---|\n")
        for c in a["sensitivity"]["crossings"]:
            md.append(f"| {c['budget']} | {c['cost_penalty']} | {c['saving_pct']:.2f}% |\n")
    out_md.write_text("".join(md), encoding="utf-8")


def main() -> int:
    REPO.joinpath("experiments").mkdir(parents=True, exist_ok=True)
    REPO.joinpath("reports").mkdir(parents=True, exist_ok=True)
    all_a = {}
    for k, cfg in PROVIDER_SNAPSHOTS.items():
        if not (REPO / cfg["outcomes"]).exists():
            print(f"skip {k}: {cfg['outcomes']} missing")
            continue
        all_a[k] = analyze_provider(k, cfg)
        # Strip task_results from heuristic to keep JSON small.
        a = all_a[k].copy()
        if "heuristic" in a and "task_results" in a["heuristic"]:
            a["heuristic"] = {kk: vv for kk, vv in a["heuristic"].items()
                              if kk != "task_results"}
        if "oracles" in a:
            a["oracles"] = {kk: vv for kk, vv in a["oracles"].items()
                            if kk not in ("qr_results", "gr_results")}
        out_json = REPO / "experiments" / f"rescue_summary_{k}.json"
        out_json.write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
        out_md = REPO / "reports" / f"RESCUE_{k.upper()}.md"
        write_provider_md(k, all_a[k], out_md)
        print(f"wrote {out_json.relative_to(REPO)}, {out_md.relative_to(REPO)}")

    # Joint comparison.
    joint = {p: {
        "oracle_cost_saving_pct": all_a[p]["oracles"]["cost_saving_pct_at_observed"],
        "oracle_success_delta_pp": all_a[p]["oracles"]["success_delta_pp"],
        "router_avg_cost": all_a[p]["oracles"]["query_router"]["aggregate"]["avg_cost"],
        "graph_avg_cost": all_a[p]["oracles"]["deliberation_graph"]["aggregate"]["avg_cost"],
        "heuristic_avg_cost": all_a[p]["heuristic"]["aggregate"]["avg_cost"],
        "router_success": all_a[p]["oracles"]["query_router"]["aggregate"]["success_rate"],
        "graph_success": all_a[p]["oracles"]["deliberation_graph"]["aggregate"]["success_rate"],
        "verifier_ablation_pass": all_a[p]["verifier_ablation"]["go_criterion_4_passed"],
        "sensitivity_any_crossing": all_a[p]["sensitivity"]["any_crossing"],
        "bootstrap_saving_ci": all_a[p]["bootstrap"]["graph_query_cost_saving_pct"],
        "n_tasks": all_a[p]["n_tasks"],
    } for p in all_a}
    (REPO / "experiments" / "rescue_summary_joint.json").write_text(
        json.dumps(joint, indent=2, default=str), encoding="utf-8")

    # Comparison report.
    md = ["# RESCUE_JOINT_COMPARISON\n\n",
          "Real-model evaluation across two providers, n=60 each.\n\n",
          "| provider | cheap | strong | router cost | graph cost | saving | router succ | graph succ |\n",
          "|---|---|---|---|---|---|---|---|\n"]
    for p, j in joint.items():
        cfg = PROVIDER_SNAPSHOTS[p]
        md.append(f"| {p} | {cfg['cheap_model']} | {cfg['strong_model']} | {j['router_avg_cost']:.3f} | {j['graph_avg_cost']:.3f} | {j['oracle_cost_saving_pct']:.2f}% | {j['router_success']:.3f} | {j['graph_success']:.3f} |\n")
    md.append("\n## Bootstrap saving (95% CI)\n\n")
    md.append("| provider | mean | ci_lo | ci_hi |\n|---|---|---|---|\n")
    for p, j in joint.items():
        c = j["bootstrap_saving_ci"]
        md.append(f"| {p} | {c['mean']:.3f} | {c['ci_lo']:.3f} | {c['ci_hi']:.3f} |\n")
    md.append("\n## Verifier Pareto ablation\n\n")
    md.append("| provider | GO criterion 4 |\n|---|---|\n")
    for p, j in joint.items():
        md.append(f"| {p} | {'PASS' if j['verifier_ablation_pass'] else 'FAIL/MIXED'} |\n")
    md.append("\n## Sensitivity\n\n")
    md.append("| provider | any sweep cell crosses 30%? |\n|---|---|\n")
    for p, j in joint.items():
        md.append(f"| {p} | {j['sensitivity_any_crossing']} |\n")
    (REPO / "reports" / "RESCUE_JOINT_COMPARISON.md").write_text("".join(md), encoding="utf-8")
    print("wrote experiments/rescue_summary_joint.json and reports/RESCUE_JOINT_COMPARISON.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
