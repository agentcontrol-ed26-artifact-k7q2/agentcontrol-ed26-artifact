"""Phase 4: regime-mapping analyses on hard-regime real-model outcomes."""
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

PROVIDERS = {
    "deepseek": {
        "outcomes": "experiments/hard_regime_outcomes_deepseek.json",
        "cheap_model": "deepseek-chat (V4 non-thinking)",
        "strong_model": "deepseek-reasoner (V4 thinking / R1)",
    },
    "together": {
        "outcomes": "experiments/hard_regime_outcomes_together.json",
        "cheap_model": "Qwen2.5-7B-Instruct-Turbo",
        "strong_model": "Llama-3.3-70B-Instruct-Turbo",
    },
}


def _fam(tid: str) -> str:
    if tid.startswith("hm"): return "math"
    if tid.startswith("hc"): return "code"
    if tid.startswith("he"): return "evidence"
    return "other"


def _regime(outcomes: dict, tid: str) -> str:
    for action_dict in outcomes[tid].values():
        if "regime" in action_dict:
            return action_dict["regime"]
    return "unknown"


def _saving_pct(base: float, cur: float) -> float:
    return 100.0 * (base - cur) / base if base > 0 else 0.0


def _per_family(rs):
    by = defaultdict(list)
    for r in rs:
        by[_fam(r["task_id"])].append(r)
    return {f: aggregate(v) for f, v in by.items()}


def _per_regime(rs, outcomes):
    by = defaultdict(list)
    for r in rs:
        by[_regime(outcomes, r["task_id"])].append(r)
    return {f: aggregate(v) for f, v in by.items()}


def _per_family_regime(rs, outcomes):
    by = defaultdict(list)
    for r in rs:
        key = (_fam(r["task_id"]), _regime(outcomes, r["task_id"]))
        by[key].append(r)
    return {f"{f}/{reg}": aggregate(v) for (f, reg), v in by.items()}


def _baselines(outcomes):
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
    for n, plan in panel.items():
        rs = run_plan_over_outcomes(n, plan, outcomes, verifier_aware=True,
                                    budget=BUDGET, cost_penalty=COST_PENALTY)
        out[n] = {
            "aggregate": aggregate(rs),
            "per_family": _per_family(rs),
            "per_regime": _per_regime(rs, outcomes),
        }
    return out


def _heuristic(outcomes):
    rs = run_plan_over_outcomes("heuristic_bdelg", PLAN_HEURISTIC_BDELG, outcomes,
                                 verifier_aware=True, budget=BUDGET, cost_penalty=COST_PENALTY)
    return {
        "aggregate": aggregate(rs),
        "per_family": _per_family(rs),
        "per_regime": _per_regime(rs, outcomes),
        "per_family_regime": _per_family_regime(rs, outcomes),
    }


def _oracles(outcomes):
    qr = best_plan_per_task(QUERY_ROUTER_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    gr = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    return {
        "query_router": {
            "aggregate": aggregate(qr),
            "per_family": _per_family(qr),
            "per_regime": _per_regime(qr, outcomes),
            "per_family_regime": _per_family_regime(qr, outcomes),
            "results": qr,
        },
        "deliberation_graph": {
            "aggregate": aggregate(gr),
            "per_family": _per_family(gr),
            "per_regime": _per_regime(gr, outcomes),
            "per_family_regime": _per_family_regime(gr, outcomes),
            "results": gr,
        },
    }


def _verifier_ablation(outcomes):
    out = []
    for b in (3.0, 5.0, 8.0, 12.0, 20.0):
        va = aggregate(best_plan_per_task(GRAPH_PLANS, outcomes,
                                          verifier_aware=True, budget=b,
                                          cost_penalty=COST_PENALTY))
        nv = aggregate(best_plan_per_task(GRAPH_PLANS, outcomes,
                                          verifier_aware=False, budget=b,
                                          cost_penalty=COST_PENALTY))
        out.append({
            "budget": b,
            "va_succ": va["success_rate"], "va_cost": va["avg_cost"],
            "nv_succ": nv["success_rate"], "nv_cost": nv["avg_cost"],
            "no_verifier_cost_premium_pct": _saving_pct(va["avg_cost"], nv["avg_cost"]),
        })
    return out


def _partial_strong_ablation(outcomes):
    full = {"cheap_answer", "cheap_repair", "strong_hint",
            "cheap_repair_after_hint", "strong_answer"}
    no_hint = full - {"strong_hint", "cheap_repair_after_hint"}
    only_strong = {"cheap_answer", "strong_answer"}
    out = {}
    for name, allowed in (("graph_with_strong_hint", full),
                          ("graph_without_strong_hint", no_hint),
                          ("graph_full_strong_only", only_strong)):
        rs = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                                budget=BUDGET, cost_penalty=COST_PENALTY,
                                allowed_actions=allowed)
        out[name] = {
            "aggregate": aggregate(rs),
            "per_family": _per_family(rs),
            "per_regime": _per_regime(rs, outcomes),
        }
    return out


def _per_task_unsupported_risk(outcomes, results):
    """Average unsupported_risk over tasks where the chosen plan ran on evidence."""
    risks = []
    for r in results:
        if _fam(r["task_id"]) != "evidence":
            continue
        risks.append(r.get("unsupported_risk", 0.0))
    return statistics.fmean(risks) if risks else 0.0


def regime_classify(family: str, regime: str, cheap_succ: float, strong_succ: float,
                    graph_saving: float, unsupported_risk: float) -> str:
    """Map a (family, regime) cell to one of the regime-map labels."""
    if cheap_succ >= 0.95 and strong_succ >= 0.95 and graph_saving < 5.0:
        return "saturation"
    if cheap_succ >= 0.85 and graph_saving < 10.0:
        return "router-sufficient"
    if graph_saving >= 25.0:
        return "graph-headroom"
    if family == "evidence" and unsupported_risk >= 0.05:
        return "verifier-risk"
    if (strong_succ - cheap_succ) >= 0.10 and graph_saving >= 10.0:
        return "graph-headroom"
    if cheap_succ < 0.6 and strong_succ < 0.6:
        return "no-signal"
    return "router-sufficient"


def analyze_provider(provider_key: str, cfg: dict) -> dict:
    path = REPO / cfg["outcomes"]
    if not path.exists():
        return {}
    outcomes = json.loads(path.read_text(encoding="utf-8"))
    res = {
        "provider": provider_key,
        "cheap_model": cfg["cheap_model"],
        "strong_model": cfg["strong_model"],
        "n_tasks": len(outcomes),
        "oracles": _oracles(outcomes),
        "baselines": _baselines(outcomes),
        "heuristic": _heuristic(outcomes),
        "verifier_ablation": _verifier_ablation(outcomes),
        "partial_strong": _partial_strong_ablation(outcomes),
    }
    o = res["oracles"]
    res["graph_query_cost_saving_pct"] = _saving_pct(
        o["query_router"]["aggregate"]["avg_cost"],
        o["deliberation_graph"]["aggregate"]["avg_cost"]
    )
    res["graph_query_success_delta_pp"] = 100.0 * (
        o["deliberation_graph"]["aggregate"]["success_rate"]
        - o["query_router"]["aggregate"]["success_rate"]
    )
    res["unsupported_evidence_risk"] = _per_task_unsupported_risk(
        outcomes, o["deliberation_graph"]["results"]
    )

    # Build per-(family, regime) cheap/strong success gap and graph saving.
    cheap_per_cell = defaultdict(list)
    strong_per_cell = defaultdict(list)
    risk_per_cell = defaultdict(list)
    for tid, action_dict in outcomes.items():
        fam = _fam(tid)
        regime = _regime(outcomes, tid)
        cheap_per_cell[(fam, regime)].append(int(action_dict.get("cheap_answer", {}).get("success", False)))
        strong_per_cell[(fam, regime)].append(int(action_dict.get("strong_answer", {}).get("success", False)))
        risk_per_cell[(fam, regime)].append(action_dict.get("cheap_answer", {}).get("unsupported_risk", 0.0))

    qr_fr = o["query_router"]["per_family_regime"]
    gr_fr = o["deliberation_graph"]["per_family_regime"]
    regime_map = []
    for cell, cheap_list in cheap_per_cell.items():
        fam, reg = cell
        key = f"{fam}/{reg}"
        cheap_succ = statistics.fmean(cheap_list)
        strong_succ = statistics.fmean(strong_per_cell[cell])
        unsup = statistics.fmean(risk_per_cell[cell])
        if key in qr_fr and key in gr_fr:
            r_cost = qr_fr[key]["avg_cost"]
            g_cost = gr_fr[key]["avg_cost"]
            saving = _saving_pct(r_cost, g_cost)
            r_succ = qr_fr[key]["success_rate"]
            g_succ = gr_fr[key]["success_rate"]
        else:
            r_cost = g_cost = 0.0
            saving = 0.0
            r_succ = g_succ = 0.0
        label = regime_classify(fam, reg, cheap_succ, strong_succ, saving, unsup)
        regime_map.append({
            "family": fam, "regime": reg, "n": len(cheap_list),
            "cheap_success": cheap_succ, "strong_success": strong_succ,
            "router_cost": r_cost, "graph_cost": g_cost,
            "graph_saving_pct": saving,
            "router_success": r_succ, "graph_success": g_succ,
            "unsupported_risk": unsup,
            "interpretation": label,
        })
    res["regime_map"] = regime_map
    return res


def write_provider_md(provider_key: str, a: dict, out_md: Path) -> None:
    o = a["oracles"]
    qr = o["query_router"]["aggregate"]
    gr = o["deliberation_graph"]["aggregate"]
    md = [f"# HARD_REGIME — {provider_key}\n",
          f"\n- cheap: {a['cheap_model']}\n",
          f"- strong: {a['strong_model']}\n",
          f"- n_tasks: {a['n_tasks']}\n",
          f"\n## Headline\n",
          f"\n- oracle query router: success {qr['success_rate']:.3f}, avg_cost {qr['avg_cost']:.3f}\n",
          f"- oracle deliberation graph: success {gr['success_rate']:.3f}, avg_cost {gr['avg_cost']:.3f}\n",
          f"- graph-vs-router cost saving: **{a['graph_query_cost_saving_pct']:.2f}%**\n",
          f"- graph-vs-router success delta: **{a['graph_query_success_delta_pp']:.2f} pp**\n",
          f"- unsupported_evidence_risk: {a['unsupported_evidence_risk']:.4f}\n",
          "\n## Baselines (aggregate)\n\n",
          "| baseline | success | avg_cost |\n|---|---|---|\n"]
    for n, b in a["baselines"].items():
        ag = b["aggregate"]
        md.append(f"| {n} | {ag['success_rate']:.3f} | {ag['avg_cost']:.3f} |\n")
    h = a["heuristic"]["aggregate"]
    md.append(f"| **heuristic_bdelg** | **{h['success_rate']:.3f}** | **{h['avg_cost']:.3f}** |\n")

    md.append("\n## Per-family\n\n")
    md.append("| family | router cost | graph cost | saving | router succ | graph succ |\n|---|---|---|---|---|---|\n")
    for fam in ("math", "code", "evidence"):
        if fam in o["query_router"]["per_family"]:
            r = o["query_router"]["per_family"][fam]
            g = o["deliberation_graph"]["per_family"][fam]
            md.append(f"| {fam} | {r['avg_cost']:.3f} | {g['avg_cost']:.3f} | {_saving_pct(r['avg_cost'], g['avg_cost']):.2f}% | {r['success_rate']:.3f} | {g['success_rate']:.3f} |\n")

    md.append("\n## Regime map\n\n")
    md.append("| family | regime | n | cheap_succ | strong_succ | router_cost | graph_cost | saving | unsup_risk | interpretation |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for cell in sorted(a["regime_map"], key=lambda c: (c["family"], c["regime"])):
        md.append(
            f"| {cell['family']} | {cell['regime']} | {cell['n']} | "
            f"{cell['cheap_success']:.3f} | {cell['strong_success']:.3f} | "
            f"{cell['router_cost']:.3f} | {cell['graph_cost']:.3f} | "
            f"{cell['graph_saving_pct']:.2f}% | {cell['unsupported_risk']:.3f} | "
            f"**{cell['interpretation']}** |\n"
        )

    md.append("\n## Verifier Pareto ablation (across budgets)\n\n")
    md.append("| budget | va_succ | va_cost | nv_succ | nv_cost | nv_cost_premium |\n|---|---|---|---|---|---|\n")
    for r in a["verifier_ablation"]:
        md.append(f"| {r['budget']} | {r['va_succ']:.3f} | {r['va_cost']:.3f} | {r['nv_succ']:.3f} | {r['nv_cost']:.3f} | {r['no_verifier_cost_premium_pct']:.2f}% |\n")

    md.append("\n## Partial-strong ablation\n\n")
    md.append("| variant | success | avg_cost |\n|---|---|---|\n")
    for k, v in a["partial_strong"].items():
        ag = v["aggregate"]
        md.append(f"| {k} | {ag['success_rate']:.3f} | {ag['avg_cost']:.3f} |\n")

    out_md.write_text("".join(md), encoding="utf-8")


def main() -> int:
    all_a = {}
    for prov, cfg in PROVIDERS.items():
        a = analyze_provider(prov, cfg)
        if not a:
            print(f"skip {prov}: outcomes file missing")
            continue
        all_a[prov] = a
        # Strip large nested results from JSON.
        a_json = {k: v for k, v in a.items()}
        a_json["oracles"] = {ok: {k: v for k, v in ov.items() if k != "results"}
                             for ok, ov in a_json["oracles"].items()}
        out = REPO / "experiments" / f"hard_regime_summary_{prov}.json"
        out.write_text(json.dumps(a_json, indent=2, default=str), encoding="utf-8")
        write_provider_md(prov, a, REPO / "reports" / f"HARD_REGIME_{prov.upper()}.md")
        print(f"wrote {out.relative_to(REPO)} and reports/HARD_REGIME_{prov.upper()}.md")

    # Joint regime map.
    if all_a:
        joint = {p: {
            "graph_query_cost_saving_pct": all_a[p]["graph_query_cost_saving_pct"],
            "graph_query_success_delta_pp": all_a[p]["graph_query_success_delta_pp"],
            "router_avg_cost": all_a[p]["oracles"]["query_router"]["aggregate"]["avg_cost"],
            "graph_avg_cost": all_a[p]["oracles"]["deliberation_graph"]["aggregate"]["avg_cost"],
            "router_success": all_a[p]["oracles"]["query_router"]["aggregate"]["success_rate"],
            "graph_success": all_a[p]["oracles"]["deliberation_graph"]["aggregate"]["success_rate"],
            "heuristic_avg_cost": all_a[p]["heuristic"]["aggregate"]["avg_cost"],
            "heuristic_success": all_a[p]["heuristic"]["aggregate"]["success_rate"],
            "unsupported_evidence_risk": all_a[p]["unsupported_evidence_risk"],
            "regime_map": all_a[p]["regime_map"],
            "n_tasks": all_a[p]["n_tasks"],
        } for p in all_a}
        (REPO / "experiments" / "hard_regime_summary_joint.json").write_text(
            json.dumps(joint, indent=2, default=str), encoding="utf-8")

        md = ["# HARD_REGIME_RESULTS\n\n",
              "Real-model evaluation on hard-regime pool (n=90 per provider, 4 model families).\n\n",
              "## Headline\n\n",
              "| provider | cheap | strong | router cost | graph cost | saving | succ_delta | router succ | graph succ |\n",
              "|---|---|---|---|---|---|---|---|---|\n"]
        for p, j in joint.items():
            cfg = PROVIDERS[p]
            md.append(f"| {p} | {cfg['cheap_model']} | {cfg['strong_model']} | {j['router_avg_cost']:.3f} | {j['graph_avg_cost']:.3f} | {j['graph_query_cost_saving_pct']:.2f}% | {j['graph_query_success_delta_pp']:.2f} pp | {j['router_success']:.3f} | {j['graph_success']:.3f} |\n")
        md.append("\n## Heuristic vs Oracle\n\n")
        md.append("| provider | heuristic_cost | heuristic_success | oracle graph cost | oracle graph success |\n|---|---|---|---|---|\n")
        for p, j in joint.items():
            md.append(f"| {p} | {j['heuristic_avg_cost']:.3f} | {j['heuristic_success']:.3f} | {j['graph_avg_cost']:.3f} | {j['graph_success']:.3f} |\n")
        (REPO / "reports" / "HARD_REGIME_RESULTS.md").write_text("".join(md), encoding="utf-8")

        # Regime map combined report.
        md2 = ["# HARD_REGIME_REGIME_MAP\n\n",
               "Per-(provider, family, regime) interpretation labels.\n\n"]
        for p, j in joint.items():
            md2.append(f"## {p}\n\n")
            md2.append("| family | regime | n | cheap | strong | router_cost | graph_cost | saving | unsup_risk | label |\n")
            md2.append("|---|---|---|---|---|---|---|---|---|---|\n")
            for cell in sorted(j["regime_map"], key=lambda c: (c["family"], c["regime"])):
                md2.append(
                    f"| {cell['family']} | {cell['regime']} | {cell['n']} | "
                    f"{cell['cheap_success']:.3f} | {cell['strong_success']:.3f} | "
                    f"{cell['router_cost']:.3f} | {cell['graph_cost']:.3f} | "
                    f"{cell['graph_saving_pct']:.2f}% | {cell['unsupported_risk']:.3f} | "
                    f"**{cell['interpretation']}** |\n"
                )
            md2.append("\n")
        (REPO / "reports" / "HARD_REGIME_REGIME_MAP.md").write_text("".join(md2), encoding="utf-8")

        # Oracle gap report.
        md3 = ["# HARD_REGIME_ORACLE_GAP\n\n",
               "Oracle deliberation graph vs oracle query router on hard regime pool.\n\n",
               "| provider | router avg_cost | graph avg_cost | saving | success_delta_pp |\n",
               "|---|---|---|---|---|\n"]
        for p, j in joint.items():
            md3.append(f"| {p} | {j['router_avg_cost']:.3f} | {j['graph_avg_cost']:.3f} | {j['graph_query_cost_saving_pct']:.2f}% | {j['graph_query_success_delta_pp']:.2f} |\n")
        md3.append("\nGO threshold: 30% cost saving OR 5 pp success delta. ")
        any_pass = any(
            j["graph_query_cost_saving_pct"] >= 30.0 or
            j["graph_query_success_delta_pp"] >= 5.0
            for j in joint.values()
        )
        md3.append(f"Any provider crosses GO threshold: **{any_pass}**\n")
        (REPO / "reports" / "HARD_REGIME_ORACLE_GAP.md").write_text("".join(md3), encoding="utf-8")
        print("wrote experiments/hard_regime_summary_joint.json and 3 reports")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
