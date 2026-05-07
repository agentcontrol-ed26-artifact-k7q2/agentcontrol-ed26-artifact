"""F2 Phase 4: oracle / baselines / heuristic / per-difficulty / bootstrap on F2 outcomes."""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from agentcontrol_method.f2_data_analysis_tasks import get_pool  # noqa: E402

PROVS = ("deepseek", "together")

# Oracle plan sets. observation action mapped via "OBS" placeholder
# (data_analysis_code uses run_code).
QUERY_ROUTER_PLANS = {
    "qr_cheap_only": ["cheap_answer"],
    "qr_strong_only": ["strong_answer"],
    "qr_cascade": ["cheap_answer", "strong_answer_if_needed"],
    "qr_automix": ["cheap_answer", "cheap_repair_if_needed", "strong_answer_if_needed"],
}
# Observation-augmented router (FAIR baseline per Codex review pass 5).
QUERY_ROUTER_PLANS_WITH_OBS = {
    **QUERY_ROUTER_PLANS,
    "qr_cheap_obs": ["cheap_answer", "OBS", "cheap_repair_after_observation_if_needed"],
    "qr_cheap_obs_then_strong": ["cheap_answer", "OBS",
                                  "cheap_repair_after_observation_if_needed",
                                  "strong_answer_if_needed"],
}
GRAPH_PLANS = {
    **QUERY_ROUTER_PLANS_WITH_OBS,
    "graph_strong_hint": ["cheap_answer", "strong_hint_if_needed",
                           "cheap_repair_after_strong_partial_if_needed",
                           "strong_answer_if_needed"],
    "graph_obs_then_hint": ["cheap_answer", "OBS",
                              "cheap_repair_after_observation_if_needed",
                              "strong_hint_if_needed",
                              "cheap_repair_after_strong_partial_if_needed",
                              "strong_answer_if_needed"],
    "graph_full": ["cheap_answer", "cheap_repair_if_needed", "OBS",
                    "cheap_repair_after_observation_if_needed",
                    "strong_hint_if_needed",
                    "cheap_repair_after_strong_partial_if_needed",
                    "strong_answer_if_needed"],
}
ANSWER_ACTIONS = {"cheap_answer", "cheap_repair", "cheap_repair_after_observation",
                  "cheap_repair_after_strong_partial", "strong_answer"}
BUDGET = 25.0
COST_PENALTY = 0.005


def evaluate_plan(plan_name, actions, outcomes, task_id):
    obs_name = "run_code"  # data_analysis_code observation
    actions_resolved = [obs_name if a == "OBS" else a for a in actions]
    out = outcomes[task_id]
    success = False; cost = 0.0; lat = 0; risk = 0.0
    actions_run = []
    for a in actions_resolved:
        base = a[:-len("_if_needed")] if a.endswith("_if_needed") else a
        cond = a.endswith("_if_needed")
        if cond and success: continue
        oa = out.get(base)
        if oa is None: continue
        if cost + oa["cost"] > BUDGET: break
        cost += oa["cost"]; lat += oa["latency_ms"]
        risk = max(risk, oa.get("unsupported_risk", 0.0))
        if base in ANSWER_ACTIONS and oa.get("success"):
            success = True
        actions_run.append(base)
    obj = (1.0 if success else 0.0) - COST_PENALTY * cost
    return {"task_id": task_id, "plan_name": plan_name,
            "success": success, "cost": cost, "latency_ms": lat,
            "unsupported_risk": risk, "objective": obj,
            "actions_run": actions_run}


def best_plan(plans, outcomes, task_id):
    cands = [evaluate_plan(name, actions, outcomes, task_id)
             for name, actions in plans.items()]
    return max(cands, key=lambda r: (r["objective"], r["success"], -r["cost"]))


def aggregate(rs):
    n = max(1, len(rs))
    return {"n": len(rs),
            "success_rate": sum(r["success"] for r in rs) / n,
            "avg_cost": sum(r["cost"] for r in rs) / n,
            "avg_objective": sum(r["objective"] for r in rs) / n,
            "avg_unsupported_risk": sum(r["unsupported_risk"] for r in rs) / n}


def saving_pct(b, c):
    return 100.0 * (b - c) / b if b > 0 else 0.0


def main() -> int:
    pool = get_pool()
    out_all = {}
    for prov in PROVS:
        path = HERE / "experiments" / f"f2_data_analysis_outcomes_{prov}.json"
        if not path.exists():
            continue
        outcomes = json.loads(path.read_text(encoding="utf-8"))
        # Compute oracles vs observation-augmented router.
        qr = [best_plan(QUERY_ROUTER_PLANS, outcomes, tid) for tid in outcomes]
        qr_obs = [best_plan(QUERY_ROUTER_PLANS_WITH_OBS, outcomes, tid) for tid in outcomes]
        gr = [best_plan(GRAPH_PLANS, outcomes, tid) for tid in outcomes]
        qr_a = aggregate(qr)
        qr_obs_a = aggregate(qr_obs)
        gr_a = aggregate(gr)

        # Per-difficulty
        by_diff = defaultdict(list)
        for r in gr:
            by_diff[pool[r["task_id"]]["difficulty"]].append(r)
        gr_by_diff = {d: aggregate(rs) for d, rs in by_diff.items()}
        by_diff_qr = defaultdict(list)
        for r in qr_obs:
            by_diff_qr[pool[r["task_id"]]["difficulty"]].append(r)
        qr_obs_by_diff = {d: aggregate(rs) for d, rs in by_diff_qr.items()}

        # Baselines
        baselines = {
            "always_cheapest": ["cheap_answer"],
            "always_strongest": ["strong_answer"],
            "frugalgpt_cascade": ["cheap_answer", "strong_answer_if_needed"],
            "automix": ["cheap_answer", "cheap_repair_if_needed", "strong_answer_if_needed"],
            "shepherding_hint": ["cheap_answer", "strong_hint_if_needed",
                                 "cheap_repair_after_strong_partial_if_needed"],
            "fixed_react": ["cheap_answer", "OBS", "cheap_repair_after_observation_if_needed"],
            "fixed_react_with_strong_fallback": ["cheap_answer", "OBS",
                                                  "cheap_repair_after_observation_if_needed",
                                                  "strong_answer_if_needed"],
            "heuristic_bdelg": ["cheap_answer", "OBS",
                                 "cheap_repair_after_observation_if_needed",
                                 "strong_hint_if_needed",
                                 "cheap_repair_after_strong_partial_if_needed",
                                 "strong_answer_if_needed"],
        }
        bl = {}
        for name, plan in baselines.items():
            rs = [evaluate_plan(name, plan, outcomes, tid) for tid in outcomes]
            bl[name] = aggregate(rs)

        # Bootstrap
        ids = sorted(outcomes.keys())
        rng = random.Random(20260427)
        savings_vs_obs = []
        succ_deltas_vs_obs = []
        savings_vs_no_obs = []
        succ_deltas_vs_no_obs = []
        # Map task_id -> result for fast lookup
        qr_map = {r["task_id"]: r for r in qr}
        qr_obs_map = {r["task_id"]: r for r in qr_obs}
        gr_map = {r["task_id"]: r for r in gr}
        for _ in range(2000):
            sample = [rng.choice(ids) for _ in ids]
            qr_cost = statistics.fmean(qr_map[t]["cost"] for t in sample)
            qr_obs_cost = statistics.fmean(qr_obs_map[t]["cost"] for t in sample)
            gr_cost = statistics.fmean(gr_map[t]["cost"] for t in sample)
            qr_succ = statistics.fmean(int(qr_map[t]["success"]) for t in sample)
            qr_obs_succ = statistics.fmean(int(qr_obs_map[t]["success"]) for t in sample)
            gr_succ = statistics.fmean(int(gr_map[t]["success"]) for t in sample)
            savings_vs_obs.append(saving_pct(qr_obs_cost, gr_cost))
            succ_deltas_vs_obs.append(100.0 * (gr_succ - qr_obs_succ))
            savings_vs_no_obs.append(saving_pct(qr_cost, gr_cost))
            succ_deltas_vs_no_obs.append(100.0 * (gr_succ - qr_succ))

        def ci(s):
            ss = sorted(s); n = len(ss)
            return {"mean": statistics.fmean(s),
                    "ci_lo": ss[int(n*0.025)], "ci_hi": ss[int(n*0.975)]}

        out_all[prov] = {
            "n_tasks": len(outcomes),
            "qr_no_obs": qr_a, "qr_with_obs": qr_obs_a, "graph": gr_a,
            "graph_vs_router_with_obs": {
                "cost_saving_pct": saving_pct(qr_obs_a["avg_cost"], gr_a["avg_cost"]),
                "success_delta_pp": 100.0 * (gr_a["success_rate"] - qr_obs_a["success_rate"]),
                "ci_saving": ci(savings_vs_obs),
                "ci_succ_delta": ci(succ_deltas_vs_obs),
            },
            "graph_vs_router_no_obs": {
                "cost_saving_pct": saving_pct(qr_a["avg_cost"], gr_a["avg_cost"]),
                "success_delta_pp": 100.0 * (gr_a["success_rate"] - qr_a["success_rate"]),
                "ci_saving": ci(savings_vs_no_obs),
                "ci_succ_delta": ci(succ_deltas_vs_no_obs),
            },
            "per_difficulty_graph": gr_by_diff,
            "per_difficulty_router_with_obs": qr_obs_by_diff,
            "baselines": bl,
        }

    out_path = HERE / "experiments" / "f2_data_analysis_oracle_summary.json"
    out_path.write_text(json.dumps(out_all, indent=2, default=str), encoding="utf-8")

    md = ["# F2_DATA_ANALYSIS_RESULTS\n",
          "\nF2 extension of data_analysis_code from n=20 → n=50. Oracle deliberation graph vs **two** router baselines: query router WITHOUT observation (the original Codex-flagged baseline) and query router WITH observation (the fair Codex-required baseline).\n\n"]
    md.append("## Headline\n\n")
    md.append("| provider | n | qr (no obs) cost | qr (with obs) cost | graph cost | "
              "graph_vs_router(no_obs) | graph_vs_router(with_obs) | succ delta vs router_with_obs | bootstrap CI saving (vs with_obs) | bootstrap CI succ delta |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for prov, d in out_all.items():
        v = d["graph_vs_router_with_obs"]; v0 = d["graph_vs_router_no_obs"]
        md.append(f"| {prov} | {d['n_tasks']} | {d['qr_no_obs']['avg_cost']:.3f} | "
                  f"{d['qr_with_obs']['avg_cost']:.3f} | {d['graph']['avg_cost']:.3f} | "
                  f"{v0['cost_saving_pct']:.2f}% / +{v0['success_delta_pp']:.2f} pp | "
                  f"**{v['cost_saving_pct']:.2f}%** / **+{v['success_delta_pp']:.2f} pp** | "
                  f"{v['success_delta_pp']:.2f} pp | "
                  f"[{v['ci_saving']['ci_lo']:.2f}, {v['ci_saving']['ci_hi']:.2f}] | "
                  f"[{v['ci_succ_delta']['ci_lo']:.2f}, {v['ci_succ_delta']['ci_hi']:.2f}] |\n")

    md.append("\n## Per-difficulty (oracle deliberation graph vs oracle router-WITH-obs)\n\n")
    for prov, d in out_all.items():
        md.append(f"### {prov}\n\n")
        md.append("| difficulty | n | router(with obs) cost | graph cost | saving | router succ | graph succ |\n|---|---|---|---|---|---|---|\n")
        for diff in ("easy", "medium", "hard"):
            r = d["per_difficulty_router_with_obs"].get(diff)
            g = d["per_difficulty_graph"].get(diff)
            if r and g:
                md.append(f"| {diff} | {r['n']} | {r['avg_cost']:.3f} | {g['avg_cost']:.3f} | "
                          f"{saving_pct(r['avg_cost'], g['avg_cost']):.2f}% | "
                          f"{r['success_rate']:.3f} | {g['success_rate']:.3f} |\n")
        md.append("\n")

    md.append("\n## Baselines (aggregate)\n\n")
    for prov, d in out_all.items():
        md.append(f"### {prov}\n\n")
        md.append("| baseline | success | avg_cost |\n|---|---|---|\n")
        for n, b in d["baselines"].items():
            md.append(f"| {n} | {b['success_rate']:.3f} | {b['avg_cost']:.3f} |\n")
        md.append("\n")

    md.append("\n## Honest interpretation\n\n")
    md.append("Note: the strict Codex-required gate is **graph vs oracle query router WITH observation included** (column 6 above), NOT vs the no-observation router (column 5). The observation-enabled router is a fair baseline.\n")
    (HERE / "reports" / "F2_DATA_ANALYSIS_RESULTS.md").write_text("".join(md),
                                                                   encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} and reports/F2_DATA_ANALYSIS_RESULTS.md")
    for prov, d in out_all.items():
        v = d["graph_vs_router_with_obs"]
        ci_s = v["ci_saving"]; ci_d = v["ci_succ_delta"]
        print(f"  [{prov}] graph_vs_router_WITH_obs: {v['cost_saving_pct']:.2f}% / +{v['success_delta_pp']:.2f} pp; "
              f"CI saving [{ci_s['ci_lo']:.2f}, {ci_s['ci_hi']:.2f}]; CI succ_delta [{ci_d['ci_lo']:.2f}, {ci_d['ci_hi']:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
