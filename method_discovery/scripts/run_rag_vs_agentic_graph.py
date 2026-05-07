"""Phase 6/7 (F1): oracle query router (with retrieval) vs oracle agentic graph
on the frozen multi-hop corpus.
"""
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

from agentcontrol_method.agentic_corpus import get_questions  # noqa: E402

PROVS = ("deepseek", "together")

# Plan format: list of action names. Each action is one cell in the cached
# outcomes JSON. `_if_needed` short-circuits after a successful answer.
QUERY_ROUTER_PLANS_RETRIEVAL = {
    "qr_cheap_cold": ["cheap_cold"],
    "qr_topk1_cheap": ["cheap_topk1"],
    "qr_topk3_cheap": ["cheap_topk3"],
    "qr_topk5_cheap": ["cheap_topk5"],
    "qr_topk5_strong": ["strong_topk5"],
    "qr_topk5_then_strong": ["cheap_topk5", "strong_topk5_if_needed"],
    "qr_topk3_then_topk5": ["cheap_topk3", "cheap_topk5_if_needed"],
}
AGENTIC_GRAPH_PLANS = {
    **QUERY_ROUTER_PLANS_RETRIEVAL,
    "graph_agentic_cheap": ["cheap_agentic"],
    "graph_agentic_then_strong": ["cheap_agentic", "strong_agentic_if_needed"],
    "graph_topk5_then_agentic": ["cheap_topk5", "cheap_agentic_if_needed",
                                  "strong_agentic_if_needed"],
    "graph_full": ["cheap_topk3", "cheap_agentic_if_needed",
                    "cheap_topk5_if_needed", "strong_topk5_if_needed",
                    "strong_agentic_if_needed"],
}

ANSWER_ACTIONS = {"cheap_cold", "cheap_topk1", "cheap_topk3", "cheap_topk5",
                  "cheap_agentic", "strong_topk5", "strong_agentic"}

BUDGET = 30.0
COST_PENALTY = 0.005


def evaluate_plan(plan_name, actions, outcomes, task_id):
    out = outcomes[task_id]
    success = False
    cost = 0.0
    lat = 0
    risk = 0.0
    actions_run = []
    for a in actions:
        base = a[:-len("_if_needed")] if a.endswith("_if_needed") else a
        conditional = a.endswith("_if_needed")
        if conditional and success:
            continue
        oa = out.get(base)
        if oa is None:
            continue
        if cost + oa["cost"] > BUDGET:
            break
        cost += oa["cost"]
        lat += oa["latency_ms"]
        risk = max(risk, oa.get("unsupported_risk", 0.0))
        if base in ANSWER_ACTIONS and oa.get("success"):
            success = True
        actions_run.append(base)
    obj = (1.0 if success else 0.0) - COST_PENALTY * cost
    return {"task_id": task_id, "plan_name": plan_name, "success": success,
            "cost": cost, "latency_ms": lat, "unsupported_risk": risk,
            "objective": obj, "actions_run": actions_run}


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


def per_difficulty(rs, questions_by_id):
    by = defaultdict(list)
    for r in rs:
        d = questions_by_id[r["task_id"]]["difficulty"]
        by[d].append(r)
    return {d: aggregate(v) for d, v in by.items()}


def main() -> int:
    questions = {q["task_id"]: q for q in get_questions()}
    out_all = {}
    for prov in PROVS:
        path = HERE / "experiments" / f"agentic_search_outcomes_{prov}.json"
        if not path.exists():
            continue
        outcomes = json.loads(path.read_text(encoding="utf-8"))
        qr = [best_plan(QUERY_ROUTER_PLANS_RETRIEVAL, outcomes, tid) for tid in outcomes]
        gr = [best_plan(AGENTIC_GRAPH_PLANS, outcomes, tid) for tid in outcomes]
        qr_a = aggregate(qr)
        gr_a = aggregate(gr)

        # bootstrap
        ids = list(outcomes.keys())
        rng = random.Random(20260427)
        savings = []
        succ_deltas = []
        for _ in range(2000):
            sample = [rng.choice(ids) for _ in ids]
            qr_cost = statistics.fmean(next(r for r in qr if r["task_id"] == t)["cost"] for t in sample)
            gr_cost = statistics.fmean(next(r for r in gr if r["task_id"] == t)["cost"] for t in sample)
            qr_succ = statistics.fmean(int(next(r for r in qr if r["task_id"] == t)["success"]) for t in sample)
            gr_succ = statistics.fmean(int(next(r for r in gr if r["task_id"] == t)["success"]) for t in sample)
            savings.append(saving_pct(qr_cost, gr_cost))
            succ_deltas.append(100.0 * (gr_succ - qr_succ))
        s_sorted = sorted(savings)
        d_sorted = sorted(succ_deltas)

        out_all[prov] = {
            "n_tasks": len(outcomes),
            "qr": qr_a, "gr": gr_a,
            "graph_query_cost_saving_pct": saving_pct(qr_a["avg_cost"], gr_a["avg_cost"]),
            "graph_query_success_delta_pp": 100.0 * (gr_a["success_rate"] - qr_a["success_rate"]),
            "bootstrap_saving_ci": {
                "mean": statistics.fmean(savings),
                "ci_lo": s_sorted[50], "ci_hi": s_sorted[1949],
            },
            "bootstrap_succ_delta_ci": {
                "mean": statistics.fmean(succ_deltas),
                "ci_lo": d_sorted[50], "ci_hi": d_sorted[1949],
            },
            "qr_per_difficulty": per_difficulty(qr, questions),
            "gr_per_difficulty": per_difficulty(gr, questions),
            "qr_results": qr,
            "gr_results": gr,
        }

    out_path = HERE / "experiments" / "agentic_search_oracle_summary.json"
    out_path.write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("qr_results", "gr_results")}
                                    for k, v in out_all.items()}, indent=2, default=str), encoding="utf-8")

    md = ["# AGENTIC_SEARCH_RESULTS\n",
          "\nAgentic deliberation graph (with iterative retrieve) vs oracle query router (with fixed top-k RAG cascades) on a frozen 60-question multi-hop corpus with deliberate distractors.\n\n"]
    md.append("## Headline\n\n")
    md.append("| provider | n | router cost | graph cost | saving | router succ | graph succ | succ delta | saving 95% CI |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for prov, d in out_all.items():
        ci = d["bootstrap_saving_ci"]
        md.append(f"| {prov} | {d['n_tasks']} | {d['qr']['avg_cost']:.3f} | {d['gr']['avg_cost']:.3f} | "
                  f"{d['graph_query_cost_saving_pct']:.2f}% | {d['qr']['success_rate']:.3f} | {d['gr']['success_rate']:.3f} | "
                  f"{d['graph_query_success_delta_pp']:.2f} pp | "
                  f"[{ci['ci_lo']:.2f}%, {ci['ci_hi']:.2f}%] |\n")
    md.append("\n## Per-difficulty\n\n")
    for prov, d in out_all.items():
        md.append(f"### {prov}\n\n")
        md.append("| difficulty | n | router cost | graph cost | saving | router succ | graph succ |\n|---|---|---|---|---|---|---|\n")
        for diff in ("multihop", "single_hop"):
            r = d["qr_per_difficulty"].get(diff)
            g = d["gr_per_difficulty"].get(diff)
            if r and g:
                md.append(f"| {diff} | {r['n']} | {r['avg_cost']:.3f} | {g['avg_cost']:.3f} | "
                          f"{saving_pct(r['avg_cost'], g['avg_cost']):.2f}% | {r['success_rate']:.3f} | {g['success_rate']:.3f} |\n")
        md.append("\n")
    md.append("\n## Interpretation\n\n")
    any_pass = any(d["graph_query_cost_saving_pct"] >= 25.0 or d["graph_query_success_delta_pp"] >= 5.0
                   for d in out_all.values())
    md.append(f"any provider crosses GO (≥25% cost OR ≥5pp success): **{any_pass}**\n\n")
    md.append("If GO is not crossed, the result is: **fixed top-k RAG with cheap-then-strong cascade is sufficient on a controlled multi-hop closed corpus; agentic iterative retrieve adds no measurable value beyond fixed RAG cascades.** This is methodologically valuable as a negative finding and strengthens the E&D paper's regime-mapping framing.\n")

    (HERE / "reports" / "AGENTIC_SEARCH_RESULTS.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} and reports/AGENTIC_SEARCH_RESULTS.md")
    for prov, d in out_all.items():
        ci = d["bootstrap_saving_ci"]
        print(f"  [{prov}] saving {d['graph_query_cost_saving_pct']:.2f}% / +{d['graph_query_success_delta_pp']:.2f} pp; "
              f"saving CI [{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
