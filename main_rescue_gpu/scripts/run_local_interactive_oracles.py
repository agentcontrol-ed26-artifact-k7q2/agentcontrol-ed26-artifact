"""Phase 4: oracle / baseline / heuristic / regime analyses on interactive outcomes.

Defines oracle query router (with cascades) vs oracle deliberation graph that
includes observation actions. Computes saving + per-family + per-(family,
regime/conditional-success-pattern) breakdown.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "src"))

# Plans use action sequences. `_if_needed` short-circuits after success.
# Observation actions inject state that subsequent actions can use; the
# oracle plans choose whether to include them.

# Plan / oracle definitions live in agentcontrol_main_rescue.interactive_oracle
# to avoid drift across analysis scripts. This script imports the canonical
# definitions and only adds the per-script reporting glue.
from agentcontrol_main_rescue.interactive_oracle import (  # noqa: E402
    QUERY_ROUTER_PLANS, GRAPH_PLANS, FAMILY_OBS, ANSWER_ACTIONS,
    BUDGET, COST_PENALTY,
)


from agentcontrol_main_rescue.interactive_oracle import (  # noqa: E402
    aggregate, best_plan, evaluate_plan, per_family as _per_family,
    saving_pct as _saving_pct,
)


def _baselines(outcomes: dict):
    """Heuristic / fixed-pipeline baselines."""
    baselines = {
        "always_cheapest": ["cheap_answer"],
        "always_strongest": ["strong_answer"],
        "frugalgpt_cascade": ["cheap_answer", "strong_answer_if_needed"],
        "automix_self_verification_cascade": ["cheap_answer", "cheap_repair_if_needed",
                                              "strong_answer_if_needed"],
        "shepherding_hint": ["cheap_answer", "strong_hint_if_needed",
                             "cheap_repair_after_strong_partial_if_needed"],
        "fixed_react": ["cheap_answer", "OBS", "cheap_repair_after_observation_if_needed"],
        "fixed_retrieve_answer_verify": ["OBS", "cheap_answer", "cheap_repair_after_observation_if_needed"],
        "heuristic_bdelg": ["cheap_answer", "OBS", "cheap_repair_after_observation_if_needed",
                            "strong_hint_if_needed", "cheap_repair_after_strong_partial_if_needed",
                            "strong_answer_if_needed"],
    }
    out = {}
    for name, actions in baselines.items():
        rs = [evaluate_plan(name, actions, outcomes, tid) for tid in outcomes]
        out[name] = {"aggregate": aggregate(rs), "per_family": _per_family(rs),
                     "results": rs}
    return out


def _conditional_breakdown(qr_results, gr_results, outcomes):
    """Per-(family, cheap-success-pattern) breakdown to diagnose where graph wins."""
    by = defaultdict(list)
    for q, g in zip(qr_results, gr_results):
        tid = q["task_id"]
        ca = outcomes[tid].get("cheap_answer", {}).get("success", False)
        cra = outcomes[tid].get("cheap_repair_after_observation", {}).get("success", False)
        sa = outcomes[tid].get("strong_answer", {}).get("success", False)
        pattern = f"cheap={int(ca)}/obs_repair={int(cra)}/strong={int(sa)}"
        by[(q["family"], pattern)].append((q, g))
    out = {}
    for (fam, pat), pairs in by.items():
        qs = [p[0] for p in pairs]
        gs = [p[1] for p in pairs]
        out[f"{fam}/{pat}"] = {
            "n": len(pairs),
            "router_avg_cost": aggregate(qs)["avg_cost"],
            "graph_avg_cost": aggregate(gs)["avg_cost"],
            "router_success": aggregate(qs)["success_rate"],
            "graph_success": aggregate(gs)["success_rate"],
            "saving_pct": _saving_pct(aggregate(qs)["avg_cost"], aggregate(gs)["avg_cost"]),
        }
    return out


def main() -> int:
    PROVS = ("deepseek", "together")
    out_all = {}
    for prov in PROVS:
        path = HERE / "experiments" / f"local_interactive_outcomes_{prov}.json"
        if not path.exists():
            continue
        outcomes = json.loads(path.read_text(encoding="utf-8"))
        qr = [best_plan(QUERY_ROUTER_PLANS, outcomes, tid) for tid in outcomes]
        gr = [best_plan(GRAPH_PLANS, outcomes, tid) for tid in outcomes]
        baselines = _baselines(outcomes)
        cond = _conditional_breakdown(qr, gr, outcomes)
        qr_a = aggregate(qr)
        gr_a = aggregate(gr)
        a = {
            "provider": prov, "n_tasks": len(outcomes),
            "oracle_query_router": {"aggregate": qr_a, "per_family": _per_family(qr)},
            "oracle_deliberation_graph": {"aggregate": gr_a, "per_family": _per_family(gr)},
            "graph_query_cost_saving_pct": _saving_pct(qr_a["avg_cost"], gr_a["avg_cost"]),
            "graph_query_success_delta_pp": 100.0 * (gr_a["success_rate"] - qr_a["success_rate"]),
            "baselines": {n: {"aggregate": b["aggregate"], "per_family": b["per_family"]}
                          for n, b in baselines.items()},
            "conditional_breakdown": cond,
        }
        out_all[prov] = a
        out_path = HERE / "experiments" / f"local_interactive_summary_{prov}.json"
        out_path.write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")

        md = [f"# LOCAL_INTERACTIVE — {prov}\n",
              f"\n- n_tasks: {a['n_tasks']}\n",
              f"- oracle query router: success {qr_a['success_rate']:.3f}, cost {qr_a['avg_cost']:.3f}\n",
              f"- oracle deliberation graph: success {gr_a['success_rate']:.3f}, cost {gr_a['avg_cost']:.3f}\n",
              f"- **graph-vs-router cost saving: {a['graph_query_cost_saving_pct']:.2f}%**\n",
              f"- **graph-vs-router success delta: {a['graph_query_success_delta_pp']:.2f} pp**\n",
              "\n## Per-family\n\n",
              "| family | router cost | graph cost | saving | router succ | graph succ |\n|---|---|---|---|---|---|\n"]
        for fam in ("code_debug_interactive", "data_analysis_code",
                    "evidence_multihop_local", "tool_planning_deterministic",
                    "math_checkpoint"):
            r = a["oracle_query_router"]["per_family"].get(fam)
            g = a["oracle_deliberation_graph"]["per_family"].get(fam)
            if r and g:
                s = _saving_pct(r["avg_cost"], g["avg_cost"])
                md.append(f"| {fam} | {r['avg_cost']:.3f} | {g['avg_cost']:.3f} | {s:.2f}% | {r['success_rate']:.3f} | {g['success_rate']:.3f} |\n")

        md.append("\n## Baselines (aggregate)\n\n| baseline | success | avg_cost |\n|---|---|---|\n")
        for n, b in a["baselines"].items():
            ag = b["aggregate"]
            md.append(f"| {n} | {ag['success_rate']:.3f} | {ag['avg_cost']:.3f} |\n")

        md.append("\n## Conditional breakdown (cheap_succ / obs_repair_succ / strong_succ)\n\n")
        md.append("| pattern | n | router_cost | graph_cost | saving | router_succ | graph_succ |\n|---|---|---|---|---|---|---|\n")
        for key, c in sorted(cond.items()):
            md.append(f"| {key} | {c['n']} | {c['router_avg_cost']:.3f} | {c['graph_avg_cost']:.3f} | {c['saving_pct']:.2f}% | {c['router_success']:.3f} | {c['graph_success']:.3f} |\n")

        report = HERE / "reports" / f"LOCAL_INTERACTIVE_{prov.upper()}.md"
        report.write_text("".join(md), encoding="utf-8")
        print(f"wrote {out_path.relative_to(REPO)} and {report.relative_to(REPO)}")
        print(f"  [{prov}] graph saving {a['graph_query_cost_saving_pct']:.2f}% / success delta {a['graph_query_success_delta_pp']:.2f} pp")

    # Joint comparison.
    if out_all:
        joint_md = ["# LOCAL_INTERACTIVE_ORACLE_GAP\n\n"]
        joint_md.append("| provider | n | router_cost | graph_cost | saving | router_succ | graph_succ | succ_delta |\n|---|---|---|---|---|---|---|---|\n")
        for prov, a in out_all.items():
            qr = a["oracle_query_router"]["aggregate"]
            gr = a["oracle_deliberation_graph"]["aggregate"]
            joint_md.append(f"| {prov} | {a['n_tasks']} | {qr['avg_cost']:.3f} | {gr['avg_cost']:.3f} | {a['graph_query_cost_saving_pct']:.2f}% | {qr['success_rate']:.3f} | {gr['success_rate']:.3f} | {a['graph_query_success_delta_pp']:.2f} pp |\n")
        joint_md.append("\nGO threshold: ≥30% cost saving OR ≥5 pp success delta. ")
        any_pass = any(
            a["graph_query_cost_saving_pct"] >= 30.0 or
            a["graph_query_success_delta_pp"] >= 5.0
            for a in out_all.values()
        )
        joint_md.append(f"Any provider crosses GO: **{any_pass}**\n")
        (HERE / "reports" / "LOCAL_INTERACTIVE_ORACLE_GAP.md").write_text(
            "".join(joint_md), encoding="utf-8")
        (HERE / "experiments" / "local_interactive_summary_joint.json").write_text(
            json.dumps({k: {"graph_query_cost_saving_pct": v["graph_query_cost_saving_pct"],
                            "graph_query_success_delta_pp": v["graph_query_success_delta_pp"],
                            "router": v["oracle_query_router"]["aggregate"],
                            "graph": v["oracle_deliberation_graph"]["aggregate"],
                            "n_tasks": v["n_tasks"]}
                       for k, v in out_all.items()}, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
