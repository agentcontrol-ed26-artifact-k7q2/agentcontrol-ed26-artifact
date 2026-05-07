"""Cached-only E&D ablations.

Runs A) verifier-aware vs no-verifier, B) partial-strong, C) repair-action,
D) action-set ablations against ``experiments/smoke_outcomes.json`` only.
No provider calls. No dataset expansion. Decision unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    PLAN_AUTOMIX,
    PLAN_HEURISTIC_BDELG,
    PLAN_SHEPHERDING,
    QUERY_ROUTER_PLANS,
    aggregate,
    best_plan_per_task,
    load_outcomes,
    per_family,
    run_plan_over_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
BUDGET = 20.0
COST_PENALTY = 0.01


def _saving_pct(base: float, cur: float) -> float:
    return 100.0 * (base - cur) / base if base > 0 else 0.0


def ablation_a_verifier(outcomes: dict) -> dict:
    """A. verifier-aware vs no-verifier for cascades and oracle graph."""
    out = {}
    for name, plan in {
        "automix_self_verification_cascade": PLAN_AUTOMIX,
        "shepherding_hint": PLAN_SHEPHERDING,
        "heuristic_bdelg": PLAN_HEURISTIC_BDELG,
    }.items():
        va = run_plan_over_outcomes(name, plan, outcomes, verifier_aware=True,
                                    budget=BUDGET, cost_penalty=COST_PENALTY)
        nv = run_plan_over_outcomes(name, plan, outcomes, verifier_aware=False,
                                    budget=BUDGET, cost_penalty=COST_PENALTY)
        agg_va = aggregate(va)
        agg_nv = aggregate(nv)
        out[name] = {
            "verifier_aware": agg_va,
            "no_verifier": agg_nv,
            "cost_saving_due_to_verifier_pct": _saving_pct(agg_nv["avg_cost"], agg_va["avg_cost"]),
            "per_family_verifier_aware": per_family(va),
            "per_family_no_verifier": per_family(nv),
        }

    # Oracle deliberation graph: verifier-aware vs no-verifier (the latter
    # cannot use ``_if_needed`` short-circuit, so per-task cost is the full
    # plan cost; the oracle still picks the best cheapest fixed plan).
    g_va = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                              budget=BUDGET, cost_penalty=COST_PENALTY)
    g_nv = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=False,
                              budget=BUDGET, cost_penalty=COST_PENALTY)
    out["oracle_deliberation_graph"] = {
        "verifier_aware": aggregate(g_va),
        "no_verifier": aggregate(g_nv),
        "cost_saving_due_to_verifier_pct": _saving_pct(
            aggregate(g_nv)["avg_cost"], aggregate(g_va)["avg_cost"]
        ),
        "per_family_verifier_aware": per_family(g_va),
        "per_family_no_verifier": per_family(g_nv),
    }
    return out


def ablation_b_partial_strong(outcomes: dict) -> dict:
    """B. partial-strong (strong_hint) ablation."""
    full_actions = {"cheap_answer", "cheap_repair", "strong_hint",
                    "cheap_repair_after_hint", "strong_answer"}
    no_hint_actions = full_actions - {"strong_hint", "cheap_repair_after_hint"}
    only_full_strong_actions = {"cheap_answer", "strong_answer"}

    g_full = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                                budget=BUDGET, cost_penalty=COST_PENALTY,
                                allowed_actions=full_actions)
    g_no_hint = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                                   budget=BUDGET, cost_penalty=COST_PENALTY,
                                   allowed_actions=no_hint_actions)
    g_full_strong_only = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                                            budget=BUDGET, cost_penalty=COST_PENALTY,
                                            allowed_actions=only_full_strong_actions)

    h_full = run_plan_over_outcomes("heuristic_bdelg_full",
                                    PLAN_HEURISTIC_BDELG, outcomes,
                                    verifier_aware=True, budget=BUDGET,
                                    cost_penalty=COST_PENALTY)
    # Heuristic without strong_hint: skip the hint & repair-after-hint steps.
    plan_no_hint = [a for a in PLAN_HEURISTIC_BDELG
                    if "strong_hint" not in a and "cheap_repair_after_hint" not in a]
    h_no_hint = run_plan_over_outcomes("heuristic_bdelg_no_hint",
                                       plan_no_hint, outcomes,
                                       verifier_aware=True, budget=BUDGET,
                                       cost_penalty=COST_PENALTY)

    return {
        "oracle_graph_with_strong_hint": {
            "aggregate": aggregate(g_full),
            "per_family": per_family(g_full),
        },
        "oracle_graph_without_strong_hint": {
            "aggregate": aggregate(g_no_hint),
            "per_family": per_family(g_no_hint),
        },
        "oracle_graph_full_strong_only": {
            "aggregate": aggregate(g_full_strong_only),
            "per_family": per_family(g_full_strong_only),
        },
        "heuristic_bdelg_with_hint": {
            "aggregate": aggregate(h_full),
            "per_family": per_family(h_full),
        },
        "heuristic_bdelg_no_hint": {
            "aggregate": aggregate(h_no_hint),
            "per_family": per_family(h_no_hint),
        },
        "_note": "code n=4; partial-strong evidence remains underpowered for any method claim.",
    }


def ablation_c_repair(outcomes: dict) -> dict:
    """C. repair-action ablation."""
    full = {"cheap_answer", "cheap_repair", "strong_hint",
            "cheap_repair_after_hint", "strong_answer"}
    no_repair = full - {"cheap_repair"}
    no_repair_after_hint = full - {"cheap_repair_after_hint"}
    only_two = {"cheap_answer", "strong_answer"}

    cells = {
        "graph_full": full,
        "graph_no_cheap_repair": no_repair,
        "graph_no_cheap_repair_after_hint": no_repair_after_hint,
        "graph_only_cheap_and_strong": only_two,
    }
    out = {}
    for name, allowed in cells.items():
        results = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                                     budget=BUDGET, cost_penalty=COST_PENALTY,
                                     allowed_actions=allowed)
        out[name] = {
            "aggregate": aggregate(results),
            "per_family": per_family(results),
        }
    return out


def ablation_d_action_set(outcomes: dict) -> dict:
    """D. action-set ablation table."""
    sets = {
        "query_only": {"cheap_answer", "strong_answer"},
        "cascade_only": {"cheap_answer", "strong_answer", "cheap_repair"},
        "graph_no_hint": {"cheap_answer", "cheap_repair", "strong_answer"},
        "graph_no_repair": {"cheap_answer", "strong_hint",
                            "cheap_repair_after_hint", "strong_answer"},
        "graph_full": {"cheap_answer", "cheap_repair", "strong_hint",
                       "cheap_repair_after_hint", "strong_answer"},
    }
    # Reference: oracle query router for cost-saving baseline.
    qr = best_plan_per_task(QUERY_ROUTER_PLANS, outcomes, verifier_aware=True,
                            budget=BUDGET, cost_penalty=COST_PENALTY)
    qr_cost = aggregate(qr)["avg_cost"]

    out = {}
    for name, allowed in sets.items():
        rs = best_plan_per_task(GRAPH_PLANS, outcomes, verifier_aware=True,
                                budget=BUDGET, cost_penalty=COST_PENALTY,
                                allowed_actions=allowed)
        agg = aggregate(rs)
        out[name] = {
            "aggregate": agg,
            "cost_saving_vs_oracle_query_router_pct": _saving_pct(qr_cost, agg["avg_cost"]),
            "per_family": per_family(rs),
        }
    out["_oracle_query_router_avg_cost"] = qr_cost
    return out


def _md_table_cascade(name: str, block: dict) -> str:
    va = block["verifier_aware"]
    nv = block["no_verifier"]
    return (
        f"### {name}\n\n"
        f"| variant | success | avg_cost | avg_objective |\n"
        f"|---|---|---|---|\n"
        f"| verifier_aware | {va['success_rate']:.3f} | {va['avg_cost']:.3f} | {va['avg_objective']:.4f} |\n"
        f"| no_verifier    | {nv['success_rate']:.3f} | {nv['avg_cost']:.3f} | {nv['avg_objective']:.4f} |\n\n"
        f"cost_saving_due_to_verifier = **{block['cost_saving_due_to_verifier_pct']:.2f}%**\n"
    )


def write_markdown(out: dict, path: Path) -> None:
    a, b, c, d = out["A_verifier"], out["B_partial_strong"], out["C_repair"], out["D_action_set"]
    parts = ["# ED_CACHED_ABLATIONS\n",
             "\nDECISION: BACKUP / E&D-only (unchanged). Cached-only ablations on n=28.\n",
             "\n## A. Verifier-aware vs no-verifier\n\n",
             "Verifier-aware policies short-circuit `_if_needed` actions on success; "
             "no-verifier policies execute the full plan unconditionally.\n\n"]
    for name, block in a.items():
        parts.append(_md_table_cascade(name, block))
    parts.append("\n## B. Partial-strong (strong_hint) ablation\n\n")
    parts.append("| variant | success | avg_cost | avg_objective |\n|---|---|---|---|\n")
    for k in ["oracle_graph_with_strong_hint", "oracle_graph_without_strong_hint",
              "oracle_graph_full_strong_only", "heuristic_bdelg_with_hint",
              "heuristic_bdelg_no_hint"]:
        agg = b[k]["aggregate"]
        parts.append(f"| {k} | {agg['success_rate']:.3f} | {agg['avg_cost']:.3f} | {agg['avg_objective']:.4f} |\n")
    parts.append(f"\n*Note: {b['_note']}*\n")
    # Per-family code highlight.
    parts.append("\n### Code-family detail (n=4)\n\n")
    parts.append("| variant | success | avg_cost |\n|---|---|---|\n")
    for k in ["oracle_graph_with_strong_hint", "oracle_graph_without_strong_hint",
              "oracle_graph_full_strong_only"]:
        f = b[k]["per_family"].get("code", {})
        if f:
            parts.append(f"| {k} | {f['success_rate']:.3f} | {f['avg_cost']:.3f} |\n")

    parts.append("\n## C. Repair-action ablation\n\n")
    parts.append("| variant | success | avg_cost | avg_objective |\n|---|---|---|---|\n")
    for k, v in c.items():
        agg = v["aggregate"]
        parts.append(f"| {k} | {agg['success_rate']:.3f} | {agg['avg_cost']:.3f} | {agg['avg_objective']:.4f} |\n")

    parts.append("\n## D. Action-set ablation\n\n")
    parts.append(f"Reference oracle-query-router avg_cost = {d['_oracle_query_router_avg_cost']:.3f}\n\n")
    parts.append("| action_set | success | avg_cost | cost_saving_vs_router_pct |\n|---|---|---|---|\n")
    for k, v in d.items():
        if k.startswith("_"):
            continue
        agg = v["aggregate"]
        parts.append(f"| {k} | {agg['success_rate']:.3f} | {agg['avg_cost']:.3f} | {v['cost_saving_vs_oracle_query_router_pct']:.2f}% |\n")

    parts.append("\n## Caveats\n")
    parts.append("- All numbers come from cached outcomes on n=28 (math=20, code=4, evidence=4).\n")
    parts.append("- Code family is severely underpowered (n=4); no method claim is supported.\n")
    parts.append("- Decision remains BACKUP / E&D-only / DO-NOT-SCALE.\n")

    path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    outcomes = load_outcomes(str(REPO / "experiments" / "smoke_outcomes.json"))
    out = {
        "budget": BUDGET,
        "cost_penalty": COST_PENALTY,
        "n_total": len(outcomes),
        "A_verifier": ablation_a_verifier(outcomes),
        "B_partial_strong": ablation_b_partial_strong(outcomes),
        "C_repair": ablation_c_repair(outcomes),
        "D_action_set": ablation_d_action_set(outcomes),
    }
    json_path = REPO / "experiments" / "ed_cached_ablations.json"
    md_path = REPO / "reports" / "ED_CACHED_ABLATIONS.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown(out, md_path)
    print(f"wrote {json_path.relative_to(REPO)} and {md_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
