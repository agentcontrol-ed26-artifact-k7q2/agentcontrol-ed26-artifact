"""Phase 4b: Baseline panel on the rescue pool."""
from __future__ import annotations

import json
from pathlib import Path

from agentcontrol.ed_sim import (
    PLAN_AUTOMIX,
    PLAN_FRUGALGPT,
    PLAN_HEURISTIC_BDELG,
    PLAN_SHEPHERDING,
    PLANS_CHEAP_ONLY,
    PLANS_STRONG_ONLY,
    aggregate,
    family_of,
    load_outcomes,
    run_plan_over_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
BUDGET = 20.0
COST_PENALTY = 0.01

# Two extra fixed-pipeline baselines mentioned in the spec.
PLAN_FIXED_REACT = ["cheap_answer", "strong_critique", "cheap_repair"]
PLAN_FIXED_RETRIEVE_ANSWER_VERIFY = ["cheap_answer", "strong_checklist",
                                      "cheap_repair_after_hint"]


def _fam(tid: str) -> str:
    if tid.startswith("rm"): return "math"
    if tid.startswith("rc"): return "code"
    if tid.startswith("re"): return "evidence"
    if tid.startswith("rt"): return "tool_use"
    return family_of(tid)


def _per_family(rs):
    by = {}
    for r in rs:
        by.setdefault(_fam(r["task_id"]), []).append(r)
    return {f: aggregate(v) for f, v in by.items()}


def main() -> int:
    outcomes = load_outcomes(str(REPO / "experiments" / "rescue_outcomes.json"))
    panel = {
        "always_cheapest": PLANS_CHEAP_ONLY,
        "always_strongest": PLANS_STRONG_ONLY,
        "frugalgpt_cascade": PLAN_FRUGALGPT,
        "automix_self_verification_cascade": PLAN_AUTOMIX,
        "shepherding_hint": PLAN_SHEPHERDING,
        "fixed_react": PLAN_FIXED_REACT,
        "fixed_retrieve_answer_verify": PLAN_FIXED_RETRIEVE_ANSWER_VERIFY,
    }
    out = {"non_decisive": True, "n_total": len(outcomes), "budget": BUDGET,
           "cost_penalty": COST_PENALTY, "baselines": {}}
    for name, plan in panel.items():
        rs = run_plan_over_outcomes(name, plan, outcomes, verifier_aware=True,
                                    budget=BUDGET, cost_penalty=COST_PENALTY)
        out["baselines"][name] = {
            "aggregate": aggregate(rs),
            "per_family": _per_family(rs),
        }

    out_json = REPO / "experiments" / "rescue_baselines_summary.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# RESCUE_BASELINES\n",
          "\n**Status:** synthetic-local pool, n=170. **NON-DECISIVE.**\n",
          "\n## Aggregate\n\n",
          "| baseline | success | avg_cost | avg_objective |\n|---|---|---|---|\n"]
    for name, b in out["baselines"].items():
        a = b["aggregate"]
        md.append(f"| {name} | {a['success_rate']:.3f} | {a['avg_cost']:.3f} | {a['avg_objective']:.4f} |\n")
    md.append("\n## Per family — avg_cost (success in parentheses)\n\n")
    md.append("| baseline | math | code | evidence | tool_use |\n|---|---|---|---|---|\n")
    for name, b in out["baselines"].items():
        cells = []
        for fam in ("math", "code", "evidence", "tool_use"):
            f = b["per_family"].get(fam)
            if f:
                cells.append(f"{f['avg_cost']:.3f} ({f['success_rate']:.3f})")
            else:
                cells.append("—")
        md.append(f"| {name} | " + " | ".join(cells) + " |\n")

    out_md = REPO / "reports" / "RESCUE_BASELINES.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO)} and {out_md.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
