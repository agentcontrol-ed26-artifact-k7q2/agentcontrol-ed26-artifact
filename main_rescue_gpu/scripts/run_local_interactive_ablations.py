"""Phase 5: ablations on the interactive pool.

Variants:
- no_observation: drop OBS action; oracle only sees cheap/strong cascades.
- no_partial_strong: drop strong_hint and strong_critique/checklist.
- no_repair: drop cheap_repair and cheap_repair_after_observation.
- no_strong: drop strong_answer.
- fixed_react_only: must use fixed plan [cheap, OBS, repair_after_obs].
- full_graph: all actions allowed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "src"))

from agentcontrol_main_rescue.interactive_oracle import (  # noqa: E402
    GRAPH_PLANS, QUERY_ROUTER_PLANS, resolve_plan, aggregate, best_plan,
    evaluate_plan, per_family, saving_pct,
)


PROVS = ("deepseek", "together")


def _filter(plan_dict, drop):
    """Drop plans whose action list contains any prefix in `drop`."""
    out = {}
    for n, actions in plan_dict.items():
        keep = True
        for a in actions:
            base = a[:-len("_if_needed")] if a.endswith("_if_needed") else a
            for d in drop:
                if base == d or base == "OBS" and d == "observation":
                    keep = False
                    break
            if not keep:
                break
        if keep:
            out[n] = actions
    return out


def _ablation_plan_set(name: str):
    if name == "full_graph":
        return GRAPH_PLANS
    if name == "no_observation":
        return _filter(GRAPH_PLANS, ["observation",
                                      "cheap_repair_after_observation",
                                      "run_tests", "run_code", "retrieve",
                                      "citation_check", "tool_observation",
                                      "checkpoint_check"])
    if name == "no_partial_strong":
        return _filter(GRAPH_PLANS, ["strong_hint", "strong_critique",
                                      "strong_checklist",
                                      "cheap_repair_after_strong_partial"])
    if name == "no_repair":
        return _filter(GRAPH_PLANS, ["cheap_repair",
                                      "cheap_repair_after_observation",
                                      "cheap_repair_after_strong_partial"])
    if name == "no_strong":
        return _filter(GRAPH_PLANS, ["strong_answer"])
    return GRAPH_PLANS


def main() -> int:
    out_all = {}
    for prov in PROVS:
        path = HERE / "experiments" / f"local_interactive_outcomes_{prov}.json"
        if not path.exists():
            continue
        outcomes = json.loads(path.read_text(encoding="utf-8"))
        full = aggregate([best_plan(GRAPH_PLANS, outcomes, tid) for tid in outcomes])
        rows = {}
        for ablation in ("full_graph", "no_observation", "no_partial_strong",
                         "no_repair", "no_strong"):
            plans = _ablation_plan_set(ablation)
            rs = [best_plan(plans, outcomes, tid) for tid in outcomes]
            rows[ablation] = aggregate(rs)
        # Reference: query router (cascade only).
        qr = aggregate([best_plan(QUERY_ROUTER_PLANS, outcomes, tid) for tid in outcomes])
        rows["query_router_baseline"] = qr
        out_all[prov] = rows

    md = ["# LOCAL_INTERACTIVE_ABLATIONS\n",
          "\nAll ablations on the same cached interactive outcomes; no re-collection.\n\n"]
    for prov, rows in out_all.items():
        full = rows["full_graph"]
        qr = rows["query_router_baseline"]
        md.append(f"## {prov}\n\n")
        md.append("| ablation | success | avg_cost | cost_delta_vs_full | succ_delta_vs_full |\n|---|---|---|---|---|\n")
        for name, r in rows.items():
            cost_delta = r["avg_cost"] - full["avg_cost"]
            succ_delta = (r["success_rate"] - full["success_rate"]) * 100.0
            md.append(f"| {name} | {r['success_rate']:.3f} | {r['avg_cost']:.3f} | {cost_delta:+.3f} | {succ_delta:+.2f} pp |\n")
        md.append(f"\n**Verdict (key check)**: removing observation should *cost more or lower success*. "
                  f"Compare `no_observation` vs `full_graph`: cost change = {(rows['no_observation']['avg_cost'] - full['avg_cost']):+.3f}, "
                  f"success change = {((rows['no_observation']['success_rate'] - full['success_rate']) * 100):+.2f} pp.\n\n")

    out_path = HERE / "experiments" / "local_interactive_ablations.json"
    out_path.write_text(json.dumps(out_all, indent=2, default=str), encoding="utf-8")
    (HERE / "reports" / "LOCAL_INTERACTIVE_ABLATIONS.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} and reports/LOCAL_INTERACTIVE_ABLATIONS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
