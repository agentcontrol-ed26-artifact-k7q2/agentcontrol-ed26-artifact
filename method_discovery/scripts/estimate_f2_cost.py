"""F2 Phase 2: preflight API cost estimate."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from agentcontrol_method.f2_data_analysis_tasks import get_pool  # noqa: E402

# 8 LLM actions per task (matches main_rescue_gpu/run_local_interactive_collection).
LLM_ACTIONS = {
    "cheap_answer":                       (180, 100),
    "cheap_repair":                       (180, 100),
    "cheap_repair_after_observation":     (260, 120),
    "strong_hint":                        (180, 80),
    "cheap_repair_after_strong_partial":  (220, 120),
    "strong_critique":                    (180, 80),
    "strong_checklist":                   (180, 80),
    "strong_answer":                      (180, 800),
}
DEEPSEEK = {
    "cheap": ("deepseek-chat", 0.27, 1.10),
    "strong": ("deepseek-reasoner", 0.55, 2.19),
}
TOGETHER = {
    "cheap": ("Qwen/Qwen2.5-7B-Instruct-Turbo", 0.30, 0.30),
    "strong": ("meta-llama/Llama-3.3-70B-Instruct-Turbo", 0.88, 0.88),
}


def _est(prov_prices: dict) -> dict:
    cheap_n = strong_n = 0
    cheap_cost = strong_cost = 0.0
    for action, (in_t, out_t) in LLM_ACTIONS.items():
        if action.startswith("cheap_"):
            name, p_in, p_out = prov_prices["cheap"]
            cheap_n += 1
            cheap_cost += (in_t * p_in + out_t * p_out) / 1e6
        else:
            name, p_in, p_out = prov_prices["strong"]
            strong_n += 1
            strong_cost += (in_t * p_in + out_t * p_out) / 1e6
    return {"cheap_actions_per_task": cheap_n, "strong_actions_per_task": strong_n,
            "per_task_estimated_usd": cheap_cost + strong_cost,
            "cheap_per_task_usd": cheap_cost, "strong_per_task_usd": strong_cost}


def main() -> int:
    pool = get_pool()
    n_new = sum(1 for t in pool.values() if t["provenance"].endswith("extension-f2"))
    n_legacy = len(pool) - n_new
    out = {
        "n_tasks_total": len(pool),
        "n_new_tasks_likely_uncached": n_new,
        "n_legacy_tasks_likely_cached": n_legacy,
        "actions_per_task": len(LLM_ACTIONS),
        "providers": {},
        "max_spend_cap_usd": 5.0,
        "approval_env_var": "AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED",
        "approval_set": os.environ.get("AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED") == "1",
    }
    for prov_name, prices in (("deepseek", DEEPSEEK), ("together", TOGETHER)):
        e = _est(prices)
        out["providers"][prov_name] = {
            "cheap_model": prices["cheap"][0],
            "strong_model": prices["strong"][0],
            "per_task_estimated_usd": e["per_task_estimated_usd"],
            "estimated_total_usd_if_no_cache": e["per_task_estimated_usd"] * len(pool),
            "estimated_total_usd_after_cache": e["per_task_estimated_usd"] * n_new,
        }
    out_path = HERE / "experiments" / "f2_cost_estimate.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# F2_API_BUDGET_GATE\n",
          f"\n- tasks total: {len(pool)} (legacy {n_legacy} + new {n_new})\n",
          f"- actions per task: {len(LLM_ACTIONS)}\n",
          f"- approval env var set: **{out['approval_set']}**\n",
          f"- max spend cap (sprint): **${out['max_spend_cap_usd']:.2f}**\n",
          "\n## Per-provider estimate\n\n",
          "| provider | per task | total (no cache) | total (after cache hit on legacy 20) |\n",
          "|---|---|---|---|\n"]
    grand_total = 0.0
    for p, d in out["providers"].items():
        md.append(f"| {p} | ${d['per_task_estimated_usd']:.5f} | "
                  f"${d['estimated_total_usd_if_no_cache']:.4f} | "
                  f"${d['estimated_total_usd_after_cache']:.4f} |\n")
        grand_total += d["estimated_total_usd_after_cache"]
    md.append(f"\n**Grand total (after cache hits) for both providers**: ${grand_total:.4f}\n")
    md.append(f"**Status**: {'OPEN — proceeding' if out['approval_set'] else 'CLOSED — set AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED=1 to enable'}\n")
    if grand_total > out["max_spend_cap_usd"]:
        md.append(f"\n**WARN**: estimated total ${grand_total:.4f} exceeds cap ${out['max_spend_cap_usd']:.2f}. STOP and ask user.\n")
    (HERE / "reports" / "F2_API_BUDGET_GATE.md").write_text("".join(md), encoding="utf-8")
    print(f"grand total estimated (after cache hits): ${grand_total:.4f}")
    print(f"approval set: {out['approval_set']}")
    return 0 if out["approval_set"] or grand_total <= out["max_spend_cap_usd"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
