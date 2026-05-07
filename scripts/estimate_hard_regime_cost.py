"""Phase 2 (hard regime): preflight cost estimate + budget gate report."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentcontrol.hard_regime_tasks import get_pool  # noqa: E402

ACTIONS = [
    "cheap_answer", "cheap_repair", "strong_hint", "cheap_repair_after_hint",
    "strong_critique", "strong_checklist", "strong_answer",
]
# Per-action token estimates (input, output) by role.
EST_TOKENS = {
    "cheap_answer": (180, 60),
    "cheap_repair": (180, 60),
    "strong_hint": (180, 80),
    "cheap_repair_after_hint": (200, 100),
    "strong_critique": (180, 80),
    "strong_checklist": (180, 80),
    "strong_answer": (180, 800),  # may include thinking trace
}
# Provider price tables (USD per 1M tokens) — match real_providers.py.
DEEPSEEK = {
    "cheap": ("deepseek-chat", 0.27, 1.10),
    "strong": ("deepseek-reasoner", 0.55, 2.19),
}
TOGETHER = {
    "cheap": ("Qwen/Qwen2.5-7B-Instruct-Turbo", 0.30, 0.30),
    "strong": ("meta-llama/Llama-3.3-70B-Instruct-Turbo", 0.88, 0.88),
}


def _est(role_actions, prices):
    name, p_in, p_out = prices
    total = 0.0
    calls = 0
    for action in role_actions:
        ti, to = EST_TOKENS[action]
        total += (ti * p_in + to * p_out) / 1e6
        calls += 1
    return name, calls, total


def main() -> int:
    pool = get_pool()
    n_tasks = len(pool)
    cheap_actions = ["cheap_answer", "cheap_repair", "cheap_repair_after_hint"]
    strong_actions = ["strong_hint", "strong_critique", "strong_checklist", "strong_answer"]

    estimates = {}
    for prov, prices in (("deepseek", DEEPSEEK), ("together", TOGETHER)):
        cn, cc, cp = _est(cheap_actions, prices["cheap"])
        sn, sc, sp = _est(strong_actions, prices["strong"])
        per_task_calls = cc + sc
        per_task_cost = cp + sp
        estimates[prov] = {
            "cheap_model": cn, "strong_model": sn,
            "calls_per_task": per_task_calls,
            "total_calls": per_task_calls * n_tasks,
            "estimated_total_usd": per_task_cost * n_tasks,
            "per_task_estimated_usd": per_task_cost,
        }
    grand_total = sum(e["estimated_total_usd"] for e in estimates.values())

    out = {
        "n_tasks": n_tasks,
        "actions_per_task": len(ACTIONS),
        "estimates": estimates,
        "grand_total_estimated_usd": grand_total,
        "max_spend_cap_usd_per_provider": 10.0,
        "approval_env_var": "AGENTCONTROL_HARD_REGIME_APPROVED",
        "approval_set": os.environ.get("AGENTCONTROL_HARD_REGIME_APPROVED") == "1",
    }
    out_path = REPO / "experiments" / "hard_regime_cost_estimate.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# HARD_REGIME_API_BUDGET_GATE\n",
          f"\n- tasks: {n_tasks}\n",
          f"- actions per task: {len(ACTIONS)}\n",
          f"- AGENTCONTROL_HARD_REGIME_APPROVED set: **{out['approval_set']}**\n",
          "\n## Per-provider preflight\n\n",
          "| provider | cheap model | strong model | calls/task | total calls | est total USD |\n",
          "|---|---|---|---|---|---|\n"]
    for prov, e in estimates.items():
        md.append(f"| {prov} | {e['cheap_model']} | {e['strong_model']} | {e['calls_per_task']} | {e['total_calls']} | ${e['estimated_total_usd']:.4f} |\n")
    md.append(f"\n**Grand total estimated spend: ${grand_total:.4f}**\n")
    md.append(f"\nUser-set sprint cap: **$20.00**. Per-provider cap: **${out['max_spend_cap_usd_per_provider']:.2f}**.\n")
    md.append(f"\nGate decision: {'OPEN' if out['approval_set'] else 'CLOSED — set AGENTCONTROL_HARD_REGIME_APPROVED=1 to enable'}.\n")

    md.append("\n## What runs if approved\n\n")
    md.append("- DeepSeek collection over 90 tasks × 7 actions = 630 calls.\n")
    md.append("- Together collection over 90 tasks × 7 actions = 630 calls.\n")
    md.append("- All calls cached and ledger-tracked.\n")
    md.append("- Replay mode for re-analysis without API calls.\n")
    md.append("- Budget hard-kill if spend > per-provider cap.\n")

    (REPO / "reports" / "HARD_REGIME_API_BUDGET_GATE.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} and reports/HARD_REGIME_API_BUDGET_GATE.md")
    print(f"grand total estimated spend: ${grand_total:.4f}")
    print(f"approval set: {out['approval_set']}")
    return 0 if out["approval_set"] or grand_total <= 20.0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
