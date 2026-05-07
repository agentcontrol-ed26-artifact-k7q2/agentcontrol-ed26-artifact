"""Phase 3: Outcome collection over the rescue task pool.

If real APIs are authorized (env AGENTCONTROL_RESCUE_APPROVED=1 AND
configs/rescue/providers_api.yaml has real_api_allowed: true), this would
issue cached real-model calls. In this run, that gate is CLOSED — we
short-circuit to the synthetic-local outcomes already produced by
``scripts/build_rescue_task_pool.py`` and label them NON-DECISIVE.

Writes a replayable JSONL trace per task and appends to the cost ledger
with cache_hit=true, cost_usd=0, actual_api_call=false.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
APPROVED = os.environ.get("AGENTCONTROL_RESCUE_APPROVED") == "1"
CFG = REPO / "configs" / "rescue" / "providers_api.yaml"
OUTCOMES = REPO / "experiments" / "rescue_outcomes.json"
LEDGER = REPO / "cache" / "cost_ledger.jsonl"
TRACES_DIR = REPO / "traces"
REPORT = REPO / "reports" / "RESCUE_OUTCOME_COLLECTION.md"


def main() -> int:
    if not OUTCOMES.exists():
        print("ERROR: run scripts/build_rescue_task_pool.py first")
        return 2
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
    real_api = bool(cfg.get("real_api_allowed", False)) and APPROVED
    outcomes = json.loads(OUTCOMES.read_text(encoding="utf-8"))

    n_tasks = len(outcomes)
    actions_per_task = 7
    total_calls = n_tasks * actions_per_task

    if real_api:
        # Defensive abort: even with the env var set, if budget is unset, refuse.
        budget = cfg.get("budget", {})
        max_spend = float(budget.get("max_total_spend_usd", 0.0))
        per_call = sum(
            float(s.get("estimated_cost_per_call_usd", 0.0))
            for s in cfg.get("provider_slots", {}).values()
        ) or 0.0
        est_total = total_calls * per_call
        print(f"projected calls: {total_calls}; per-call estimate: ${per_call:.4f}; total estimate: ${est_total:.2f}")
        if est_total <= 0 or max_spend <= 0 or est_total > max_spend:
            print("API_BUDGET_GATE: refusing to proceed; see reports/API_BUDGET_GATE.md")
            return 3
        print("Real-API path would execute here. NOT IMPLEMENTED in this rescue run.")
        return 4

    # Replay path: write traces from synthetic outcomes; append cache-hit ledger rows.
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    traces_by_family: dict[str, list[dict]] = {}
    n_rows = 0
    with LEDGER.open("a", encoding="utf-8") as ledger:
        for tid, action_outcomes in outcomes.items():
            family = ("math" if tid.startswith("rm")
                      else "code" if tid.startswith("rc")
                      else "evidence" if tid.startswith("re")
                      else "tool_use" if tid.startswith("rt")
                      else "other")
            for action, obs in action_outcomes.items():
                row = {
                    "task_id": tid,
                    "family": family,
                    "action": action,
                    "model": "dummy-cheap" if "cheap" in action else "dummy-pro",
                    "provider": "dummy",
                    "input_tokens": 60,
                    "output_tokens": 8,
                    "latency_ms": int(obs["latency_ms"]),
                    "cost_usd": 0.0,
                    "cache_hit": True,
                    "actual_api_call": False,
                    "request_hash": f"rescue::{tid}::{action}",
                    "verifier": {
                        "success": bool(obs["success"]),
                        "objective": float(obs["success"]),
                        "unsupported_risk": float(obs.get("unsupported_risk", 0.0)),
                    },
                    "non_decisive": True,
                }
                ledger.write(json.dumps(row) + "\n")
                traces_by_family.setdefault(family, []).append(row)
                n_rows += 1

    for fam, rows in traces_by_family.items():
        p = TRACES_DIR / f"rescue_{fam}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    md = ["# RESCUE_OUTCOME_COLLECTION\n",
          "\n**Status:** synthetic-local replay only. **Non-decisive** for Main Track.\n",
          f"\n- AGENTCONTROL_RESCUE_APPROVED: {os.environ.get('AGENTCONTROL_RESCUE_APPROVED', '<unset>')}\n",
          f"- real_api_allowed (config): {cfg.get('real_api_allowed', False)}\n",
          f"- gate decision: {'OPEN' if real_api else 'CLOSED → DummyProvider replay'}\n",
          f"- tasks: {n_tasks}\n",
          f"- ledger rows appended: {n_rows} (all cache_hit=true, cost_usd=0)\n",
          f"- traces written: {sorted(traces_by_family)}\n",
          "\n## Honesty\n\n",
          "Outcomes are deterministic synthetic functions of engineered ",
          "difficulty profiles. They test the protocol on n=170 with a richer ",
          "distribution than the original n=28 smoke, but they are NOT real-model ",
          "evidence. To collect real-model outcomes, see `reports/API_BUDGET_GATE.md`.\n"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(md), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(REPO)}; ledger rows={n_rows}; traces={len(traces_by_family)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
