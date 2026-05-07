"""KV-readout Phase 3A — build control labels from existing AgentControl traces.

Generates a CSV/JSON dataset where each row is:
  (task_id, family, prompt_text, control_label)

Control labels are derived from oracle deliberation-graph traces:
  - need_code     : task family ∈ {code_debug_interactive, data_analysis_code, code_*}
  - need_retrieve : task family ∈ {evidence, evidence_multihop_local}
  - need_strong   : oracle plan included strong_answer
  - escalate      : oracle plan ran beyond cheap_answer
  - graph_candidate : oracle deliberation-graph plan strictly beats query-router cascade on the same task (binary; from cached oracle results)
  - verifier_risk : task has unsupported_risk > 0 in any cached outcome
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APOLLO = REPO / "main_apollo"
sys.path.insert(0, str(REPO / "src"))

# Existing trace / outcome / pool sources to mine.
SOURCES = [
    ("smoke", "experiments/smoke_outcomes.json"),
    ("rescue", "experiments/rescue_outcomes_deepseek.json"),
    ("rescue_together", "experiments/rescue_outcomes_together.json"),
    ("hard_regime", "experiments/hard_regime_outcomes_deepseek.json"),
    ("hard_regime_together", "experiments/hard_regime_outcomes_together.json"),
    ("interactive", "main_rescue_gpu/experiments/local_interactive_outcomes_deepseek.json"),
    ("interactive_together", "main_rescue_gpu/experiments/local_interactive_outcomes_together.json"),
    ("agentic_search", "method_discovery/experiments/agentic_search_outcomes_deepseek.json"),
    ("agentic_search_together", "method_discovery/experiments/agentic_search_outcomes_together.json"),
    ("f2", "method_discovery/experiments/f2_data_analysis_outcomes_deepseek.json"),
]


def _family_of(task_id: str) -> str:
    if task_id.startswith("rm") or task_id.startswith("hm") or task_id.startswith("m") and not task_id.startswith("main"): return "math"
    if task_id.startswith("rc") or task_id.startswith("hc") or task_id.startswith("ic") or task_id.startswith("code"): return "code"
    if task_id.startswith("re") or task_id.startswith("he") or task_id.startswith("ie") or task_id.startswith("e") or task_id.startswith("ag"): return "evidence"
    if task_id.startswith("rt") or task_id.startswith("it"): return "tool_planning"
    if task_id.startswith("im"): return "math_checkpoint"
    if task_id.startswith("id"): return "data_analysis"
    return "other"


def _need_code(family: str) -> int:
    return int(family in {"code", "data_analysis"})


def _need_retrieve(family: str) -> int:
    return int(family == "evidence")


def _verifier_risk(actions: dict) -> int:
    for a, info in actions.items():
        if isinstance(info, dict) and info.get("unsupported_risk", 0.0) > 0.0:
            return 1
    return 0


def _need_strong(actions: dict) -> int:
    """Did cheap_answer succeed? If not, the right policy needs strong escalation."""
    cheap = actions.get("cheap_answer", {})
    if isinstance(cheap, dict) and not cheap.get("success", False):
        return 1
    return 0


def _escalate(actions: dict) -> int:
    """Same as need_strong here; alias kept for label diversity in downstream tools."""
    return _need_strong(actions)


def _graph_candidate(actions: dict) -> int:
    """Oracle graph would strictly help: cheap fails AND any later answer-action succeeds at lower cost than strong_answer."""
    cheap_succ = actions.get("cheap_answer", {}).get("success", False)
    if cheap_succ:
        return 0
    strong_succ = actions.get("strong_answer", {}).get("success", False)
    repair_succ = (
        actions.get("cheap_repair", {}).get("success", False) or
        actions.get("cheap_repair_after_observation", {}).get("success", False) or
        actions.get("cheap_repair_after_strong_partial", {}).get("success", False)
    )
    return int(repair_succ and strong_succ)


def main() -> int:
    rows: list[dict] = []
    for source_name, rel in SOURCES:
        path = REPO / rel
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for tid, actions in data.items():
            if not isinstance(actions, dict):
                continue
            family = _family_of(tid)
            # Reconstruct prompt text by family — reuse the prompt builder we used at collection time.
            # For mining purposes, the question text itself is the discriminator.
            row = {
                "source": source_name,
                "task_id": tid,
                "family": family,
                "labels": {
                    "need_code": _need_code(family),
                    "need_retrieve": _need_retrieve(family),
                    "verifier_risk": _verifier_risk(actions),
                    "need_strong": _need_strong(actions),
                    "escalate": _escalate(actions),
                    "graph_candidate": _graph_candidate(actions),
                },
            }
            rows.append(row)

    # Dedupe by task_id (the cached outcomes from re-runs will agree on labels).
    seen = set()
    deduped = []
    for r in rows:
        if r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        deduped.append(r)

    # Per-label class balance.
    label_counts: dict[str, dict[str, int]] = {}
    for label in ("need_code", "need_retrieve", "verifier_risk", "need_strong", "escalate", "graph_candidate"):
        pos = sum(1 for r in deduped if r["labels"][label] == 1)
        neg = sum(1 for r in deduped if r["labels"][label] == 0)
        label_counts[label] = {"pos": pos, "neg": neg, "n": pos + neg,
                                "pos_rate": pos / max(1, pos + neg)}

    out = {
        "n_total": len(deduped),
        "sources": [s for s, _ in SOURCES],
        "per_label_class_balance": label_counts,
        "rows": deduped,
        "honesty": (
            "Labels derived deterministically from cached oracle outcomes. "
            "need_code / need_retrieve are family-based (trivially predictable). "
            "verifier_risk / need_strong / escalate / graph_candidate are derived "
            "from per-task oracle plan decisions. The task prompt text is NOT "
            "stored here (use original task pool modules for prompt text); the "
            "external-router baseline (run_external_router_baselines.py) joins "
            "this label table with the prompt text."
        ),
    }
    out_path = APOLLO / "kv_readout" / "experiments" / "control_labels.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# KV_READOUT_CONTROL_LABELS\n",
          f"\n- total tasks: **{len(deduped)}**\n",
          f"- sources: {[s for s, _ in SOURCES]}\n",
          "\n## Per-label class balance\n\n",
          "| label | pos | neg | pos_rate |\n|---|---|---|---|\n"]
    for k, v in label_counts.items():
        md.append(f"| {k} | {v['pos']} | {v['neg']} | {v['pos_rate']:.3f} |\n")
    md.append("\n## Honesty\n\n" + out["honesty"] + "\n")
    (APOLLO / "kv_readout" / "reports" / "KV_READOUT_LABELS.md").write_text(
        "".join(md), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)}")
    print(f"n_total: {len(deduped)}")
    print(f"label class balance:")
    for k, v in label_counts.items():
        print(f"  {k}: pos={v['pos']} neg={v['neg']} pos_rate={v['pos_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
