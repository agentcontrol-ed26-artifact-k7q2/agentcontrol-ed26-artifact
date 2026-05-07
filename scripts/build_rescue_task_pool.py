"""Phase 1: Expand the task pool deterministically without any model calls.

Generates synthetic-local tasks with per-action outcomes derived from a
difficulty profile. Each task has a deterministic verifier outcome under
the price/latency model in ``configs/rescue/task_pool.yaml``. Tasks are
PROTOCOL test data, NOT real-model evidence.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CFG = REPO / "configs" / "rescue" / "task_pool.yaml"
OUT_JSON = REPO / "experiments" / "rescue_task_pool_manifest.json"
OUT_OUTCOMES = REPO / "experiments" / "rescue_outcomes.json"
OUT_MD = REPO / "reports" / "RESCUE_TASK_POOL.md"

ACTIONS = [
    "cheap_answer",
    "cheap_repair",
    "strong_hint",
    "cheap_repair_after_hint",
    "strong_critique",
    "strong_checklist",
    "strong_answer",
]


def _outcomes_for_difficulty(d: str, family: str, price: dict, latency: dict,
                             rng: random.Random, ev_unsup: dict | None) -> dict:
    """Map difficulty class → per-action (success, cost, latency, risk).

    Outcomes encode the deliberation-graph structure:
    - easy            : cheap_answer succeeds.
    - medium          : cheap fails, cheap_repair succeeds.
    - partial_strong  : cheap & cheap_repair fail; strong_hint produces a usable
                        hint (success=False because hint isn't an answer) and
                        cheap_repair_after_hint succeeds at low cost.
    - hard            : only strong_answer succeeds.

    All paths set strong_answer.success=True (the strong arm is reliable).
    """
    s = {a: False for a in ACTIONS}
    if d == "easy":
        s["cheap_answer"] = True
        s["cheap_repair"] = True
        s["cheap_repair_after_hint"] = True
        s["strong_answer"] = True
    elif d == "medium":
        s["cheap_answer"] = False
        s["cheap_repair"] = True
        s["cheap_repair_after_hint"] = True
        s["strong_answer"] = True
    elif d == "partial_strong":
        s["cheap_answer"] = False
        s["cheap_repair"] = False
        s["strong_hint"] = False  # hint is not an answer
        s["cheap_repair_after_hint"] = True
        s["strong_answer"] = True
    elif d == "hard":
        s["cheap_answer"] = False
        s["cheap_repair"] = False
        s["strong_hint"] = False
        s["cheap_repair_after_hint"] = False
        s["strong_answer"] = True
    else:
        raise ValueError(f"unknown difficulty {d}")

    out = {}
    for a in ACTIONS:
        risk = 0.0
        if family == "evidence" and ev_unsup is not None:
            if a == "cheap_answer" and not s[a]:
                # Cheap answers may produce unsupported citations on failures.
                risk = ev_unsup.get("cheap_answer_unsupported_prob", 0.0) if rng.random() < 0.5 else 0.0
            if a == "strong_hint":
                risk = ev_unsup.get("strong_hint_unsupported_prob", 0.0)
        out[a] = {
            "cost": float(price.get(a, 1.0)),
            "latency_ms": int(latency.get(a, 1)),
            "success": bool(s[a]),
            "unsupported_risk": float(risk),
        }
    return out


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    seed = cfg.get("seed", 0)
    rng = random.Random(seed)
    price = cfg["price_model"]
    latency = cfg["latency_model_ms"]

    families: dict[str, list[str]] = {}
    outcomes: dict[str, dict[str, dict]] = {}
    provenance: dict[str, str] = {}
    difficulty_assignments: dict[str, str] = {}

    for fam_name, fam_cfg in cfg["families"].items():
        n = int(fam_cfg["n"])
        prov = fam_cfg.get("provenance", "synthetic-local")
        mix = fam_cfg["difficulty_mix"]
        ev_unsup = fam_cfg.get("unsupported_risk_profile")
        # Build the difficulty roster deterministically.
        roster = []
        for d, frac in mix.items():
            roster.extend([d] * round(n * frac))
        # Pad/truncate to n.
        while len(roster) < n:
            roster.append("medium")
        roster = roster[:n]
        rng.shuffle(roster)

        prefix = {"math": "rm", "code": "rc", "evidence": "re", "tool_use": "rt"}.get(fam_name, "rx")
        ids = []
        for i, d in enumerate(roster, 1):
            tid = f"{prefix}_{i:03d}"
            ids.append(tid)
            outcomes[tid] = _outcomes_for_difficulty(d, fam_name, price, latency, rng, ev_unsup)
            provenance[tid] = prov
            difficulty_assignments[tid] = d
        families[fam_name] = ids

    n_total = sum(len(v) for v in families.values())

    manifest = {
        "seed": seed,
        "n_total": n_total,
        "family_counts": {f: len(ids) for f, ids in families.items()},
        "family_difficulty_distribution": {
            fam: {
                d: sum(1 for tid in ids if difficulty_assignments[tid] == d)
                for d in ("easy", "medium", "partial_strong", "hard")
            }
            for fam, ids in families.items()
        },
        "provenance_summary": {p: sum(1 for v in provenance.values() if v == p)
                               for p in set(provenance.values())},
        "price_model": price,
        "latency_model_ms": latency,
        "task_ids_by_family": families,
        "tasks": [
            {
                "task_id": tid,
                "family": next(f for f, ids in families.items() if tid in ids),
                "difficulty": difficulty_assignments[tid],
                "provenance": provenance[tid],
            }
            for tid in sorted(outcomes)
        ],
        "config_path": "configs/rescue/task_pool.yaml",
        "honesty_note": (
            "Synthetic-local pool. Outcomes are deterministic functions of "
            "engineered difficulty profiles, not real model behavior. Use to "
            "test the harness/protocol on a richer distribution; do NOT cite "
            "as real-model evidence."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    OUT_OUTCOMES.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")

    md = ["# RESCUE_TASK_POOL\n",
          "\n**Status:** synthetic-local pool (Phase 1). Not real-model evidence.\n",
          f"\n- total tasks: **{n_total}**\n",
          f"- seed: {seed}\n",
          "\n## Family counts and difficulty distribution\n\n",
          "| family | n | easy | medium | partial_strong | hard | provenance |\n",
          "|---|---|---|---|---|---|---|\n"]
    for fam, ids in families.items():
        d = manifest["family_difficulty_distribution"][fam]
        prov = cfg["families"][fam].get("provenance", "synthetic-local")
        md.append(f"| {fam} | {len(ids)} | {d['easy']} | {d['medium']} | {d['partial_strong']} | {d['hard']} | {prov} |\n")
    md.append("\n## Honesty\n\n")
    md.append(manifest["honesty_note"] + "\n\n")
    md.append("- Every task has a deterministic verifier (synthetic outcome).\n")
    md.append("- No real model was called.\n")
    md.append("- Pool intentionally engineered so the cheap arm does not saturate.\n")
    md.append("- Headroom on this pool tests the **protocol**, not real-model claims.\n")
    md.append("\n## Files\n\n")
    md.append("- manifest: `experiments/rescue_task_pool_manifest.json`\n")
    md.append("- outcomes: `experiments/rescue_outcomes.json`\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(REPO)}, {OUT_OUTCOMES.relative_to(REPO)}, {OUT_MD.relative_to(REPO)}")
    print(f"task counts: {manifest['family_counts']}; total {n_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
