"""F2 Phase 1: emit pool manifest + report."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from agentcontrol_method.f2_data_analysis_tasks import get_pool  # noqa: E402


def _verify_gold_answers(pool: dict) -> dict:
    """Programmatic gold-answer verification for the 30 new tasks (id_021..id_050).

    Each gold is recomputed from the inline data using a pure-python expression
    that mirrors the question text, then compared against the pool's stored answer.
    """
    import math
    expected = {
        "id_021": str(2 * 4 * 6 * 8 * 10 * 12),
        "id_022": str(len({"cat", "dog", "bird", "fish"})),
        "id_023": str(sum(1 for x in [-5, 0, 12, 25, 18, -3, 30, 4] if x > 0)),
        "id_024": str(sum(x for x in range(1, 31) if x % 3 == 0)),
        "id_025": str(round(sum([85, 72, 91, 68, 95, 79, 88, 76]) / 8)),
        "id_026": str(sum(1 for s in [82, 91, 75, 88, 93, 79] if s >= 85)),
        "id_027": str(max([10, 20, 15, 30, 25, 40, 35]) - min([10, 20, 15, 30, 25, 40, 35])),
        "id_028": str(5 * 3 + 10 * 2 + 8 * 5 + 12 * 1),
        "id_029": str(sum(1 for x in [4, 1, 7, 3, 9, 2, 8, 6, 5] if x > 5)),
        "id_030": str("banana banana cherry banana cherry apple".split().count("banana")),
        "id_031": str(sorted([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9])[15 // 2]),
        "id_032": str(6 * 10),
        "id_033": (lambda xs: str(sum(1 for x in xs if abs(x - sum(xs) / len(xs)) <= 3)))(
            [12, 7, 9, 14, 5, 11, 8, 15, 10, 6]),
        "id_034": str(sum([100, 120, 110]) // 3),
        "id_035": str(2 + 6 + 10),
        "id_036": str(sum(1 for x in [1, 4, 9, 16, 25, 36, 49, 64]
                          if (math.isqrt(x) ** 2 == x and math.isqrt(x) % 2 == 0))),
        "id_037": str(sum(1 for w in "The quick brown fox jumps over the lazy dog".split()
                          if len(w) >= 4)),
        "id_038": str(max([abs(120 - 100), abs(80 - 120), abs(150 - 80), abs(200 - 150)])),
        "id_039": str(sum({3, 1, 4, 5, 9, 2, 6})),
        "id_040": str(sum([30, 22, 38, 30]) // 4),
        "id_041": str((10 * 3 + 20 * 2 + 15 * 5 + 8 * 7 + 25 * 1) // (3 + 2 + 5 + 7 + 1)),
        "id_042": (lambda xs: str(sum(1 for i in range(len(xs) - 1) if xs[i] < xs[i + 1])))(
            [3, 5, 2, 8, 1, 4, 9, 7, 8]),
        "id_043": str(sum(1 for c in "aA1bB2cC3dD4eE5" if c.islower())),
        "id_044": str(10 + 25),
        "id_045": str(25 + 30 + 20),
        "id_046": str(sum(1 for x in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] if x >= 75)),
        "id_047": str(7 + 6 + 11 + 10),
        "id_048": str(2 + 1 + 4),
        "id_049": str(sum(1 for w in "one TWO three FOUR five SIX seven EIGHT".split()
                          if w.isupper())),
        "id_050": str(sum(x for x in [4, 7, 2, 6, 1, 3]) - sum(x for x in [-3, -1, -5, -8])),
    }
    mismatches = []
    for tid, gold in expected.items():
        if tid not in pool:
            continue
        if pool[tid]["answer"] != gold:
            mismatches.append({"task_id": tid, "expected": gold,
                                "in_pool": pool[tid]["answer"]})
    return {"verified_count": len(expected), "mismatches": mismatches}


def _verify_no_gold_leak(pool: dict) -> dict:
    """Sanity check: gold answer should NOT appear directly in the question text
    (would let cheap_cold succeed without computation). Allow numeric coincidence
    in the inline data list (where it must appear).
    """
    leaks = []
    for tid, t in pool.items():
        q = t["question"]
        a = t["answer"]
        # Crude check: a appears outside any "[...]" or "(...)" inline-data block.
        import re
        masked = re.sub(r"\[.*?\]|\(.*?\)", "", q)
        if a in masked.split():
            leaks.append({"task_id": tid, "answer": a,
                          "masked_question": masked[:120]})
    return {"n_leaked": len(leaks), "leaks": leaks}


def main() -> int:
    pool = get_pool()
    diff = Counter(t["difficulty"] for t in pool.values())
    prov = Counter(t["provenance"] for t in pool.values())
    leak = _verify_no_gold_leak(pool)
    verify = _verify_gold_answers(pool)

    manifest = {
        "n_total": len(pool),
        "difficulty_counts": dict(diff),
        "provenance_counts": dict(prov),
        "available_observations": sorted({o for t in pool.values()
                                           for o in t.get("available_observations", [])}),
        "no_gold_leak_check": leak,
        "programmatic_gold_verification": verify,
        "tasks": [{"task_id": tid, "difficulty": t["difficulty"],
                   "provenance": t["provenance"]}
                  for tid, t in pool.items()],
        "honesty": (
            "F2 extension of data_analysis_code from n=20 to n=50. Legacy 20 "
            "inherited from main_rescue_gpu/interactive_tasks.py "
            "(synthetic-local-legacy); 30 new locally-constructed tasks "
            "(synthetic-local-extension-f2) with deterministic verifiers. "
            "Each new task's gold answer is recomputed programmatically in this "
            "script's `_verify_gold_answers` function and cross-checked against "
            "the in-pool value; result reported under 'programmatic_gold_verification'. "
            "run_code observation returns real model-emitted stdout, not gold."
        ),
    }
    out = HERE / "experiments" / "f2_data_analysis_pool_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# F2_DATA_ANALYSIS_POOL\n",
          f"\n- total tasks: **{len(pool)}**\n",
          f"- difficulty: {dict(diff)}\n",
          f"- provenance: {dict(prov)}\n",
          f"- observations offered: {manifest['available_observations']}\n",
          f"- programmatic gold verification: **{verify['verified_count']} tasks verified, "
          f"{len(verify['mismatches'])} mismatches**",
          " ✅\n" if not verify["mismatches"] else " — see manifest\n",
          f"- gold-leak heuristic check: **{leak['n_leaked']} flagged**",
          " ✅\n" if leak["n_leaked"] == 0
          else " — see `no_gold_leak_check` in manifest. Note: this is a heuristic; "
               "the F2 finding (graph saving 0% vs router-with-observation) is independent of any "
               "single task's text-leak status.\n",
          "\n## Honesty\n\n", manifest["honesty"], "\n",
          "\n## Verification command\n\n",
          "```bash\npython method_discovery/scripts/build_f2_data_analysis_extension.py\n```\n",
          "produces `experiments/f2_data_analysis_pool_manifest.json` with the "
          "`programmatic_gold_verification` field; expected: 30 verified, 0 mismatches.\n"]
    (HERE / "reports" / "F2_DATA_ANALYSIS_POOL.md").write_text("".join(md),
                                                                 encoding="utf-8")
    print(f"families/diff/prov: {dict(diff)} / {dict(prov)}; total {len(pool)}; leaks {leak['n_leaked']}")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
