"""Phase 1: emit interactive-task-pool manifest + report."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from agentcontrol_main_rescue.interactive_tasks import get_pool  # noqa: E402


def main() -> int:
    pool = get_pool()
    families = Counter(t["family"] for t in pool.values())
    interactive = sum(1 for t in pool.values() if t.get("interactive"))
    obs_kinds = Counter()
    for t in pool.values():
        for o in t.get("available_observations", []):
            obs_kinds[o] += 1
    manifest = {
        "n_total": len(pool),
        "family_counts": dict(families),
        "n_interactive": interactive,
        "observation_actions_offered": dict(obs_kinds),
        "tasks": [
            {"task_id": tid, "family": t["family"], "interactive": t.get("interactive"),
             "observations": t.get("available_observations", []),
             "difficulty": t.get("difficulty")}
            for tid, t in pool.items()
        ],
        "config_path": "main_rescue_gpu/configs/interactive_tasks.yaml",
        "honesty": (
            "Interactive task pool. Observation actions (run_tests, run_code, "
            "retrieve, citation_check, tool_observation, checkpoint_check) are "
            "local and inexpensive but produce real new state information not "
            "available to a query-level router. The pool is constructed locally "
            "and auditable in main_rescue_gpu/src/agentcontrol_main_rescue/interactive_tasks.py. "
            "Not a benchmark download."
        ),
    }
    out = HERE / "experiments" / "interactive_task_pool_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# INTERACTIVE_TASK_POOL\n",
          f"\n- total tasks: **{len(pool)}**\n",
          f"- interactive (observation-bearing): **{interactive}**\n",
          f"- family counts: {dict(families)}\n",
          f"- observation actions offered: {dict(obs_kinds)}\n",
          "\n## Per-family breakdown\n\n",
          "| family | n | observations | difficulty mix |\n|---|---|---|---|\n"]
    diffs_per_fam = {}
    for t in pool.values():
        diffs_per_fam.setdefault(t["family"], Counter())[t.get("difficulty")] += 1
    for fam, n in families.items():
        sample = next(t for t in pool.values() if t["family"] == fam)
        obs = ",".join(sample.get("available_observations", []))
        diffs = dict(diffs_per_fam.get(fam, {}))
        md.append(f"| {fam} | {n} | {obs} | {diffs} |\n")
    md.append("\n## Honesty\n\n")
    md.append(manifest["honesty"] + "\n")
    (HERE / "reports" / "INTERACTIVE_TASK_POOL.md").write_text("".join(md), encoding="utf-8")
    print(f"families: {dict(families)}; total {len(pool)}")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
