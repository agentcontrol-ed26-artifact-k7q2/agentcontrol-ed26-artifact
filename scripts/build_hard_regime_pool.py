"""Phase 1 (hard regime): emit pool manifest + report. No API calls."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentcontrol.hard_regime_tasks import get_pool  # noqa: E402


def main() -> int:
    pool = get_pool()
    families = Counter(t["family"] for t in pool.values())
    regimes = Counter(t["regime"] for t in pool.values())
    fam_regime = Counter((t["family"], t["regime"]) for t in pool.values())
    manifest = {
        "n_total": len(pool),
        "family_counts": dict(families),
        "regime_counts": dict(regimes),
        "family_regime_counts": {f"{f}/{r}": n for (f, r), n in fam_regime.items()},
        "tasks": [
            {"task_id": tid, "family": t["family"], "regime": t["regime"]}
            for tid, t in pool.items()
        ],
        "config_path": "configs/rescue/hard_regime.yaml",
        "honesty": (
            "Synthetic-local hard pool with regime tags. Tasks are auditable "
            "in src/agentcontrol/hard_regime_tasks.py. NOT a benchmark download."
        ),
    }
    out = REPO / "experiments" / "hard_regime_pool_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# HARD_REGIME_POOL\n",
          f"\n- total tasks: **{len(pool)}**\n",
          f"- family counts: {dict(families)}\n",
          f"- regime counts: {dict(regimes)}\n",
          "\n## Family × regime\n\n",
          "| family | easy_saturation | medium_headroom | hard_strong_gap | weak_verifier_risk | evidence_support_risk |\n",
          "|---|---|---|---|---|---|\n"]
    for fam in ("math", "code", "evidence"):
        row = [fam]
        for r in ("easy_saturation", "medium_headroom", "hard_strong_gap",
                  "weak_verifier_risk", "evidence_support_risk"):
            row.append(str(fam_regime.get((fam, r), 0)))
        md.append("| " + " | ".join(row) + " |\n")
    md.append("\n## Honesty\n\n")
    md.append(manifest["honesty"] + "\n")
    md.append("\nEach task has a deterministic verifier. Math: numeric exact match on last integer in output. "
              "Code: sandboxed unit-test execution. Evidence: gold-answer phrase + at least one [doc_xxx] "
              "citation in the authorized citation set.\n")

    md.append("\n## Files\n\n")
    md.append("- `experiments/hard_regime_pool_manifest.json`\n")
    md.append("- `src/agentcontrol/hard_regime_tasks.py` (full task list, verifiers)\n")
    (REPO / "reports" / "HARD_REGIME_POOL.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)} and reports/HARD_REGIME_POOL.md")
    print(f"families: {dict(families)}; regimes: {dict(regimes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
