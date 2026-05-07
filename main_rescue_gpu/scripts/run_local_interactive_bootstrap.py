"""Phase 6a: bootstrap CIs on interactive outcomes."""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "src"))

from agentcontrol_main_rescue.interactive_oracle import (  # noqa: E402
    GRAPH_PLANS, QUERY_ROUTER_PLANS, best_plan, saving_pct,
)

PROVS = ("deepseek", "together")
SEED = 20260427
N_BOOT = 2000


def _fam(tid):
    if tid.startswith("ic"): return "code_debug_interactive"
    if tid.startswith("id"): return "data_analysis_code"
    if tid.startswith("ie"): return "evidence_multihop_local"
    if tid.startswith("it"): return "tool_planning_deterministic"
    if tid.startswith("im"): return "math_checkpoint"
    return "other"


def _ci(s):
    s2 = sorted(s)
    n = len(s2)
    return {"mean": statistics.fmean(s), "ci_lo": s2[int(n * 0.025)],
            "ci_hi": s2[int(n * 0.975)]}


def main() -> int:
    out_all = {}
    for prov in PROVS:
        path = HERE / "experiments" / f"local_interactive_outcomes_{prov}.json"
        if not path.exists():
            continue
        outcomes = json.loads(path.read_text(encoding="utf-8"))
        ids = sorted(outcomes.keys())
        qr = {tid: best_plan(QUERY_ROUTER_PLANS, outcomes, tid) for tid in ids}
        gr = {tid: best_plan(GRAPH_PLANS, outcomes, tid) for tid in ids}

        fams = defaultdict(list)
        for t in ids:
            fams[_fam(t)].append(t)

        rng = random.Random(SEED)
        s_unstrat = []
        s_fam = []
        succ_delta = []
        for _ in range(N_BOOT):
            unstrat = [rng.choice(ids) for _ in ids]
            s_unstrat.append(saving_pct(
                statistics.fmean(qr[t]["cost"] for t in unstrat),
                statistics.fmean(gr[t]["cost"] for t in unstrat)))
            fam_s = []
            for fam, tids in fams.items():
                fam_s.extend(rng.choice(tids) for _ in tids)
            s_fam.append(saving_pct(
                statistics.fmean(qr[t]["cost"] for t in fam_s),
                statistics.fmean(gr[t]["cost"] for t in fam_s)))
            succ_delta.append(100.0 * (
                statistics.fmean(int(gr[t]["success"]) for t in fam_s) -
                statistics.fmean(int(qr[t]["success"]) for t in fam_s)))

        out_all[prov] = {
            "n_resamples": N_BOOT, "seed": SEED,
            "graph_query_costsaving_pct": {
                "unstratified": _ci(s_unstrat),
                "family_stratified": _ci(s_fam),
            },
            "graph_query_success_delta_pp": _ci(succ_delta),
        }

    out_path = HERE / "experiments" / "local_interactive_bootstrap.json"
    out_path.write_text(json.dumps(out_all, indent=2), encoding="utf-8")

    md = ["# LOCAL_INTERACTIVE_BOOTSTRAP\n",
          f"\n2000 resamples, seed {SEED}.\n\n"]
    for prov, b in out_all.items():
        md.append(f"## {prov}\n\n")
        md.append("| metric | strat | mean | ci_lo | ci_hi |\n|---|---|---|---|---|\n")
        for s, name in (("unstratified", "unstrat"), ("family_stratified", "fam-strat")):
            ci = b["graph_query_costsaving_pct"][s]
            md.append(f"| graph_query_costsaving_pct | {name} | {ci['mean']:.3f} | {ci['ci_lo']:.3f} | {ci['ci_hi']:.3f} |\n")
        ci = b["graph_query_success_delta_pp"]
        md.append(f"| graph_query_success_delta_pp | fam-strat | {ci['mean']:.3f} | {ci['ci_lo']:.3f} | {ci['ci_hi']:.3f} |\n\n")
    (HERE / "reports" / "LOCAL_INTERACTIVE_BOOTSTRAP.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} and reports/LOCAL_INTERACTIVE_BOOTSTRAP.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
