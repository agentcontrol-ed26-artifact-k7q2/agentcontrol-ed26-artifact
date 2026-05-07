"""Phase 5a: bootstrap CIs on hard-regime real outcomes."""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from agentcontrol.ed_sim import (
    GRAPH_PLANS,
    PLAN_AUTOMIX,
    PLAN_FRUGALGPT,
    PLAN_HEURISTIC_BDELG,
    PLAN_SHEPHERDING,
    PLANS_CHEAP_ONLY,
    PLANS_STRONG_ONLY,
    QUERY_ROUTER_PLANS,
    aggregate,
    best_plan_per_task,
    run_plan_over_outcomes,
)

REPO = Path(__file__).resolve().parent.parent
SEED = 20260427
N_BOOT = 2000
BUDGET = 20.0
COST_PENALTY = 0.01

PROVIDERS = {
    "deepseek": "experiments/hard_regime_outcomes_deepseek.json",
    "together": "experiments/hard_regime_outcomes_together.json",
}


def _fam(tid):
    if tid.startswith("hm"): return "math"
    if tid.startswith("hc"): return "code"
    if tid.startswith("he"): return "evidence"
    return "other"


def _regime(outcomes, tid):
    for v in outcomes[tid].values():
        if "regime" in v:
            return v["regime"]
    return "unknown"


def _saving_pct(b, c):
    return 100.0 * (b - c) / b if b > 0 else 0.0


def _ci(s):
    s2 = sorted(s)
    n = len(s2)
    return {"mean": statistics.fmean(s), "ci_lo": s2[int(n * 0.025)],
            "ci_hi": s2[int(n * 0.975)]}


def _bootstrap_provider(outcomes):
    ids = sorted(outcomes.keys())
    qr = {r["task_id"]: r for r in best_plan_per_task(
        QUERY_ROUTER_PLANS, outcomes, verifier_aware=True, budget=BUDGET, cost_penalty=COST_PENALTY)}
    gr = {r["task_id"]: r for r in best_plan_per_task(
        GRAPH_PLANS, outcomes, verifier_aware=True, budget=BUDGET, cost_penalty=COST_PENALTY)}
    h = {r["task_id"]: r for r in run_plan_over_outcomes(
        "h", PLAN_HEURISTIC_BDELG, outcomes, verifier_aware=True, budget=BUDGET, cost_penalty=COST_PENALTY)}

    fams = defaultdict(list)
    regs = defaultdict(list)
    fr = defaultdict(list)
    for t in ids:
        fams[_fam(t)].append(t)
        regs[_regime(outcomes, t)].append(t)
        fr[(_fam(t), _regime(outcomes, t))].append(t)

    def _resample(rng, groups):
        sample = []
        for k, items in groups.items():
            sample.extend(rng.choice(items) for _ in items)
        return sample

    rng = random.Random(SEED)
    s_unstrat, s_fam, s_reg, s_fr = [], [], [], []
    cheap_strong_gap = []  # success gap on full sample
    cheap_succ_per = {tid: int(outcomes[tid].get("cheap_answer", {}).get("success", False)) for tid in ids}
    strong_succ_per = {tid: int(outcomes[tid].get("strong_answer", {}).get("success", False)) for tid in ids}
    fam_strat_csg = []

    for _ in range(N_BOOT):
        unstrat = [rng.choice(ids) for _ in ids]
        s_unstrat.append(_saving_pct(
            statistics.fmean(qr[t]["cost"] for t in unstrat),
            statistics.fmean(gr[t]["cost"] for t in unstrat)))
        fam_strat = _resample(rng, fams)
        s_fam.append(_saving_pct(
            statistics.fmean(qr[t]["cost"] for t in fam_strat),
            statistics.fmean(gr[t]["cost"] for t in fam_strat)))
        reg_strat = _resample(rng, regs)
        s_reg.append(_saving_pct(
            statistics.fmean(qr[t]["cost"] for t in reg_strat),
            statistics.fmean(gr[t]["cost"] for t in reg_strat)))
        fr_strat = _resample(rng, fr)
        s_fr.append(_saving_pct(
            statistics.fmean(qr[t]["cost"] for t in fr_strat),
            statistics.fmean(gr[t]["cost"] for t in fr_strat)))
        fam_strat_csg.append(
            statistics.fmean(strong_succ_per[t] for t in fam_strat) -
            statistics.fmean(cheap_succ_per[t] for t in fam_strat))

    return {
        "n_resamples": N_BOOT,
        "seed": SEED,
        "graph_query_cost_saving_pct": {
            "unstratified": _ci(s_unstrat),
            "family_stratified": _ci(s_fam),
            "regime_stratified": _ci(s_reg),
            "family_regime_stratified": _ci(s_fr),
        },
        "cheap_vs_strong_success_gap_pp": _ci([100.0 * x for x in fam_strat_csg]),
    }


def main() -> int:
    out = {}
    for p, path in PROVIDERS.items():
        f = REPO / path
        if not f.exists():
            continue
        outcomes = json.loads(f.read_text(encoding="utf-8"))
        out[p] = _bootstrap_provider(outcomes)
    out_path = REPO / "experiments" / "hard_regime_bootstrap.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# HARD_REGIME_BOOTSTRAP\n",
          f"\n2000-resample bootstrap, seed {SEED}.\n",
          "\n## Graph-query cost saving (95% CI)\n\n",
          "| provider | strat | mean | ci_lo | ci_hi |\n|---|---|---|---|---|\n"]
    for p, b in out.items():
        for s, name in [("unstratified", "unstratified"),
                        ("family_stratified", "family-strat"),
                        ("regime_stratified", "regime-strat"),
                        ("family_regime_stratified", "family×regime")]:
            ci = b["graph_query_cost_saving_pct"][s]
            md.append(f"| {p} | {name} | {ci['mean']:.3f} | {ci['ci_lo']:.3f} | {ci['ci_hi']:.3f} |\n")

    md.append("\n## Cheap-vs-strong success gap (pp, family-stratified)\n\n")
    md.append("| provider | mean | ci_lo | ci_hi |\n|---|---|---|---|\n")
    for p, b in out.items():
        ci = b["cheap_vs_strong_success_gap_pp"]
        md.append(f"| {p} | {ci['mean']:.3f} | {ci['ci_lo']:.3f} | {ci['ci_hi']:.3f} |\n")
    (REPO / "reports" / "HARD_REGIME_BOOTSTRAP.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} and reports/HARD_REGIME_BOOTSTRAP.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
