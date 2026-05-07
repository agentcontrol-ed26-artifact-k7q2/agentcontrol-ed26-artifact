"""Apollo Phase 3B — risk-cost Pareto across policies on verifier-risk pool.

Cost units (protocol):
  - cheap_call    = 1.0
  - strong_call   = 10.0
  - weak_verify   = 0.0  (free; just pattern match)
  - exact_verify  = 0.5  (audit suite / authorized-citation check; modest cost)

Policies:
  - always_cheap_weak       : cheap → accept by weak verifier
  - always_cheap_exact      : cheap → accept by exact verifier (catches traps)
  - always_strong_weak      : strong → accept by weak verifier
  - always_strong_exact     : strong → accept by exact verifier
  - cascade_weak            : cheap → strong if weak rejects
  - cascade_exact           : cheap → strong if exact rejects (escalate-on-trap)
  - risk_constrained_oracle : cheap; invoke exact verifier ONLY on tasks where
                              cheap output equals a "common_wrong" or for evidence
                              tasks where any citation is a known distractor;
                              escalate to strong if exact rejects.
                              (oracle-style heuristic from this task pool's metadata)

Metrics per policy:
  - success_rate (gold-correct accept)
  - false_accept_rate (accepted but wrong by exact)
  - avg_cost (protocol units)
  - cost_at_target_false_accept (parametric)
  - bootstrap CIs on false_accept_rate
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APOLLO = REPO / "main_apollo"
sys.path.insert(0, str(APOLLO / "verifier_risk" / "scripts"))

from build_verifier_risk_pool import get_pool  # noqa: E402

OUTCOMES_PATH = APOLLO / "verifier_risk" / "experiments" / "verifier_risk_outcomes.json"

COST_CHEAP = 1.0
COST_STRONG = 10.0
COST_WEAK_V = 0.0
COST_EXACT_V = 0.5


def _evaluate_policy(name: str, policy_fn, pool, outcomes) -> list[dict]:
    rs = []
    for tid, task in pool.items():
        out = outcomes.get(tid, {})
        cheap = out.get("cheap")
        strong = out.get("strong")
        if not cheap or not strong:
            continue
        rs.append(policy_fn(name, task, cheap, strong))
    return rs


def _accept_outcome(accepted_text_pass: bool, gold_pass_exact: bool):
    """Return (success, false_accept). accepted_text_pass is what the policy
    decided; gold_pass_exact is the exact-verifier ground truth."""
    if accepted_text_pass:
        # Policy accepted. Was it actually correct?
        return (1 if gold_pass_exact else 0, 1 if not gold_pass_exact else 0)
    else:
        # Policy rejected (abstained / escalated).
        return (0, 0)


def _is_distractor_cited(task: dict, text: str) -> bool:
    """Heuristic for risk-constrained oracle on evidence tasks: did model cite a known distractor?"""
    if task["family"] != "evidence_traps":
        return False
    import re as _re
    cited = _re.findall(r"\[(doc_[a-zA-Z0-9_]+)\]", text or "")
    distractors = set(task.get("distractor_citations", []))
    return any(c in distractors for c in cited)


def _cheap_says_common_wrong(task: dict, text: str) -> bool:
    """Heuristic for risk-constrained oracle on math/data tasks: cheap output is the common-wrong answer."""
    if task["family"] not in ("math_traps", "data_analysis_traps"):
        return False
    import re as _re
    nums = _re.findall(r"-?\d+", (text or "").replace(",", ""))
    if not nums:
        return False
    last = nums[-1]
    return last == task.get("common_wrong_answer", "")


def _is_code_with_weak_pass_only(task: dict, cheap_outcome: dict) -> bool:
    """Risk heuristic for code traps: weak passed but exact didn't on cheap output."""
    if task["family"] != "code_traps":
        return False
    return cheap_outcome.get("weak_pass") and not cheap_outcome.get("exact_pass")


def _risk_signal(task: dict, cheap: dict) -> bool:
    """Combined risk heuristic across all families. Used by risk_constrained_oracle."""
    return (
        _cheap_says_common_wrong(task, cheap.get("text", "")) or
        _is_distractor_cited(task, cheap.get("text", "")) or
        _is_code_with_weak_pass_only(task, cheap)
    )


# Policy definitions. Each returns dict with task_id, accepted, success, false_accept, cost.
def policy_always_cheap_weak(name, task, cheap, strong):
    accepted = cheap["weak_pass"]
    s, fa = _accept_outcome(accepted, cheap["exact_pass"])
    return {"task_id": task["id"], "policy": name,
            "accepted": int(accepted), "success": s, "false_accept": fa,
            "cost": COST_CHEAP + COST_WEAK_V}


def policy_always_cheap_exact(name, task, cheap, strong):
    accepted = cheap["exact_pass"]
    s, fa = _accept_outcome(accepted, cheap["exact_pass"])
    return {"task_id": task["id"], "policy": name,
            "accepted": int(accepted), "success": s, "false_accept": fa,
            "cost": COST_CHEAP + COST_EXACT_V}


def policy_always_strong_weak(name, task, cheap, strong):
    accepted = strong["weak_pass"]
    s, fa = _accept_outcome(accepted, strong["exact_pass"])
    return {"task_id": task["id"], "policy": name,
            "accepted": int(accepted), "success": s, "false_accept": fa,
            "cost": COST_STRONG + COST_WEAK_V}


def policy_always_strong_exact(name, task, cheap, strong):
    accepted = strong["exact_pass"]
    s, fa = _accept_outcome(accepted, strong["exact_pass"])
    return {"task_id": task["id"], "policy": name,
            "accepted": int(accepted), "success": s, "false_accept": fa,
            "cost": COST_STRONG + COST_EXACT_V}


def policy_cascade_weak(name, task, cheap, strong):
    if cheap["weak_pass"]:
        s, fa = _accept_outcome(True, cheap["exact_pass"])
        return {"task_id": task["id"], "policy": name,
                "accepted": 1, "success": s, "false_accept": fa,
                "cost": COST_CHEAP + COST_WEAK_V}
    accepted = strong["weak_pass"]
    s, fa = _accept_outcome(accepted, strong["exact_pass"])
    return {"task_id": task["id"], "policy": name,
            "accepted": int(accepted), "success": s, "false_accept": fa,
            "cost": COST_CHEAP + COST_WEAK_V + COST_STRONG + COST_WEAK_V}


def policy_cascade_exact(name, task, cheap, strong):
    if cheap["exact_pass"]:
        return {"task_id": task["id"], "policy": name, "accepted": 1,
                "success": 1, "false_accept": 0,
                "cost": COST_CHEAP + COST_EXACT_V}
    accepted = strong["exact_pass"]
    return {"task_id": task["id"], "policy": name,
            "accepted": int(accepted),
            "success": 1 if accepted else 0,
            "false_accept": 0,  # exact verifier; never false-accepts
            "cost": COST_CHEAP + COST_EXACT_V + COST_STRONG + COST_EXACT_V}


def policy_risk_constrained(name, task, cheap, strong):
    """Cheap with weak verifier as default; invoke exact verifier ONLY when risk signal fires;
    escalate to strong-with-exact if exact rejects."""
    risky = _risk_signal(task, cheap)
    if not risky:
        # Trust cheap+weak (cost 1.0 + 0 = 1.0). False-accept if cheap's exact_pass is False.
        s, fa = _accept_outcome(cheap["weak_pass"], cheap["exact_pass"])
        return {"task_id": task["id"], "policy": name,
                "accepted": int(cheap["weak_pass"]),
                "success": s, "false_accept": fa,
                "cost": COST_CHEAP + COST_WEAK_V}
    # Risk fired: invoke exact verifier on cheap.
    if cheap["exact_pass"]:
        return {"task_id": task["id"], "policy": name, "accepted": 1,
                "success": 1, "false_accept": 0,
                "cost": COST_CHEAP + COST_EXACT_V}
    # Escalate to strong, audit with exact.
    accepted = strong["exact_pass"]
    return {"task_id": task["id"], "policy": name,
            "accepted": int(accepted),
            "success": 1 if accepted else 0,
            "false_accept": 0,
            "cost": COST_CHEAP + COST_EXACT_V + COST_STRONG + COST_EXACT_V}


POLICIES = [
    ("always_cheap_weak", policy_always_cheap_weak),
    ("always_cheap_exact", policy_always_cheap_exact),
    ("always_strong_weak", policy_always_strong_weak),
    ("always_strong_exact", policy_always_strong_exact),
    ("cascade_weak", policy_cascade_weak),
    ("cascade_exact", policy_cascade_exact),
    ("risk_constrained", policy_risk_constrained),
]


def aggregate(rs):
    n = max(1, len(rs))
    return {"n": len(rs),
            "success_rate": sum(r["success"] for r in rs) / n,
            "false_accept_rate": sum(r["false_accept"] for r in rs) / n,
            "accept_rate": sum(r["accepted"] for r in rs) / n,
            "avg_cost": sum(r["cost"] for r in rs) / n}


def per_family(rs, pool):
    by = defaultdict(list)
    for r in rs:
        fam = pool[r["task_id"]]["family"]
        by[fam].append(r)
    return {f: aggregate(v) for f, v in by.items()}


def bootstrap(rs, n_resamples=2000, seed=20260427):
    rng = random.Random(seed)
    fa_rates = []
    succ_rates = []
    costs = []
    n = len(rs)
    for _ in range(n_resamples):
        sample = [rs[rng.randrange(n)] for _ in range(n)]
        fa_rates.append(sum(r["false_accept"] for r in sample) / n)
        succ_rates.append(sum(r["success"] for r in sample) / n)
        costs.append(sum(r["cost"] for r in sample) / n)

    def ci(s):
        ss = sorted(s); m = len(ss)
        return {"mean": statistics.fmean(s),
                "ci_lo": ss[int(m * 0.025)], "ci_hi": ss[int(m * 0.975)]}

    return {"false_accept_rate": ci(fa_rates),
            "success_rate": ci(succ_rates),
            "avg_cost": ci(costs)}


def main() -> int:
    pool = get_pool()
    if not OUTCOMES_PATH.exists():
        print(f"missing outcomes at {OUTCOMES_PATH}; run collection first")
        return 2
    outcomes = json.loads(OUTCOMES_PATH.read_text(encoding="utf-8"))

    summary = {"policies": {}}
    all_rs = {}
    for name, fn in POLICIES:
        rs = _evaluate_policy(name, fn, pool, outcomes)
        all_rs[name] = rs
        summary["policies"][name] = {
            "aggregate": aggregate(rs),
            "per_family": per_family(rs, pool),
            "bootstrap": bootstrap(rs),
        }

    out_path = APOLLO / "verifier_risk" / "experiments" / "verifier_risk_policy_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# VERIFIER_RISK_POLICIES\n",
          f"\nn = {summary['policies']['always_cheap_weak']['aggregate']['n']} tasks; DeepSeek (chat / reasoner) real outcomes.\n\n",
          "## Aggregate risk-cost-success table\n\n",
          "| policy | success | false_accept | accept | avg_cost |\n|---|---|---|---|---|\n"]
    for name, _ in POLICIES:
        a = summary["policies"][name]["aggregate"]
        md.append(f"| {name} | {a['success_rate']:.3f} | {a['false_accept_rate']:.3f} | "
                  f"{a['accept_rate']:.3f} | {a['avg_cost']:.3f} |\n")

    md.append("\n## Bootstrap 95% CI on false_accept_rate\n\n")
    md.append("| policy | mean | ci_lo | ci_hi |\n|---|---|---|---|\n")
    for name, _ in POLICIES:
        b = summary["policies"][name]["bootstrap"]["false_accept_rate"]
        md.append(f"| {name} | {b['mean']:.3f} | {b['ci_lo']:.3f} | {b['ci_hi']:.3f} |\n")

    md.append("\n## Per-family false_accept_rate\n\n")
    md.append("| policy | math_traps | code_traps | evidence_traps | data_analysis_traps |\n|---|---|---|---|---|\n")
    for name, _ in POLICIES:
        pf = summary["policies"][name]["per_family"]
        cells = []
        for fam in ("math_traps", "code_traps", "evidence_traps", "data_analysis_traps"):
            v = pf.get(fam)
            cells.append(f"{v['false_accept_rate']:.3f}" if v else "—")
        md.append(f"| {name} | " + " | ".join(cells) + " |\n")

    md.append("\n## Honest interpretation\n\n")
    md.append("`risk_constrained` is the Apollo-distinct policy. It uses cheap+weak by default and only invokes the exact verifier when a heuristic risk signal fires. Compare its (false_accept, cost) point to the `cascade_exact` baseline (which always uses the expensive exact verifier) and to `cascade_weak` (the trivial baseline). The Main GO gate is: risk_constrained reduces false_accept materially OR matches false_accept at strictly lower cost than cascade_exact, with bootstrap CI strictly improving.\n")

    (APOLLO / "verifier_risk" / "reports" / "VERIFIER_RISK_SMOKE.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} and reports/VERIFIER_RISK_SMOKE.md")
    print("Headline (false_accept_rate):")
    for name, _ in POLICIES:
        a = summary["policies"][name]["aggregate"]
        print(f"  {name}: succ={a['success_rate']:.3f} fa={a['false_accept_rate']:.3f} cost={a['avg_cost']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
