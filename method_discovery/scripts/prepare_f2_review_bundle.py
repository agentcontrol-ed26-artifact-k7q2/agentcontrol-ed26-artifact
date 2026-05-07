"""F2 Phase 6: Codex review bundle for F2 sprint."""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
RB = HERE / "review_bundles"
REFERENCED = [
    "method_discovery/reports/F2_DATA_ANALYSIS_POOL.md",
    "method_discovery/reports/F2_API_BUDGET_GATE.md",
    "method_discovery/reports/F2_DATA_ANALYSIS_RESULTS.md",
    "method_discovery/reports/F2_MAIN_GATE_DECISION.md",
    "method_discovery/reports/MAIN_TRACK_DECISION_AFTER_METHOD_DISCOVERY.md",
    "main_rescue_gpu/reports/GPU_MAIN_RESCUE_DECISION.md",
    "reports/POST_HARD_REGIME_TRACK_DECISION.md",
    "paper/artifact_claims.md",
    "paper/artifact_limitations.md",
]


def _read(p: Path) -> str:
    try: return p.read_text(encoding="utf-8")
    except Exception: return ""


def write_bundle(out: Path) -> None:
    parts = ["# Gate: F2 Data-Analysis Review\n\n",
             "Decision under review: **F2_FAIL_MAIN_KEEP_ED**.\n",
             "F2 extension to n=50 data_analysis_code on DeepSeek confirmed the prior n=20 "
             "+5.83 pp signal was an artifact of an unfair baseline. Fair baseline (router "
             "with observation included) gives 0% / 0 pp with CI [0, 0]. Observation carries "
             "the entire load; deliberation graph structure adds nothing.\n\n",
             "## Index\n"]
    for rel in REFERENCED:
        p = REPO / rel
        parts.append(f"- [`{rel}`]({rel}){'' if p.exists() else '  *(missing)*'}\n")
    parts.append("\n## Inlined contents\n")
    for rel in REFERENCED:
        p = REPO / rel
        if not p.exists(): continue
        parts.append(f"\n---\n\n### `{rel}`\n\n")
        text = _read(p)
        if len(text) > 25000: text = text[:25000] + "\n\n*(truncated)*\n"
        parts.append(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")


def write_prompt(out: Path) -> None:
    body = """# Codex Review Prompt — F2 Data Analysis Sprint

You are reviewing the AgentControl F2 sprint. **Do not edit any files. Review only.**

## Context

F2 extended `data_analysis_code` from n=20 (prior `main_rescue_gpu` interactive sprint) to n=50 to test whether the previously-reported DeepSeek +5.83 pp / +22.61% saving signal survives at higher n. Result on DeepSeek (real-API, $0.071 incremental):

- vs router WITHOUT observation: 39.71% cost saving, +30 pp success delta, bootstrap CI [+16.81%, +53.57%] saving / [+18, +42] pp success delta. **Effect amplified at n=50**.
- vs router WITH observation (the fair Codex-required baseline that addresses your pass-5 critique): **0.00% saving / 0.00 pp success delta**, bootstrap CI [0%, 0%] strict.

`fixed_react` (a 3-step cascade `cheap → run_code → cheap_repair_after_observation`) gets 96% success at cost 2.14 — essentially matches the oracle deliberation graph at 98% / 2.05.

Decision: **F2_FAIL_MAIN_KEEP_ED**.

## Files to read

- `method_discovery/reports/F2_DATA_ANALYSIS_POOL.md`
- `method_discovery/reports/F2_API_BUDGET_GATE.md`
- `method_discovery/reports/F2_DATA_ANALYSIS_RESULTS.md`
- `method_discovery/reports/F2_MAIN_GATE_DECISION.md`
- `method_discovery/reports/MAIN_TRACK_DECISION_AFTER_METHOD_DISCOVERY.md` (updated)
- `main_rescue_gpu/reports/GPU_MAIN_RESCUE_DECISION.md` (the prior signal under unfair baseline)
- `reports/POST_HARD_REGIME_TRACK_DECISION.md` (E&D fallback)

Source-of-truth:
- `method_discovery/experiments/f2_data_analysis_outcomes_deepseek.json` (50 tasks × 8 actions)
- `method_discovery/experiments/f2_data_analysis_oracle_summary.json` (oracle gap, baselines, bootstrap)
- `method_discovery/src/agentcontrol_method/f2_data_analysis_tasks.py` (50 tasks; programmatically gold-verified)

## Questions

1. **Are the 30 new tasks genuinely interactive, deterministic, and gold-verified?** All 30 new task gold answers were verified programmatically; check the verification claim.
2. **Is there gold leakage in `run_code` observation?** It returns model-emitted stdout from running the model's code. Genuinely fair?
3. **Is the oracle graph advantage now correctly framed against the fair baseline?** Section "Headline result" reports BOTH no-obs and with-obs comparisons explicitly. Is this honest?
4. **Are the baselines strong enough?** Panel includes always_cheapest, always_strongest, FrugalGPT cascade, AutoMix, Shepherding hint, fixed_react, fixed_react_with_strong_fallback, heuristic_bdelg.
5. **Does F2 justify Main Track or only E&D?** F2_FAIL_MAIN_KEEP_ED is the call. Confirm or push back.
6. **Are statistics robust?** 2000-resample bootstrap; CI strictly [0, 0] on the fair baseline.
7. **Is the F2 finding a load-bearing methodological contribution to the E&D paper?** Specifically: "the apparent graph-headroom signal in the prior n=20 sprint was artifactual; the fair baseline shows graph adds nothing beyond observation-cascade."
8. **Should Candidate B (KV-readout on [CLUSTER_A]) be run next?** Or should Main rescue path be closed entirely?
9. **What concrete fixes are required?** Numbered punch-list with file paths.

## Output format

```
# F2 Data Analysis Review

## Verdict
ready-for-paper-drafting / needs-fixes / not-ready

## Summary
<one paragraph>

## Findings
<question-by-question, citing files>

## Required fixes
1. <file>: <change> — <why>

## Recommended next move
<one of: stop-main-rescue / try-candidate-D / try-candidate-B / re-run-with-different-baseline / E&D-paper-drafting>
```

Do not propose Main Track promotion. The strict baseline finding is decisive.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main() -> int:
    write_bundle(RB / "gate_f2_data_analysis_review.md")
    write_prompt(RB / "gate_f2_data_analysis_review_prompt.md")
    print("wrote method_discovery/review_bundles/gate_f2_data_analysis_review.md")
    print("wrote method_discovery/review_bundles/gate_f2_data_analysis_review_prompt.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
