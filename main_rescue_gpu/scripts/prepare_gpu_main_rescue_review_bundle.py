"""Phase 10: Codex review bundle for GPU Main Rescue Fork."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
RB = HERE / "review_bundles"

REFERENCED = [
    "main_rescue_gpu/reports/GPU_ENVIRONMENT_AUDIT.md",
    "main_rescue_gpu/reports/LOCAL_MODEL_LADDER_PLAN.md",
    "main_rescue_gpu/reports/INTERACTIVE_TASK_POOL.md",
    "main_rescue_gpu/reports/LOCAL_INTERACTIVE_DEEPSEEK.md",
    "main_rescue_gpu/reports/LOCAL_INTERACTIVE_TOGETHER.md",
    "main_rescue_gpu/reports/LOCAL_INTERACTIVE_ORACLE_GAP.md",
    "main_rescue_gpu/reports/LOCAL_INTERACTIVE_ABLATIONS.md",
    "main_rescue_gpu/reports/LOCAL_INTERACTIVE_BOOTSTRAP.md",
    "main_rescue_gpu/reports/LOCAL_INTERACTIVE_SENSITIVITY.md",
    "main_rescue_gpu/reports/CONTROLLER_TRAINING_GATE.md",
    "main_rescue_gpu/reports/GPU_MAIN_RESCUE_DECISION.md",
    "reports/POST_HARD_REGIME_TRACK_DECISION.md",
    "reports/FINAL_TRACK_DECISION.md",
    "reports/MAIN_RESCUE_OPTION.md",
    "paper/artifact_claims.md",
    "paper/artifact_limitations.md",
    "paper/ed_final_results.md",
]


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def write_bundle(out: Path) -> None:
    parts = ["# Gate: GPU Main Rescue Fork Review\n\n",
             "Decision under review: **MAIN-NEEDS-API-VALIDATION**.\n",
             "First non-zero graph-headroom signal in the project: DeepSeek interactive +5.83 pp success at −22.61% cost.\n\n",
             "## Index\n"]
    for rel in REFERENCED:
        p = REPO / rel
        marker = "" if p.exists() else "  *(missing)*"
        parts.append(f"- [`{rel}`]({rel}){marker}\n")
    parts.append("\n## Inlined contents\n")
    for rel in REFERENCED:
        p = REPO / rel
        if not p.exists():
            continue
        parts.append(f"\n---\n\n### `{rel}`\n\n")
        text = _read(p)
        if len(text) > 25000:
            text = text[:25000] + "\n\n*(truncated; see source)*\n"
        parts.append(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")


def write_prompt(out: Path) -> None:
    body = """# Codex Review Prompt — GPU Main Rescue Fork

You are reviewing the AgentControl GPU Main Rescue Fork. **Do not edit any files. Review only.**

This pass evaluates the first non-zero graph-headroom result in the project. The interactive task pool (n=120, observation actions like run_tests/run_code/retrieve/citation_check/tool_observation/checkpoint_check) was collected on real DeepSeek and Together AI APIs ($0.41 incremental). The result: oracle deliberation graph saves 22.61% cost over oracle query router on DeepSeek with **+5.83 pp success delta** (95% bootstrap CI [+2.50, +9.17] family-stratified). Together corroborates direction at 20.88% / +1.67 pp.

The decision is **MAIN-NEEDS-API-VALIDATION**, not MAIN-CANDIDATE — see `main_rescue_gpu/reports/GPU_MAIN_RESCUE_DECISION.md`. The E&D-STRONG-ACCEPT-CANDIDATE fallback remains intact.

## Files to read

- `main_rescue_gpu/reports/GPU_MAIN_RESCUE_DECISION.md` (the decision)
- `main_rescue_gpu/reports/LOCAL_INTERACTIVE_{DEEPSEEK,TOGETHER,ORACLE_GAP,ABLATIONS,BOOTSTRAP,SENSITIVITY}.md`
- `main_rescue_gpu/reports/CONTROLLER_TRAINING_GATE.md`
- `main_rescue_gpu/reports/GPU_ENVIRONMENT_AUDIT.md`
- `main_rescue_gpu/reports/INTERACTIVE_TASK_POOL.md`
- `main_rescue_gpu/reports/LOCAL_MODEL_LADDER_PLAN.md`
- `reports/POST_HARD_REGIME_TRACK_DECISION.md` and `reports/FINAL_TRACK_DECISION.md` (prior E&D state)
- `paper/artifact_claims.md`, `paper/artifact_limitations.md`

Optional source-of-truth:
- `main_rescue_gpu/experiments/local_interactive_outcomes_{deepseek,together}.json`
- `main_rescue_gpu/experiments/local_interactive_{summary_*,ablations,bootstrap,sensitivity}.json`
- `main_rescue_gpu/src/agentcontrol_main_rescue/interactive_tasks.py` (task pool source)
- `main_rescue_gpu/src/agentcontrol_main_rescue/interactive_oracle.py` (oracle plan logic)
- `main_rescue_gpu/scripts/run_local_interactive_collection.py` (collection logic)
- `cache/cost_ledger.jsonl` (real-API spend; $0.41 sprint incremental)

## Questions to answer

1. **Does the GPU detection / TIER0_NO_GPU decision make sense?** Quadro P1000 4GB is too small for any LLM serving. The fork pivoted to API-only; is that pivot honest?
2. **Are the tasks genuinely interactive?** Observation actions (run_tests, run_code, retrieve, citation_check, tool_observation, checkpoint_check) execute locally and produce real new state. Are they fair, deterministic, and non-leaky?
3. **Is the oracle graph advantage fair?** Oracle query router has access to fixed cascades (cheap_only, strong_only, FrugalGPT, AutoMix) but NOT observation actions. Oracle graph adds observation-then-repair routes. Is excluding observation from the router the right comparison? Or should the router also see observation? See `main_rescue_gpu/src/agentcontrol_main_rescue/interactive_oracle.py`.
4. **Does query router have fair access?** The router has 4 plans; the graph adds 4 more. Is the graph getting an unfair structural advantage independent of observation? Check the ablations.
5. **Are ablations convincing?** Removing observation reduces DeepSeek graph success by 5.83 pp and raises cost by 0.354 (graph collapses to query router exactly). Is this the right test for the verifier-state Pareto gate?
6. **Is the partial-strong evidence meaningful?** Tool-planning n=15 shows 56.56% saving; data-analysis n=20 shows 51.18% saving on DeepSeek. Underpowered or sufficient given bootstrap CI is strictly positive?
7. **Is controller training justified now?** The CONTROLLER_TRAINING_GATE is OPEN-PENDING-USER-APPROVAL. Is the oracle gap strong enough to justify the small-imitation-MLP path the gate proposes? What additional evidence would Codex want first?
8. **Is the MAIN-NEEDS-API-VALIDATION label correct?** Or should it be MAIN-CANDIDATE (criterion 1 strictly passes on DeepSeek via +5.83 pp success delta with strictly-positive 95% CI)? Or back to MAIN-NO?
9. **Should we keep E&D instead?** Compare the strength of the new signal to the cost of pursuing Main Track. Does this sprint legitimately upgrade the artifact or is the per-family n=15-20 too small?
10. **What concrete fixes are required before NeurIPS submission?** Numbered punch-list with file paths.

## Output format

```
# GPU Main Rescue Fork Review

## Verdict
ready-for-paper-drafting / needs-fixes / not-ready

## Summary
<one paragraph>

## Findings
<question-by-question, citing files>

## Required fixes
1. <file>: <change> — <why>
```

Do not propose Main Track promotion if the evidence is borderline. Do not propose dataset / API / GPU expansion in the punch-list (those require user approval). Do not damage the E&D fallback.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main() -> int:
    write_bundle(RB / "gate_gpu_main_rescue_review.md")
    write_prompt(RB / "gate_gpu_main_rescue_review_prompt.md")
    print("wrote main_rescue_gpu/review_bundles/gate_gpu_main_rescue_review.md")
    print("wrote main_rescue_gpu/review_bundles/gate_gpu_main_rescue_review_prompt.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
