"""Phase 8: Codex review bundle for method-discovery sprint."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
RB = HERE / "review_bundles"

REFERENCED = [
    "method_discovery/reports/CURRENT_DIAGNOSIS.md",
    "method_discovery/reports/CANDIDATE_MAIN_THESES.md",
    "method_discovery/reports/MINI_COLLISION_CHECK.md",
    "method_discovery/reports/EXPERIMENT_SELECTION.md",
    "method_discovery/reports/BUDGET_AND_COMPUTE_GATE.md",
    "method_discovery/reports/AGENTIC_SEARCH_POOL.md",
    "method_discovery/reports/AGENTIC_SEARCH_RESULTS.md",
    "method_discovery/reports/METHOD_DISCOVERY_RESULTS.md",
    "method_discovery/reports/MAIN_TRACK_DECISION_AFTER_METHOD_DISCOVERY.md",
    "main_rescue_gpu/reports/GPU_MAIN_RESCUE_DECISION.md",
    "reports/POST_HARD_REGIME_TRACK_DECISION.md",
    "paper/artifact_claims.md",
    "paper/artifact_limitations.md",
]


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def write_bundle(out: Path) -> None:
    parts = ["# Gate: Method Discovery Review\n\n",
             "Decision under review: **MAIN-TRACK-POSSIBLE-BUT-NEEDS-MORE**.\n",
             "Sprint executed Phase 0-7: candidate theses, mini collision check, experiment selection, "
             "agentic-search-vs-fixed-RAG real-API run on n=60 frozen multi-hop corpus.\n\n",
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
    body = """# Codex Review Prompt — Method Discovery Sprint

You are reviewing the AgentControl Method Discovery sprint. **Do not edit any files. Review only.**

This pass evaluates the autonomous-research output: 6 candidate theses, mini collision check, experiment selection, real-API agentic-search-vs-fixed-RAG run, and the final decision.

## Files to read

- `method_discovery/reports/CURRENT_DIAGNOSIS.md`
- `method_discovery/reports/CANDIDATE_MAIN_THESES.md`
- `method_discovery/reports/MINI_COLLISION_CHECK.md`
- `method_discovery/reports/EXPERIMENT_SELECTION.md`
- `method_discovery/reports/BUDGET_AND_COMPUTE_GATE.md`
- `method_discovery/reports/AGENTIC_SEARCH_POOL.md`
- `method_discovery/reports/AGENTIC_SEARCH_RESULTS.md`
- `method_discovery/reports/METHOD_DISCOVERY_RESULTS.md`
- `method_discovery/reports/MAIN_TRACK_DECISION_AFTER_METHOD_DISCOVERY.md`
- `main_rescue_gpu/reports/GPU_MAIN_RESCUE_DECISION.md` (the prior interactive-feedback signal)
- `reports/POST_HARD_REGIME_TRACK_DECISION.md` (E&D fallback)
- `paper/artifact_claims.md`, `paper/artifact_limitations.md`

Optional source-of-truth:
- `method_discovery/experiments/agentic_search_outcomes_{deepseek,together}.json`
- `method_discovery/experiments/agentic_search_oracle_summary.json`
- `method_discovery/src/agentcontrol_method/agentic_corpus.py` (the corpus)

## Questions

1. **Are the candidate theses genuinely different from prior routing/orchestration work?** Six candidates listed; A=interactive feedback, B=KV readout, C=regime detector, D=verifier-risk, E=partial-strong (DEAD), F=agentic search. Top three by score: F, A, C. Are F and A actually distinct from prior work (Self-RAG, IRCoT, FLARE for F; AutoMix, FrugalGPT for A)?
2. **Which candidate has the best Main Track chance?** A is the only one with strict GO PASS via DeepSeek +5.83 pp success delta. F produced a controlled negative result. B not run. Is the right path A's n=50 extension, or B (KV-readout on [CLUSTER_A]), or D (verifier-risk pool)?
3. **Are the selected experiments decisive?** F1 (agentic search) is decisive negative. F2 (interactive extension) was deferred. Should F2 have been run instead of F1? Or was F1 the right falsification first?
4. **Are gates strict enough?** GO criteria require ≥30% saving OR ≥5 pp success at matched cost. Together's +5 pp at +1.78% cost is borderline — is the strict matched-cost interpretation correct, or should we accept "asymmetric" success-cost trades?
5. **Is there any overclaiming?** The decision is MAIN-TRACK-POSSIBLE-BUT-NEEDS-MORE, not MAIN-TRACK-CANDIDATE. Is this hedge appropriate, or is the evidence too weak to even claim that?
6. **Is E&D fallback preserved?** New material lives under `method_discovery/`; no E&D doc overwritten. Verify.
7. **Should we spend API/GPU budget on follow-ups?** F2 extension ($0.30) seems clearly worth it. B (KV-readout, 6h [CLUSTER_A] GPU, $0) is higher-risk-higher-reward. D ($2) fills an empty regime cell. Which is the right next bet?
8. **What should be killed immediately?** Candidate E is dead. Should anything else be killed (C? D? B?)?

## Output format

```
# Method Discovery Review

## Verdict
ready-for-paper-drafting / needs-fixes / not-ready

## Summary
<one paragraph>

## Findings
<question-by-question, citing files>

## Required fixes
1. <file>: <change> — <why>

## Recommended next experiment
<one of: F2-extension / B-kv-readout / D-verifier-risk / stop>
```

Do not propose Main Track promotion if evidence is weak. Do not damage E&D fallback. Do not propose API/GPU spend without explicit approval gate.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main() -> int:
    write_bundle(RB / "gate_method_discovery_review.md")
    write_prompt(RB / "gate_method_discovery_review_prompt.md")
    print("wrote method_discovery/review_bundles/gate_method_discovery_review.md")
    print("wrote method_discovery/review_bundles/gate_method_discovery_review_prompt.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
