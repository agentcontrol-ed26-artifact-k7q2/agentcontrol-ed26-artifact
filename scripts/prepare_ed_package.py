"""Prepare the E&D artifact package.

Verifies smoke artifacts and decision-doc invariants, scans for forbidden-escalation
signals, writes ``experiments/ed_artifact_manifest.json``, refreshes the consolidated
``review_bundles/gate_ed_artifact_review.md``, and writes the paste-ready Codex review
prompt at ``review_bundles/gate_ed_artifact_review_prompt.md``.

This script does NOT call real APIs, train controllers, or expand datasets. It is a
read-only check + bundle generator.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_SMOKE_ARTIFACTS = [
    "experiments/aggregate_summary.json",
    "experiments/baselines_summary.json",
    "experiments/heuristic_bdelg_summary.json",
    "experiments/oracle_gap_summary.json",
    "experiments/smoke_outcomes.json",
    "traces/smoke_math_code.jsonl",
    "traces/smoke_evidence.jsonl",
    "cache/cost_ledger.jsonl",
    "reports/SMOKE_DECISION.md",
    "reports/ORACLE_GAP_DECISION.md",
    "reports/TRACK_DECISION.md",
    "reports/COMPUTE_ESCALATION_DECISION.md",
]

DECISION_DOC_CHECKS = [
    ("reports/SMOKE_DECISION.md", "must_contain_any", ["BACKUP", "E&D-only", "E&D-PACKAGE", "E&D fallback"]),
    ("reports/ORACLE_GAP_DECISION.md", "must_not_contain", ["DECISION: GO"]),
    ("reports/TRACK_DECISION.md", "must_contain_any", ["BACKUP", "E&D"]),
    ("reports/COMPUTE_ESCALATION_DECISION.md", "must_contain_any", ["DO-NOT-SCALE"]),
    ("reports/ED_PACKAGE_DECISION.md", "must_contain_any", ["DECISION: E&D-PACKAGE"]),
    ("reports/MAIN_RESCUE_OPTION.md", "must_contain_any", ["DECISION: NOT ACTIVE"]),
]

REVIEW_BUNDLE_REFERENCED = [
    "PROJECT_SUMMARY.md",
    "reports/SMOKE_DECISION.md",
    "reports/ORACLE_GAP_DECISION.md",
    "reports/TRACK_DECISION.md",
    "reports/COMPUTE_ESCALATION_DECISION.md",
    "reports/ED_PACKAGE_DECISION.md",
    "reports/MAIN_RESCUE_OPTION.md",
    "paper/ed_positioning.md",
    "paper/artifact_claims.md",
    "paper/evaluation_protocol.md",
    "paper/reproducibility_checklist.md",
    "paper/artifact_limitations.md",
    "paper/ed_abstract.md",
    "paper/ed_outline.md",
    "paper/ed_fallback.md",
    "paper/claims.md",
]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _check_required_files() -> tuple[list[str], list[str]]:
    present, missing = [], []
    for rel in REQUIRED_SMOKE_ARTIFACTS:
        p = REPO_ROOT / rel
        (present if p.exists() else missing).append(rel)
    return present, missing


def _check_decision_docs() -> list[dict]:
    findings = []
    for rel, mode, needles in DECISION_DOC_CHECKS:
        p = REPO_ROOT / rel
        if not p.exists():
            findings.append({"file": rel, "mode": mode, "needles": needles, "status": "MISSING"})
            continue
        text = _read(p)
        if mode == "must_contain_any":
            ok = any(n in text for n in needles)
            findings.append({"file": rel, "mode": mode, "needles": needles, "status": "OK" if ok else "VIOLATION"})
        elif mode == "must_not_contain":
            hits = [n for n in needles if n in text]
            findings.append({"file": rel, "mode": mode, "needles": needles, "hits": hits,
                             "status": "OK" if not hits else "VIOLATION"})
    return findings


def _scan_forbidden_escalation() -> list[dict]:
    findings = []

    # Cost ledger: any actual_api_call=true with non-dummy provider is a violation.
    ledger = REPO_ROOT / "cache" / "cost_ledger.jsonl"
    if ledger.exists():
        with ledger.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                provider = (row.get("provider") or "").lower()
                if row.get("actual_api_call") and provider not in {"dummy", ""}:
                    findings.append({
                        "kind": "real_provider_call",
                        "severity": "error",
                        "file": "cache/cost_ledger.jsonl",
                        "line": i + 1,
                        "provider": provider,
                        "row": row,
                    })

    # Trained-controller / readout artifact directories.
    for rel in [
        "checkpoints", "models", "runs",
        "experiments/controller_train", "experiments/control_readout_train",
        "artifacts/controller", "artifacts/control_readouts",
    ]:
        p = REPO_ROOT / rel
        if p.exists() and any(p.iterdir()):
            findings.append({
                "kind": "training_artifact_present",
                "severity": "error",
                "path": rel,
                "note": "controller / control-readout training is forbidden in BACKUP/E&D-only",
            })

    # GPU logs.
    for rel in ["logs/gpu", "logs/slurm", "logs/runpod"]:
        p = REPO_ROOT / rel
        if p.exists() and any(p.iterdir()):
            findings.append({
                "kind": "gpu_logs_present",
                "severity": "warning",
                "path": rel,
                "note": "GPU jobs are not authorized in current decision state",
            })

    # Dataset expansion: outcome counts above smoke baseline (math=20, code=4, evidence=4 → n=28).
    out = REPO_ROOT / "experiments" / "smoke_outcomes.json"
    if out.exists():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            n = 0
            if isinstance(data, dict):
                if "tasks" in data and isinstance(data["tasks"], list):
                    n = len(data["tasks"])
                elif "outcomes" in data and isinstance(data["outcomes"], list):
                    n = len(data["outcomes"])
                else:
                    # Sum over family keys if structure is per-family.
                    for v in data.values():
                        if isinstance(v, list):
                            n += len(v)
            if n > 28 and not (REPO_ROOT / "reports" / "DATASET_EXPANSION_APPROVAL.md").exists():
                findings.append({
                    "kind": "dataset_expansion_without_approval",
                    "severity": "error",
                    "n_observed": n,
                    "n_expected": 28,
                })
        except Exception:
            pass

    return findings


def _smoke_numbers() -> dict:
    """Pull the headline numbers from oracle_gap and aggregate summaries."""
    nums = {
        "pytest": "must be re-run; expected 18/18 passing",
        "n_total": 28,
        "n_math": 20,
        "n_code": 4,
        "n_evidence": 4,
    }
    og = REPO_ROOT / "experiments" / "oracle_gap_summary.json"
    if og.exists():
        try:
            d = json.loads(og.read_text(encoding="utf-8"))
            nums.update({
                "oracle_query_router_avg_cost": round(d["query_router"]["avg_cost"], 4),
                "oracle_deliberation_graph_avg_cost": round(d["deliberation_graph"]["avg_cost"], 4),
                "oracle_query_router_success_rate": d["query_router"]["success_rate"],
                "oracle_deliberation_graph_success_rate": d["deliberation_graph"]["success_rate"],
                "cost_saving_pct_at_observed": round(d["cost_saving_pct_at_observed"], 4),
                "success_delta_pp": round(d["success_delta_pp"], 4),
                "decision": d.get("decision", "BACKUP"),
            })
        except Exception:
            pass
    h = REPO_ROOT / "experiments" / "heuristic_bdelg_summary.json"
    if h.exists():
        try:
            d = json.loads(h.read_text(encoding="utf-8"))
            nums.update({
                "heuristic_bdelg_avg_cost": round(d.get("avg_cost", 0.0), 4),
                "heuristic_bdelg_success_rate": d.get("success_rate", 0.0),
                "heuristic_bdelg_avg_unsupported_risk": d.get("avg_unsupported_risk", 0.0),
            })
        except Exception:
            pass
    return nums


def _write_review_bundle(out_path: Path) -> None:
    parts = [
        "# Gate: E&D Artifact Review\n",
        "Consolidated review bundle for the AgentControl E&D artifact package.\n",
        "Decision: **BACKUP / E&D-only** (Main Track NOT viable on current smoke).\n",
        "\n## Bundle index\n",
    ]
    for rel in REVIEW_BUNDLE_REFERENCED:
        p = REPO_ROOT / rel
        marker = "" if p.exists() else "  *(missing)*"
        parts.append(f"- [`{rel}`]({rel}){marker}\n")
    parts.append("\n## Inlined contents\n")
    for rel in REVIEW_BUNDLE_REFERENCED:
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        parts.append(f"\n---\n\n### `{rel}`\n\n")
        text = _read(p)
        # Cap each inlined doc to keep bundle reviewable.
        if len(text) > 30000:
            text = text[:30000] + "\n\n*(truncated for bundle; see source file)*\n"
        parts.append(text)
        parts.append("\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")


def _write_codex_prompt(out_path: Path) -> None:
    body = """# Codex Review Prompt — AgentControl E&D Artifact

You are reviewing the AgentControl repository as a paper / artifact reviewer.
**Do not edit any files. Review only.** Read the files listed below from the repo root and produce a review report.

## What you are reviewing

A 48h smoke-only research harness that has been pivoted from a Main Track method submission to an **E&D (Evaluation & Demonstration) artifact**. The decision pivot is:

- pytest 18/18 passing.
- Smoke: n=28 (math=20, code=4, evidence=4), DummyProvider only, no real APIs.
- Oracle deliberation graph saves **26.09%** cost over oracle query router at matched 100% success — **below** the pre-registered 30% GO threshold.
- Per-family: math 0% / code 58.1% (n=4) / evidence 0%.
- Heuristic BDelG ties or loses to per-family-best baseline.
- Verifier-state ablation untested.
- Decision: **BACKUP / E&D-only / DO-NOT-SCALE.** No controller, no readout training, no GPU, no real APIs.

## Files to read (repo root)

Decision docs:
- `reports/SMOKE_DECISION.md`
- `reports/ORACLE_GAP_DECISION.md`
- `reports/TRACK_DECISION.md`
- `reports/COMPUTE_ESCALATION_DECISION.md`
- `reports/ED_PACKAGE_DECISION.md`
- `reports/MAIN_RESCUE_OPTION.md`

Paper / E&D drafts:
- `paper/ed_positioning.md`
- `paper/artifact_claims.md`
- `paper/evaluation_protocol.md`
- `paper/reproducibility_checklist.md`
- `paper/artifact_limitations.md`
- `paper/ed_abstract.md`
- `paper/ed_outline.md`
- `paper/ed_fallback.md`
- `paper/claims.md`

Consolidated bundle:
- `review_bundles/gate_ed_artifact_review.md`

Optional source-of-truth:
- `experiments/oracle_gap_summary.json`
- `experiments/aggregate_summary.json`
- `experiments/heuristic_bdelg_summary.json`
- `experiments/baselines_summary.json`

## What to check

Answer each question concretely with file:line citations where useful.

1. **Honesty of E&D positioning.** Does `paper/ed_positioning.md` and `paper/ed_abstract.md` honestly frame the artifact as E&D, not as a Main Track method paper? Are collisions (FrugalGPT / RouteLLM / RouterBench / Router-R1 / ToolOrchestra / LLM Shepherding / TRIM / prefill activation routers / IntroLM / etc.) acknowledged as a reason to avoid broad method claims?
2. **Removal of Main Track claims.** Search `paper/` for any residual Main Track method claim (state-of-the-art, "our method", "we propose", learned-controller superiority, KV-amortized readout as contribution). List them.
3. **Support for artifact claims.** Are the claims in `paper/artifact_claims.md` and `paper/claims.md` actually supported by the smoke artifacts in `experiments/`? Identify any unsupported claim.
4. **Oracle vs router framing.** Is the oracle-graph vs oracle-query-router comparison framed as **evaluation methodology** (a structural upper bound on multi-step controller benefit on this data) rather than as a method-superiority claim?
5. **Reproducibility / cache / cost ledger / trace replay / no-scale.** Does `paper/reproducibility_checklist.md` cover deterministic cache keys, append-only cost ledger, replayable JSONL traces, no-real-API verification, and the no-scale invariants? Any gaps?
6. **KV-amortized readout.** Confirm the artifact does NOT claim KV-amortized readout as a contribution. It should appear only as motivation / future work.
7. **DeepSeek V4 anchors.** Are the V4 anchors (Quick Instruction tokens, agentic-search-vs-RAG) cited only as motivation, with no reproduction or improvement claim?
8. **Decision-document consistency.** Are `SMOKE_DECISION.md`, `ORACLE_GAP_DECISION.md`, `TRACK_DECISION.md`, `COMPUTE_ESCALATION_DECISION.md`, `ED_PACKAGE_DECISION.md`, and `MAIN_RESCUE_OPTION.md` mutually consistent? Any contradiction?
9. **Forbidden-escalation hygiene.** Any sign in the repo of real API calls, controller training, KV/readout training, GPU jobs, or dataset expansion that should not be there given BACKUP / E&D-only?
10. **Concrete required fixes.** List, as a numbered punch-list, any concrete fixes required before this E&D artifact is ready for submission. Be specific (file path, what to change, why).

## Output format

```
# E&D Artifact Review

## Verdict
ready-for-submission / needs-fixes / not-ready

## Summary
<one paragraph>

## Findings
<question-by-question, citing files>

## Required fixes
1. <file>: <change> — <why>
2. ...
```

Do not propose Main Track promotion. Do not propose KV-amortized readout as a contribution. Do not propose dataset / API / GPU expansion in the punch-list (those are gated behind `reports/MAIN_RESCUE_OPTION.md` and require user approval).
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")


def main() -> int:
    os.chdir(REPO_ROOT)

    present, missing = _check_required_files()
    decision_findings = _check_decision_docs()
    forbidden_findings = _scan_forbidden_escalation()
    smoke = _smoke_numbers()

    bundle_path = REPO_ROOT / "review_bundles" / "gate_ed_artifact_review.md"
    prompt_path = REPO_ROOT / "review_bundles" / "gate_ed_artifact_review_prompt.md"
    _write_review_bundle(bundle_path)
    _write_codex_prompt(prompt_path)

    manifest = {
        "package_decision": "E&D-PACKAGE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke_numbers": smoke,
        "required_files_present": present,
        "missing_files": missing,
        "decision_doc_findings": decision_findings,
        "forbidden_escalation_findings": forbidden_findings,
        "codex_review_status": "pending",
        "reproduction_commands": [
            "python -m pytest tests -q",
            "python scripts/run_smoke_math_code.py",
            "python scripts/run_smoke_evidence.py",
            "python scripts/enumerate_oracle_graph.py",
            "python scripts/run_baselines.py",
            "python scripts/run_heuristic_bdelg.py",
            "python scripts/aggregate_results.py",
            "python scripts/make_figures.py",
            "python scripts/prepare_ed_package.py",
        ],
        "generated_review_bundle": str(bundle_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "generated_codex_prompt": str(prompt_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    out_manifest = REPO_ROOT / "experiments" / "ed_artifact_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Console summary.
    print(f"wrote {out_manifest.relative_to(REPO_ROOT)}")
    print(f"wrote {bundle_path.relative_to(REPO_ROOT)}")
    print(f"wrote {prompt_path.relative_to(REPO_ROOT)}")
    if missing:
        print(f"WARNING: missing required files: {missing}")
    violations = [f for f in decision_findings if f.get("status") == "VIOLATION"]
    if violations:
        print(f"WARNING: decision-doc violations: {violations}")
    errs = [f for f in forbidden_findings if f.get("severity") == "error"]
    if errs:
        print(f"ERROR: forbidden-escalation findings: {errs}")
    print("If pytest was not run by this script, run: python -m pytest tests -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
