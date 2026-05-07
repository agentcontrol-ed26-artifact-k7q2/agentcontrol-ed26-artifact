# AgentControl: A Fair-Baseline Protocol for Evaluating Budgeted LLM Agent Orchestration

**Track**: NeurIPS 2026 **Evaluations & Datasets**.

## What this is

AgentControl is a **reproducible cached-replayable evaluation harness** plus a **fair-baseline regime-mapping methodology** for budgeted LLM agent orchestration. The contribution is the **methodology that catches over-claims arising from missing baselines**, plus the calibrated regime map and three worked falsifications.

Apparent gains from LLM agent orchestration can be artifacts of unfair baselines. AgentControl introduces a three-rule fair-baseline protocol — action-set fairness, strongest-admissible baselines, and leakage-resistant evaluation — and demonstrates it through three cached, reproducible falsifications.

## The three fair-baseline rules

1. **Action-set fairness.** Any action available to the proposed method must also be available to the baseline (or the action must be justified as structurally specific with bootstrap-validated end-to-end utility).
2. **Strongest-admissible baseline.** The reported bar is `max(production-admissible non-method baselines)` per label / metric, with bootstrap CI strictly above.
3. **Leakage-resistant evaluation.** Labels evaluated under family-held-out and source-held-out splits, not just random splits.

## The three falsifications (load-bearing)

| # | rule | falsification | result |
|---|---|---|---|
| 1 | action-set fairness | F2: graph-vs-router-with-observation on `data_analysis_code` (n=50) | apparent +22.61% saving collapses to **0.00% with bootstrap CI [0%, 0%] strict** |
| 2 | strongest-admissible baseline | Apollo verifier-risk on n=60 adversarial pool | `risk_constrained` is **strictly Pareto-dominated by `always_cheap_exact`** at low/moderate exact-verifier cost |
| 3 | leakage-resistant evaluation | Apollo A-GPU CPU pre-screen on n=388 cached tasks | random-CV AUROC 0.726–0.857 **collapses to chance ≈ 0.500 under family-LOFO splits**; the labels are not in the prompts within our task pool |

## Minimal no-spend reproduction

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q                    # 32 / 32 pass; $0; CPU-only
```

This validates the harness, cache, trace replay, oracle DP, and verifier discipline. It does NOT regenerate the paper's reported numbers — see `REPRODUCIBILITY.md` for the full cache-only replay (≈ 30 commands, ≈ 5 CPU-min, **$0** incremental real-API spend).

## Artifact contents

```
src/                  # core harness + sprint-6/7/8 forks
tests/                # 32 unit tests
scripts/              # 30+ pipeline scripts (smoke, real-API, analyses, audit, build)
configs/              # YAML configs (anonymized cluster placeholders)
cache/                # cost ledger ($0.65 cumulative) + cached real-API responses
traces/               # replayable JSONL traces
experiments/          # per-sprint summary JSONs
figures/              # Pareto + 4 hard-regime figures
paper/                # paper drafts (ed_final_*.md + assembled neurips_ed_final_draft.md)
reports/              # ≈ 40 decision docs (some marked HISTORICAL)
review_bundles/       # 9 Codex review bundles + outputs
main_rescue_gpu/      # sprint-6 fork (interactive)
method_discovery/     # sprint-7 (F1) + sprint-8 (F2) forks
main_apollo/          # sprint-9 verifier-risk + A-GPU CPU pre-screen
submission/           # NeurIPS submission package (16 docs incl. final manifests)
```

## What this artifact does NOT claim

To set reviewer expectations explicitly:

- ❌ no Main-Track method-superiority claim;
- ❌ no SOTA on any benchmark;
- ❌ no learned-controller / KV-amortized-readout contribution (kill-switched on CPU evidence; cumulative GPU hours = 0);
- ❌ no DeepSeek V4 reproduction;
- ❌ no kernel-level / sparse-attention / TensorRT-LLM / Blackwell serving claim (orthogonal; see `reports/GVR_IMPACT_DECISION.md`);
- ❌ no production-deployment claim;
- ❌ no provider-ranking claim.

## Engineering invariants

- 32 / 32 unit tests pass.
- 2,400+ real-API calls; **$0.65 cumulative real spend** across 9 sprints.
- **Zero** GPU jobs submitted; **zero** controller training; **zero** KV/readout training.
- All real calls cached with deterministic SHA256 keys; second-pass replay = $0.
- Append-only cost ledger with deterministic-request-hash reconciliation manifest.
- Replayable JSONL traces.
- 9 successful Codex review passes; 45 cumulative needs-fixes items addressed.

## Where to go next

- `REPRODUCIBILITY.md` — full no-spend replay command list.
- `ARTIFACT_OVERVIEW.md` — high-level tour of the artifact layout.
- `DATASET_METADATA.md` — task-pool provenance, intended use, prohibited use.
- `LICENSE` — code MIT; data CC-BY-4.0; cached LLM responses included for academic reproducibility subject to provider ToS.
- `submission/` — NeurIPS submission package (title, abstract, OpenReview metadata, claim audit, anonymization audit, bundle manifest, checklists, human sign-off).
- `paper/ed_final_*.md` — the nine submission-facing source files; assembled into `submission/neurips_ed_submission_draft.md`.

## Reproducibility one-line summary

```
$ python -m pip install -e ".[dev]" && python -m pytest tests -q
```

Expected: `32 passed`. Expected real-API spend: **$0**. Expected GPU hours: **0**.
