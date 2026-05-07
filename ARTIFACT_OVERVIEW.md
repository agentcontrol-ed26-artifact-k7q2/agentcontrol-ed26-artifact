# ARTIFACT_OVERVIEW

A high-level tour of what this repository contains, what each top-level directory is for, and where to start reading.

## TL;DR for reviewers

- **Start with**: [`README.md`](README.md) (title, track, three rules, three falsifications, no-spend repro).
- **For the full repro**: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`submission/REPRODUCIBILITY_COMMANDS.md`](submission/REPRODUCIBILITY_COMMANDS.md).
- **For the paper**: assembly map at [`submission/section_map.md`](submission/section_map.md); master draft at [`submission/neurips_ed_submission_draft.md`](submission/neurips_ed_submission_draft.md); component sources under [`paper/ed_final_*.md`](paper/).
- **For the claim audit**: [`submission/FINAL_CLAIM_AUDIT.md`](submission/FINAL_CLAIM_AUDIT.md).
- **For the data-side metadata**: [`DATASET_METADATA.md`](DATASET_METADATA.md) and [`submission/croissant_agentcontrol.json`](submission/croissant_agentcontrol.json).
- **For licensing**: [`LICENSE`](LICENSE) (code MIT; data CC-BY-4.0; cached LLM responses subject to provider ToS).

## Top-level layout

```
src/                    AgentControl core harness + sprint-6/7/8 forks
tests/                  32 unit tests; pytest entry point
scripts/                30+ pipeline scripts (smoke, real-API, analyses, audit, build)
configs/                YAML configs (cluster names anonymized via [ANONYMIZED_*] placeholders)
cache/                  cost_ledger.jsonl + provider/*.json (cached real-API responses)
traces/                 replayable JSONL traces (one row per task-action)
experiments/            per-sprint summary JSONs feeding the paper's tables / figures
figures/                Pareto + 4 hard-regime figures
paper/                  paper sources (ed_final_*.md) + master draft (neurips_ed_final_draft.md)
reports/                ≈ 40 decision docs (some marked HISTORICAL)
review_bundles/         9 Codex review bundles + outputs
main_rescue_gpu/        sprint-6 fork (interactive n=120)
method_discovery/       sprint-7 (F1 agentic search) + sprint-8 (F2 falsification) forks
main_apollo/            sprint-9 verifier-risk + A-GPU CPU pre-screen
submission/             NeurIPS submission package (16+ documents incl. manifests)

README.md               reviewer-facing entry point
REPRODUCIBILITY.md      one-command sanity check + minimal repro + full replay path
LICENSE                 three-class licensing (code / data / cached LLM responses)
DATASET_METADATA.md     data-side provenance, intended use, prohibited use, RAI notes
ARTIFACT_OVERVIEW.md    this file
PROJECT_SUMMARY.md      cumulative project summary
pyproject.toml          install metadata
.env.example            template for real-API re-collection (NOT used for paper repro)
```

## Asset classes

| asset class | location | release | role |
|---|---|---|---|
| Source code | `src/`, `scripts/`, `tests/`, `configs/`, `main_rescue_gpu/`, `method_discovery/`, `main_apollo/` | MIT | the harness + the falsifications |
| Task pools | code under `*/tasks_*.py`, `*/build_*_pool.py`, `*/build_*_dataset.py` | CC-BY-4.0 | what is evaluated |
| Traces | `traces/*.jsonl` | CC-BY-4.0 | per-action records |
| Experiment summaries | `experiments/*.json`, `*/experiments/*.json` | CC-BY-4.0 | what the paper cites |
| Figures | `figures/*.pdf` | CC-BY-4.0 | what the paper plots |
| Decision docs | `reports/`, `*/reports/`, `submission/` | CC-BY-4.0 | the project's reasoning trail |
| Codex review bundles | `*/review_bundles/` | CC-BY-4.0 | external-review trail |
| Paper drafts | `paper/`, `submission/neurips_ed_submission_draft.md` | CC-BY-4.0 | what's submitted |
| Cached LLM responses | `cache/provider/*.json` | for academic reproducibility, subject to provider ToS | makes second-pass replay $0 |
| Cost ledger | `cache/cost_ledger.jsonl` | CC-BY-4.0 | append-only, ground-truth real-API spend |

## Active vs historical decision docs

The project went through 9 sprints. Several decision docs from earlier sprints were superseded by later ones. Those are explicitly marked at the top with a `HISTORICAL` banner pointing to the canonical superseding doc, e.g.:

```
> HISTORICAL (frozen post-hard-regime / sprint 5). Superseded by submission/TRACK_DECISION_FINAL.md.
```

The active reviewer-path is exclusively the `submission/` directory plus the `paper/ed_final_*.md` files. Everything in `reports/`, `*/reports/`, `*/review_bundles/` should be treated as the project's reasoning trail, not as a current claim source.

## Engineering invariants

- 32 / 32 unit tests pass (`pytest tests -q`).
- All real-API calls go through `CachedProvider` with deterministic SHA256 keys.
- Append-only cost ledger; deterministic-request-hash reconciliation manifest at `experiments/hard_regime_ledger_manifest.json`.
- Trace replay never calls APIs.
- Budget gates per sprint (six distinct env vars) enforced.
- 9 successful Codex review passes; 45 cumulative needs-fixes items addressed.
- Cumulative real spend: **$0.65**. Cumulative GPU hours: **0**.

## What this artifact does NOT include

- No model weights.
- No fine-tuning data.
- No external benchmark datasets (everything is synthetic-local; see `DATASET_METADATA.md`).
- No human-subjects data; no PII.
- No commercial / production deployment policy.

## Pointers to in-repo files reviewers may want directly

| topic | file |
|---|---|
| Three rules | `paper/ed_final_fair_baseline_methodology.md` |
| Three falsifications (compact table) | `paper/table_fair_baseline_falsifications.md` |
| Three falsifications (full discussion) | `paper/ed_final_two_falsifications.md` |
| Regime map (24 cells) | `paper/ed_final_regime_map.md` |
| Limitations (16 numbered) | `paper/ed_final_limitations.md` |
| Reproducibility (one-liner + full replay) | `paper/ed_final_reproducibility.md` + `submission/REPRODUCIBILITY_COMMANDS.md` |
| Broader impact | `paper/ed_final_broader_impact.md` |
| Anonymization audit | `submission/FINAL_ANONYMIZATION_AUDIT.md` |
| Bundle manifest (SHA256) | `submission/FINAL_BUNDLE_MANIFEST.md` |
| Clean-unpack test report | `submission/FINAL_REPRO_UNPACK_TEST.md` |
| Human sign-off checklist | `submission/HUMAN_SIGNOFF.md` |
| GVR (kernel-level Top-K) orthogonality | `reports/GVR_IMPACT_DECISION.md` |
