# LEAN_TREE_MANIFEST

This is the **lean reviewer-facing artifact** for the NeurIPS 2026 Evaluations & Datasets submission of **AgentControl**. It is hosted at the anonymous artifact URL `https://anonymous.4open.science/r/agentcontrol-ed26-artifact-16A1/` and at the anonymous GitHub repo it mirrors.

## What this tree contains

| dir / file | purpose |
|---|---|
| `src/` | Core AgentControl harness (deliberation graph, oracle DP, providers, traces, metrics, policies). |
| `tests/` | 32 pytest unit tests covering the harness; pass in <1 s with no API calls. |
| `configs/` | Smoke / rescue / baseline YAML configs. |
| `cache/` | Cached LLM responses (deterministic-key-addressed) for review-time replay without spending. |
| `traces/` | Replayable JSONL traces for the smoke and rescue runs. |
| `experiments/` | Experiment summaries (JSON) referenced by the paper. |
| `figures/` | Figure-generation outputs referenced by the paper. |
| `method_discovery/` | Method-discovery sweep harness (configs, experiments, scripts, src, traces). |
| `main_rescue_gpu/` | Rescue / hard-regime evaluation harness (configs, experiments, scripts, src, traces). |
| `main_apollo/configs/` | Apollo-cycle configs. |
| `main_apollo/verifier_risk/` | Verifier-risk falsification (experiments, scripts). |
| `main_apollo/kv_readout/` | KV-readout falsification (experiments, scripts). |
| `scripts/` | Research scripts (data-pool builders, baselines, oracles, ablations, bootstrap, sensitivity, table / figure generators). Submission-pipeline tooling is excluded. |
| `submission/REPRODUCIBILITY_COMMANDS.md` | Operator-readable command list to reproduce every reported number from cache. |
| `submission/CACHE_TRACE_POLICY.md` | License + reuse policy for cache and traces. |
| `README.md`, `REPRODUCIBILITY.md`, `DATASET_METADATA.md`, `ARTIFACT_OVERVIEW.md`, `LICENSE`, `pyproject.toml`, `.env.example` | Standard reviewer-facing root files. |

## What this tree does NOT contain (intentional)

`paper/`, `latex/`, `reports/`, `review_bundles/`, sub-project `review_bundles/` and `reports/`, OpenReview operator / upload-packet docs, war-room / historical decision logs, bundle-builder scripts (`build_final_anon_bundle.py`), URL-substitution scripts (`finalize_with_artifact_url.py`), cluster-cleanroom scripts (`cluster_cleanroom_check.{py,sh}`), nested PDFs / LaTeX files / zip files, `agentcontrol_chatgpt_final_review.zip`, `next_main_project/`, original `.git/` history, real `.env` files or secrets.

## Anonymisation

All of the following have been scrubbed across the tree (0 hits in grep audit):

- author names (`Bowen` / `bowenzhu` / `bdju2` / `Zhu`)
- email addresses (real)
- institution names (`MIT` outside "MIT License", `CSAIL`, `NCSA`, `ORCD`)
- institutional domains (`mit.edu`)
- cluster identifiers (`orcd-login`, `slurm-login`, `engaging` / `Engaging`)
- auth methods (`Kerberos`, `GlobusAuth`, `Duo`)
- absolute home paths (`C:\Users\bdju2`, `/home/bdju2`, `/Users/bdju2`)
- compiled bytecode (`__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.egg-info/`)
- API key shapes (real `sk-…`, `AKIA…`, populated `*_API_KEY=…`)
- placeholder `[ANON_ARTIFACT_URL]` (substituted everywhere)

## Reviewer reproduction

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q                # expect 32 passed
# full pipeline: see submission/REPRODUCIBILITY_COMMANDS.md
```

No API calls. No GPU jobs. No paid spend.
