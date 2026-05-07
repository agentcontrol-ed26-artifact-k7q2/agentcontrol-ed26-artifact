# REPRODUCIBILITY_COMMANDS

All commands required to reproduce the paper's reported numbers from the cached artifact. **No real-API calls are issued by any command in this file.** All scripts read from `cache/provider/*.json` and `cache/cost_ledger.jsonl`.

Prerequisite: Python ≥ 3.10. Tested on Linux + macOS + Windows (under `bash`).

## 0. Install

```bash
python -m pip install -e ".[dev]"
```

## 1. Sanity check (≈ 30 seconds)

```bash
python -m pytest tests -q
# expected: 32 passed
```

This validates the harness, cache, trace replay, oracle DP, and verifier discipline. **It does NOT regenerate paper numbers** — for those, run the full pipeline replay below.

## 2. Full pipeline replay (≈ 3–5 minutes; cache-only; $0 real spend)

### Sprints 1–3: harness + DummyProvider validation

```bash
python scripts/run_smoke_math_code.py
python scripts/run_smoke_evidence.py
python scripts/enumerate_oracle_graph.py
python scripts/run_baselines.py
python scripts/run_heuristic_bdelg.py
python scripts/aggregate_results.py
python scripts/make_figures.py
python scripts/prepare_ed_package.py
```

### Sprint 3.5: ED ablations + sensitivity + bootstrap

```bash
python scripts/run_ed_cached_ablations.py
python scripts/run_ed_sensitivity.py
python scripts/run_ed_bootstrap_ci.py
python scripts/run_ed_family_reweighting.py
python scripts/run_ed_oracle_sanity_checks.py
python scripts/make_ed_tables.py
```

### Sprints 4–5: real-API rescue + hard regime (cache-only here)

```bash
python scripts/run_rescue_analyses.py
python scripts/run_hard_regime_analyses.py
python scripts/run_hard_regime_bootstrap.py
python scripts/run_hard_regime_sensitivity.py
python scripts/make_hard_regime_tables.py
python scripts/make_hard_regime_figures.py
python scripts/reconcile_hard_regime_ledger.py
```

### Sprint 6: GPU Main Rescue Fork (interactive, n=120)

```bash
python main_rescue_gpu/scripts/run_local_interactive_oracles.py
python main_rescue_gpu/scripts/run_local_interactive_ablations.py
python main_rescue_gpu/scripts/run_local_interactive_bootstrap.py
python main_rescue_gpu/scripts/run_local_interactive_sensitivity.py
```

### Sprint 7: Method Discovery (F1 agentic search, n=60)

```bash
python method_discovery/scripts/run_rag_vs_agentic_graph.py
```

### Sprint 8: F2 falsification (data_analysis_code, n=50)

```bash
python method_discovery/scripts/run_f2_data_analysis_oracles.py
```

### Sprint 9: Apollo verifier-risk + A-GPU CPU pre-screen

```bash
python main_apollo/verifier_risk/scripts/run_verifier_risk_policies.py
python main_apollo/kv_readout/scripts/audit_a_gpu_env.py
python main_apollo/kv_readout/scripts/build_a_gpu_readout_dataset.py
python main_apollo/kv_readout/scripts/run_a_gpu_baselines.py
```

## 3. Output verification

After replay, the following key files should exist and match the paper:

| file | reproduces |
|---|---|
| `experiments/hard_regime_summary_joint.json` | Table: regime map (24 cells); 0 graph-headroom |
| `experiments/hard_regime_bootstrap.json` | CI [0%, 0%] for graph-vs-router |
| `method_discovery/experiments/f2_data_analysis_oracle_summary.json` | F2 falsification: 0% saving with fair router |
| `method_discovery/experiments/f1_agentic_vs_fixed_topk.json` | F1: −0.66% / −1.78% (within noise) |
| `main_apollo/verifier_risk/experiments/verifier_risk_policy_summary.json` | always_cheap_exact dominates risk_constrained at exact_cost ≤ 2.0 |
| `main_apollo/kv_readout/experiments/a_gpu_baselines.json` | family-LOFO AUROC ≈ 0.5 on all admissible-hard labels |
| `figures/pareto_main.pdf`, `figures/hard_regime_*.pdf` | paper figures |
| `paper/table_regime_map.md` | regime-map table |

## 4. Real-API re-collection (gated, optional)

To re-collect from scratch with new providers/models, set the appropriate env var per sprint and run the corresponding `run_*_real_collection.py`:

```bash
export AGENTCONTROL_RESCUE_APPROVED=1                  # sprint 4
export AGENTCONTROL_HARD_REGIME_APPROVED=1             # sprint 5
export AGENTCONTROL_GPU_MAIN_RESCUE_APPROVED=1         # sprint 6
export AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED=1    # sprints 7, 8
export AGENTCONTROL_APOLLO_API_APPROVED=1              # sprint 9 verifier-risk
# (AGENTCONTROL_APOLLO_GPU_APPROVED was never used; A-GPU kill-switch fired)
```

Real-API scripts also require `.env` with `DEEPSEEK_API_KEY=...` and `TOGETHER_API_KEY=...`. See `.env.example`.

## 5. Determinism

- All cache keys are SHA256 over the canonical request body (provider, model, messages, temperature, max_tokens, system).
- Distractor selection in evidence pools uses md5 (process-stable, independent of `PYTHONHASHSEED`).
- All bootstrap analyses use a fixed seed (2026); resampling order is reproducible.
- Second-pass replay produces only cache hits; cumulative incremental real-API spend is **$0**.

## 6. Troubleshooting

- "ModuleNotFoundError" → re-run `python -m pip install -e ".[dev]"`.
- "FileNotFoundError: cache/provider/<sha>.json" → cache was not bundled; redownload the artifact ZIP.
- "ledger row missing" → re-run the originating sprint's analyzer, not the collection script (collection scripts append; analyzers read).
- pytest hang on macOS → none expected; if observed, run with `-x` and report the first failure.
