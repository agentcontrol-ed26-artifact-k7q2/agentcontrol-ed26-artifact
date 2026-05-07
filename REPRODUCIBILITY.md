# REPRODUCIBILITY

How to reproduce the paper's reported numbers from the cached artifact. **No real-API calls are issued by any command in this file unless you explicitly set an `AGENTCONTROL_*_APPROVED=1` env var.**

Tested on Linux + macOS + Windows (under `bash`). Python ≥ 3.10.

## 1. One-command sanity check

```bash
python -m pip install -e ".[dev]" && python -m pytest tests -q
```

**Expected**: `32 passed`. **Expected real-API spend**: **$0**. **Expected GPU hours**: **0**.

This validates the harness, cache, trace replay, oracle DP, and verifier discipline. It does NOT regenerate the paper's reported numbers — for those, run the full pipeline replay below.

## 2. Minimal reproduction (key paper numbers, ≈ 30 seconds)

```bash
python main_apollo/kv_readout/scripts/audit_a_gpu_env.py
python main_apollo/kv_readout/scripts/build_a_gpu_readout_dataset.py
python main_apollo/kv_readout/scripts/run_a_gpu_baselines.py
```

This reproduces the Apollo A-GPU CPU pre-screen Table from §5 / `paper/ed_final_results.md`:

| label | random_5fold | family_LOFO_macro |
|---|---|---|
| verifier_risk | 0.857 | **0.500** |
| need_strong | 0.726 | **0.522** |
| escalate | 0.726 | **0.522** |
| graph_candidate | 0.800 | **0.517** |

## 3. Full pipeline replay (≈ 5 CPU-min; cache-only; **$0** real spend)

The full ≈ 30-script command list is documented in [`submission/REPRODUCIBILITY_COMMANDS.md`](submission/REPRODUCIBILITY_COMMANDS.md). It runs the analyses for sprints 1–9 strictly against `cache/provider/*.json` and `cache/cost_ledger.jsonl`; no provider HTTP calls.

## 4. API-gated commands (clearly marked) — **DO NOT RUN unless re-collecting**

The following commands issue **real-API calls** and will spend money. They refuse to run unless the corresponding `AGENTCONTROL_*_APPROVED=1` env var is set:

```bash
# DO NOT SET THESE unless you are intentionally re-collecting from scratch
# with new providers / models / pools and have funded the appropriate accounts.
export AGENTCONTROL_RESCUE_APPROVED=1                  # sprint 4
export AGENTCONTROL_HARD_REGIME_APPROVED=1             # sprint 5
export AGENTCONTROL_GPU_MAIN_RESCUE_APPROVED=1         # sprint 6
export AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED=1    # sprints 7, 8
export AGENTCONTROL_APOLLO_API_APPROVED=1              # sprint 9 verifier-risk
# Note: AGENTCONTROL_APOLLO_GPU_APPROVED was never set; A-GPU kill-switch fired.
```

**Warning**: do not set these env vars during reviewer-side reproduction. The paper's reported numbers are reproducible without them, from cache only.

## 5. No-spend replay invariants

- Cache key construction: deterministic SHA256 over canonical-JSON of the provider request body. Re-running the same task on the same code yields the same hash and a guaranteed cache hit.
- `cache/cost_ledger.jsonl` byte-identical before vs after a cache-only replay.
- No new entries in `cache/provider/`.
- All scripts respect their `AGENTCONTROL_*_APPROVED` env-var gates.

## 6. Output verification table

After replay, the following key files must exist and match the paper:

| file | reproduces |
|---|---|
| `experiments/hard_regime_summary_joint.json` | regime map (24 cells); 0 graph-headroom |
| `experiments/hard_regime_bootstrap.json` | CI [0%, 0%] for graph-vs-router |
| `method_discovery/experiments/f2_data_analysis_oracle_summary.json` | F2 falsification: 0% saving with fair router |
| `method_discovery/experiments/f1_agentic_vs_fixed_topk.json` | F1: −0.66% / −1.78% (within noise) |
| `main_apollo/verifier_risk/experiments/verifier_risk_policy_summary.json` | `always_cheap_exact` dominates `risk_constrained` at exact_cost ≤ 2.0 |
| `main_apollo/kv_readout/experiments/a_gpu_baselines.json` | family-LOFO AUROC ≈ 0.50 on every admissible-hard label |

## 7. Expected outcomes summary

- pytest: **32 / 32** pass in < 1 second.
- Full pipeline replay: deterministic; second pass is 100% cache hits.
- Cumulative incremental real-API spend during reviewer-side replay: **$0**.
- Cumulative GPU hours during reviewer-side replay: **0**.

## 8. Troubleshooting

- `ModuleNotFoundError`: re-run `python -m pip install -e ".[dev]"`.
- `FileNotFoundError: cache/provider/<sha>.json`: cache was not bundled; redownload the artifact ZIP.
- `RealApiCallNotApproved`: a script attempted a provider HTTP call without the corresponding `AGENTCONTROL_*_APPROVED=1` env var. This is the gate working as designed — **do not set the env var to silence it; re-run the analysis script (which uses cache only) instead of the collection script.**
- Hash mismatch on rebuild: `submission/FINAL_BUNDLE_MANIFEST.md` carries the SHA256 of the canonical bundle.
