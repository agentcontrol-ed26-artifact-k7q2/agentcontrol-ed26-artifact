# CACHE_TRACE_POLICY

Policy for releasing cached LLM provider responses and replayable traces alongside the artifact.

## What we ship

| asset | size | rows | content | release decision |
|---|---|---|---|---|
| `cache/provider/*.json` | ≈ 30 MB | 2,400+ files | full provider request+response payloads (model, prompt, completion, usage, latency) | **RELEASE in full** |
| `cache/cost_ledger.jsonl` | < 1 MB | 2,400+ rows | per-call accounting (request_hash, provider, model, tokens, cost_usd, latency_ms, cache_hit, actual_api_call) | **RELEASE in full** |
| `traces/*.jsonl` | ≈ 5 MB | per-task action trajectories | task_id, family, action, success, cost, latency, cache_hit | **RELEASE in full** |

## Why release in full

Releasing the cached responses is what makes the paper reproducible at $0. Without them, every reviewer who wants to re-derive the numbers must spend ≥ $0.65 of their own real-API budget. With them, the full pipeline replay is deterministic and free.

## Provider Terms of Service

Cached responses are **included** in the artifact bundle for academic reproducibility. Whether they may be **publicly redistributed** depends on each provider's current Terms of Service; this is the authors' best understanding as of the dates noted, and authors will re-verify provider terms before camera-ready distribution.

- **DeepSeek API ToS** (read 2026-04): the authors read the terms to support user output rights and academic / research use. Cached payloads include `model: "deepseek-chat"` or `"deepseek-reasoner"` and the timestamp of collection (visible in ledger). Final redistribution scope will be re-verified against DeepSeek's terms at camera-ready.
- **Together AI ToS** (read 2026-04): the authors read the terms to support user content/output rights. Cached payloads for `Qwen/Qwen2.5-7B-Instruct-Turbo` and `meta-llama/Llama-3.3-70B-Instruct-Turbo` preserve model attribution in the JSON `model` field. Final redistribution scope will be re-verified against Together AI's terms at camera-ready.

In all cases, redistribution is academic-reproducibility-only, attribution-required, no commercial redistribution, and no model-training / fine-tuning / distillation reuse.

We will add a one-paragraph notice in the release `README.md`:

> Cached LLM responses in `cache/provider/` are included for academic reproducibility, subject to the original providers' Terms of Service; public release should be reviewed against provider terms before camera-ready distribution. Reuse for model training, fine-tuning, or distillation is prohibited. Attribution is required (each cached payload's `model` field preserves model identity); no commercial redistribution.

## What is NOT in the cache

- No API keys (verified by `submission/SECURITY_SECRETS_AUDIT.md`).
- No `Authorization` headers.
- No user-identifying metadata (account ID, organization ID).
- No PII in prompts or responses (task pools are synthetic-local with deterministic verifiers; spot-checked).

## Trace format

Each trace row is one action step:

```json
{
  "task_id": "math_001",
  "family": "math",
  "regime": "easy_saturation",
  "action": "cheap_answer",
  "success": true,
  "cost_usd": 0.00012,
  "input_tokens": 87,
  "output_tokens": 24,
  "latency_ms": 612,
  "cache_hit": true,
  "actual_api_call": false,
  "request_hash": "<sha256>",
  "provider": "deepseek",
  "model": "deepseek-chat",
  "timestamp": "2026-04-18T19:32:11Z"
}
```

`cache_hit: true / actual_api_call: false` rows dominate any second-pass replay.

## Redactions applied

- None required. Synthetic-local task pools mean prompts contain no PII.
- Distractor documents in evidence-QA tasks are programmatically generated; no copyrighted source material.

## Hash determinism

Cache key construction (see `src/agentcontrol/cached_provider.py`):

```python
key = sha256(canonical_json({
    "provider": provider,
    "model": model,
    "messages": messages,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "system": system_prompt,
})).hexdigest()
```

Canonical JSON: keys sorted, no whitespace, NFC unicode. Independent of `PYTHONHASHSEED`. Re-running the same task on the same code produces the same hash and a guaranteed cache hit.

## Replay invariants enforced in code

- `CachedProvider.call(...)` raises `RealApiCallNotApproved` if the per-sprint env-var gate is unset AND the request is a cache miss.
- `actual_api_call=true` rows in the ledger correspond 1:1 with provider HTTP calls; verified by `scripts/reconcile_hard_regime_ledger.py`.
- Trace replay scripts (`run_*_analyses.py`) never construct a real provider client; they read the cache directly.
