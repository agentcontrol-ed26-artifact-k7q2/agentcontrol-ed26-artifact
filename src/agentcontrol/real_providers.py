"""Real-API provider implementations for DeepSeek and Together AI.

Both implement the ``Provider`` protocol from ``providers.py`` so they slot
into ``CachedProvider`` directly. Cost is computed from per-token rates
returned in the API response (or fallback rates).
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict

import requests

from .providers import Provider, ProviderRequest, ProviderResponse, deterministic_cache_key


# DeepSeek pricing (USD / 1M tokens). Cache hit/miss rates differ; we treat
# every uncached call as cache-miss-priced (conservative).
DEEPSEEK_PRICES = {
    "deepseek-chat":    {"input": 0.27, "output": 1.10},     # V3.2-Exp / V4 non-thinking
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},    # V4 thinking / R1
}
TOGETHER_PRICES = {
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": {"input": 0.06, "output": 0.06},
    "Qwen/Qwen2.5-7B-Instruct-Turbo":   {"input": 0.30, "output": 0.30},
    "Qwen/Qwen2.5-72B-Instruct-Turbo":  {"input": 1.20, "output": 1.20},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
}
DEFAULT_PRICE = {"input": 1.0, "output": 1.0}


class _BaseHTTPProvider:
    name = "base"
    base_url = ""
    auth_env = ""
    prices = {}

    def __init__(self, timeout_s: int = 90):
        self.timeout_s = timeout_s
        self.api_key = os.environ.get(self.auth_env, "")
        if not self.api_key:
            raise RuntimeError(f"{self.auth_env} is not set in environment")

    def _post_chat(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        if r.status_code >= 400:
            raise RuntimeError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.prices.get(model, DEFAULT_PRICE)
        return (input_tokens / 1e6) * p["input"] + (output_tokens / 1e6) * p["output"]

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        msgs = []
        if request.system:
            msgs.append({"role": "system", "content": request.system})
        msgs.extend(request.messages)
        payload = {
            "model": request.model,
            "messages": msgs,
            "temperature": request.params.get("temperature", 0.0),
            "max_tokens": request.params.get("max_tokens", 1024),
        }
        start = time.time()
        body = self._post_chat(payload)
        latency_ms = int((time.time() - start) * 1000)
        choice = body["choices"][0]
        text = choice["message"]["content"] or ""
        usage = body.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", 0)) or max(1, sum(len(m["content"]) for m in msgs) // 4)
        out_tok = int(usage.get("completion_tokens", 0)) or max(1, len(text) // 4)
        return ProviderResponse(
            text=text,
            provider=request.provider,
            model=request.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self._cost(request.model, in_tok, out_tok),
            latency_ms=latency_ms,
            cache_hit=False,
            request_hash=deterministic_cache_key(request),
        )


class DeepSeekProvider(_BaseHTTPProvider):
    name = "deepseek"
    base_url = "https://api.deepseek.com"
    auth_env = "DEEPSEEK_API_KEY"
    prices = DEEPSEEK_PRICES


class TogetherProvider(_BaseHTTPProvider):
    name = "together"
    base_url = "https://api.together.xyz/v1"
    auth_env = "TOGETHER_API_KEY"
    prices = TOGETHER_PRICES
