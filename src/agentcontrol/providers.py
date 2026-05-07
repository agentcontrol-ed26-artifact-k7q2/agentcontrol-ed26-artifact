from __future__ import annotations
import json, re, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from .utils import append_jsonl, ensure_dir, sha256_json


@dataclass(frozen=True)
class ProviderRequest:
    provider: str
    model: str
    messages: list[dict[str, str]]
    params: dict[str, Any] = field(default_factory=dict)
    system: str = ""
    tool_schema: dict[str, Any] | None = None


@dataclass
class ProviderResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    cache_hit: bool = False
    request_hash: str = ""


class Provider(Protocol):
    def generate(self, request: ProviderRequest) -> ProviderResponse: ...


def deterministic_cache_key(request: ProviderRequest) -> str:
    return sha256_json(asdict(request))


class DummyProvider:
    def __init__(self, provider_name: str = 'dummy') -> None:
        self.provider_name = provider_name
        self.call_count = 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.call_count += 1
        start = time.time()
        prompt = '\n'.join(m.get('content', '') for m in request.messages)
        text = self._respond(prompt, request.model)
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(text) // 4)
        if self._is_strong(request.model):
            in_rate, out_rate = 0.000002, 0.000008
        else:
            in_rate, out_rate = 0.0000002, 0.0000006
        return ProviderResponse(
            text=text,
            provider=request.provider,
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=input_tokens * in_rate + output_tokens * out_rate,
            latency_ms=max(1, int((time.time() - start) * 1000)),
            cache_hit=False,
            request_hash=deterministic_cache_key(request),
        )

    def _is_strong(self, model: str) -> bool:
        m = model.lower()
        return 'strong' in m or 'pro' in m or 'frontier' in m

    def _marker(self, prompt: str, name: str) -> str | None:
        m = re.search(rf'{re.escape(name)}\s*:\s*(.+)', prompt)
        return m.group(1).strip().splitlines()[0].strip() if m else None

    def _respond(self, prompt: str, model: str) -> str:
        if 'CODE_TASK:' in prompt:
            return self._code_response(prompt, model)
        if 'EVIDENCE_QA:' in prompt:
            if self._is_strong(model) or 'REPAIR_WITH_HINT' in prompt:
                return self._marker(prompt, 'DUMMY_STRONG') or self._marker(prompt, 'DUMMY_ANSWER') or 'I do not know.'
            return self._marker(prompt, 'DUMMY_CHEAP') or 'Unsupported answer.'
        if 'STRONG_HINT' in prompt:
            return self._marker(prompt, 'DUMMY_HINT') or 'Hint: verify intermediate steps.'
        if 'REPAIR_WITH_HINT' in prompt:
            return self._marker(prompt, 'DUMMY_REPAIR') or self._marker(prompt, 'DUMMY_STRONG') or '42'
        if self._is_strong(model):
            return self._marker(prompt, 'DUMMY_STRONG') or self._marker(prompt, 'DUMMY_ANSWER') or '42'
        return self._marker(prompt, 'DUMMY_CHEAP') or '0'

    def _code_response(self, prompt: str, model: str) -> str:
        strong = self._is_strong(model) or 'REPAIR_WITH_HINT' in prompt
        if 'code_add' in prompt:
            return 'def add(a, b):\n    return a + b\n'
        if 'code_is_even' in prompt:
            return 'def is_even(n):\n    return n % 2 == 0\n' if strong else 'def is_even(n):\n    return n % 2 == 1\n'
        if 'code_factorial' in prompt:
            return 'def factorial(n):\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n' if strong else 'def factorial(n):\n    return 0\n'
        if 'code_reverse_words' in prompt:
            return "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n" if strong else 'def reverse_words(s):\n    return s[::-1]\n'
        return 'def solution(*args, **kwargs):\n    return None\n'


class CachedProvider:
    def __init__(self, provider: Provider, cache_dir: str | Path = 'cache/provider', ledger_path: str | Path = 'cache/cost_ledger.jsonl', replay_only: bool = False) -> None:
        self.provider = provider
        self.cache_dir = ensure_dir(cache_dir)
        self.ledger_path = Path(ledger_path)
        ensure_dir(self.ledger_path.parent)
        self.replay_only = replay_only

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        key = deterministic_cache_key(request)
        path = self.cache_dir / f'{key}.json'
        if path.exists():
            raw = json.loads(path.read_text(encoding='utf-8'))
            resp = ProviderResponse(**raw)
            resp.cache_hit = True
            resp.cost_usd = 0.0
            resp.request_hash = key
            self._append_ledger(request, resp, actual_api_call=False)
            return resp
        if self.replay_only:
            raise RuntimeError(f'Replay-only mode: missing cache entry {key}')
        resp = self.provider.generate(request)
        resp.cache_hit = False
        resp.request_hash = key
        path.write_text(json.dumps(asdict(resp), indent=2, sort_keys=True), encoding='utf-8')
        self._append_ledger(request, resp, actual_api_call=True)
        return resp

    def _append_ledger(self, request: ProviderRequest, response: ProviderResponse, actual_api_call: bool) -> None:
        append_jsonl(self.ledger_path, {
            'request_hash': response.request_hash,
            'provider': request.provider,
            'model': request.model,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
            'cost_usd': response.cost_usd,
            'latency_ms': response.latency_ms,
            'cache_hit': response.cache_hit,
            'actual_api_call': actual_api_call,
        })


def total_actual_spend_usd(ledger_path: str | Path) -> float:
    p = Path(ledger_path)
    if not p.exists():
        return 0.0
    total = 0.0
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('actual_api_call'):
                total += float(row.get('cost_usd', 0.0))
    return total
