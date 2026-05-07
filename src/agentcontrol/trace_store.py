from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from .utils import append_jsonl, read_jsonl, sha256_json


@dataclass
class TraceEvent:
    task_id: str
    family: str
    step: int
    action: str
    state_hash: str
    prompt_hash: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    cost_usd: float = 0.0
    latency_ms: int = 0
    observation: str = ""
    verifier_result: dict[str, Any] = field(default_factory=dict)
    budget_remaining: float = 0.0
    success_final: bool | None = None


class TraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: TraceEvent) -> None:
        append_jsonl(self.path, asdict(event))

    def read(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path)


def state_hash(state: dict[str, Any]) -> str:
    return sha256_json(state)


class ReplayTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.rows = read_jsonl(path)
        self.idx = 0

    def next(self) -> dict[str, Any]:
        if self.idx >= len(self.rows):
            raise RuntimeError('Replay exhausted')
        row = self.rows[self.idx]
        self.idx += 1
        return row
