"""Hard-regime ledger reconciliation by request-hash recomputation.

Recomputes the request_hash for every (task, action, provider, model) tuple
the hard-regime collection issued, then cross-references against
`cache/cost_ledger.jsonl` to identify exactly which actual_api_call rows
belong to the hard-regime sprint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentcontrol.providers import ProviderRequest, deterministic_cache_key  # noqa: E402
from agentcontrol.hard_regime_tasks import get_pool  # noqa: E402

# Mirror the prompt builder + per-action max_tokens from
# scripts/run_hard_regime_collection.py exactly (kept short here).
ACTIONS = [
    "cheap_answer", "cheap_repair", "strong_hint", "cheap_repair_after_hint",
    "strong_critique", "strong_checklist", "strong_answer",
]


def _system(task):
    f = task["family"]
    if f == "math":
        return ("Careful arithmetic / number-theory assistant. Answer with the "
                "final integer or short numeric answer ONLY, on the last line, "
                "no explanation prose, no units.")
    if f == "code":
        return ("Python coder. Output exactly one Python code block containing "
                "the requested function definition. No tests, no commentary.")
    if f == "evidence":
        return ("Citation-grounded QA assistant. Answer using ONLY the provided "
                "evidence. End your response with the answer phrase followed by "
                "exactly one bracketed citation [doc_xxx] referencing the supporting document.")
    return "Careful assistant."


def _user(task, action):
    f = task["family"]
    if f == "math":
        q = task["question"]
        m = {
            "cheap_answer": f"Question: {q}\nFinal answer:",
            "cheap_repair": f"Question: {q}\nA prior cheap attempt may be wrong. Re-derive carefully and give the final answer:",
            "strong_hint": f"Question: {q}\nGive ONE short hint on the key step. Do NOT give the final answer.",
            "cheap_repair_after_hint": f"Question: {q}\nUse careful step-by-step reasoning, then output ONLY the final number on the last line.",
            "strong_critique": f"Question: {q}\nName the most common mistake to avoid. One sentence. No final answer.",
            "strong_checklist": f"Question: {q}\nList 2 verification checks (no answer).",
            "strong_answer": f"Question: {q}\nThink carefully, then output the final answer on the last line.",
        }
        return m[action]
    if f == "code":
        prompt = task["prompt"]
        m = {
            "cheap_answer": f"Task: {prompt}\nWrite the function in a single Python code block.",
            "cheap_repair": f"Task: {prompt}\nA prior attempt may have a bug. Write a correct version in a Python code block.",
            "strong_hint": f"Task: {prompt}\nGive one short algorithmic hint (no code).",
            "cheap_repair_after_hint": f"Task: {prompt}\nWrite the correct function carefully, attending to typical edge cases (zero, empty, negative, duplicates).",
            "strong_critique": f"Task: {prompt}\nName the most common bug to avoid. One sentence, no code.",
            "strong_checklist": f"Task: {prompt}\nList 2 short test cases worth checking (no code).",
            "strong_answer": f"Task: {prompt}\nThink carefully, then write the function in a single Python code block.",
        }
        return m[action]
    if f == "evidence":
        q = task["question"]
        ev = "\n".join(f"{k}: {v}" for k, v in task["evidence"].items())
        m = {
            "cheap_answer": f"Evidence:\n{ev}\nQuestion: {q}\nAnswer with the answer phrase and append exactly one [doc_xxx] citation from the evidence above.",
            "cheap_repair": f"Evidence:\n{ev}\nQuestion: {q}\nReread carefully and answer with the answer phrase and exactly one [doc_xxx] citation.",
            "strong_hint": f"Evidence:\n{ev}\nQuestion: {q}\nGive one short hint that names the relevant doc id, no answer.",
            "cheap_repair_after_hint": f"Evidence:\n{ev}\nQuestion: {q}\nAnswer with the answer phrase and exactly one [doc_xxx] citation strictly from the supporting evidence above. Do not cite a distractor.",
            "strong_critique": f"Evidence:\n{ev}\nQuestion: {q}\nState the most common citation mistake. One sentence. No answer.",
            "strong_checklist": f"Evidence:\n{ev}\nQuestion: {q}\nList 2 verification checks (no answer).",
            "strong_answer": f"Evidence:\n{ev}\nQuestion: {q}\nThink carefully, then output the answer phrase followed by exactly one [doc_xxx] citation strictly from the evidence.",
        }
        return m[action]
    return task.get("question", "")


def _max_tokens(action):
    if action == "strong_answer": return 1024
    if action.startswith("strong_"): return 200
    if action == "cheap_answer": return 256
    return 512


PROVIDER_MODELS = {
    "deepseek": {"cheap": "deepseek-chat", "strong": "deepseek-reasoner"},
    "together": {"cheap": "Qwen/Qwen2.5-7B-Instruct-Turbo",
                 "strong": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
}


def _model_for(action, mods):
    if action in ("cheap_answer", "cheap_repair", "cheap_repair_after_hint"):
        return mods["cheap"]
    return mods["strong"]


def main() -> int:
    pool = get_pool()
    expected_hashes_per_provider = {p: set() for p in PROVIDER_MODELS}
    for tid, task in pool.items():
        for action in ACTIONS:
            for prov, mods in PROVIDER_MODELS.items():
                req = ProviderRequest(
                    provider=prov,
                    model=_model_for(action, mods),
                    system=_system(task),
                    messages=[{"role": "user", "content": _user(task, action)}],
                    params={"temperature": 0.0, "max_tokens": _max_tokens(action)},
                )
                expected_hashes_per_provider[prov].add(deterministic_cache_key(req))

    ledger = REPO / "cache" / "cost_ledger.jsonl"
    real_per_provider = {p: 0 for p in PROVIDER_MODELS}
    real_spend_per_provider = {p: 0.0 for p in PROVIDER_MODELS}
    matched_hashes_per_provider = {p: set() for p in PROVIDER_MODELS}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        prov = row.get("provider")
        if prov not in expected_hashes_per_provider:
            continue
        h = row.get("request_hash")
        if h in expected_hashes_per_provider[prov] and row.get("actual_api_call"):
            real_per_provider[prov] += 1
            real_spend_per_provider[prov] += float(row.get("cost_usd", 0.0))
            matched_hashes_per_provider[prov].add(h)

    manifest = {
        "expected_calls_per_provider": {p: len(s) for p, s in expected_hashes_per_provider.items()},
        "real_actual_api_call_rows_per_provider": real_per_provider,
        "real_spend_usd_per_provider": real_spend_per_provider,
        "real_spend_usd_total": sum(real_spend_per_provider.values()),
        "n_distinct_request_hashes_matched_per_provider": {p: len(s) for p, s in matched_hashes_per_provider.items()},
        "protocol": (
            "Reconciliation recomputes the deterministic request_hash for every "
            "(task, action, provider, model) tuple from get_pool() using the "
            "exact same prompt builder as scripts/run_hard_regime_collection.py, "
            "then matches against rows in cache/cost_ledger.jsonl with "
            "actual_api_call=true and matching provider. This isolates the "
            "hard-regime sprint's real API spend from prior rescue calls."
        ),
    }
    out = REPO / "experiments" / "hard_regime_ledger_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    print(f"expected calls per provider: {manifest['expected_calls_per_provider']}")
    print(f"real actual_api_call rows per provider: {real_per_provider}")
    print(f"real spend per provider: {real_spend_per_provider}")
    print(f"real spend total: ${manifest['real_spend_usd_total']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
