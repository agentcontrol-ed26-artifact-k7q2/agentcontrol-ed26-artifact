"""Phase 5/6 (F1): collect cheap/strong outcomes on the agentic-search pool
under multiple retrieval regimes:
  - no retrieval (cold)
  - fixed top-k RAG for k in {1, 3, 5}
  - agentic iterative retrieve (entity-driven, 2 rounds, deterministic local)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "src"))

from agentcontrol.providers import CachedProvider, ProviderRequest  # noqa: E402
from agentcontrol.real_providers import DeepSeekProvider, TogetherProvider  # noqa: E402
from agentcontrol_method.agentic_corpus import (  # noqa: E402
    get_questions, get_corpus, retrieve_topk, verify_answer,
)

load_dotenv(REPO / ".env")

# Action set (LLM): each is one API call. Retrieval actions are local + free.
LLM_ACTIONS = [
    "cheap_cold",           # cheap, no retrieval
    "cheap_topk1",          # cheap with top-1 fixed RAG
    "cheap_topk3",          # cheap with top-3 fixed RAG
    "cheap_topk5",          # cheap with top-5 fixed RAG
    "cheap_agentic",        # cheap with agentic iterative retrieve (2 rounds)
    "strong_topk5",         # strong with top-5 fixed RAG (best fixed-RAG reference)
    "strong_agentic",       # strong with agentic iterative retrieve
]
# Cost units (protocol price model).
COST_UNITS = {
    "cheap_cold": 1.0,
    "cheap_topk1": 1.3, "cheap_topk3": 1.3, "cheap_topk5": 1.3,
    "cheap_agentic": 1.5,    # cheap (1.0) + 2 retrieve rounds (0.25 each)
    "strong_topk5": 10.3,
    "strong_agentic": 10.5,
    "retrieve_topk": 0.3,
    "retrieve_iterative": 0.5,
    "citation_check": 0.2,
}


def _system() -> str:
    return ("Citation-grounded QA assistant. Use ONLY the provided evidence. "
            "End your response with the answer phrase followed by exactly one "
            "bracketed citation [doc_xxx] referencing the supporting document.")


def _system_cold() -> str:
    return ("QA assistant. Answer with the answer phrase. If you do not know, "
            "say 'I do not know'.")


def _format_evidence(docs: list[tuple[str, str, float]]) -> str:
    return "\n".join(f"{d}: {t}" for d, t, _ in docs)


def _agentic_iterative(question: str, max_rounds: int = 2) -> list[tuple[str, str, float]]:
    """Deterministic local agentic search: round 1 retrieves on the question;
    round 2 retrieves on entities mined from round 1 docs that share lexical
    features with the question (i.e. expand on linking nouns).
    """
    seen_ids: set[str] = set()
    all_docs: list[tuple[str, str, float]] = []
    # Round 1: retrieve on the bare question.
    round1 = retrieve_topk(question, k=3)
    for d, t, s in round1:
        if d not in seen_ids:
            seen_ids.add(d)
            all_docs.append((d, t, s))
    # Round 2: extract proper nouns / capitalized terms from round-1 docs and
    # add them to the query; retrieve again excluding seen.
    if max_rounds >= 2 and round1:
        seed_text = " ".join(t for _, t, _ in round1)
        # cheap proper-noun extractor: capitalized words not at sentence start.
        nouns = re.findall(r"(?<=[a-z\.,\s])([A-Z][a-zA-Z]{3,})", seed_text)
        expanded = question + " " + " ".join(set(nouns))
        round2 = retrieve_topk(expanded, k=3, exclude=seen_ids)
        for d, t, s in round2:
            if d not in seen_ids:
                seen_ids.add(d)
                all_docs.append((d, t, s))
    return all_docs[:5]   # cap at 5 docs to keep prompts bounded


def _user_prompt(question: dict, action: str) -> tuple[str, list[str]]:
    """Build the model prompt and return (text, retrieved_doc_ids)."""
    q = question["question"]
    if action == "cheap_cold":
        return f"Question: {q}\nAnswer with the answer phrase only. If unknown, say 'I do not know'.", []
    k_map = {"cheap_topk1": 1, "cheap_topk3": 3, "cheap_topk5": 5, "strong_topk5": 5}
    if action in k_map:
        docs = retrieve_topk(q, k=k_map[action])
        ev = _format_evidence(docs)
        return (f"Evidence:\n{ev}\nQuestion: {q}\n"
                f"Answer with the answer phrase and exactly one [doc_xxx] citation."), [d for d, _, _ in docs]
    if action in ("cheap_agentic", "strong_agentic"):
        docs = _agentic_iterative(q, max_rounds=2)
        ev = _format_evidence(docs)
        return (f"Evidence (assembled by an iterative retrieve over 2 rounds):\n{ev}\n"
                f"Question: {q}\n"
                f"Answer with the answer phrase and exactly one [doc_xxx] citation that supports the answer."), [d for d, _, _ in docs]
    return q, []


def _model_for(action: str, mods: dict) -> tuple[str, str]:
    if action.startswith("cheap_"):
        return mods["cheap"]
    return mods["strong"]


def _max_tokens(action: str) -> int:
    if action.startswith("strong_"):
        return 1024
    return 256


def _spend(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    total = 0.0
    try:
        text = ledger.read_text(encoding="utf-8")
    except OSError:
        return 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("actual_api_call") and r.get("provider") in ("deepseek", "together"):
            total += float(r.get("cost_usd", 0.0))
    return total


def _api_call(cps, providers_cfg, question: dict, action: str) -> dict:
    pname, model = _model_for(action, providers_cfg)
    cp = cps[pname]
    user, retrieved_ids = _user_prompt(question, action)
    sys_prompt = _system_cold() if action == "cheap_cold" else _system()
    req = ProviderRequest(
        provider=pname, model=model, system=sys_prompt,
        messages=[{"role": "user", "content": user}],
        params={"temperature": 0.0, "max_tokens": _max_tokens(action)},
    )
    resp = cp.generate(req)
    text = resp.text or ""
    success, risk = verify_answer(question, text)
    return {
        "task_id": question["task_id"], "action": action,
        "success": bool(success), "unsupported_risk": float(risk),
        "retrieved_ids": retrieved_ids,
        "cost_usd": float(resp.cost_usd),
        "input_tokens": int(resp.input_tokens),
        "output_tokens": int(resp.output_tokens),
        "latency_ms": int(resp.latency_ms),
        "cache_hit": bool(resp.cache_hit),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-spend-usd", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--provider-mode", choices=["deepseek", "together"], default="deepseek")
    args = p.parse_args()

    if os.environ.get("AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED") != "1":
        print("AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED is not 1; refusing API run.")
        return 2

    questions = get_questions()
    print(f"agentic-search pool: {len(questions)} questions")
    ledger = REPO / "cache" / "cost_ledger.jsonl"
    initial = _spend(ledger)
    print(f"prior real-spend: ${initial:.4f}")

    if args.provider_mode == "deepseek":
        providers_cfg = {"cheap": ("deepseek", "deepseek-chat"),
                         "strong": ("deepseek", "deepseek-reasoner")}
        cps = {"deepseek": CachedProvider(provider=DeepSeekProvider(),
                                          cache_dir=REPO / "cache" / "provider",
                                          ledger_path=ledger)}
    else:
        providers_cfg = {"cheap": ("together", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
                         "strong": ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo")}
        cps = {"together": CachedProvider(provider=TogetherProvider(),
                                          cache_dir=REPO / "cache" / "provider",
                                          ledger_path=ledger)}

    jobs = [(q, a) for q in questions for a in LLM_ACTIONS]
    cap = initial + args.max_spend_usd
    abort = threading.Event()
    lock = threading.Lock()
    progress = {"done": 0, "errors": 0}
    results = []

    def worker(q, a):
        if abort.is_set():
            return None
        try:
            r = _api_call(cps, providers_cfg, q, a)
        except Exception as e:
            with lock:
                progress["errors"] += 1
            return {"task_id": q["task_id"], "action": a, "error": str(e)[:200],
                    "success": False, "unsupported_risk": 0.0,
                    "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                    "latency_ms": 0, "cache_hit": False, "retrieved_ids": []}
        with lock:
            progress["done"] += 1
            if _spend(ledger) > cap:
                abort.set()
        return r

    t0 = time.time()
    last_print = t0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(worker, q, a) for q, a in jobs]
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                results.append(r)
            now = time.time()
            if now - last_print > 5 or progress["done"] >= len(jobs):
                cur = _spend(ledger) - initial
                print(f"  progress {progress['done']}/{len(jobs)} | spend ${cur:.4f} | err {progress['errors']} | {int(now - t0)}s")
                last_print = now
    final = _spend(ledger) - initial
    print(f"done; incremental real spend ${final:.4f}; errors {progress['errors']}; elapsed {int(time.time() - t0)}s")

    outcomes: dict[str, dict] = {}
    for r in results:
        if "error" in r and r.get("error"):
            continue
        per_action = outcomes.setdefault(r["task_id"], {})
        per_action[r["action"]] = {
            "cost": COST_UNITS.get(r["action"], 1.0),
            "latency_ms": int(r["latency_ms"]),
            "success": bool(r["success"]),
            "unsupported_risk": float(r["unsupported_risk"]),
            "real_cost_usd": float(r["cost_usd"]),
            "real_input_tokens": int(r["input_tokens"]),
            "real_output_tokens": int(r["output_tokens"]),
            "retrieved_ids": r["retrieved_ids"],
        }
    out_path = HERE / "experiments" / f"agentic_search_outcomes_{args.provider_mode}.json"
    out_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")

    summary = {"provider_mode": args.provider_mode,
               "n_tasks": len(outcomes),
               "n_jobs": len(jobs),
               "n_errors": progress["errors"],
               "incremental_spend_usd": final,
               "per_action_success_rate": {}}
    for a in LLM_ACTIONS:
        n = sum(1 for tid, d in outcomes.items() if a in d)
        s = sum(1 for tid, d in outcomes.items() if a in d and d[a].get("success"))
        if n:
            summary["per_action_success_rate"][a] = {"n": n, "successes": s,
                                                     "rate": s / n}
    sum_path = HERE / "experiments" / f"agentic_search_collection_summary_{args.provider_mode}.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)}")
    print(f"wrote {sum_path.relative_to(REPO)}")
    print("per-action success:")
    for a, d in summary["per_action_success_rate"].items():
        print(f"  {a}: {d['successes']}/{d['n']} = {d['rate']:.3f}")
    return 0 if not abort.is_set() else 3


if __name__ == "__main__":
    raise SystemExit(main())
