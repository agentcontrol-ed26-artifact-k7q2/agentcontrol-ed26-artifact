"""Phase 3 (hard regime): real-API outcome collection across providers.

Mirrors run_rescue_real_collection.py but uses the harder regime-tagged pool
and respects AGENTCONTROL_HARD_REGIME_APPROVED.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentcontrol.providers import CachedProvider, ProviderRequest  # noqa: E402
from agentcontrol.real_providers import DeepSeekProvider, TogetherProvider  # noqa: E402
from agentcontrol.hard_regime_tasks import get_pool, verify  # noqa: E402

load_dotenv(REPO / ".env")

ACTIONS = [
    "cheap_answer", "cheap_repair", "strong_hint", "cheap_repair_after_hint",
    "strong_critique", "strong_checklist", "strong_answer",
]
ANSWER_ACTIONS = {"cheap_answer", "cheap_repair", "cheap_repair_after_hint", "strong_answer"}
COST_UNITS = {
    "cheap_answer": 1.0, "cheap_repair": 1.0,
    "strong_hint": 2.0, "cheap_repair_after_hint": 1.0,
    "strong_critique": 2.0, "strong_checklist": 2.0,
    "strong_answer": 10.0,
}


def _system(task: dict) -> str:
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


def _user(task: dict, action: str) -> str:
    f = task["family"]
    if f == "math":
        q = task["question"]
        return {
            "cheap_answer": f"Question: {q}\nFinal answer:",
            "cheap_repair": f"Question: {q}\nA prior cheap attempt may be wrong. Re-derive carefully and give the final answer:",
            "strong_hint": f"Question: {q}\nGive ONE short hint on the key step. Do NOT give the final answer.",
            "cheap_repair_after_hint": f"Question: {q}\nUse careful step-by-step reasoning, then output ONLY the final number on the last line.",
            "strong_critique": f"Question: {q}\nName the most common mistake to avoid. One sentence. No final answer.",
            "strong_checklist": f"Question: {q}\nList 2 verification checks (no answer).",
            "strong_answer": f"Question: {q}\nThink carefully, then output the final answer on the last line.",
        }[action]
    if f == "code":
        prompt = task["prompt"]
        return {
            "cheap_answer": f"Task: {prompt}\nWrite the function in a single Python code block.",
            "cheap_repair": f"Task: {prompt}\nA prior attempt may have a bug. Write a correct version in a Python code block.",
            "strong_hint": f"Task: {prompt}\nGive one short algorithmic hint (no code).",
            "cheap_repair_after_hint": f"Task: {prompt}\nWrite the correct function carefully, attending to typical edge cases (zero, empty, negative, duplicates).",
            "strong_critique": f"Task: {prompt}\nName the most common bug to avoid. One sentence, no code.",
            "strong_checklist": f"Task: {prompt}\nList 2 short test cases worth checking (no code).",
            "strong_answer": f"Task: {prompt}\nThink carefully, then write the function in a single Python code block.",
        }[action]
    if f == "evidence":
        q = task["question"]
        ev = "\n".join(f"{k}: {v}" for k, v in task["evidence"].items())
        return {
            "cheap_answer": f"Evidence:\n{ev}\nQuestion: {q}\nAnswer with the answer phrase and append exactly one [doc_xxx] citation from the evidence above.",
            "cheap_repair": f"Evidence:\n{ev}\nQuestion: {q}\nReread carefully and answer with the answer phrase and exactly one [doc_xxx] citation.",
            "strong_hint": f"Evidence:\n{ev}\nQuestion: {q}\nGive one short hint that names the relevant doc id, no answer.",
            "cheap_repair_after_hint": f"Evidence:\n{ev}\nQuestion: {q}\nAnswer with the answer phrase and exactly one [doc_xxx] citation strictly from the supporting evidence above. Do not cite a distractor.",
            "strong_critique": f"Evidence:\n{ev}\nQuestion: {q}\nState the most common citation mistake. One sentence. No answer.",
            "strong_checklist": f"Evidence:\n{ev}\nQuestion: {q}\nList 2 verification checks (no answer).",
            "strong_answer": f"Evidence:\n{ev}\nQuestion: {q}\nThink carefully, then output the answer phrase followed by exactly one [doc_xxx] citation strictly from the evidence.",
        }[action]
    return task.get("question", "")


def _max_tokens(action: str) -> int:
    if action == "strong_answer":
        return 1024
    if action.startswith("strong_"):
        return 200
    if action == "cheap_answer":
        return 256
    return 512


def _model_for(action: str, providers_cfg: dict) -> tuple[str, str]:
    if action in ("cheap_answer", "cheap_repair", "cheap_repair_after_hint"):
        return providers_cfg["cheap"]
    return providers_cfg["strong"]


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


def _call_one(cps: dict, providers_cfg: dict, task: dict, action: str) -> dict:
    pname, model = _model_for(action, providers_cfg)
    cp = cps[pname]
    req = ProviderRequest(
        provider=pname,
        model=model,
        system=_system(task),
        messages=[{"role": "user", "content": _user(task, action)}],
        params={"temperature": 0.0, "max_tokens": _max_tokens(action)},
    )
    resp = cp.generate(req)
    text = resp.text or ""
    success = False
    risk = 0.0
    if action in ANSWER_ACTIONS:
        s, r = verify(task, text)
        success, risk = bool(s), float(r)
    return {
        "task_id": task["id"], "family": task["family"], "regime": task["regime"],
        "action": action, "success": success, "unsupported_risk": risk,
        "cost_usd": float(resp.cost_usd),
        "input_tokens": int(resp.input_tokens),
        "output_tokens": int(resp.output_tokens),
        "latency_ms": int(resp.latency_ms),
        "cache_hit": bool(resp.cache_hit),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-spend-usd", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--provider-mode", choices=["deepseek", "together"],
                   default="deepseek")
    args = p.parse_args()

    if os.environ.get("AGENTCONTROL_HARD_REGIME_APPROVED") != "1":
        print("AGENTCONTROL_HARD_REGIME_APPROVED is not 1; refusing.")
        return 2

    pool = get_pool()
    print(f"hard-regime pool: {len(pool)} tasks across families {sorted({t['family'] for t in pool.values()})}")

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

    jobs = [(task, action) for task in pool.values() for action in ACTIONS]
    print(f"jobs: {len(jobs)}")
    cap = initial + args.max_spend_usd
    abort = threading.Event()
    lock = threading.Lock()
    results = []
    progress = {"done": 0, "errors": 0}

    def worker(task, action):
        if abort.is_set():
            return None
        try:
            r = _call_one(cps, providers_cfg, task, action)
        except Exception as e:
            with lock:
                progress["errors"] += 1
            return {"task_id": task["id"], "family": task["family"], "regime": task["regime"],
                    "action": action, "success": False, "unsupported_risk": 0.0,
                    "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                    "latency_ms": 0, "cache_hit": False, "error": str(e)[:200]}
        with lock:
            progress["done"] += 1
            if _spend(ledger) > cap:
                abort.set()
        return r

    t0 = time.time()
    last_print = t0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(worker, t, a) for t, a in jobs]
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                results.append(r)
            now = time.time()
            if now - last_print > 5 or progress["done"] >= len(jobs):
                cur = _spend(ledger) - initial
                print(f"  progress {progress['done']}/{len(jobs)} | spend ${cur:.4f} | err {progress['errors']} | {int(now-t0)}s")
                last_print = now
    final = _spend(ledger) - initial
    print(f"done; incremental real spend ${final:.4f}; errors {progress['errors']}; elapsed {int(time.time()-t0)}s")

    outcomes = {}
    for r in results:
        if "error" in r:
            continue
        outcomes.setdefault(r["task_id"], {})[r["action"]] = {
            "cost": COST_UNITS[r["action"]],
            "latency_ms": int(r["latency_ms"]),
            "success": bool(r["success"]),
            "unsupported_risk": float(r["unsupported_risk"]),
            "real_cost_usd": float(r["cost_usd"]),
            "real_input_tokens": int(r["input_tokens"]),
            "real_output_tokens": int(r["output_tokens"]),
            "regime": r["regime"],
        }

    out_path = REPO / "experiments" / f"hard_regime_outcomes_{args.provider_mode}.json"
    out_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    summary = {
        "provider_mode": args.provider_mode,
        "n_tasks": len(outcomes),
        "n_jobs": len(jobs),
        "n_errors": progress["errors"],
        "incremental_spend_usd": final,
        "per_action_success_rate": {
            a: {
                "n": sum(1 for r in results if r["action"] == a and "error" not in r),
                "successes": sum(1 for r in results if r["action"] == a and r.get("success")),
            }
            for a in ACTIONS
        },
    }
    sum_path = REPO / "experiments" / f"hard_regime_collection_summary_{args.provider_mode}.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)} and {sum_path.relative_to(REPO)}")
    print("per-action success:")
    for a, d in summary["per_action_success_rate"].items():
        rate = d["successes"] / d["n"] if d["n"] else 0
        print(f"  {a}: {d['successes']}/{d['n']} = {rate:.3f}")
    return 0 if not abort.is_set() else 3


if __name__ == "__main__":
    raise SystemExit(main())
