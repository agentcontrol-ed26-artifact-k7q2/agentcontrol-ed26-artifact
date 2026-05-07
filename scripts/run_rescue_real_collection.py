"""Phase 3 (real-API): collect real-model outcomes on the rescue task pool.

Uses ``CachedProvider`` over DeepSeek (primary) — and optionally Together —
to populate ``experiments/rescue_outcomes.json`` with REAL model outcomes
verified deterministically locally.

Gate: respects ``AGENTCONTROL_RESCUE_APPROVED=1`` and refuses to spend more
than the configured per-provider cap.
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
from agentcontrol.rescue_tasks import get_pool, verify  # noqa: E402

load_dotenv(REPO / ".env")

ACTIONS = [
    "cheap_answer",
    "cheap_repair",
    "strong_hint",
    "cheap_repair_after_hint",
    "strong_critique",
    "strong_checklist",
    "strong_answer",
]

# Which actions count as ANSWER actions (verified for success).
ANSWER_ACTIONS = {"cheap_answer", "cheap_repair", "cheap_repair_after_hint", "strong_answer"}
# Cost units in the deliberation graph price model (must match ed_sim).
COST_UNITS = {
    "cheap_answer": 1.0,
    "cheap_repair": 1.0,
    "strong_hint": 2.0,
    "cheap_repair_after_hint": 1.0,
    "strong_critique": 2.0,
    "strong_checklist": 2.0,
    "strong_answer": 10.0,
}


def _system_prompt(task: dict) -> str:
    fam = task["family"]
    if fam == "math":
        return ("You are a careful arithmetic assistant. Answer with the final "
                "numeric or short-text answer only, on one line, without explanation.")
    if fam == "code":
        return ("You are a Python coder. Write a single Python function definition "
                "named exactly as requested. Output ONLY a Python code block. "
                "Do not include tests or commentary.")
    if fam == "evidence":
        return ("You are a careful citation-grounded QA assistant. Answer using "
                "ONLY the provided evidence. Append a single bracketed citation "
                "of the form [doc_xxx] referencing the supporting document.")
    return "You are a careful assistant."


def _user_prompt(task: dict, action: str) -> str:
    fam = task["family"]
    if fam == "math":
        q = task["question"]
        if action == "cheap_answer":
            return f"Question: {q}\nAnswer:"
        if action == "cheap_repair":
            return f"Question: {q}\nA prior cheap attempt may have been wrong. Re-derive carefully and answer:"
        if action == "strong_hint":
            return f"Question: {q}\nProvide a short hint (one sentence) on the key step. Do NOT give the final answer."
        if action == "cheap_repair_after_hint":
            return f"Question: {q}\nUse careful step-by-step reasoning, then output only the final answer on the last line."
        if action == "strong_critique":
            return f"Question: {q}\nList the most common mistake to avoid in solving this. One sentence. No final answer."
        if action == "strong_checklist":
            return f"Question: {q}\nList 2 verification checks (no answer)."
        if action == "strong_answer":
            return f"Question: {q}\nThink carefully step-by-step, then output the final answer on the last line."
    if fam == "code":
        prompt = task["prompt"]
        if action == "cheap_answer":
            return f"Task: {prompt}\nWrite the function in a Python code block."
        if action == "cheap_repair":
            return f"Task: {prompt}\nA prior attempt may have had a bug. Write a correct version in a Python code block."
        if action == "strong_hint":
            return f"Task: {prompt}\nGive one short hint (no code) about the algorithm or edge case."
        if action == "cheap_repair_after_hint":
            return f"Task: {prompt}\nWrite the correct function carefully, attending to standard edge cases (zero, empty, negative)."
        if action == "strong_critique":
            return f"Task: {prompt}\nName the most common bug to avoid. One sentence, no code."
        if action == "strong_checklist":
            return f"Task: {prompt}\nList 2 short test cases to consider (no code)."
        if action == "strong_answer":
            return f"Task: {prompt}\nThink carefully, then write the function in a Python code block."
    if fam == "evidence":
        q = task["question"]
        ev = "\n".join(f"{k}: {v}" for k, v in task["evidence"].items())
        if action == "cheap_answer":
            return f"Evidence:\n{ev}\nQuestion: {q}\nAnswer with the answer phrase and append a single [doc_xxx] citation."
        if action == "cheap_repair":
            return f"Evidence:\n{ev}\nQuestion: {q}\nReread the evidence and answer with the answer phrase and a [doc_xxx] citation."
        if action == "strong_hint":
            return f"Evidence:\n{ev}\nQuestion: {q}\nGive one short hint that names the relevant doc id, no answer."
        if action == "cheap_repair_after_hint":
            return f"Evidence:\n{ev}\nQuestion: {q}\nAnswer with the answer phrase and exactly one [doc_xxx] citation matching the evidence above."
        if action == "strong_critique":
            return f"Evidence:\n{ev}\nQuestion: {q}\nState the most common citation mistake. One sentence. No answer."
        if action == "strong_checklist":
            return f"Evidence:\n{ev}\nQuestion: {q}\nList 2 verification checks (no answer)."
        if action == "strong_answer":
            return f"Evidence:\n{ev}\nQuestion: {q}\nThink carefully, then output the answer phrase followed by exactly one [doc_xxx] citation."
    return task.get("question", "")


def _model_for_action(action: str, providers_cfg: dict) -> tuple[str, str]:
    """Return (provider_name, model). 'cheap' actions use the cheap model;
    everything else uses the strong model."""
    if action in ("cheap_answer", "cheap_repair", "cheap_repair_after_hint"):
        return providers_cfg["cheap"]
    return providers_cfg["strong"]


def _max_tokens_for(action: str) -> int:
    if action == "strong_answer":
        return 1024
    if action in ("strong_hint", "strong_critique", "strong_checklist"):
        return 200
    if action == "cheap_answer":
        return 256
    return 512


def _call_one(cached_providers: dict, providers_cfg: dict, task: dict,
              action: str) -> dict:
    pname, model = _model_for_action(action, providers_cfg)
    cp = cached_providers[pname]
    user = _user_prompt(task, action)
    req = ProviderRequest(
        provider=pname,
        model=model,
        system=_system_prompt(task),
        messages=[{"role": "user", "content": user}],
        params={"temperature": 0.0, "max_tokens": _max_tokens_for(action)},
    )
    resp = cp.generate(req)
    text = resp.text or ""
    success = False
    risk = 0.0
    if action in ANSWER_ACTIONS:
        succ, risk = verify(task, text)
        success = bool(succ)
    return {
        "task_id": task["id"],
        "family": task["family"],
        "action": action,
        "success": success,
        "unsupported_risk": float(risk),
        "cost_usd": float(resp.cost_usd),
        "input_tokens": int(resp.input_tokens),
        "output_tokens": int(resp.output_tokens),
        "latency_ms": int(resp.latency_ms),
        "cache_hit": bool(resp.cache_hit),
    }


def _spend_so_far(ledger_path: Path) -> float:
    if not ledger_path.exists():
        return 0.0
    total = 0.0
    try:
        text = ledger_path.read_text(encoding="utf-8")
    except OSError:
        return total
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Tolerate partial writes during concurrent collection.
            continue
        if row.get("actual_api_call"):
            total += float(row.get("cost_usd", 0.0))
    return total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-spend-usd", type=float, default=3.0,
                   help="Hard cap on this run's incremental spend.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--provider-mode", choices=["deepseek", "together", "deepseek+together"],
                   default="deepseek")
    p.add_argument("--together-subsample", type=int, default=30,
                   help="When together is in the mix, only run on this many tasks.")
    p.add_argument("--families", nargs="*", default=["math", "code", "evidence"])
    args = p.parse_args()

    if os.environ.get("AGENTCONTROL_RESCUE_APPROVED") != "1":
        print("AGENTCONTROL_RESCUE_APPROVED is not 1. Refusing real-API run.")
        print("Set the env var and re-run.")
        return 2

    pool_all = get_pool()
    pool = {tid: t for tid, t in pool_all.items() if t["family"] in args.families}
    print(f"task pool: {len(pool)} tasks across families {sorted({t['family'] for t in pool.values()})}")

    ledger_path = REPO / "cache" / "cost_ledger.jsonl"
    initial_spend = _spend_so_far(ledger_path)
    print(f"prior actual_api_call spend in ledger: ${initial_spend:.4f}")

    # Build provider+model assignments.
    providers_cfg: dict[str, tuple[str, str]] = {}
    cached_providers: dict[str, CachedProvider] = {}
    if "deepseek" in args.provider_mode:
        providers_cfg["cheap"] = ("deepseek", "deepseek-chat")
        providers_cfg["strong"] = ("deepseek", "deepseek-reasoner")
        cached_providers["deepseek"] = CachedProvider(
            provider=DeepSeekProvider(),
            cache_dir=REPO / "cache" / "provider",
            ledger_path=ledger_path,
        )
    elif args.provider_mode == "together":
        # Weaker cheap (Qwen-2.5-7B) vs strong (Llama-3.3-70B). Chosen so the
        # cheap arm fails differentially with task difficulty, exposing the
        # cheap-vs-strong gap that DeepSeek-chat masks via saturation. Both
        # are confirmed serverless on Together.
        providers_cfg["cheap"] = ("together", "Qwen/Qwen2.5-7B-Instruct-Turbo")
        providers_cfg["strong"] = ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        cached_providers["together"] = CachedProvider(
            provider=TogetherProvider(),
            cache_dir=REPO / "cache" / "provider",
            ledger_path=ledger_path,
        )
    if "+together" in args.provider_mode:
        cached_providers["together"] = CachedProvider(
            provider=TogetherProvider(),
            cache_dir=REPO / "cache" / "provider",
            ledger_path=ledger_path,
        )

    # If together-only, restrict pool to subsample.
    if args.provider_mode == "together":
        ids_sorted = sorted(pool.keys())[: args.together_subsample]
        pool = {k: pool[k] for k in ids_sorted}
        print(f"together subsample: {len(pool)} tasks")

    # Schedule all (task, action) calls.
    jobs = []
    for tid, task in pool.items():
        for action in ACTIONS:
            jobs.append((task, action))
    print(f"total jobs: {len(jobs)}")

    spend_cap = initial_spend + args.max_spend_usd
    spend_lock = threading.Lock()
    abort_event = threading.Event()
    results: list[dict] = []
    progress = {"done": 0, "errors": 0}

    def worker(task, action):
        if abort_event.is_set():
            return None
        try:
            r = _call_one(cached_providers, providers_cfg, task, action)
        except Exception as e:
            with spend_lock:
                progress["errors"] += 1
            return {"task_id": task["id"], "family": task["family"],
                    "action": action, "success": False, "unsupported_risk": 0.0,
                    "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                    "latency_ms": 0, "cache_hit": False, "error": str(e)[:200]}
        # Check budget.
        with spend_lock:
            progress["done"] += 1
            cur = _spend_so_far(ledger_path)
            if cur > spend_cap:
                abort_event.set()
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(worker, t, a) for t, a in jobs]
        last_print = t0
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                results.append(r)
            now = time.time()
            if now - last_print > 5.0 or progress["done"] >= len(jobs):
                cur = _spend_so_far(ledger_path) - initial_spend
                print(f"  progress: {progress['done']}/{len(jobs)} jobs, spend ${cur:.4f}, errors {progress['errors']}, elapsed {int(now-t0)}s")
                last_print = now
    final_spend = _spend_so_far(ledger_path)
    incremental = final_spend - initial_spend
    print(f"done. incremental spend ${incremental:.4f} (cap ${args.max_spend_usd:.2f}); errors {progress['errors']}; elapsed {int(time.time()-t0)}s")

    # Build outcomes JSON in the same shape as smoke_outcomes.json.
    outcomes: dict[str, dict[str, dict]] = {}
    for r in results:
        if "error" in r:
            continue
        outcomes.setdefault(r["task_id"], {})[r["action"]] = {
            "cost": COST_UNITS[r["action"]],   # protocol cost units (not USD)
            "latency_ms": int(r["latency_ms"]),
            "success": bool(r["success"]),
            "unsupported_risk": float(r["unsupported_risk"]),
            "real_cost_usd": float(r["cost_usd"]),
            "real_input_tokens": int(r["input_tokens"]),
            "real_output_tokens": int(r["output_tokens"]),
        }

    suffix = {"deepseek": "_deepseek", "together": "_together",
              "deepseek+together": ""}.get(args.provider_mode, "")
    out_path = REPO / "experiments" / f"rescue_outcomes{suffix}.json"
    out_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    # Also write to canonical rescue_outcomes.json for downstream scripts.
    if args.provider_mode == "together":
        (REPO / "experiments" / "rescue_outcomes.json").write_text(
            json.dumps(outcomes, indent=2), encoding="utf-8")

    # Per-action success summary.
    summary = {
        "decisive": True,
        "provider_mode": args.provider_mode,
        "n_tasks": len(outcomes),
        "incremental_spend_usd": incremental,
        "n_jobs": len(jobs),
        "n_errors": progress["errors"],
        "per_action_success_rate": {
            a: {
                "n": sum(1 for r in results if r["action"] == a and "error" not in r),
                "successes": sum(1 for r in results if r["action"] == a and r.get("success")),
            }
            for a in ACTIONS
        },
        "per_family_n": {fam: sum(1 for t in outcomes if outcomes[t] and any(True for _ in outcomes[t]) and pool[t]["family"] == fam) for fam in {pool[t]["family"] for t in outcomes}},
    }
    summary_path = REPO / "experiments" / f"rescue_collection_summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.provider_mode == "together":
        (REPO / "experiments" / "rescue_collection_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {out_path.relative_to(REPO)}")
    print(f"wrote {summary_path.relative_to(REPO)}")
    print("per-action success rates:")
    for a, d in summary["per_action_success_rate"].items():
        rate = d["successes"] / d["n"] if d["n"] else 0
        print(f"  {a}: {d['successes']}/{d['n']} = {rate:.3f}")
    return 0 if not abort_event.is_set() else 3


if __name__ == "__main__":
    raise SystemExit(main())
