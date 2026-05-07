"""F2 Phase 3: real-API outcome collection on n=50 data_analysis_code pool.

Uses the same 8-action prompt schema as main_rescue_gpu's interactive
collection so the legacy n=20 cache-hits and only the new n=30 are billed.
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

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "main_rescue_gpu" / "src"))
sys.path.insert(0, str(HERE / "src"))

from agentcontrol.providers import CachedProvider, ProviderRequest  # noqa: E402
from agentcontrol.real_providers import DeepSeekProvider, TogetherProvider  # noqa: E402
from agentcontrol_method.f2_data_analysis_tasks import get_pool, verify  # noqa: E402
from agentcontrol_main_rescue.interactive_tasks import (  # noqa: E402
    run_code_observation,
)

load_dotenv(REPO / ".env")

LLM_ACTIONS = [
    "cheap_answer", "cheap_repair", "cheap_repair_after_observation",
    "strong_hint", "cheap_repair_after_strong_partial",
    "strong_critique", "strong_checklist", "strong_answer",
]
ANSWER_ACTIONS = {"cheap_answer", "cheap_repair", "cheap_repair_after_observation",
                  "cheap_repair_after_strong_partial", "strong_answer"}
COST_UNITS = {
    "cheap_answer": 1.0, "cheap_repair": 1.0,
    "cheap_repair_after_observation": 1.0,
    "cheap_repair_after_strong_partial": 1.0,
    "strong_hint": 2.0, "strong_critique": 2.0, "strong_checklist": 2.0,
    "strong_answer": 10.0,
    "run_code": 0.5,
}


def _system(task):
    return ("Data analysis assistant. Write Python that prints the answer "
            "as the LAST line of stdout. Use only the inline data given.")


def _user(task, action, ctx=None):
    """Mirror of main_rescue_gpu's _user_for() data_analysis branch verbatim
    so cache keys match for legacy n=20 tasks (id_001..id_020)."""
    ctx = ctx or {}
    q = task["question"]
    return {
        "cheap_answer": f"{q}\nWrite Python that prints the final answer on the last line of stdout.",
        "cheap_repair": f"{q}\nReread the question carefully and write Python that prints the answer on the last line.",
        "cheap_repair_after_observation": (f"{q}\nYour previous code printed:\n{ctx.get('observation','')}\n"
                                           f"If this matches the answer, restate it; otherwise correct the code."),
        "strong_hint": f"{q}\nGive one short hint about the right calculation. No final answer.",
        "cheap_repair_after_strong_partial": f"{q}\nHint: {ctx.get('strong_hint','')}\nWrite Python that prints the answer.",
        "strong_critique": f"{q}\nName the most common arithmetic mistake on this. One sentence.",
        "strong_checklist": f"{q}\nList 2 verification checks (no answer).",
        "strong_answer": f"{q}\nThink step by step and print the final answer on the last line.",
    }[action]


def _model_for(action, mods):
    if action.startswith("cheap_"):
        return mods["cheap"]
    return mods["strong"]


def _max_tokens(action):
    if action == "strong_answer": return 1024
    if action.startswith("strong_"): return 200
    if action == "cheap_answer": return 512
    return 512


def _spend(ledger):
    if not ledger.exists(): return 0.0
    total = 0.0
    try: text = ledger.read_text(encoding="utf-8")
    except OSError: return 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if r.get("actual_api_call") and r.get("provider") in ("deepseek", "together"):
            total += float(r.get("cost_usd", 0.0))
    return total


def _api_call(cps, providers_cfg, task, action, ctx):
    pname, model = _model_for(action, providers_cfg)
    cp = cps[pname]
    req = ProviderRequest(
        provider=pname, model=model, system=_system(task),
        messages=[{"role": "user", "content": _user(task, action, ctx)}],
        params={"temperature": 0.0, "max_tokens": _max_tokens(action)},
    )
    resp = cp.generate(req)
    text = resp.text or ""
    success = False; risk = 0.0
    if action in ANSWER_ACTIONS:
        s, r = verify(task, text)
        success, risk = bool(s), float(r)
    return {"task_id": task["id"], "action": action, "text": text,
            "success": success, "unsupported_risk": risk,
            "cost_usd": float(resp.cost_usd),
            "input_tokens": int(resp.input_tokens),
            "output_tokens": int(resp.output_tokens),
            "latency_ms": int(resp.latency_ms),
            "cache_hit": bool(resp.cache_hit)}


def _process_task(cps, providers_cfg, task, abort, ledger, cap):
    if abort.is_set(): return None
    out = {"task_id": task["id"], "actions": {}}
    # 1. cheap_answer
    r = _api_call(cps, providers_cfg, task, "cheap_answer", None)
    out["actions"]["cheap_answer"] = r
    cheap_text = r["text"]
    # 2. local observation: run_code on cheap_answer's code
    obs_ok, obs_text = run_code_observation(task, cheap_text)
    out["actions"]["run_code"] = {
        "task_id": task["id"], "action": "run_code", "text": obs_text,
        "success": bool(obs_ok), "unsupported_risk": 0.0,
        "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
        "latency_ms": 0, "cache_hit": False,
    }
    # 3. cheap_repair_after_observation
    r = _api_call(cps, providers_cfg, task, "cheap_repair_after_observation",
                  {"observation": obs_text})
    out["actions"]["cheap_repair_after_observation"] = r
    # 4. cheap_repair (blind)
    r = _api_call(cps, providers_cfg, task, "cheap_repair", None)
    out["actions"]["cheap_repair"] = r
    # 5. strong_hint
    r = _api_call(cps, providers_cfg, task, "strong_hint", None)
    out["actions"]["strong_hint"] = r
    hint = r["text"]
    # 6. cheap_repair_after_strong_partial
    r = _api_call(cps, providers_cfg, task, "cheap_repair_after_strong_partial",
                  {"strong_hint": hint})
    out["actions"]["cheap_repair_after_strong_partial"] = r
    # 7. strong_critique / strong_checklist
    r = _api_call(cps, providers_cfg, task, "strong_critique", None)
    out["actions"]["strong_critique"] = r
    r = _api_call(cps, providers_cfg, task, "strong_checklist", None)
    out["actions"]["strong_checklist"] = r
    # 8. strong_answer
    r = _api_call(cps, providers_cfg, task, "strong_answer", None)
    out["actions"]["strong_answer"] = r
    if _spend(ledger) > cap:
        abort.set()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-spend-usd", type=float, default=3.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--provider-mode", choices=["deepseek", "together"], default="deepseek")
    args = p.parse_args()

    if os.environ.get("AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED") != "1":
        print("AGENTCONTROL_METHOD_DISCOVERY_API_APPROVED is not 1; refusing API run.")
        return 2

    pool = get_pool()
    print(f"F2 pool: {len(pool)} tasks (data_analysis_code)")

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

    cap = initial + args.max_spend_usd
    abort = threading.Event()
    lock = threading.Lock()
    progress = {"done": 0, "errors": 0}
    results = []

    def worker(task):
        if abort.is_set(): return None
        try:
            r = _process_task(cps, providers_cfg, task, abort, ledger, cap)
        except Exception as e:
            with lock: progress["errors"] += 1
            return {"task_id": task["id"], "actions": {}, "error": str(e)[:200]}
        with lock: progress["done"] += 1
        return r

    t0 = time.time()
    last = t0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(worker, t) for t in pool.values()]
        for f in as_completed(futs):
            r = f.result()
            if r is not None: results.append(r)
            now = time.time()
            if now - last > 5 or progress["done"] >= len(pool):
                print(f"  progress {progress['done']}/{len(pool)} | spend ${_spend(ledger)-initial:.4f} | err {progress['errors']} | {int(now-t0)}s")
                last = now
    final = _spend(ledger) - initial
    print(f"done; incremental spend ${final:.4f}; errors {progress['errors']}; elapsed {int(time.time()-t0)}s")

    outcomes = {}
    for r in results:
        if not r or "error" in r: continue
        per = {}
        for a, ad in r["actions"].items():
            per[a] = {
                "cost": COST_UNITS.get(a, 0.0),
                "latency_ms": int(ad.get("latency_ms", 0)),
                "success": bool(ad.get("success", False)),
                "unsupported_risk": float(ad.get("unsupported_risk", 0.0)),
                "real_cost_usd": float(ad.get("cost_usd", 0.0)),
                "real_input_tokens": int(ad.get("input_tokens", 0)),
                "real_output_tokens": int(ad.get("output_tokens", 0)),
            }
        outcomes[r["task_id"]] = per

    out_path = HERE / "experiments" / f"f2_data_analysis_outcomes_{args.provider_mode}.json"
    out_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")

    # Trace JSONL
    traces_path = HERE / "traces" / f"f2_data_analysis_{args.provider_mode}.jsonl"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with traces_path.open("w", encoding="utf-8") as f:
        for r in results:
            if not r: continue
            for a, ad in r["actions"].items():
                f.write(json.dumps({"task_id": r["task_id"], "action": a,
                                    **{k: ad.get(k) for k in ("success", "cost_usd",
                                                                "input_tokens", "output_tokens",
                                                                "latency_ms", "cache_hit")}}) + "\n")

    summary = {"provider_mode": args.provider_mode, "n_tasks": len(outcomes),
               "incremental_spend_usd": final,
               "per_action_success_rate": {}}
    for a in LLM_ACTIONS + ["run_code"]:
        n = sum(1 for tid in outcomes if a in outcomes[tid])
        s = sum(1 for tid in outcomes if a in outcomes[tid] and outcomes[tid][a].get("success"))
        if n: summary["per_action_success_rate"][a] = {"n": n, "successes": s, "rate": s/n}
    sum_path = HERE / "experiments" / f"f2_data_analysis_collection_summary_{args.provider_mode}.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)}")
    print("per-action success:")
    for a, d in summary["per_action_success_rate"].items():
        print(f"  {a}: {d['successes']}/{d['n']} = {d['rate']:.3f}")
    return 0 if not abort.is_set() else 3


if __name__ == "__main__":
    raise SystemExit(main())
