"""Phase 3: interactive-pool outcome collection.

Per task, collects:
  cheap_answer                       — cheap blind first attempt
  cheap_repair                       — cheap blind retry (no observation)
  observation                        — local execution producing state info
  cheap_repair_after_observation     — cheap with observation in context
  strong_hint                        — strong arm short hint (no observation)
  cheap_repair_after_strong_partial  — cheap with strong hint in context
  strong_critique                    — strong critique (no observation)
  strong_checklist                   — strong checklist (no observation)
  strong_answer                      — strong full answer

Observations execute locally (free); LLM actions go through CachedProvider.
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
sys.path.insert(0, str(HERE / "src"))

from agentcontrol.providers import CachedProvider, ProviderRequest  # noqa: E402
from agentcontrol.real_providers import DeepSeekProvider, TogetherProvider  # noqa: E402
from agentcontrol_main_rescue.interactive_tasks import (  # noqa: E402
    get_pool,
    run_tests_observation,
    run_code_observation,
    retrieve_observation,
    citation_check_observation,
    checkpoint_check_observation,
    tool_observation_observation,
    verify,
)

load_dotenv(REPO / ".env")

# Action set. "observation" actions run locally; the rest are LLM API actions.
LLM_ACTIONS = [
    "cheap_answer",
    "cheap_repair",
    "cheap_repair_after_observation",
    "strong_hint",
    "cheap_repair_after_strong_partial",
    "strong_critique",
    "strong_checklist",
    "strong_answer",
]
OBSERVATION_ACTIONS = [
    "run_tests", "run_code", "retrieve", "citation_check",
    "tool_observation", "checkpoint_check",
]
ANSWER_ACTIONS = {
    "cheap_answer", "cheap_repair", "cheap_repair_after_observation",
    "cheap_repair_after_strong_partial", "strong_answer",
}
COST_UNITS = {
    "cheap_answer": 1.0, "cheap_repair": 1.0,
    "cheap_repair_after_observation": 1.0,
    "cheap_repair_after_strong_partial": 1.0,
    "strong_hint": 2.0, "strong_critique": 2.0, "strong_checklist": 2.0,
    "strong_answer": 10.0,
    "run_tests": 0.5, "run_code": 0.5, "retrieve": 0.3,
    "citation_check": 0.2, "tool_observation": 0.5, "checkpoint_check": 0.3,
    "stop": 0.0,
}


def _system(task: dict) -> str:
    f = task["family"]
    if f == "code_debug_interactive":
        return ("Python coder. Output exactly one Python code block containing "
                "the requested function definition. No tests, no commentary outside the block.")
    if f == "data_analysis_code":
        return ("Data analysis assistant. Write Python that prints the answer "
                "as the LAST line of stdout. Use only the inline data given.")
    if f == "evidence_multihop_local":
        return ("Citation-grounded QA assistant. Use ONLY the provided evidence. "
                "Answer with the answer phrase followed by exactly one [doc_xxx] citation.")
    if f == "tool_planning_deterministic":
        return ("Reasoning assistant. Output the final integer answer on the last line, no units.")
    if f == "math_checkpoint":
        return ("Multi-step arithmetic assistant. Compute and print intermediate values "
                "first, then output the final integer answer on the last line.")
    return "Careful assistant."


def _user_for(task: dict, action: str, ctx: dict | None = None) -> str:
    f = task["family"]
    ctx = ctx or {}
    if f == "code_debug_interactive":
        prompt = task["prompt"]
        if action == "cheap_answer":
            return f"Task: {prompt}\nWrite the function in a single Python code block."
        if action == "cheap_repair":
            return f"Task: {prompt}\nA prior attempt may have a bug. Write a correct version."
        if action == "cheap_repair_after_observation":
            obs = ctx.get("observation", "")
            return (f"Task: {prompt}\n\nYour previous attempt produced these test results:\n{obs}\n\n"
                    f"Fix the bugs and write a corrected function in a single Python code block.")
        if action == "strong_hint":
            return f"Task: {prompt}\nGive ONE short algorithmic hint (no code, one sentence)."
        if action == "cheap_repair_after_strong_partial":
            hint = ctx.get("strong_hint", "")
            return f"Task: {prompt}\n\nHint: {hint}\n\nWrite the function carefully in a single Python code block."
        if action == "strong_critique":
            return f"Task: {prompt}\nName the most common bug to avoid. One sentence, no code."
        if action == "strong_checklist":
            return f"Task: {prompt}\nList 3 short test cases worth checking, including edge cases. No code."
        if action == "strong_answer":
            return f"Task: {prompt}\nThink carefully step by step, then write the function in a single Python code block."
    if f == "data_analysis_code":
        q = task["question"]
        if action == "cheap_answer":
            return f"{q}\nWrite Python that prints the final answer on the last line of stdout."
        if action == "cheap_repair":
            return f"{q}\nReread the question carefully and write Python that prints the answer on the last line."
        if action == "cheap_repair_after_observation":
            obs = ctx.get("observation", "")
            return f"{q}\nYour previous code printed:\n{obs}\nIf this matches the answer, restate it; otherwise correct the code."
        if action == "strong_hint":
            return f"{q}\nGive one short hint about the right calculation. No final answer."
        if action == "cheap_repair_after_strong_partial":
            hint = ctx.get("strong_hint", "")
            return f"{q}\nHint: {hint}\nWrite Python that prints the answer."
        if action == "strong_critique":
            return f"{q}\nName the most common arithmetic mistake on this. One sentence."
        if action == "strong_checklist":
            return f"{q}\nList 2 verification checks (no answer)."
        if action == "strong_answer":
            return f"{q}\nThink step by step and print the final answer on the last line."
    if f == "evidence_multihop_local":
        q = task["question"]
        ev = "\n".join(f"{k}: {v}" for k, v in task["evidence"].items())
        if action == "cheap_answer":
            return f"Evidence:\n{ev}\nQuestion: {q}\nAnswer with the answer phrase and exactly one [doc_xxx] citation."
        if action == "cheap_repair":
            return f"Evidence:\n{ev}\nQuestion: {q}\nReread carefully and answer with the answer phrase and exactly one [doc_xxx] citation."
        if action == "cheap_repair_after_observation":
            obs = ctx.get("observation", "")
            return (f"Evidence:\n{ev}\nQuestion: {q}\n\nYour previous citation was checked:\n{obs}\n\n"
                    f"Answer with the answer phrase and exactly one [doc_xxx] citation that strictly supports the answer.")
        if action == "strong_hint":
            return f"Evidence:\n{ev}\nQuestion: {q}\nGive one short hint that names the relevant doc id, no answer."
        if action == "cheap_repair_after_strong_partial":
            hint = ctx.get("strong_hint", "")
            return f"Evidence:\n{ev}\nQuestion: {q}\nHint: {hint}\nAnswer with the answer phrase and exactly one [doc_xxx] citation."
        if action == "strong_critique":
            return f"Evidence:\n{ev}\nQuestion: {q}\nState the most common citation mistake. One sentence."
        if action == "strong_checklist":
            return f"Evidence:\n{ev}\nQuestion: {q}\nList 2 verification checks (no answer)."
        if action == "strong_answer":
            return f"Evidence:\n{ev}\nQuestion: {q}\nThink carefully, then output the answer phrase followed by exactly one [doc_xxx] citation strictly from the evidence."
    if f == "tool_planning_deterministic":
        q = task["question"]
        if action == "cheap_answer":
            return f"{q}\nFinal integer:"
        if action == "cheap_repair":
            return f"{q}\nReconsider carefully. Final integer:"
        if action == "cheap_repair_after_observation":
            obs = ctx.get("observation", "")
            return f"{q}\nA tool reports:\n{obs}\nFinal integer:"
        if action == "strong_hint":
            return f"{q}\nGive one short reasoning hint (no answer)."
        if action == "cheap_repair_after_strong_partial":
            hint = ctx.get("strong_hint", "")
            return f"{q}\nHint: {hint}\nFinal integer:"
        if action == "strong_critique":
            return f"{q}\nName the typical mistake. One sentence."
        if action == "strong_checklist":
            return f"{q}\nList 2 verification checks (no answer)."
        if action == "strong_answer":
            return f"{q}\nThink carefully, then give the final integer."
    if f == "math_checkpoint":
        q = task["question"]
        if action == "cheap_answer":
            return f"{q}\nFirst show the intermediate value, then the final answer on the last line."
        if action == "cheap_repair":
            return f"{q}\nReconsider carefully and show intermediate then final answer on last line."
        if action == "cheap_repair_after_observation":
            obs = ctx.get("observation", "")
            return f"{q}\nA verifier reports about your last attempt:\n{obs}\nNow give the corrected intermediate and final answer on the last line."
        if action == "strong_hint":
            return f"{q}\nGive one short reasoning hint (no answer)."
        if action == "cheap_repair_after_strong_partial":
            hint = ctx.get("strong_hint", "")
            return f"{q}\nHint: {hint}\nShow intermediate then final answer."
        if action == "strong_critique":
            return f"{q}\nName the typical arithmetic slip. One sentence."
        if action == "strong_checklist":
            return f"{q}\nList 2 verification checks (no answer)."
        if action == "strong_answer":
            return f"{q}\nThink carefully, show intermediate, then give the final answer on the last line."
    return task.get("question", "")


def _model_for(action: str, mods: dict) -> tuple[str, str]:
    if action.startswith("cheap_"):
        return mods["cheap"]
    return mods["strong"]


def _max_tokens(action: str) -> int:
    if action == "strong_answer":
        return 1024
    if action.startswith("strong_"):
        return 200
    if action == "cheap_answer":
        return 512
    return 512


def _execute_observation(task: dict, action: str, model_output: str) -> tuple[bool, str]:
    """Run the local observation action; returns (observation_success, text)."""
    f = task["family"]
    if f == "code_debug_interactive":
        return run_tests_observation(task, model_output)
    if f == "data_analysis_code":
        return run_code_observation(task, model_output)
    if f == "evidence_multihop_local":
        if action == "retrieve":
            return True, retrieve_observation(task)
        return citation_check_observation(task, model_output)
    if f == "tool_planning_deterministic":
        return tool_observation_observation(task, model_output)
    if f == "math_checkpoint":
        return checkpoint_check_observation(task, model_output)
    return False, "no observation"


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


def _api_call(cps: dict, providers_cfg: dict, task: dict, action: str, ctx: dict | None) -> dict:
    pname, model = _model_for(action, providers_cfg)
    cp = cps[pname]
    user = _user_for(task, action, ctx)
    req = ProviderRequest(
        provider=pname, model=model,
        system=_system(task),
        messages=[{"role": "user", "content": user}],
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
        "task_id": task["id"], "family": task["family"],
        "action": action, "text": text,
        "success": success, "unsupported_risk": risk,
        "cost_usd": float(resp.cost_usd),
        "input_tokens": int(resp.input_tokens),
        "output_tokens": int(resp.output_tokens),
        "latency_ms": int(resp.latency_ms),
        "cache_hit": bool(resp.cache_hit),
    }


def _process_task(cps: dict, providers_cfg: dict, task: dict, abort: threading.Event,
                   ledger: Path, cap: float) -> dict:
    """Sequentially execute the 8 LLM actions for one task with proper observation chaining.

    Order matters: cheap_answer → run_tests/code/citation/etc. observation →
    cheap_repair_after_observation → strong_hint → cheap_repair_after_strong_partial →
    strong_critique → strong_checklist → strong_answer → cheap_repair (independent blind).
    """
    if abort.is_set():
        return {}
    out = {"task_id": task["id"], "family": task["family"], "actions": {}}
    # 1. cheap_answer
    r = _api_call(cps, providers_cfg, task, "cheap_answer", None)
    out["actions"]["cheap_answer"] = r
    cheap_text = r["text"]

    # 2. observation (local, free in dollars)
    obs_kind = task.get("available_observations", ["run_tests"])[0] if task.get("available_observations") else None
    obs_ok, obs_text = (False, "")
    if task.get("interactive") and obs_kind:
        obs_ok, obs_text = _execute_observation(task, obs_kind, cheap_text)
        out["actions"][obs_kind] = {
            "task_id": task["id"], "family": task["family"], "action": obs_kind,
            "success": bool(obs_ok), "text": obs_text,
            "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
            "latency_ms": 0, "cache_hit": False, "unsupported_risk": 0.0,
        }

    # 3. cheap_repair_after_observation
    ctx_obs = {"observation": obs_text}
    r = _api_call(cps, providers_cfg, task, "cheap_repair_after_observation", ctx_obs)
    out["actions"]["cheap_repair_after_observation"] = r

    # 4. cheap_repair (blind retry, independent of observation)
    r = _api_call(cps, providers_cfg, task, "cheap_repair", None)
    out["actions"]["cheap_repair"] = r

    # 5. strong_hint
    r = _api_call(cps, providers_cfg, task, "strong_hint", None)
    out["actions"]["strong_hint"] = r
    hint_text = r["text"]

    # 6. cheap_repair_after_strong_partial
    r = _api_call(cps, providers_cfg, task, "cheap_repair_after_strong_partial",
                  {"strong_hint": hint_text})
    out["actions"]["cheap_repair_after_strong_partial"] = r

    # 7. strong_critique / strong_checklist
    r = _api_call(cps, providers_cfg, task, "strong_critique", None)
    out["actions"]["strong_critique"] = r
    r = _api_call(cps, providers_cfg, task, "strong_checklist", None)
    out["actions"]["strong_checklist"] = r

    # 8. strong_answer
    r = _api_call(cps, providers_cfg, task, "strong_answer", None)
    out["actions"]["strong_answer"] = r

    # Budget check.
    if _spend(ledger) > cap:
        abort.set()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-spend-usd", type=float, default=5.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--provider-mode", choices=["deepseek", "together"], default="deepseek")
    p.add_argument("--families", nargs="*", default=None,
                   help="Limit to these families (else all).")
    args = p.parse_args()

    if os.environ.get("AGENTCONTROL_GPU_MAIN_RESCUE_APPROVED") != "1":
        print("AGENTCONTROL_GPU_MAIN_RESCUE_APPROVED is not 1; refusing API run.")
        return 2

    pool = get_pool()
    if args.families:
        pool = {tid: t for tid, t in pool.items() if t["family"] in args.families}
    print(f"interactive pool: {len(pool)} tasks")

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
    results: list[dict] = []
    progress = {"done": 0, "errors": 0}
    lock = threading.Lock()

    def worker(task):
        if abort.is_set():
            return None
        try:
            r = _process_task(cps, providers_cfg, task, abort, ledger, cap)
        except Exception as e:
            with lock:
                progress["errors"] += 1
            return {"task_id": task["id"], "family": task["family"],
                    "actions": {}, "error": str(e)[:200]}
        with lock:
            progress["done"] += 1
        return r

    t0 = time.time()
    last_print = t0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(worker, t) for t in pool.values()]
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                results.append(r)
            now = time.time()
            if now - last_print > 5 or progress["done"] >= len(pool):
                cur = _spend(ledger) - initial
                print(f"  progress {progress['done']}/{len(pool)} | spend ${cur:.4f} | err {progress['errors']} | {int(now - t0)}s")
                last_print = now
    final = _spend(ledger) - initial
    print(f"done; incremental spend ${final:.4f}; errors {progress['errors']}; elapsed {int(time.time()-t0)}s")

    # Build outcomes JSON.
    outcomes = {}
    for r in results:
        if not r or "error" in r:
            continue
        per_action = {}
        for action_name, ad in r["actions"].items():
            per_action[action_name] = {
                "cost": COST_UNITS.get(action_name, 0.0),
                "latency_ms": int(ad.get("latency_ms", 0)),
                "success": bool(ad.get("success", False)),
                "unsupported_risk": float(ad.get("unsupported_risk", 0.0)),
                "real_cost_usd": float(ad.get("cost_usd", 0.0)),
                "real_input_tokens": int(ad.get("input_tokens", 0)),
                "real_output_tokens": int(ad.get("output_tokens", 0)),
                "is_observation": action_name in OBSERVATION_ACTIONS,
            }
        outcomes[r["task_id"]] = per_action

    out_path = HERE / "experiments" / f"local_interactive_outcomes_{args.provider_mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")

    # Trace JSONL.
    traces_path = HERE / "traces" / f"local_interactive_{args.provider_mode}.jsonl"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with traces_path.open("w", encoding="utf-8") as f:
        for r in results:
            if not r:
                continue
            for action_name, ad in r["actions"].items():
                row = {"task_id": r["task_id"], "family": r["family"],
                       "action": action_name,
                       **{k: ad.get(k) for k in ("success", "unsupported_risk",
                                                  "cost_usd", "input_tokens",
                                                  "output_tokens", "latency_ms",
                                                  "cache_hit")}}
                f.write(json.dumps(row) + "\n")

    summary = {
        "provider_mode": args.provider_mode,
        "n_tasks": len(outcomes),
        "n_errors": progress["errors"],
        "incremental_spend_usd": final,
        "per_action_success_rate": {},
    }
    for action in (LLM_ACTIONS + OBSERVATION_ACTIONS):
        n = 0
        succ = 0
        for tid, ad in outcomes.items():
            if action in ad:
                n += 1
                if ad[action].get("success"):
                    succ += 1
        if n > 0:
            summary["per_action_success_rate"][action] = {"n": n, "successes": succ,
                                                          "rate": succ / n}
    sum_path = HERE / "experiments" / f"local_interactive_collection_summary_{args.provider_mode}.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)}")
    print(f"wrote {sum_path.relative_to(REPO)}")
    print("per-action success:")
    for a, d in summary["per_action_success_rate"].items():
        print(f"  {a}: {d['successes']}/{d['n']} = {d['rate']:.3f}")
    return 0 if not abort.is_set() else 3


if __name__ == "__main__":
    raise SystemExit(main())
