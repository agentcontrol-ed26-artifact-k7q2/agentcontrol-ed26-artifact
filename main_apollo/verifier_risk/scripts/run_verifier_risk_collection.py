"""Apollo Phase 3B — collect cheap and strong outcomes on the verifier-risk pool,
apply weak and exact verifiers, and compute the risk-cost Pareto.

Single script combining collection + policy evaluation since the pool is small (n=60).
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

REPO = Path(__file__).resolve().parents[3]
APOLLO = REPO / "main_apollo"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(APOLLO / "verifier_risk" / "scripts"))

from agentcontrol.providers import CachedProvider, ProviderRequest  # noqa: E402
from agentcontrol.real_providers import DeepSeekProvider, TogetherProvider  # noqa: E402
from build_verifier_risk_pool import get_pool, _EV_CORPUS  # noqa: E402

load_dotenv(REPO / ".env")

# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

def _last_int(s: str) -> str | None:
    nums = re.findall(r"-?\d+", (s or "").replace(",", ""))
    return nums[-1] if nums else None


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def weak_verifier(task: dict, output: str) -> bool:
    """Weak / loose verifier — easy to false-accept by design."""
    fam = task["family"]
    if fam == "math_traps" or fam == "data_analysis_traps":
        # Weak: any integer present in output (not even checked against gold).
        return _last_int(output) is not None
    if fam == "code_traps":
        # Weak: only the weak_tests subset is run.
        code = _extract_python(output)
        if not code:
            return False
        ns: dict = {}
        try:
            exec(compile(code, "<weak>", "exec"), ns, ns)
        except Exception:
            return False
        if task["entry_point"] not in ns:
            return False
        try:
            for t in task["weak_tests"]:
                exec(compile(t, "<weak>", "exec"), ns, ns)
        except Exception:
            return False
        return True
    if fam == "evidence_traps":
        # Weak: gold phrase + ANY citation (authorized OR distractor).
        out_norm = _normalize(output)
        gold = _normalize(task["gold_answer"])
        cited = re.findall(r"\[(doc_[a-zA-Z0-9_]+)\]", output or "")
        return bool(cited and gold in out_norm)
    return False


def exact_verifier(task: dict, output: str) -> bool:
    """Exact / audit verifier — catches the trap cases."""
    fam = task["family"]
    if fam == "math_traps" or fam == "data_analysis_traps":
        last = _last_int(output)
        return last is not None and last == task["gold_answer"]
    if fam == "code_traps":
        code = _extract_python(output)
        if not code:
            return False
        ns: dict = {}
        try:
            exec(compile(code, "<exact>", "exec"), ns, ns)
        except Exception:
            return False
        if task["entry_point"] not in ns:
            return False
        try:
            for t in task["audit_tests"]:
                exec(compile(t, "<exact>", "exec"), ns, ns)
        except Exception:
            return False
        return True
    if fam == "evidence_traps":
        out_norm = _normalize(output)
        gold = _normalize(task["gold_answer"])
        cited = re.findall(r"\[(doc_[a-zA-Z0-9_]+)\]", output or "")
        authorized = task["authorized_citations"]
        return bool(any(c in authorized for c in cited) and gold in out_norm)
    return False


def _extract_python(output: str) -> str:
    if not output:
        return ""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    return output.strip()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _system_for(task: dict) -> str:
    fam = task["family"]
    if fam == "math_traps":
        return ("Careful arithmetic assistant. Give the final answer as a single integer on the last line.")
    if fam == "code_traps":
        return ("Python coder. Output exactly one Python code block containing the requested function. "
                "No tests, no commentary outside the block.")
    if fam == "evidence_traps":
        return ("Citation-grounded QA assistant. Answer using ONLY the provided evidence. "
                "End with the answer phrase followed by exactly one [doc_xxx] citation.")
    if fam == "data_analysis_traps":
        return ("Data analysis assistant. Write Python that prints the final answer on the last line of stdout.")
    return "Careful assistant."


def _user_for(task: dict, action: str) -> str:
    fam = task["family"]
    if fam == "math_traps":
        return f"Question: {task['question']}\nFinal answer:"
    if fam == "code_traps":
        return f"Task: {task['prompt']}\nWrite the function in a single Python code block."
    if fam == "evidence_traps":
        ev = "\n".join(f"{k}: {v}" for k, v in task["evidence"].items())
        return f"Evidence:\n{ev}\nQuestion: {task['question']}\nAnswer with the answer phrase and exactly one [doc_xxx] citation."
    if fam == "data_analysis_traps":
        return f"{task['question']}\nWrite Python that prints the final answer on the last line of stdout."
    return ""


def _model_for(action: str, mods: dict) -> tuple[str, str]:
    if action == "cheap":
        return mods["cheap"]
    return mods["strong"]


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


def _api_call(cps, providers_cfg, task, action) -> dict:
    pname, model = _model_for(action, providers_cfg)
    cp = cps[pname]
    req = ProviderRequest(
        provider=pname, model=model,
        system=_system_for(task),
        messages=[{"role": "user", "content": _user_for(task, action)}],
        params={"temperature": 0.0,
                "max_tokens": 1024 if action == "strong" else 512},
    )
    resp = cp.generate(req)
    text = resp.text or ""
    return {"task_id": task["id"], "family": task["family"], "action": action,
            "text": text,
            "weak_pass": weak_verifier(task, text),
            "exact_pass": exact_verifier(task, text),
            "cost_usd": float(resp.cost_usd),
            "input_tokens": int(resp.input_tokens),
            "output_tokens": int(resp.output_tokens),
            "latency_ms": int(resp.latency_ms),
            "cache_hit": bool(resp.cache_hit)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-spend-usd", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--provider-mode", choices=["deepseek"], default="deepseek")
    args = p.parse_args()

    if os.environ.get("AGENTCONTROL_APOLLO_API_APPROVED") != "1":
        print("AGENTCONTROL_APOLLO_API_APPROVED is not 1; refusing API run.")
        return 2

    pool = get_pool()
    print(f"verifier-risk pool: {len(pool)} tasks")
    ledger = REPO / "cache" / "cost_ledger.jsonl"
    initial = _spend(ledger)
    print(f"prior real-spend: ${initial:.4f}")

    providers_cfg = {"cheap": ("deepseek", "deepseek-chat"),
                     "strong": ("deepseek", "deepseek-reasoner")}
    cps = {"deepseek": CachedProvider(provider=DeepSeekProvider(),
                                      cache_dir=REPO / "cache" / "provider",
                                      ledger_path=ledger)}

    cap = initial + args.max_spend_usd
    abort = threading.Event()
    lock = threading.Lock()
    progress = {"done": 0, "errors": 0}
    results = []

    jobs = [(t, "cheap") for t in pool.values()] + [(t, "strong") for t in pool.values()]

    def worker(task, action):
        if abort.is_set(): return None
        try:
            r = _api_call(cps, providers_cfg, task, action)
        except Exception as e:
            with lock: progress["errors"] += 1
            return {"task_id": task["id"], "family": task["family"],
                    "action": action, "error": str(e)[:200],
                    "weak_pass": False, "exact_pass": False, "cost_usd": 0.0,
                    "input_tokens": 0, "output_tokens": 0, "latency_ms": 0,
                    "cache_hit": False}
        with lock: progress["done"] += 1
        return r

    t0 = time.time(); last = t0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(worker, t, a) for t, a in jobs]
        for f in as_completed(futs):
            r = f.result()
            if r: results.append(r)
            if abort.is_set(): break
            now = time.time()
            if now - last > 5 or progress["done"] >= len(jobs):
                cur = _spend(ledger) - initial
                print(f"  progress {progress['done']}/{len(jobs)} | spend ${cur:.4f} | err {progress['errors']} | {int(now-t0)}s")
                last = now
                if cur > args.max_spend_usd:
                    abort.set()

    final = _spend(ledger) - initial
    print(f"done; incremental spend ${final:.4f}; errors {progress['errors']}; elapsed {int(time.time()-t0)}s")

    # Build outcomes (per task: cheap.weak_pass, cheap.exact_pass, cheap.cost; strong.* same)
    outcomes = {}
    for r in results:
        if "error" in r and r.get("error"): continue
        outcomes.setdefault(r["task_id"], {})[r["action"]] = r

    out_path = APOLLO / "verifier_risk" / "experiments" / "verifier_risk_outcomes.json"
    out_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
