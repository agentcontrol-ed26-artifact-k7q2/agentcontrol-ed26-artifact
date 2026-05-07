#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentcontrol.oracle import evaluate_plan, summarize_results
from agentcontrol.policies import BASELINE_PLANS
from agentcontrol.utils import read_json, write_json


def main() -> None:
    outcomes = read_json('experiments/smoke_outcomes.json', {})
    if not outcomes: raise SystemExit('Missing smoke outcomes.')
    summary = {}
    for name, plan in BASELINE_PLANS.items():
        results = [evaluate_plan(tid, name, plan, outcomes, budget=20.0, cost_penalty=0.01) for tid in sorted(outcomes)]
        summary[name] = summarize_results(results)
    write_json('experiments/baselines_summary.json', summary)
    print('wrote experiments/baselines_summary.json')

if __name__ == '__main__': main()
