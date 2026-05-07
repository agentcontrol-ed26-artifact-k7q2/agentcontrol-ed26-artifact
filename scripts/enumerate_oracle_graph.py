#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentcontrol.oracle import oracle_gap_summary
from agentcontrol.utils import read_json, write_json


def decision_from_gap(summary: dict) -> str:
    if summary['success_delta_pp'] >= 5.0 or summary['cost_saving_pct_at_observed'] >= 30.0:
        return 'GO-CONDITIONAL'
    return 'BACKUP'


def main() -> None:
    outcomes = read_json('experiments/smoke_outcomes.json', {})
    if not outcomes:
        raise SystemExit('Missing experiments/smoke_outcomes.json. Run smoke scripts first.')
    summary = oracle_gap_summary(outcomes, budget=20.0, cost_penalty=0.01)
    decision = decision_from_gap(summary)
    summary['decision'] = decision
    write_json('experiments/oracle_gap_summary.json', summary)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/ORACLE_GAP_DECISION.md').write_text(f"""# ORACLE_GAP_DECISION

DECISION: {decision}

## Summary

- success_delta_pp: {summary['success_delta_pp']:.3f}
- avg_cost_delta: {summary['avg_cost_delta']:.3f}
- avg_objective_delta: {summary['avg_objective_delta']:.3f}
- cost_saving_pct_at_observed: {summary['cost_saving_pct_at_observed']:.3f}

## Rule

GO/GO-CONDITIONAL requires oracle deliberation graph to beat oracle query router by >=5 pp at same cost or save >=30% cost at same success.

## Raw artifact

See `experiments/oracle_gap_summary.json`.
""", encoding='utf-8')
    print('wrote experiments/oracle_gap_summary.json and reports/ORACLE_GAP_DECISION.md')

if __name__ == '__main__': main()
