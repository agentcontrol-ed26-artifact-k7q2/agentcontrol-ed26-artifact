#!/usr/bin/env python3
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentcontrol.analysis import write_aggregate_summary
from agentcontrol.utils import read_json


def main() -> None:
    summary = write_aggregate_summary('experiments/aggregate_summary.json')
    oracle = read_json('experiments/oracle_gap_summary.json', {})
    heuristic = read_json('experiments/heuristic_bdelg_summary.json', {})
    decision = oracle.get('decision', 'PENDING')
    Path('reports/SMOKE_DECISION.md').write_text('# SMOKE_DECISION\n\nDECISION: ' + decision + '\n\n## Aggregate\n\n```json\n' + json.dumps(summary, indent=2, ensure_ascii=False) + '\n```\n\n## Heuristic BDelG\n\n```json\n' + json.dumps(heuristic, indent=2, ensure_ascii=False) + '\n```\n\n## Oracle\n\nSee `reports/ORACLE_GAP_DECISION.md`.\n', encoding='utf-8')
    print('wrote experiments/aggregate_summary.json and reports/SMOKE_DECISION.md')

if __name__ == '__main__': main()
