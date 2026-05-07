#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentcontrol.plotting import make_pareto_plot
from agentcontrol.utils import read_json


def main() -> None:
    points = []
    baselines = read_json('experiments/baselines_summary.json', {})
    for name, row in baselines.items(): points.append({'label': name, 'cost': row.get('avg_cost', 0.0), 'success': row.get('success_rate', 0.0)})
    heuristic = read_json('experiments/heuristic_bdelg_summary.json', {})
    if heuristic: points.append({'label': 'heuristic_bdelg', 'cost': heuristic.get('avg_cost', 0.0), 'success': heuristic.get('success_rate', 0.0)})
    oracle = read_json('experiments/oracle_gap_summary.json', {})
    if oracle:
        q, g = oracle.get('query_router', {}), oracle.get('deliberation_graph', {})
        points.append({'label': 'oracle_query', 'cost': q.get('avg_cost', 0.0), 'success': q.get('success_rate', 0.0)})
        points.append({'label': 'oracle_graph', 'cost': g.get('avg_cost', 0.0), 'success': g.get('success_rate', 0.0)})
    make_pareto_plot(points, 'figures/pareto.png')
    print('wrote figures/pareto.png')

if __name__ == '__main__': main()
