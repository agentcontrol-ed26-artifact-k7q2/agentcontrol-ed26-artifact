from __future__ import annotations
from pathlib import Path
from typing import Any
from .utils import read_json, read_jsonl, write_json


def aggregate_traces(trace_dir: str | Path = 'traces') -> dict[str, Any]:
    rows = []
    for path in sorted(Path(trace_dir).glob('*.jsonl')):
        rows.extend(read_jsonl(path))
    families = sorted(set(r.get('family', '') for r in rows))
    by_family = {}
    for fam in families:
        fam_rows = [r for r in rows if r.get('family') == fam]
        by_family[fam] = {'events': len(fam_rows), 'cost_usd': sum(float(r.get('cost_usd', 0.0)) for r in fam_rows), 'cache_hits': sum(1 for r in fam_rows if r.get('cache_hit'))}
    return {'events': len(rows), 'families': by_family}


def aggregate_experiments(experiment_dir: str | Path = 'experiments') -> dict[str, Any]:
    p = Path(experiment_dir)
    return {'oracle_gap': read_json(p / 'oracle_gap_summary.json', {}), 'baselines': read_json(p / 'baselines_summary.json', {}), 'heuristic_bdelg': read_json(p / 'heuristic_bdelg_summary.json', {})}


def write_aggregate_summary(out_path: str | Path = 'experiments/aggregate_summary.json') -> dict[str, Any]:
    summary = {'traces': aggregate_traces(), 'experiments': aggregate_experiments()}
    write_json(out_path, summary)
    return summary
