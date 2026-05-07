from __future__ import annotations
from pathlib import Path


def oracle_gap_allows_readout_training(report_path: str | Path = 'reports/ORACLE_GAP_DECISION.md') -> bool:
    p = Path(report_path)
    if not p.exists():
        return False
    text = p.read_text(encoding='utf-8').upper()
    return ('DECISION: GO\n' in text or 'DECISION: GO-CONDITIONAL' in text) and 'DECISION: PENDING' not in text


def train_control_readouts(*args, **kwargs):
    if not oracle_gap_allows_readout_training():
        raise RuntimeError('KV/control readout training refused: oracle graph headroom not established.')
    return {'status': 'placeholder', 'trained': False}
