from __future__ import annotations
from pathlib import Path


def oracle_gap_allows_training(report_path: str | Path = 'reports/ORACLE_GAP_DECISION.md') -> bool:
    p = Path(report_path)
    if not p.exists():
        return False
    text = p.read_text(encoding='utf-8').upper()
    return ('DECISION: GO\n' in text or 'DECISION: GO-CONDITIONAL' in text) and 'DECISION: PENDING' not in text


def train_controller(*args, **kwargs):
    if not oracle_gap_allows_training():
        raise RuntimeError('Controller training refused: ORACLE_GAP_DECISION.md does not say GO or GO-CONDITIONAL.')
    return {'status': 'placeholder', 'trained': False}
