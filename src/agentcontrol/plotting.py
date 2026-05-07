from __future__ import annotations
from pathlib import Path
from typing import Any
from .utils import ensure_dir


def make_pareto_plot(points: list[dict[str, Any]], out_path: str | Path = 'figures/pareto.png') -> None:
    out = Path(out_path)
    ensure_dir(out.parent)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        out.with_suffix('.txt').write_text(str(points), encoding='utf-8')
        return
    plt.figure()
    for row in points:
        plt.scatter([row['cost']], [row['success']])
        plt.text(row['cost'], row['success'], row.get('label', 'policy'))
    plt.xlabel('Average cost')
    plt.ylabel('Success rate')
    plt.title('Smoke Pareto points')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
