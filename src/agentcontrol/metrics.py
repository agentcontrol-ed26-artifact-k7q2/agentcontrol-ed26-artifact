from __future__ import annotations
from typing import Any


def success_rate(rows: list[dict[str, Any]]) -> float:
    return sum(1 for r in rows if r.get('success')) / len(rows) if rows else 0.0


def mean_cost(rows: list[dict[str, Any]]) -> float:
    return sum(float(r.get('cost', r.get('cost_usd', 0.0))) for r in rows) / len(rows) if rows else 0.0


def mean_latency_ms(rows: list[dict[str, Any]]) -> float:
    return sum(float(r.get('latency_ms', 0.0)) for r in rows) / len(rows) if rows else 0.0


def pareto_auc(points: list[tuple[float, float]]) -> float:
    if not points: return 0.0
    pts = sorted(points)
    envelope, best = [], 0.0
    for c, s in pts:
        best = max(best, s)
        envelope.append((c, best))
    if len(envelope) == 1: return envelope[0][1]
    min_c, max_c = envelope[0][0], envelope[-1][0]
    if max_c == min_c: return max(s for _, s in envelope)
    area = 0.0
    for (c1, s1), (c2, s2) in zip(envelope, envelope[1:]):
        area += (c2 - c1) * (s1 + s2) / 2
    return area / (max_c - min_c)


def cost_at_target_success(points: list[tuple[float, float]], target_success: float) -> float | None:
    vals = [cost for cost, success in points if success >= target_success]
    return min(vals) if vals else None


def regret(oracle_score: float, policy_score: float) -> float:
    return oracle_score - policy_score


def unsupported_accept_rate(rows: list[dict[str, Any]]) -> float:
    accepted = [r for r in rows if r.get('accepted', r.get('success', False))]
    return sum(1 for r in accepted if float(r.get('unsupported_risk', 0.0)) > 0.0) / len(accepted) if accepted else 0.0
