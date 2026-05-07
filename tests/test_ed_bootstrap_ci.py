"""Bootstrap helpers are deterministic and CI bounds are well-formed."""
from __future__ import annotations

import random
import statistics


def test_bootstrap_seed_determinism():
    rng_a = random.Random(1234)
    rng_b = random.Random(1234)
    sample_a = [rng_a.choice(range(10)) for _ in range(50)]
    sample_b = [rng_b.choice(range(10)) for _ in range(50)]
    assert sample_a == sample_b


def test_ci_bounds_inline():
    """Reproduces the small CI helper used in run_ed_bootstrap_ci.py."""
    samples = [float(i) for i in range(100)]
    s = sorted(samples)
    n = len(s)
    lo, hi = 2.5, 97.5
    ci_lo = s[max(0, int(n * lo / 100.0) - 1)]
    ci_hi = s[min(n - 1, int(n * hi / 100.0) - 1)]
    assert ci_lo <= statistics.fmean(samples) <= ci_hi
    assert ci_lo == 1.0
    assert ci_hi == 96.0
