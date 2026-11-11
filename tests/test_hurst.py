from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_memory.hurst import hurst_test, log_returns, rs_hurst

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "hurst"


def _rs_loop(returns, min_size=10, n_sizes=20):
    n = len(returns)
    sizes = np.unique(np.logspace(np.log10(min_size), np.log10(n // 2), n_sizes, dtype=int))
    out = []
    for s in sizes:
        vals = []
        for i in range(0, n - s + 1, s):
            seg = returns[i:i + s]
            c = np.cumsum(seg - seg.mean())
            sd = seg.std(ddof=0)
            if sd > 0 and (c.max() - c.min()) > 0:
                vals.append((c.max() - c.min()) / sd)
        out.append(np.mean(vals) if vals else np.nan)
    out = np.array(out)
    k = np.isfinite(out)
    return np.polyfit(np.log(sizes[k]), np.log(out[k]), 1)[0]


def test_vectorized_equals_loop():
    r = np.random.default_rng(1).normal(size=777)
    assert rs_hurst(r) == pytest.approx(_rs_loop(r), abs=1e-12)


def test_white_noise_near_half():
    r = np.random.default_rng(0).normal(size=4000)
    assert 0.45 < rs_hurst(r) < 0.62


def test_shuffle_test_on_persistent_series():
    rng = np.random.default_rng(0)
    e = rng.normal(size=3000)
    x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = 0.5 * x[t - 1] + e[t]
    prices = 100 * np.exp(np.cumsum(0.01 * x))
    res = hurst_test(prices, n_shuffles=100, seed=0)
    assert res["H"] > res["null_hi"] and res["p_value"] < 0.05


@pytest.mark.skipif(not (RAW / "S&P 500_1mo.csv").exists(), reason="원본 데이터 없음")
@pytest.mark.parametrize("name,unit,expected", [
    ("S&P 500", "1mo", 0.6102), ("KOSPI", "1mo", 0.5765), ("Bitcoin", "1mo", 0.7886),
    ("S&P 500", "1wk", 0.5547), ("tesla", "1wk", 0.5990),
])
def test_reproduces_original_values(name, unit, expected):
    p = pd.read_csv(RAW / f"{name}_{unit}.csv")["Close"]
    assert rs_hurst(log_returns(p)) == pytest.approx(expected, abs=5e-5)
