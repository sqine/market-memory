import numpy as np
import pandas as pd
import pytest

from market_memory.chaos import (correlation_dimension, detrend_with_cpi, embed, lyapunov_rosenstein,
                                  phase_surrogate)


def logistic(n=1500, r=4.0, x0=0.1234):
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = r * x[t - 1] * (1 - x[t - 1])
    return x


def test_embed_shape():
    X = embed(np.arange(10), m=3, tau=2)
    assert X.shape == (6, 3) and list(X[0]) == [0, 2, 4]


def test_lyapunov_logistic_positive_noise_free():
    res = lyapunov_rosenstein(logistic(1200), m=2, tau=1, max_t=12, fit_range=(1, 5))
    assert 0.45 < res["lyap"] < 0.9


def test_correlation_dimension_low_for_map_high_for_noise():
    radii = np.logspace(-2.5, 0.5, 40)
    d_map = correlation_dimension((logistic(1500) - 0.5) / 0.35, m=4, radii=radii)
    d_noise = correlation_dimension(np.random.default_rng(0).normal(size=1500), m=4, radii=radii)
    assert d_map < 1.6 and d_noise > 3.0


def test_phase_surrogate_keeps_spectrum():
    x = np.cumsum(np.random.default_rng(2).normal(size=512))
    s = phase_surrogate(x, np.random.default_rng(3))
    assert np.allclose(np.abs(np.fft.rfft(x - x.mean())), np.abs(np.fft.rfft(s - s.mean())), atol=1e-8)


def test_detrend_removes_cpi_and_time():
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    cpi = pd.Series(np.exp(np.linspace(4.8, 5.2, 120)), index=idx)
    noise = np.random.default_rng(0).normal(0, 0.01, 120)
    price = pd.Series(np.exp(1.0 + 1.5 * np.log(cpi.values) + 0.002 * np.arange(120) + noise), index=idx)
    res = detrend_with_cpi(price, cpi)
    assert abs(np.corrcoef(res.values, noise)[0, 1]) > 0.99
