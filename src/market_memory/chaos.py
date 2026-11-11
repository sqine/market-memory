from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (x - x.mean()) / (x.std() + 1e-12)


def embed(x, m: int, tau: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x) - (m - 1) * tau
    if m < 1 or tau < 1 or n <= 1:
        raise ValueError("series too short for m, tau")
    return np.column_stack([x[i * tau: i * tau + n] for i in range(m)])


def correlation_sum(x, m: int, radii: np.ndarray, tau: int = 1) -> np.ndarray:
    d = np.sort(pdist(embed(x, m, tau), metric="chebyshev"))
    return np.searchsorted(d, radii, side="left") / len(d)


def correlation_dimension(x, m: int, radii: np.ndarray, tau: int = 1, c_range: tuple = (1e-3, 1e-1)) -> float:
    c = correlation_sum(x, m, radii, tau)
    ok = (c >= c_range[0]) & (c <= c_range[1])
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(radii[ok]), np.log(c[ok]), 1)[0])


def lyapunov_rosenstein(x, m: int = 5, tau: int = 1, min_tsep: int | None = None,
                        max_t: int = 30, fit_range: tuple = (1, 10)) -> dict:
    X = embed(zscore(x), m, tau)
    n = len(X)
    min_tsep = max(2, m * tau) if min_tsep is None else min_tsep
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    idx = np.arange(n)
    D[np.abs(idx[:, None] - idx[None, :]) <= min_tsep] = np.inf
    nn = np.argmin(D, axis=1)
    has = np.isfinite(D[idx, nn])
    t_vals = np.arange(max_t)
    log_div = np.full(max_t, np.nan)
    for t in t_vals:
        ok = has & (idx + t < n) & (nn + t < n)
        if not ok.any():
            continue
        d = np.linalg.norm(X[idx[ok] + t] - X[nn[ok] + t], axis=1)
        d = d[d > 0]
        if len(d):
            log_div[t] = np.log(d).mean()
    a, b = fit_range
    sel = (t_vals >= a) & (t_vals <= b) & np.isfinite(log_div)
    slope = float(np.polyfit(t_vals[sel], log_div[sel], 1)[0]) if sel.sum() >= 3 else float("nan")
    return {"lyap": slope, "t": t_vals, "log_div": log_div}


def shuffle_surrogate(x, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(np.asarray(x, dtype=float))


def phase_surrogate(x, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n) + x.mean()


def surrogate_test(stat, x, kind: str, n: int = 19, seed: int = 0, side: str = "less") -> dict:
    rng = np.random.default_rng(seed)
    gen = {"shuffle": shuffle_surrogate, "phase": phase_surrogate}[kind]
    obs = stat(x)
    sur = np.array([stat(gen(x, rng)) for _ in range(n)])
    rank = np.sum(sur <= obs) if side == "less" else np.sum(sur >= obs)
    return {"observed": float(obs), "surr_mean": float(np.nanmean(sur)), "surr_min": float(np.nanmin(sur)),
            "surr_max": float(np.nanmax(sur)), "p_value": float((rank + 1) / (n + 1)), "surrogates": sur}


def detrend_with_cpi(prices: pd.Series, cpi: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": prices})
    df["key"] = df.index.to_period("M").to_timestamp()
    c = cpi.copy()
    c.index = pd.to_datetime(c.index).to_period("M").to_timestamp()
    df = df.join(c.rename("cpi"), on="key").dropna()
    y = np.log(df["p"].to_numpy())
    A = np.column_stack([np.ones(len(df)), np.log(df["cpi"].to_numpy()), np.arange(len(df))])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return pd.Series(y - A @ beta, index=df.index, name="detrended_log_price")
