from __future__ import annotations

import numpy as np


def log_returns(prices) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    return np.log(p[1:] / p[:-1])


def _sizes(n: int, min_size: int = 10, n_sizes: int = 20) -> np.ndarray:
    return np.unique(np.logspace(np.log10(min_size), np.log10(n // 2), n_sizes, dtype=int))


def rs_curve(returns: np.ndarray, min_size: int = 10, n_sizes: int = 20) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sizes = _sizes(n, min_size, n_sizes)
    rs = np.full(len(sizes), np.nan)
    for i, s in enumerate(sizes):
        k = n // s
        seg = r[: k * s].reshape(k, s)
        cum = np.cumsum(seg - seg.mean(axis=1, keepdims=True), axis=1)
        R = cum.max(axis=1) - cum.min(axis=1)
        S = seg.std(axis=1, ddof=0)
        ok = (S > 0) & (R > 0)
        if ok.any():
            rs[i] = np.mean(R[ok] / S[ok])
    keep = np.isfinite(rs)
    return sizes[keep], rs[keep]


def rs_hurst(returns: np.ndarray, min_size: int = 10, n_sizes: int = 20) -> float:
    sizes, rs = rs_curve(returns, min_size, n_sizes)
    return float(np.polyfit(np.log(sizes), np.log(rs), 1)[0])


def shuffle_null(returns: np.ndarray, n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    return np.array([rs_hurst(rng.permutation(r)) for _ in range(n)])


def hurst_test(prices, n_shuffles: int = 500, seed: int = 0) -> dict:
    r = log_returns(prices)
    h = rs_hurst(r)
    null = shuffle_null(r, n_shuffles, seed)
    return {
        "n_returns": len(r),
        "H": h,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_lo": float(np.quantile(null, 0.025)),
        "null_hi": float(np.quantile(null, 0.975)),
        "excess": float(h - null.mean()),
        "p_value": float((np.sum(null >= h) + 1) / (len(null) + 1)),
    }
