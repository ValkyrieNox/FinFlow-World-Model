"""The five Cont (2001) stylized facts of financial log-return series.

Each function accepts ``returns`` of shape ``[n_paths, n_steps]`` (a batch of
sequences). Per-path metrics are averaged across paths so reports are robust
to single-path artifacts; pooled metrics (kurtosis, tail index) are computed
on flattened data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Low-level statistics
# ---------------------------------------------------------------------------


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("returns must have shape [n_paths, n_steps]")
    if arr.shape[1] < 2:
        raise ValueError("each path must have at least 2 steps")
    return arr


def kurtosis(returns: np.ndarray) -> float:
    """Sample (non-excess) kurtosis of the pooled returns."""

    flat = np.asarray(returns, dtype=np.float64).reshape(-1)
    if flat.size < 2:
        raise ValueError("kurtosis needs at least 2 samples")
    mean = flat.mean()
    centered = flat - mean
    m2 = (centered ** 2).mean()
    m4 = (centered ** 4).mean()
    if m2 <= 0:
        return 0.0
    return float(m4 / (m2 ** 2))


def _path_acf(x: np.ndarray, lags: int) -> np.ndarray:
    """Autocorrelation of a single 1D series up to ``lags``."""

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if lags >= n:
        raise ValueError(f"lags ({lags}) must be < series length ({n})")
    centered = x - x.mean()
    var = (centered ** 2).mean()
    if var <= 0:
        return np.zeros(lags, dtype=np.float64)
    out = np.empty(lags, dtype=np.float64)
    for k in range(1, lags + 1):
        out[k - 1] = (centered[k:] * centered[:-k]).mean() / var
    return out


def autocorrelation(returns: np.ndarray, lags: int = 20) -> np.ndarray:
    """Per-path ACF averaged across paths. Returns shape ``[lags]``."""

    arr = _validate_returns(returns)
    accs = np.stack([_path_acf(arr[i], lags) for i in range(arr.shape[0])], axis=0)
    return accs.mean(axis=0)


def absolute_return_acf(returns: np.ndarray, lags: int = 50) -> np.ndarray:
    """ACF of ``|r_t|`` (volatility clustering metric)."""

    arr = _validate_returns(returns)
    accs = np.stack(
        [_path_acf(np.abs(arr[i]), lags) for i in range(arr.shape[0])],
        axis=0,
    )
    return accs.mean(axis=0)


def leverage_correlation(returns: np.ndarray, lags: int = 10) -> np.ndarray:
    """``corr(r_t, r^2_{t+k})`` for k=1..lags, averaged across paths.

    A negative value indicates the leverage effect (returns drive future
    volatility upward when they are negative).
    """

    arr = _validate_returns(returns)
    n_paths, n_steps = arr.shape
    if lags >= n_steps:
        raise ValueError(f"lags ({lags}) must be < series length ({n_steps})")
    out = np.empty(lags, dtype=np.float64)
    sq = arr ** 2
    for k in range(1, lags + 1):
        a = arr[:, : n_steps - k]
        b = sq[:, k:]
        corr_per_path = np.empty(n_paths, dtype=np.float64)
        for i in range(n_paths):
            ai = a[i] - a[i].mean()
            bi = b[i] - b[i].mean()
            denom = np.sqrt((ai ** 2).sum() * (bi ** 2).sum())
            corr_per_path[i] = (ai * bi).sum() / denom if denom > 0 else 0.0
        out[k - 1] = corr_per_path.mean()
    return out


def aggregational_kurtosis(
    returns: np.ndarray,
    scales: Sequence[int] = (1, 5, 21),
) -> dict[int, float]:
    """Kurtosis of returns aggregated to coarser scales.

    Cont 2001: kurtosis should decay toward 3 (Gaussianity) as scale grows.
    """

    arr = _validate_returns(returns)
    out: dict[int, float] = {}
    for scale in scales:
        scale = int(scale)
        if scale <= 0:
            raise ValueError("scales must be positive integers")
        if scale > arr.shape[1]:
            continue
        n_blocks = arr.shape[1] // scale
        if n_blocks == 0:
            continue
        truncated = arr[:, : n_blocks * scale]
        aggregated = truncated.reshape(arr.shape[0], n_blocks, scale).sum(axis=-1)
        if aggregated.size < 2:
            continue
        out[scale] = kurtosis(aggregated)
    return out


def tail_index_hill(returns: np.ndarray, frac: float = 0.05) -> float:
    """Hill tail index estimator on the upper ``frac`` of ``|returns|``.

    A heavier tail produces a smaller index. ``frac=0.05`` matches Cont (2001).
    """

    arr = _validate_returns(returns).reshape(-1)
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1)")
    abs_returns = np.sort(np.abs(arr))[::-1]
    k = max(2, int(len(abs_returns) * frac))
    threshold = abs_returns[k]
    if threshold <= 0:
        return float("inf")
    top = abs_returns[:k]
    log_ratio = np.log(top / threshold)
    mean_log_ratio = log_ratio.mean()
    if mean_log_ratio <= 0:
        return float("inf")
    return float(1.0 / mean_log_ratio)


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------


@dataclass
class StylizedFactReport:
    """Container for the five Cont (2001) stylized facts."""

    kurtosis: float
    return_acf: np.ndarray
    absolute_return_acf: np.ndarray
    leverage_correlation: np.ndarray
    aggregational_kurtosis: dict[int, float]
    tail_index: float

    def to_dict(self) -> dict[str, object]:
        return {
            "kurtosis": self.kurtosis,
            "return_acf": self.return_acf.tolist(),
            "absolute_return_acf": self.absolute_return_acf.tolist(),
            "leverage_correlation": self.leverage_correlation.tolist(),
            "aggregational_kurtosis": {str(k): v for k, v in self.aggregational_kurtosis.items()},
            "tail_index": self.tail_index,
        }


def stylized_fact_report(
    returns: np.ndarray,
    *,
    return_acf_lags: int = 20,
    absolute_acf_lags: int = 50,
    leverage_lags: int = 10,
    aggregation_scales: Sequence[int] = (1, 5, 21),
    hill_frac: float = 0.05,
) -> StylizedFactReport:
    """Compute the full stylized-fact report for one batch.

    Lag arguments are clipped to ``series_length - 1`` so short test paths
    do not blow up on the default settings. The lag values actually used end
    up in the returned arrays' shape.
    """

    arr = _validate_returns(returns)
    cap = arr.shape[1] - 1
    return_acf_lags = max(1, min(int(return_acf_lags), cap))
    absolute_acf_lags = max(1, min(int(absolute_acf_lags), cap))
    leverage_lags = max(1, min(int(leverage_lags), cap))
    return StylizedFactReport(
        kurtosis=kurtosis(arr),
        return_acf=autocorrelation(arr, lags=return_acf_lags),
        absolute_return_acf=absolute_return_acf(arr, lags=absolute_acf_lags),
        leverage_correlation=leverage_correlation(arr, lags=leverage_lags),
        aggregational_kurtosis=aggregational_kurtosis(arr, scales=aggregation_scales),
        tail_index=tail_index_hill(arr, frac=hill_frac),
    )


def compare_stylized_facts(
    real: StylizedFactReport,
    fake: StylizedFactReport,
) -> dict[str, object]:
    """Summarize differences between two stylized-fact reports."""

    return {
        "kurtosis_real": real.kurtosis,
        "kurtosis_fake": fake.kurtosis,
        "kurtosis_abs_diff": abs(real.kurtosis - fake.kurtosis),
        "return_acf_l1": float(np.abs(real.return_acf - fake.return_acf).mean()),
        "absolute_return_acf_l1": float(
            np.abs(real.absolute_return_acf - fake.absolute_return_acf).mean()
        ),
        "leverage_correlation_l1": float(
            np.abs(real.leverage_correlation - fake.leverage_correlation).mean()
        ),
        "aggregational_kurtosis_l1": {
            str(s): abs(real.aggregational_kurtosis.get(s, np.nan)
                        - fake.aggregational_kurtosis.get(s, np.nan))
            for s in set(real.aggregational_kurtosis) & set(fake.aggregational_kurtosis)
        },
        "tail_index_real": real.tail_index,
        "tail_index_fake": fake.tail_index,
        "tail_index_abs_diff": abs(real.tail_index - fake.tail_index),
    }
