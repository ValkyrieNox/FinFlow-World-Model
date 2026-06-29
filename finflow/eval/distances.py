"""Distance metrics between empirical distributions of paths."""

from __future__ import annotations

import numpy as np


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Empirical 1-Wasserstein distance between two 1D samples.

    When both samples have the same length, falls back to the sorted L1 mean;
    otherwise uses the CDF-integral formulation.
    """

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("inputs must be non-empty")

    if x.size == y.size:
        return float(np.abs(np.sort(x) - np.sort(y)).mean())

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    all_vals = np.concatenate([x_sorted, y_sorted])
    all_vals.sort()
    cdf_x = np.searchsorted(x_sorted, all_vals, side="right") / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, all_vals, side="right") / y_sorted.size
    widths = np.diff(all_vals)
    avg = 0.5 * (np.abs(cdf_x[:-1] - cdf_y[:-1]) + np.abs(cdf_x[1:] - cdf_y[1:]))
    return float((widths * avg).sum())


def marginal_wasserstein_curve(real: np.ndarray, fake: np.ndarray) -> np.ndarray:
    """Per-step Wasserstein-1 distance between real and fake batches.

    Both inputs are ``[n_paths, n_steps]`` arrays of the same step count.
    Returns an array of shape ``[n_steps]``.
    """

    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("real and fake must be 2D")
    if real.shape[1] != fake.shape[1]:
        raise ValueError(
            f"step dimension mismatch: real has {real.shape[1]}, fake has {fake.shape[1]}"
        )
    out = np.empty(real.shape[1], dtype=np.float64)
    for t in range(real.shape[1]):
        out[t] = wasserstein_1d(real[:, t], fake[:, t])
    return out


def path_wasserstein(
    real_paths: np.ndarray,
    fake_paths: np.ndarray,
    *,
    reducer: str = "sum",
) -> float:
    """Wasserstein-1 between path summaries (default: cumulative sum at T)."""

    real_paths = np.asarray(real_paths, dtype=np.float64)
    fake_paths = np.asarray(fake_paths, dtype=np.float64)
    if real_paths.ndim != 2 or fake_paths.ndim != 2:
        raise ValueError("paths must be 2D [n_paths, n_steps]")
    if reducer == "sum":
        real_summary = real_paths.sum(axis=1)
        fake_summary = fake_paths.sum(axis=1)
    elif reducer == "max":
        real_summary = real_paths.max(axis=1)
        fake_summary = fake_paths.max(axis=1)
    elif reducer == "abs_sum":
        real_summary = np.abs(real_paths).sum(axis=1)
        fake_summary = np.abs(fake_paths).sum(axis=1)
    else:
        raise ValueError(f"unknown reducer '{reducer}'")
    return wasserstein_1d(real_summary, fake_summary)
