"""Truncated path-signature features and signature Wasserstein distances."""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from finflow.eval.distances import wasserstein_1d


def returns_to_time_cumsum_paths(returns: np.ndarray) -> np.ndarray:
    """Convert ``[N, T]`` returns into ``[N, T+1, 2]`` time/cumsum paths."""

    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError("returns must have shape [n_paths, n_steps]")
    n_paths, n_steps = returns.shape
    time = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
    out = np.empty((n_paths, n_steps + 1, 2), dtype=np.float64)
    out[:, :, 0] = time[None, :]
    out[:, 0, 1] = 0.0
    out[:, 1:, 1] = np.cumsum(returns, axis=1)
    return out


def _segment_signature(dx: np.ndarray, depth: int) -> list[np.ndarray]:
    levels = [np.array([1.0], dtype=np.float64)]
    for level in range(1, depth + 1):
        tensor = dx
        for _ in range(level - 1):
            tensor = np.multiply.outer(tensor, dx).reshape(-1)
        levels.append(tensor / math.factorial(level))
    return levels


def _combine_signatures(left: list[np.ndarray], right: list[np.ndarray], depth: int) -> list[np.ndarray]:
    combined = [np.array([1.0], dtype=np.float64)]
    for level in range(1, depth + 1):
        terms: list[np.ndarray] = []
        for left_level in range(level + 1):
            right_level = level - left_level
            terms.append(
                np.multiply.outer(left[left_level], right[right_level]).reshape(-1)
            )
        combined.append(np.sum(terms, axis=0))
    return combined


def signature_features(paths: np.ndarray, depth: int = 3) -> np.ndarray:
    """Compute flattened truncated signatures for piecewise-linear paths.

    ``paths`` must have shape ``[n_paths, n_points, dim]``. Depth is capped at
    4 by design because the intended P1 use is a lightweight report metric.
    """

    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("paths must have shape [n_paths, n_points, dim]")
    if not 1 <= depth <= 4:
        raise ValueError("depth must be in [1, 4]")
    n_paths, n_points, dim = paths.shape
    if n_points < 2:
        raise ValueError("paths need at least two points")
    if dim <= 0:
        raise ValueError("path dim must be positive")

    feature_dim = sum(dim ** level for level in range(1, depth + 1))
    out = np.empty((n_paths, feature_dim), dtype=np.float64)
    for i in range(n_paths):
        sig = [np.array([1.0], dtype=np.float64)]
        sig.extend(np.zeros(dim ** level, dtype=np.float64) for level in range(1, depth + 1))
        increments = np.diff(paths[i], axis=0)
        for dx in increments:
            sig = _combine_signatures(sig, _segment_signature(dx, depth), depth)
        out[i] = np.concatenate(sig[1:])
    return out


def signature_coordinate_names(dim: int = 2, depth: int = 3) -> list[str]:
    """Names for flattened signature coordinates."""

    if not 1 <= depth <= 4:
        raise ValueError("depth must be in [1, 4]")
    alphabet = [str(i) for i in range(dim)]
    names: list[str] = []
    for level in range(1, depth + 1):
        names.extend("".join(word) for word in product(alphabet, repeat=level))
    return names


def signature_wasserstein(
    real_returns: np.ndarray,
    fake_returns: np.ndarray,
    *,
    depth: int = 3,
) -> dict[str, object]:
    """Compare return-path distributions through time/cumsum signatures."""

    real_paths = returns_to_time_cumsum_paths(real_returns)
    fake_paths = returns_to_time_cumsum_paths(fake_returns)
    real_features = signature_features(real_paths, depth=depth)
    fake_features = signature_features(fake_paths, depth=depth)
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("signature feature dimensions do not match")

    per_coordinate = np.array(
        [
            wasserstein_1d(real_features[:, i], fake_features[:, i])
            for i in range(real_features.shape[1])
        ],
        dtype=np.float64,
    )
    return {
        "depth": int(depth),
        "mean": float(per_coordinate.mean()),
        "max": float(per_coordinate.max()),
        "per_coordinate": per_coordinate.tolist(),
        "coordinate_names": signature_coordinate_names(dim=2, depth=depth),
    }
