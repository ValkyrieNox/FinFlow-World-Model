"""Heston data generation for the V3 autoregressive world-model pipeline.

The variance process uses Andersen's quadratic-exponential (QE) scheme so that
``v_t`` stays non-negative without truncation; the log-return update is the
QE-M discretization that conditions on both ``v_t`` and ``v_{t+1}``.

This module also supports a regime-switching variant in which
``(kappa, theta, xi, rho, mu)`` switches across discrete macro regimes
(``normal / high_vol / crash`` by default) driven by a per-path Markov chain.
The active regime at each step is recorded as the action label ``a_t`` and
flows into the transition tensors so the FM models can be conditioned on it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """Default Heston parameters used in the V3 notes."""

    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    v0: float = 0.04
    s0: float = 100.0
    mu: float = 0.05
    dt: float = 1.0 / 252.0


@dataclass(frozen=True)
class RegimeParams:
    """Per-regime overrides for ``(kappa, theta, xi, rho, mu)``."""

    name: str
    kappa: float
    theta: float
    xi: float
    rho: float = -0.7
    mu: float = 0.05


DEFAULT_REGIMES: tuple[RegimeParams, ...] = (
    RegimeParams(name="normal", kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, mu=0.05),
    RegimeParams(name="high_vol", kappa=3.0, theta=0.09, xi=0.5, rho=-0.7, mu=0.05),
    RegimeParams(name="crash", kappa=4.0, theta=0.16, xi=0.8, rho=-0.7, mu=-0.20),
)


DEFAULT_TRANSITION_MATRIX: np.ndarray = np.array(
    [
        [0.992, 0.006, 0.002],   # from normal
        [0.040, 0.950, 0.010],   # from high_vol
        [0.050, 0.050, 0.900],   # from crash
    ],
    dtype=np.float64,
)


def _validate_params(params: HestonParams) -> None:
    if params.kappa <= 0:
        raise ValueError("kappa must be positive")
    if params.theta <= 0:
        raise ValueError("theta must be positive")
    if params.xi < 0:
        raise ValueError("xi must be non-negative")
    if not -1.0 <= params.rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    if params.v0 < 0:
        raise ValueError("v0 must be non-negative")
    if params.s0 <= 0:
        raise ValueError("s0 must be positive")
    if params.dt <= 0:
        raise ValueError("dt must be positive")


def _validate_regimes(
    regimes: Sequence[RegimeParams],
    transition_matrix: np.ndarray,
) -> None:
    if len(regimes) == 0:
        raise ValueError("regimes must be non-empty")
    n = len(regimes)
    for r in regimes:
        if r.kappa <= 0 or r.theta <= 0 or r.xi <= 0:
            raise ValueError(f"regime '{r.name}': kappa, theta, xi must be positive")
        if not -1.0 <= r.rho <= 1.0:
            raise ValueError(f"regime '{r.name}': rho must be in [-1, 1]")
    if transition_matrix.shape != (n, n):
        raise ValueError(f"transition_matrix must have shape ({n}, {n})")
    if np.any(transition_matrix < 0):
        raise ValueError("transition_matrix entries must be non-negative")
    row_sums = transition_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("transition_matrix rows must sum to 1")


def _qe_variance_step_core(
    v_t: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    kappa: np.ndarray | float,
    theta: np.ndarray | float,
    xi: np.ndarray | float,
    dt: float,
    psi_c: float,
) -> np.ndarray:
    """Vectorized Andersen QE variance step.

    ``kappa``, ``theta`` and ``xi`` may be scalars or per-element arrays;
    broadcasting follows numpy rules. ``z`` and ``u`` must have the same shape
    as ``v_t`` and provide the per-element standard normal / uniform draws.
    """

    eps = np.finfo(np.float64).eps
    exp_kdt = np.exp(-np.asarray(kappa) * dt)
    m = theta + (v_t - theta) * exp_kdt
    s2 = (
        v_t * xi ** 2 * exp_kdt * (1.0 - exp_kdt) / kappa
        + theta * xi ** 2 * (1.0 - exp_kdt) ** 2 / (2.0 * kappa)
    )
    m = np.broadcast_to(np.maximum(m, eps), v_t.shape).astype(np.float64, copy=True)
    s2 = np.broadcast_to(s2, v_t.shape).astype(np.float64, copy=True)
    psi = s2 / np.maximum(m * m, eps)

    v_next = np.empty_like(v_t, dtype=np.float64)

    quadratic = psi <= psi_c
    if np.any(quadratic):
        psi_q = np.maximum(psi[quadratic], eps)
        inv_psi = 2.0 / psi_q
        b2 = inv_psi - 1.0 + np.sqrt(inv_psi) * np.sqrt(np.maximum(inv_psi - 1.0, 0.0))
        a = m[quadratic] / (1.0 + b2)
        v_next[quadratic] = a * (np.sqrt(np.maximum(b2, 0.0)) + z[quadratic]) ** 2

    exponential = ~quadratic
    if np.any(exponential):
        psi_e = psi[exponential]
        p = np.clip((psi_e - 1.0) / (psi_e + 1.0), 0.0, 1.0 - eps)
        beta = (1.0 - p) / m[exponential]
        u_e = u[exponential]
        draws = np.zeros_like(u_e)
        nonzero = u_e > p
        draws[nonzero] = np.log((1.0 - p[nonzero]) / (1.0 - u_e[nonzero])) / beta[nonzero]
        v_next[exponential] = draws

    return np.maximum(v_next, 0.0)


def _qe_m_log_return_core(
    v_t: np.ndarray,
    v_next: np.ndarray,
    z: np.ndarray,
    mu: np.ndarray | float,
    kappa: np.ndarray | float,
    theta: np.ndarray | float,
    xi: np.ndarray | float,
    rho: np.ndarray | float,
    dt: float,
    gamma1: float,
    gamma2: float,
) -> np.ndarray:
    """Vectorized QE-M log-return update conditional on the variance transition."""

    k0 = -rho * kappa * theta * dt / xi
    k1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = gamma1 * dt * (1.0 - rho * rho)
    k4 = gamma2 * dt * (1.0 - rho * rho)
    conditional_var = np.maximum(k3 * v_t + k4 * v_next, 0.0)
    return mu * dt + k0 + k1 * v_t + k2 * v_next + np.sqrt(conditional_var) * z


def _qe_variance_step(
    v_t: np.ndarray,
    rng: np.random.Generator,
    params: HestonParams,
    psi_c: float,
) -> np.ndarray:
    if params.xi == 0.0:
        exp_kdt = np.exp(-params.kappa * params.dt)
        return params.theta + (v_t - params.theta) * exp_kdt
    z = rng.standard_normal(v_t.shape)
    u = rng.random(v_t.shape)
    return _qe_variance_step_core(
        v_t, z, u, params.kappa, params.theta, params.xi, params.dt, psi_c
    )


def _qe_m_log_return(
    v_t: np.ndarray,
    v_next: np.ndarray,
    rng: np.random.Generator,
    params: HestonParams,
    gamma1: float,
    gamma2: float,
) -> np.ndarray:
    if params.xi == 0.0:
        vol = np.sqrt(np.maximum(v_t, 0.0) * params.dt)
        z = rng.standard_normal(v_t.shape)
        return (params.mu - 0.5 * v_t) * params.dt + vol * z
    z = rng.standard_normal(v_t.shape)
    return _qe_m_log_return_core(
        v_t, v_next, z,
        params.mu, params.kappa, params.theta, params.xi, params.rho,
        params.dt, gamma1, gamma2,
    )


def simulate_heston_qe(
    n_paths: int,
    n_steps: int = 252,
    params: HestonParams | None = None,
    seed: int | None = None,
    dtype: np.dtype[Any] = np.float32,
    psi_c: float = 1.5,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
) -> dict[str, np.ndarray]:
    """Single-regime Heston simulation with Andersen QE variance and QE-M returns."""

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if psi_c <= 0:
        raise ValueError("psi_c must be positive")
    if gamma1 < 0 or gamma2 < 0:
        raise ValueError("gamma1 and gamma2 must be non-negative")

    params = params or HestonParams()
    _validate_params(params)
    rng = np.random.default_rng(seed)

    v_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_s_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_returns = np.empty((n_paths, n_steps), dtype=np.float64)
    v_paths[:, 0] = params.v0
    log_s_paths[:, 0] = np.log(params.s0)

    for step in range(n_steps):
        v_t = v_paths[:, step]
        v_next = _qe_variance_step(v_t, rng, params, psi_c)
        r_t = _qe_m_log_return(v_t, v_next, rng, params, gamma1, gamma2)
        v_paths[:, step + 1] = v_next
        log_returns[:, step] = r_t
        log_s_paths[:, step + 1] = log_s_paths[:, step] + r_t

    return {
        "s_paths": np.exp(log_s_paths).astype(dtype, copy=False),
        "v_paths": v_paths.astype(dtype, copy=False),
        "log_returns": log_returns.astype(dtype, copy=False),
    }


def _sample_actions(
    n_paths: int,
    n_steps: int,
    transition_matrix: np.ndarray,
    initial_regime: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample per-path Markov chain over regime indices, shape ``[n_paths, n_steps]``."""

    actions = np.empty((n_paths, n_steps), dtype=np.int8)
    actions[:, 0] = initial_regime
    cdf_rows = np.cumsum(transition_matrix, axis=1)
    for t in range(1, n_steps):
        u = rng.random(n_paths)
        prev = actions[:, t - 1].astype(np.int64)
        cdf = cdf_rows[prev]
        # Inverse-CDF sampling: count how many prefix CDF values u exceeds.
        actions[:, t] = (u[:, None] > cdf[:, :-1]).sum(axis=1).astype(np.int8)
    return actions


def simulate_regime_switching_heston(
    n_paths: int,
    n_steps: int = 252,
    regimes: Sequence[RegimeParams] = DEFAULT_REGIMES,
    transition_matrix: np.ndarray | None = None,
    v0: float = 0.04,
    s0: float = 100.0,
    dt: float = 1.0 / 252.0,
    initial_regime: int = 0,
    seed: int | None = None,
    dtype: np.dtype[Any] = np.float32,
    psi_c: float = 1.5,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
) -> dict[str, np.ndarray]:
    """Heston paths with per-step Markov regime switching.

    Returns a dict containing ``s_paths`` / ``v_paths`` / ``log_returns`` plus
    ``actions`` of shape ``[n_paths, n_steps]``. ``actions[i, t]`` is the regime
    index that drives the ``t -> t+1`` transition.
    """

    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if v0 < 0 or s0 <= 0 or dt <= 0:
        raise ValueError("v0 >= 0, s0 > 0, dt > 0 required")
    if not 0 <= initial_regime < len(regimes):
        raise ValueError(f"initial_regime must be in [0, {len(regimes)})")

    regimes = tuple(regimes)
    if transition_matrix is None:
        if len(regimes) != len(DEFAULT_REGIMES):
            raise ValueError(
                "must supply transition_matrix when regimes count differs from default"
            )
        transition_matrix = DEFAULT_TRANSITION_MATRIX
    transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
    _validate_regimes(regimes, transition_matrix)

    kappa_table = np.array([r.kappa for r in regimes], dtype=np.float64)
    theta_table = np.array([r.theta for r in regimes], dtype=np.float64)
    xi_table = np.array([r.xi for r in regimes], dtype=np.float64)
    rho_table = np.array([r.rho for r in regimes], dtype=np.float64)
    mu_table = np.array([r.mu for r in regimes], dtype=np.float64)

    rng = np.random.default_rng(seed)
    actions = _sample_actions(n_paths, n_steps, transition_matrix, initial_regime, rng)

    v_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_s_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_returns = np.empty((n_paths, n_steps), dtype=np.float64)
    v_paths[:, 0] = v0
    log_s_paths[:, 0] = np.log(s0)

    for step in range(n_steps):
        a = actions[:, step].astype(np.int64)
        kappa_arr = kappa_table[a]
        theta_arr = theta_table[a]
        xi_arr = xi_table[a]
        rho_arr = rho_table[a]
        mu_arr = mu_table[a]

        v_t = v_paths[:, step]
        z_v = rng.standard_normal(n_paths)
        u_v = rng.random(n_paths)
        v_next = _qe_variance_step_core(
            v_t, z_v, u_v, kappa_arr, theta_arr, xi_arr, dt, psi_c
        )
        z_r = rng.standard_normal(n_paths)
        r_t = _qe_m_log_return_core(
            v_t, v_next, z_r, mu_arr, kappa_arr, theta_arr, xi_arr, rho_arr, dt, gamma1, gamma2
        )
        v_paths[:, step + 1] = v_next
        log_returns[:, step] = r_t
        log_s_paths[:, step + 1] = log_s_paths[:, step] + r_t

    return {
        "s_paths": np.exp(log_s_paths).astype(dtype, copy=False),
        "v_paths": v_paths.astype(dtype, copy=False),
        "log_returns": log_returns.astype(dtype, copy=False),
        "actions": actions,
    }


def build_transition_arrays(
    v_paths: np.ndarray,
    log_returns: np.ndarray,
    actions: np.ndarray | None = None,
    eps: float = 1e-8,
    dtype: np.dtype[Any] = np.float32,
    include_index: bool = True,
) -> dict[str, np.ndarray]:
    """Flatten Heston paths into V3 one-step transition samples.

    Alignment:
        condition at step ``t``: ``(v_t, r_{t-1}, a_t)``, with ``r_{-1}=0``
        target at step ``t``: ``(v_{t+1}, r_t)``

    ``a_t`` is the regime active during the ``t -> t+1`` transition (i.e., the
    regime that produced ``(v_{t+1}, r_t)`` from ``v_t``).
    """

    if v_paths.ndim != 2:
        raise ValueError("v_paths must have shape [n_paths, n_steps + 1]")
    if log_returns.ndim != 2:
        raise ValueError("log_returns must have shape [n_paths, n_steps]")
    n_paths, n_steps = log_returns.shape
    if v_paths.shape != (n_paths, n_steps + 1):
        raise ValueError("v_paths shape must match log_returns shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if actions is not None and actions.shape != (n_paths, n_steps):
        raise ValueError("actions must have shape [n_paths, n_steps]")

    previous_returns = np.zeros_like(log_returns)
    previous_returns[:, 1:] = log_returns[:, :-1]

    transitions: dict[str, np.ndarray] = {
        "v_t": v_paths[:, :-1].reshape(-1).astype(dtype, copy=False),
        "r_t": previous_returns.reshape(-1).astype(dtype, copy=False),
        "v_next": v_paths[:, 1:].reshape(-1).astype(dtype, copy=False),
        "r_next": log_returns.reshape(-1).astype(dtype, copy=False),
        "log_v_t": np.log(v_paths[:, :-1].reshape(-1) + eps).astype(dtype, copy=False),
        "log_v_next": np.log(v_paths[:, 1:].reshape(-1) + eps).astype(dtype, copy=False),
    }
    if actions is not None:
        transitions["action"] = actions.reshape(-1).astype(np.int8, copy=False)
    if include_index:
        transitions["path_index"] = np.repeat(np.arange(n_paths, dtype=np.int32), n_steps)
        transitions["step_index"] = np.tile(np.arange(n_steps, dtype=np.int16), n_paths)
    return transitions


def _stats_from_train(train: dict[str, np.ndarray], eps: float) -> dict[str, float]:
    log_v = np.log(train["v_paths"] + eps)
    returns = train["log_returns"]
    return {
        "log_v_mean": float(log_v.mean()),
        "log_v_std": float(log_v.std(ddof=0) + eps),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=0) + eps),
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def generate_heston_dataset(
    output_dir: str | Path,
    n_train: int = 50_000,
    n_val: int = 5_000,
    n_test: int = 10_000,
    n_steps: int = 252,
    params: HestonParams | None = None,
    regimes: Sequence[RegimeParams] | None = None,
    transition_matrix: np.ndarray | None = None,
    initial_regime: int = 0,
    seed: int = 1234,
    save_transitions: bool = True,
    dtype: np.dtype[Any] = np.float32,
) -> dict[str, Any]:
    """Generate train/val/test Heston datasets and write them to disk.

    Two modes:
    - Single regime (default): uses ``params`` (or ``HestonParams()``); the
      resulting files do not include an ``action`` array.
    - Regime switching: pass ``regimes`` (and optionally ``transition_matrix``).
      Each path's per-step regime index is stored as ``actions`` (full paths)
      / ``action`` (transition tensors). ``params`` is still used to pick
      ``v0``, ``s0`` and ``dt`` if supplied (otherwise defaults are used).
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    use_regimes = regimes is not None
    if use_regimes:
        regimes = tuple(regimes)
        if transition_matrix is None:
            if len(regimes) != len(DEFAULT_REGIMES):
                raise ValueError(
                    "must supply transition_matrix when regimes count differs from default"
                )
            transition_matrix = DEFAULT_TRANSITION_MATRIX
        transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        _validate_regimes(regimes, transition_matrix)
        v0 = params.v0 if params is not None else 0.04
        s0 = params.s0 if params is not None else 100.0
        dt = params.dt if params is not None else 1.0 / 252.0
    else:
        params = params or HestonParams()

    split_sizes = {"train": n_train, "val": n_val, "test": n_test}
    split_seeds = {"train": seed, "val": seed + 1, "test": seed + 2}
    generated: dict[str, dict[str, np.ndarray]] = {}

    for split, size in split_sizes.items():
        if use_regimes:
            arrays = simulate_regime_switching_heston(
                n_paths=size,
                n_steps=n_steps,
                regimes=regimes,
                transition_matrix=transition_matrix,
                v0=v0, s0=s0, dt=dt,
                initial_regime=initial_regime,
                seed=split_seeds[split],
                dtype=dtype,
            )
        else:
            arrays = simulate_heston_qe(
                n_paths=size,
                n_steps=n_steps,
                params=params,
                seed=split_seeds[split],
                dtype=dtype,
            )
        generated[split] = arrays
        _save_npz(output / f"{split}.npz", arrays)
        if save_transitions:
            transitions = build_transition_arrays(
                arrays["v_paths"],
                arrays["log_returns"],
                actions=arrays.get("actions"),
                dtype=dtype,
            )
            _save_npz(output / f"{split}_transitions.npz", transitions)

    stats = _stats_from_train(generated["train"], eps=1e-8)
    metadata: dict[str, Any] = {
        "n_steps": n_steps,
        "split_sizes": split_sizes,
        "seed": seed,
        "save_transitions": save_transitions,
        "transition_alignment": {
            "condition": "(v_t, r_{t-1}, a_t) with r_{-1}=0",
            "target": "(v_{t+1}, r_t)",
        },
        "normalization": stats,
        "v0": float(v0 if use_regimes else params.v0),
        "s0": float(s0 if use_regimes else params.s0),
        "dt": float(dt if use_regimes else params.dt),
    }
    if use_regimes:
        metadata["regime_switching"] = True
        metadata["regimes"] = [asdict(r) for r in regimes]
        metadata["transition_matrix"] = transition_matrix.tolist()
        metadata["initial_regime"] = int(initial_regime)
        metadata["num_actions"] = len(regimes)
    else:
        metadata["regime_switching"] = False
        metadata["params"] = asdict(params)
        metadata["num_actions"] = 1
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
