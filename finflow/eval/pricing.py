"""Option-pricing evaluation: MC on generated paths vs Carr-Madan reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from finflow.data import HestonParams, price_heston_grid


@dataclass
class PricingComparison:
    moneynesses: np.ndarray
    maturities: np.ndarray
    strikes: np.ndarray
    mc_prices: np.ndarray        # [M, K]
    reference_prices: np.ndarray # [M, K]
    rmse_per_maturity: np.ndarray
    rmse_overall: float
    mape_overall: float

    def to_dict(self) -> dict[str, object]:
        return {
            "moneynesses": self.moneynesses.tolist(),
            "maturities": self.maturities.tolist(),
            "strikes": self.strikes.tolist(),
            "mc_prices": self.mc_prices.tolist(),
            "reference_prices": self.reference_prices.tolist(),
            "rmse_per_maturity": self.rmse_per_maturity.tolist(),
            "rmse_overall": float(self.rmse_overall),
            "mape_overall": float(self.mape_overall),
        }


def mc_call_prices_grid(
    s_paths: np.ndarray,
    *,
    dt: float,
    moneynesses: Sequence[float],
    maturities: Sequence[float],
    s0: float | None = None,
    r: float = 0.0,
) -> dict[str, np.ndarray]:
    """Monte Carlo European-call prices on the ``(K, T)`` grid.

    ``s_paths`` has shape ``[n_paths, n_steps + 1]``; index 0 is the initial
    price. ``dt`` is the year-fraction per step.

    ``s0`` defaults to the first column of ``s_paths``; ``r`` is the discount
    rate (set to your simulation drift to match the P measure on which the
    paths were simulated).
    """

    s_paths = np.asarray(s_paths, dtype=np.float64)
    if s_paths.ndim != 2:
        raise ValueError("s_paths must have shape [n_paths, n_steps + 1]")
    if dt <= 0:
        raise ValueError("dt must be positive")
    moneynesses_arr = np.asarray(moneynesses, dtype=np.float64)
    maturities_arr = np.asarray(maturities, dtype=np.float64)
    if maturities_arr.ndim != 1 or moneynesses_arr.ndim != 1:
        raise ValueError("moneynesses and maturities must be 1D")

    s0_resolved = float(s0) if s0 is not None else float(s_paths[0, 0])
    strikes = s0_resolved * moneynesses_arr

    indices = np.round(maturities_arr / dt).astype(int)
    max_index = s_paths.shape[1] - 1
    if (indices < 1).any() or (indices > max_index).any():
        raise ValueError(
            f"maturities {maturities_arr.tolist()} (in years) translate to step "
            f"indices {indices.tolist()}, outside available range [1, {max_index}]"
        )

    prices = np.zeros((maturities_arr.size, moneynesses_arr.size), dtype=np.float64)
    stderr = np.zeros_like(prices)
    n_paths = s_paths.shape[0]
    for i, idx in enumerate(indices):
        s_t = s_paths[:, idx]
        payoff = np.maximum(s_t[:, None] - strikes[None, :], 0.0)
        discount = float(np.exp(-r * maturities_arr[i]))
        prices[i] = discount * payoff.mean(axis=0)
        stderr[i] = discount * payoff.std(axis=0, ddof=0) / np.sqrt(n_paths)
    return {
        "strikes": strikes,
        "maturities": maturities_arr,
        "moneynesses": moneynesses_arr,
        "prices": prices,
        "stderr": stderr,
        "s0": s0_resolved,
    }


def mc_arithmetic_asian_call_prices_grid(
    s_paths: np.ndarray,
    *,
    dt: float,
    moneynesses: Sequence[float],
    maturities: Sequence[float],
    s0: float | None = None,
    r: float = 0.0,
) -> dict[str, np.ndarray]:
    """Monte Carlo arithmetic-Asian call prices on the ``(K, T)`` grid.

    For each maturity, the payoff is ``max(mean(S_1, ..., S_T) - K, 0)``.
    This intentionally uses the full path up to the maturity, unlike the
    European-call evaluator which only uses the terminal value ``S_T``.
    """

    s_paths = np.asarray(s_paths, dtype=np.float64)
    if s_paths.ndim != 2:
        raise ValueError("s_paths must have shape [n_paths, n_steps + 1]")
    if dt <= 0:
        raise ValueError("dt must be positive")
    moneynesses_arr = np.asarray(moneynesses, dtype=np.float64)
    maturities_arr = np.asarray(maturities, dtype=np.float64)
    if maturities_arr.ndim != 1 or moneynesses_arr.ndim != 1:
        raise ValueError("moneynesses and maturities must be 1D")

    s0_resolved = float(s0) if s0 is not None else float(s_paths[0, 0])
    strikes = s0_resolved * moneynesses_arr

    indices = np.round(maturities_arr / dt).astype(int)
    max_index = s_paths.shape[1] - 1
    if (indices < 1).any() or (indices > max_index).any():
        raise ValueError(
            f"maturities {maturities_arr.tolist()} (in years) translate to step "
            f"indices {indices.tolist()}, outside available range [1, {max_index}]"
        )

    prices = np.zeros((maturities_arr.size, moneynesses_arr.size), dtype=np.float64)
    stderr = np.zeros_like(prices)
    n_paths = s_paths.shape[0]
    for i, idx in enumerate(indices):
        average_s = s_paths[:, 1 : idx + 1].mean(axis=1)
        payoff = np.maximum(average_s[:, None] - strikes[None, :], 0.0)
        discount = float(np.exp(-r * maturities_arr[i]))
        prices[i] = discount * payoff.mean(axis=0)
        stderr[i] = discount * payoff.std(axis=0, ddof=0) / np.sqrt(n_paths)
    return {
        "strikes": strikes,
        "maturities": maturities_arr,
        "moneynesses": moneynesses_arr,
        "prices": prices,
        "stderr": stderr,
        "s0": s0_resolved,
    }


def pricing_rmse_vs_reference(
    mc_prices: np.ndarray,
    reference_prices: np.ndarray,
    moneynesses: np.ndarray,
    maturities: np.ndarray,
    strikes: np.ndarray,
) -> PricingComparison:
    """Compute RMSE / MAPE between MC prices and a reference price grid."""

    mc_prices = np.asarray(mc_prices, dtype=np.float64)
    reference_prices = np.asarray(reference_prices, dtype=np.float64)
    if mc_prices.shape != reference_prices.shape:
        raise ValueError(
            f"shape mismatch: mc {mc_prices.shape} vs reference {reference_prices.shape}"
        )

    diff = mc_prices - reference_prices
    rmse_per_maturity = np.sqrt((diff ** 2).mean(axis=1))
    rmse_overall = float(np.sqrt((diff ** 2).mean()))
    denom = np.maximum(np.abs(reference_prices), 1e-8)
    mape_overall = float(np.mean(np.abs(diff) / denom))
    return PricingComparison(
        moneynesses=np.asarray(moneynesses, dtype=np.float64),
        maturities=np.asarray(maturities, dtype=np.float64),
        strikes=np.asarray(strikes, dtype=np.float64),
        mc_prices=mc_prices,
        reference_prices=reference_prices,
        rmse_per_maturity=rmse_per_maturity,
        rmse_overall=rmse_overall,
        mape_overall=mape_overall,
    )


def pricing_rmse_vs_carr_madan(
    s_paths: np.ndarray,
    *,
    dt: float,
    moneynesses: Sequence[float],
    maturities: Sequence[float],
    params: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
    alpha: float = 1.5,
    n_fft: int = 4096,
    eta: float = 0.25,
    s0: float | None = None,
) -> PricingComparison:
    """End-to-end pricing comparison: MC on generated paths vs Carr-Madan grid."""

    mc = mc_call_prices_grid(
        s_paths, dt=dt, moneynesses=moneynesses, maturities=maturities,
        s0=s0, r=r,
    )
    grid = price_heston_grid(
        params=params,
        moneynesses=moneynesses, maturities=maturities,
        r=r, q=q, alpha=alpha, n_fft=n_fft, eta=eta,
    )
    return pricing_rmse_vs_reference(
        mc_prices=mc["prices"],
        reference_prices=grid.prices,
        moneynesses=mc["moneynesses"],
        maturities=mc["maturities"],
        strikes=mc["strikes"],
    )


def pricing_rmse_vs_mc_oracle(
    s_paths: np.ndarray,
    oracle_s_paths: np.ndarray,
    *,
    dt: float,
    moneynesses: Sequence[float],
    maturities: Sequence[float],
    s0: float | None = None,
    r: float = 0.0,
) -> PricingComparison:
    """Compare generated-path MC prices against an independent MC oracle grid."""

    s_paths = np.asarray(s_paths, dtype=np.float64)
    oracle_s_paths = np.asarray(oracle_s_paths, dtype=np.float64)
    if s_paths.ndim != 2 or oracle_s_paths.ndim != 2:
        raise ValueError("s_paths and oracle_s_paths must have shape [n_paths, n_steps + 1]")
    s0_resolved = float(s0) if s0 is not None else float(s_paths[0, 0])
    mc = mc_call_prices_grid(
        s_paths,
        dt=dt,
        moneynesses=moneynesses,
        maturities=maturities,
        s0=s0_resolved,
        r=r,
    )
    oracle = mc_call_prices_grid(
        oracle_s_paths,
        dt=dt,
        moneynesses=moneynesses,
        maturities=maturities,
        s0=s0_resolved,
        r=r,
    )
    return pricing_rmse_vs_reference(
        mc_prices=mc["prices"],
        reference_prices=oracle["prices"],
        moneynesses=mc["moneynesses"],
        maturities=mc["maturities"],
        strikes=mc["strikes"],
    )


def asian_pricing_rmse_vs_mc_oracle(
    s_paths: np.ndarray,
    oracle_s_paths: np.ndarray,
    *,
    dt: float,
    moneynesses: Sequence[float],
    maturities: Sequence[float],
    s0: float | None = None,
    r: float = 0.0,
) -> PricingComparison:
    """Compare arithmetic-Asian call MC prices against an MC oracle grid."""

    s_paths = np.asarray(s_paths, dtype=np.float64)
    oracle_s_paths = np.asarray(oracle_s_paths, dtype=np.float64)
    if s_paths.ndim != 2 or oracle_s_paths.ndim != 2:
        raise ValueError("s_paths and oracle_s_paths must have shape [n_paths, n_steps + 1]")
    s0_resolved = float(s0) if s0 is not None else float(s_paths[0, 0])
    mc = mc_arithmetic_asian_call_prices_grid(
        s_paths,
        dt=dt,
        moneynesses=moneynesses,
        maturities=maturities,
        s0=s0_resolved,
        r=r,
    )
    oracle = mc_arithmetic_asian_call_prices_grid(
        oracle_s_paths,
        dt=dt,
        moneynesses=moneynesses,
        maturities=maturities,
        s0=s0_resolved,
        r=r,
    )
    return pricing_rmse_vs_reference(
        mc_prices=mc["prices"],
        reference_prices=oracle["prices"],
        moneynesses=mc["moneynesses"],
        maturities=mc["maturities"],
        strikes=mc["strikes"],
    )
