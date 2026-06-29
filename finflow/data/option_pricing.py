"""Carr-Madan FFT pricer for Heston European call options.

Implements the Albrecher / "little Heston trap" characteristic function
formulation (numerically stable for long maturities) and the Carr-Madan
damping-factor FFT for European call prices on a log-strike grid; the result
is interpolated to user-specified strikes / moneynesses.

Reference:
- Carr & Madan, "Option Valuation Using the Fast Fourier Transform", JCF 1999.
- Albrecher et al., "The Little Heston Trap", Wilmott 2007.
- Heston, "A Closed-Form Solution for Options with Stochastic Volatility", RFS 1993.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from finflow.data.heston import HestonParams


def heston_characteristic_function(
    u: np.ndarray,
    T: float,
    params: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Heston characteristic function under the risk-neutral measure.

    Returns ``E^Q[exp(i u log S_T)]`` evaluated at complex argument ``u``.
    Uses the Albrecher et al. (2007) ``g2`` formulation to avoid branch cuts.
    """

    u = np.asarray(u, dtype=np.complex128)
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho
    v0 = params.v0
    s0 = params.s0

    iu = 1j * u
    a = rho * xi * iu - kappa
    d = np.sqrt(a * a + (xi * xi) * (iu + u * u))
    numer = kappa - rho * xi * iu - d
    denom = kappa - rho * xi * iu + d
    g2 = numer / denom

    exp_dT = np.exp(-d * T)
    log_arg = (1.0 - g2 * exp_dT) / (1.0 - g2)

    A_term = (kappa * theta / (xi * xi)) * (numer * T - 2.0 * np.log(log_arg))
    B_term = (numer / (xi * xi)) * (1.0 - exp_dT) / (1.0 - g2 * exp_dT)

    return np.exp(iu * (np.log(s0) + (r - q) * T) + A_term + B_term * v0)


def _simpson_weights(eta: float, n: int) -> np.ndarray:
    """Composite Simpson 1/3 weights for the Carr-Madan FFT grid.

    Pattern is ``[1, 4, 2, 4, 2, ..., 4, 1] * (eta / 3)``; for even ``n`` the
    right endpoint is approximated as ``4/3`` (one-point error, negligible
    at ``n >= 1024``).
    """

    j = np.arange(n)
    base = 3.0 + (-1.0) ** (j + 1.0) - (j == 0).astype(np.float64)
    weights = (eta / 3.0) * base
    return weights


def carr_madan_call_prices(
    log_strikes: np.ndarray,
    T: float,
    params: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
    alpha: float = 1.5,
    n_fft: int = 4096,
    eta: float = 0.25,
) -> np.ndarray:
    """Return European call prices via the Carr-Madan FFT.

    ``log_strikes`` is a 1D array of log-strikes ``k = log(K)``. The FFT
    produces prices on a uniform log-strike grid centered at 0; we linearly
    interpolate to the requested strikes.

    ``alpha`` is the Carr-Madan damping factor; ``alpha = 1.5`` is standard.
    """

    log_strikes = np.asarray(log_strikes, dtype=np.float64)
    if log_strikes.ndim != 1:
        raise ValueError("log_strikes must be 1D")
    if T <= 0:
        raise ValueError("T must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive (Carr-Madan damping)")
    if n_fft <= 0 or (n_fft & (n_fft - 1)) != 0:
        raise ValueError("n_fft must be a positive power of 2")
    if eta <= 0:
        raise ValueError("eta must be positive")

    lambd = 2.0 * np.pi / (n_fft * eta)
    b = n_fft * lambd / 2.0

    j = np.arange(n_fft, dtype=np.float64)
    nu = j * eta
    k_grid = -b + j * lambd

    u = nu - (alpha + 1.0) * 1j
    phi = heston_characteristic_function(u, T, params, r=r, q=q)
    denom = (alpha * alpha + alpha - nu * nu) + 1j * (2.0 * alpha + 1.0) * nu
    psi = np.exp(-r * T) * phi / denom

    weights = _simpson_weights(eta, n_fft)
    y = np.exp(1j * b * nu) * psi * weights
    fft_out = np.fft.fft(y)
    call_grid = (np.exp(-alpha * k_grid) / np.pi) * fft_out.real

    return np.interp(log_strikes, k_grid, call_grid)


@dataclass(frozen=True)
class HestonOptionGrid:
    """Result of pricing the (moneyness, maturity) grid."""

    moneynesses: np.ndarray  # shape [n_k]
    maturities: np.ndarray   # shape [n_t]
    strikes: np.ndarray      # shape [n_k]
    prices: np.ndarray       # shape [n_t, n_k]
    r: float
    q: float
    alpha: float

    def as_dict(self) -> dict[str, list]:
        return {
            "moneynesses": self.moneynesses.tolist(),
            "maturities": self.maturities.tolist(),
            "strikes": self.strikes.tolist(),
            "prices": self.prices.tolist(),
            "r": self.r,
            "q": self.q,
            "alpha": self.alpha,
        }


def price_heston_grid(
    params: HestonParams,
    moneynesses: Sequence[float] = (0.85, 0.90, 0.95, 1.00, 1.05),
    maturities: Sequence[float] = (0.25, 0.5, 1.0),
    r: float = 0.0,
    q: float = 0.0,
    alpha: float = 1.5,
    n_fft: int = 4096,
    eta: float = 0.25,
) -> HestonOptionGrid:
    """Price the V3 default 15-point ``(K, T)`` grid via Carr-Madan FFT."""

    moneynesses = np.asarray(moneynesses, dtype=np.float64)
    maturities = np.asarray(maturities, dtype=np.float64)
    if moneynesses.ndim != 1 or maturities.ndim != 1:
        raise ValueError("moneynesses and maturities must be 1D")

    strikes = params.s0 * moneynesses
    log_strikes = np.log(strikes)
    prices = np.empty((len(maturities), len(moneynesses)), dtype=np.float64)
    for i, T in enumerate(maturities):
        prices[i] = carr_madan_call_prices(
            log_strikes, float(T), params,
            r=r, q=q, alpha=alpha, n_fft=n_fft, eta=eta,
        )
    return HestonOptionGrid(
        moneynesses=moneynesses,
        maturities=maturities,
        strikes=strikes,
        prices=prices,
        r=r, q=q, alpha=alpha,
    )


def black_scholes_call(
    s0: float,
    strikes: np.ndarray,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Plain-vanilla Black-Scholes call (used for unit tests against the FFT in the
    zero-vol-of-vol limit, where Heston degenerates to BS with constant volatility
    ``sqrt(theta)``)."""

    from math import erf, log, sqrt

    strikes = np.asarray(strikes, dtype=np.float64)
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    sqrtT = sqrt(T)
    out = np.empty_like(strikes)
    for i, K in enumerate(strikes):
        d1 = (log(s0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        n_d1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
        n_d2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)))
        out[i] = s0 * np.exp(-q * T) * n_d1 - K * np.exp(-r * T) * n_d2
    return out
