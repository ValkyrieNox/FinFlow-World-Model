import numpy as np

from finflow.data import (
    HestonParams,
    black_scholes_call,
    carr_madan_call_prices,
    price_heston_grid,
)


def test_atm_call_matches_bs_in_low_vol_of_vol_limit():
    # When xi -> 0 and v0 = theta, Heston degenerates to BS with sigma = sqrt(theta).
    params = HestonParams(kappa=2.0, theta=0.04, xi=1e-3, rho=0.0, v0=0.04, s0=100.0)
    heston = carr_madan_call_prices(
        log_strikes=np.log(np.array([100.0])), T=1.0, params=params, r=0.0,
    )
    bs = black_scholes_call(s0=100.0, strikes=np.array([100.0]), T=1.0, sigma=0.2, r=0.0)
    assert abs(heston[0] - bs[0]) < 1e-2


def test_atm_call_full_heston_close_to_bs():
    # With v0 = theta and standard stoch-vol params, the ATM price is within
    # 10% of BS (negative rho => slightly lower ATM call value).
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, s0=100.0)
    heston = carr_madan_call_prices(
        log_strikes=np.log(np.array([100.0])), T=1.0, params=params, r=0.0,
    )
    bs = black_scholes_call(s0=100.0, strikes=np.array([100.0]), T=1.0, sigma=0.2, r=0.0)
    assert abs(heston[0] - bs[0]) < 0.5


def test_call_price_monotone_decreasing_in_strike():
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, s0=100.0)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = carr_madan_call_prices(np.log(strikes), T=1.0, params=params, r=0.0)
    # Strictly decreasing in K for calls
    diffs = np.diff(prices)
    assert np.all(diffs <= 0.0)


def test_call_price_above_intrinsic_value():
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, s0=100.0)
    strikes = np.array([80.0, 100.0, 120.0])
    prices = carr_madan_call_prices(np.log(strikes), T=0.5, params=params, r=0.0)
    intrinsic = np.maximum(params.s0 - strikes, 0.0)
    upper = np.full_like(prices, params.s0)
    assert np.all(prices >= intrinsic - 1e-3)
    assert np.all(prices <= upper + 1e-3)


def test_price_heston_grid_shapes_and_metadata():
    params = HestonParams()
    grid = price_heston_grid(
        params=params,
        moneynesses=(0.85, 0.95, 1.05),
        maturities=(0.25, 1.0),
    )
    assert grid.prices.shape == (2, 3)
    assert grid.strikes.shape == (3,)
    assert np.all(grid.prices >= 0.0)
    # Longer maturity => higher call value (forward call at r=0)
    assert np.all(grid.prices[1] >= grid.prices[0] - 1e-3)
