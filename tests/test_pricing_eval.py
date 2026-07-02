import numpy as np

from finflow.data import HestonParams, simulate_heston_qe
from finflow.eval import (
    asian_pricing_rmse_vs_mc_oracle,
    mc_arithmetic_asian_call_prices_grid,
    mc_call_prices_grid,
    pricing_rmse_vs_carr_madan,
    pricing_rmse_vs_mc_oracle,
    pricing_rmse_vs_reference,
)


def test_mc_call_prices_grid_shapes_and_positivity():
    arrays = simulate_heston_qe(n_paths=128, n_steps=252, seed=0)
    res = mc_call_prices_grid(
        arrays["s_paths"], dt=1.0 / 252.0,
        moneynesses=(0.9, 1.0, 1.1),
        maturities=(0.25, 1.0),
        r=0.05,
    )
    assert res["prices"].shape == (2, 3)
    assert np.all(res["prices"] >= 0.0)
    # ATM increases in maturity (with r=0.05, mostly increasing for OTM/ATM)
    assert res["prices"][1, 1] >= res["prices"][0, 1] - 0.5


def test_pricing_rmse_vs_carr_madan_finite():
    arrays = simulate_heston_qe(n_paths=2048, n_steps=252, seed=1)
    params = HestonParams()
    cmp = pricing_rmse_vs_carr_madan(
        arrays["s_paths"], dt=1.0 / 252.0,
        moneynesses=(0.95, 1.0, 1.05),
        maturities=(0.5, 1.0),
        params=params,
        r=params.mu,  # mu == r => P-measure MC ≈ Q-measure Carr-Madan
    )
    assert np.isfinite(cmp.rmse_overall)
    assert cmp.rmse_overall < 5.0


def test_pricing_rmse_vs_reference_zero_on_identical():
    mc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cmp = pricing_rmse_vs_reference(
        mc, mc.copy(),
        moneynesses=np.array([0.9, 1.0, 1.1]),
        maturities=np.array([0.25, 1.0]),
        strikes=np.array([90, 100, 110]),
    )
    assert cmp.rmse_overall == 0.0
    assert cmp.mape_overall == 0.0


def test_pricing_rmse_vs_mc_oracle_zero_on_same_paths():
    arrays = simulate_heston_qe(n_paths=128, n_steps=64, seed=4)
    cmp = pricing_rmse_vs_mc_oracle(
        arrays["s_paths"],
        arrays["s_paths"].copy(),
        dt=1.0 / 252.0,
        moneynesses=(0.95, 1.0, 1.05),
        maturities=(0.1, 0.2),
        r=0.0,
    )
    assert cmp.rmse_overall == 0.0
    assert cmp.mape_overall == 0.0


def test_mc_arithmetic_asian_call_prices_grid_shapes_and_uses_path_average():
    s_paths = np.array([
        [100.0, 100.0, 120.0],
        [100.0, 100.0, 80.0],
    ])
    res = mc_arithmetic_asian_call_prices_grid(
        s_paths,
        dt=0.5,
        moneynesses=(1.0,),
        maturities=(1.0,),
        r=0.0,
    )
    # Averages over S_1 and S_2 are 110 and 90, so the mean call payoff is 5.
    assert res["prices"].shape == (1, 1)
    assert res["prices"][0, 0] == 5.0


def test_asian_pricing_rmse_vs_mc_oracle_zero_on_same_paths():
    arrays = simulate_heston_qe(n_paths=128, n_steps=64, seed=7)
    cmp = asian_pricing_rmse_vs_mc_oracle(
        arrays["s_paths"],
        arrays["s_paths"].copy(),
        dt=1.0 / 252.0,
        moneynesses=(0.95, 1.0, 1.05),
        maturities=(0.1, 0.2),
        r=0.0,
    )
    assert cmp.rmse_overall == 0.0
    assert cmp.mape_overall == 0.0
