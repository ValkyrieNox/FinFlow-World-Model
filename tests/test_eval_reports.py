import numpy as np

from finflow.data import HestonParams, simulate_heston_qe
from finflow.eval import build_full_report
from scripts.evaluate_rollout import _metadata_dt, _normal_params_from_metadata


def test_build_full_report_runs_with_pricing():
    real = simulate_heston_qe(n_paths=256, n_steps=126, seed=0)
    fake = simulate_heston_qe(n_paths=256, n_steps=126, seed=1)

    params = HestonParams()
    report = build_full_report(
        real_returns=real["log_returns"],
        fake_returns=fake["log_returns"],
        real_s_paths=real["s_paths"],
        fake_s_paths=fake["s_paths"],
        params=params,
        moneynesses=(0.95, 1.0, 1.05),
        maturities=(0.25, 0.5),
        dt=1.0 / 252.0,
        pricing_r=params.mu,
    )
    assert "real_facts" in report and "fake_facts" in report
    assert "pricing_fake_vs_carr_madan" in report
    assert "pricing_real_vs_carr_madan" in report
    assert np.isfinite(report["distances"]["total_return_wasserstein"])
    assert "signature_wasserstein" in report["distances"]
    assert report["distances"]["signature_wasserstein"]["depth"] == 3


def test_build_full_report_no_pricing_when_params_missing():
    real = simulate_heston_qe(n_paths=128, n_steps=64, seed=2)
    fake = simulate_heston_qe(n_paths=128, n_steps=64, seed=3)
    report = build_full_report(
        real_returns=real["log_returns"],
        fake_returns=fake["log_returns"],
    )
    assert "pricing_fake_vs_carr_madan" not in report
    assert "distances" in report


def test_build_full_report_with_mc_oracle_pricing():
    real = simulate_heston_qe(n_paths=128, n_steps=64, seed=4)
    fake = simulate_heston_qe(n_paths=128, n_steps=64, seed=5)
    oracle = simulate_heston_qe(n_paths=128, n_steps=64, seed=6)
    report = build_full_report(
        real_returns=real["log_returns"],
        fake_returns=fake["log_returns"],
        real_s_paths=real["s_paths"],
        fake_s_paths=fake["s_paths"],
        oracle_s_paths=oracle["s_paths"],
        moneynesses=(0.95, 1.0),
        maturities=(0.1, 0.2),
        dt=1.0 / 252.0,
    )
    assert "pricing_fake_vs_mc_oracle" in report
    assert "pricing_real_vs_mc_oracle" in report
    assert "asian_pricing_fake_vs_mc_oracle" in report
    assert "asian_pricing_real_vs_mc_oracle" in report


def test_evaluate_rollout_metadata_dt_reads_single_regime_params():
    metadata = {
        "n_steps": 10,
        "regime_switching": False,
        "params": {
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.3,
            "rho": -0.7,
            "mu": 0.05,
            "v0": 0.04,
            "s0": 100.0,
            "dt": 0.1,
        },
    }
    assert _metadata_dt(metadata) == 0.1
    params = _normal_params_from_metadata(metadata, _metadata_dt(metadata))
    assert params.dt == 0.1
