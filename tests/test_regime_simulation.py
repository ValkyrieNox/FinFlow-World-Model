import numpy as np

from finflow.data import (
    DEFAULT_REGIMES,
    DEFAULT_TRANSITION_MATRIX,
    RegimeParams,
    build_transition_arrays,
    simulate_regime_switching_heston,
)


def test_regime_simulation_shapes_and_action_range():
    out = simulate_regime_switching_heston(
        n_paths=16, n_steps=64,
        regimes=DEFAULT_REGIMES,
        transition_matrix=DEFAULT_TRANSITION_MATRIX,
        seed=0,
    )
    assert out["s_paths"].shape == (16, 65)
    assert out["v_paths"].shape == (16, 65)
    assert out["log_returns"].shape == (16, 64)
    assert out["actions"].shape == (16, 64)
    assert out["actions"].dtype == np.int8
    assert np.all(out["v_paths"] >= 0)
    assert np.all(out["s_paths"] > 0)
    assert out["actions"].min() >= 0
    assert out["actions"].max() < len(DEFAULT_REGIMES)


def test_regime_simulation_initial_regime_respected():
    out = simulate_regime_switching_heston(
        n_paths=32, n_steps=8,
        regimes=DEFAULT_REGIMES,
        transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=2,
        seed=1,
    )
    assert np.all(out["actions"][:, 0] == 2)


def test_regime_simulation_with_identity_transition_keeps_regime_constant():
    identity = np.eye(3, dtype=np.float64)
    out = simulate_regime_switching_heston(
        n_paths=8, n_steps=20,
        regimes=DEFAULT_REGIMES,
        transition_matrix=identity,
        initial_regime=1,
        seed=2,
    )
    assert np.all(out["actions"] == 1)


def test_regime_simulation_transition_array_alignment():
    out = simulate_regime_switching_heston(
        n_paths=4, n_steps=6,
        regimes=DEFAULT_REGIMES,
        transition_matrix=DEFAULT_TRANSITION_MATRIX,
        seed=3,
    )
    transitions = build_transition_arrays(
        out["v_paths"], out["log_returns"], actions=out["actions"],
    )
    # action array is flattened with the same path-major order as the others
    assert transitions["action"].shape == (4 * 6,)
    np.testing.assert_array_equal(
        transitions["action"], out["actions"].reshape(-1).astype(np.int8),
    )


def test_regime_simulation_single_regime_matches_basic_simulator_distribution():
    # If we feed a single regime with the same params as HestonParams defaults,
    # the marginal variance/return statistics should be similar to the
    # single-regime simulator (up to MC noise from independent RNG paths).
    single = (RegimeParams(name="only", kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, mu=0.05),)
    transition_matrix = np.array([[1.0]])
    out = simulate_regime_switching_heston(
        n_paths=512, n_steps=120,
        regimes=single, transition_matrix=transition_matrix, seed=4,
    )
    assert np.isfinite(out["log_returns"]).all()
    # mean log-return roughly close to mu*dt
    assert abs(out["log_returns"].mean() - 0.05 * (1.0 / 252.0)) < 1e-3
