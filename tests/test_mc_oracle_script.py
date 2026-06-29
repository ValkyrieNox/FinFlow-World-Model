from pathlib import Path

import numpy as np

from finflow.data import DEFAULT_REGIMES, DEFAULT_TRANSITION_MATRIX, generate_heston_dataset
from scripts.generate_mc_oracle import generate_mc_oracle


def test_generate_mc_oracle_single_regime(tmp_path: Path):
    data_dir = tmp_path / "single"
    generate_heston_dataset(
        data_dir,
        n_train=4,
        n_val=2,
        n_test=2,
        n_steps=8,
        seed=1,
        save_transitions=False,
    )
    out = tmp_path / "single_oracle.npz"
    info = generate_mc_oracle(data_dir=data_dir, output=out, n_paths=5, seed=2)
    assert info["mode"] == "single_regime"
    arr = np.load(out)
    assert arr["s_paths"].shape == (5, 9)
    assert arr["log_returns"].shape == (5, 8)
    arr.close()


def test_generate_mc_oracle_regime_switching(tmp_path: Path):
    data_dir = tmp_path / "regime"
    generate_heston_dataset(
        data_dir,
        n_train=4,
        n_val=2,
        n_test=2,
        n_steps=8,
        regimes=DEFAULT_REGIMES,
        transition_matrix=DEFAULT_TRANSITION_MATRIX,
        seed=3,
        save_transitions=False,
    )
    out = tmp_path / "regime_oracle.npz"
    info = generate_mc_oracle(data_dir=data_dir, output=out, n_paths=5, seed=4)
    assert info["mode"] == "regime_switching"
    arr = np.load(out)
    assert arr["s_paths"].shape == (5, 9)
    assert arr["actions"].shape == (5, 8)
    arr.close()
