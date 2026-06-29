from pathlib import Path

import numpy as np

from finflow.data import (
    HestonParams,
    HestonTransitionDataset,
    build_transition_arrays,
    generate_heston_dataset,
    simulate_heston_qe,
)


def test_simulate_heston_qe_shapes_and_positivity() -> None:
    arrays = simulate_heston_qe(n_paths=8, n_steps=12, seed=7)

    assert arrays["s_paths"].shape == (8, 13)
    assert arrays["v_paths"].shape == (8, 13)
    assert arrays["log_returns"].shape == (8, 12)
    assert np.all(arrays["s_paths"] > 0)
    assert np.all(arrays["v_paths"] >= 0)
    np.testing.assert_allclose(
        np.diff(np.log(arrays["s_paths"]), axis=1),
        arrays["log_returns"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_transition_alignment() -> None:
    v_paths = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    log_returns = np.array([[0.01, -0.02]], dtype=np.float32)

    transitions = build_transition_arrays(v_paths, log_returns, include_index=True)

    np.testing.assert_allclose(transitions["v_t"], np.array([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(transitions["r_t"], np.array([0.0, 0.01], dtype=np.float32))
    np.testing.assert_allclose(transitions["v_next"], np.array([0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(transitions["r_next"], np.array([0.01, -0.02], dtype=np.float32))
    np.testing.assert_array_equal(transitions["path_index"], np.array([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(transitions["step_index"], np.array([0, 1], dtype=np.int16))


def test_generate_heston_dataset_writes_expected_files(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path,
        n_train=4,
        n_val=2,
        n_test=2,
        n_steps=5,
        params=HestonParams(),
        seed=11,
        save_transitions=True,
    )

    assert metadata["split_sizes"] == {"train": 4, "val": 2, "test": 2}
    assert metadata["dt"] == HestonParams().dt
    assert metadata["s0"] == HestonParams().s0
    assert metadata["v0"] == HestonParams().v0
    assert (tmp_path / "metadata.json").exists()
    for split, n_paths in [("train", 4), ("val", 2), ("test", 2)]:
        paths = np.load(tmp_path / f"{split}.npz")
        transitions = np.load(tmp_path / f"{split}_transitions.npz")
        assert paths["v_paths"].shape == (n_paths, 6)
        assert paths["log_returns"].shape == (n_paths, 5)
        assert transitions["v_t"].shape == (n_paths * 5,)
        assert transitions["r_next"].shape == (n_paths * 5,)


def test_heston_transition_dataset_loader(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path,
        n_train=3,
        n_val=1,
        n_test=1,
        n_steps=4,
        seed=22,
        save_transitions=True,
    )
    norm = metadata["normalization"]
    dataset = HestonTransitionDataset(
        tmp_path / "train_transitions.npz",
        normalize=True,
        log_v_mean=norm["log_v_mean"],
        log_v_std=norm["log_v_std"],
        return_mean=norm["return_mean"],
        return_std=norm["return_std"],
    )

    item = dataset[0]
    assert len(dataset) == 12
    assert item["condition"].shape == (2,)
    assert item["target"].shape == (2,)
    assert item["condition"].dtype.is_floating_point
