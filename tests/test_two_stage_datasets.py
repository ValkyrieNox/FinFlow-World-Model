from pathlib import Path

import torch

from finflow.data import (
    DEFAULT_REGIMES,
    DEFAULT_TRANSITION_MATRIX,
    HestonRetTransitionDataset,
    HestonVolTransitionDataset,
    generate_heston_dataset,
)


def test_vol_dataset_shape_and_action_onehot(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path, n_train=4, n_val=1, n_test=1, n_steps=6,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=11, save_transitions=True,
    )
    num_actions = metadata["num_actions"]
    ds = HestonVolTransitionDataset(
        tmp_path / "train_transitions.npz",
        normalize=True,
        log_v_mean=metadata["normalization"]["log_v_mean"],
        log_v_std=metadata["normalization"]["log_v_std"],
        num_actions=num_actions,
    )
    assert len(ds) == 4 * 6
    item = ds[0]
    assert item["condition"].shape == (1 + num_actions,)
    assert item["target"].shape == (1,)
    # One-hot encoded action: exactly one 1.0 in the action slice
    action_slice = item["condition"][1:]
    assert torch.isclose(action_slice.sum(), torch.tensor(1.0))
    # The hot index should match the stored action
    assert int(action_slice.argmax()) == int(item["action"])


def test_ret_dataset_shape_and_condition_layout(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path, n_train=3, n_val=1, n_test=1, n_steps=5,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=12, save_transitions=True,
    )
    num_actions = metadata["num_actions"]
    ds = HestonRetTransitionDataset(
        tmp_path / "train_transitions.npz",
        normalize=True,
        log_v_mean=metadata["normalization"]["log_v_mean"],
        log_v_std=metadata["normalization"]["log_v_std"],
        return_mean=metadata["normalization"]["return_mean"],
        return_std=metadata["normalization"]["return_std"],
        num_actions=num_actions,
    )
    assert len(ds) == 3 * 5
    item = ds[0]
    assert item["condition"].shape == (3 + num_actions,)
    assert item["target"].shape == (1,)
    # condition layout: [log_v_next, log_v_t, r_t, action_onehot]
    expected = torch.stack([item["log_v_next"], item["log_v_t"], item["r_t"]])
    assert torch.allclose(item["condition"][:3], expected)


def test_vol_dataset_without_actions_uses_single_dim(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path, n_train=3, n_val=1, n_test=1, n_steps=4,
        seed=13, save_transitions=True,
    )
    # No regimes => single-regime data, action array is absent (defaults to 0)
    ds = HestonVolTransitionDataset(
        tmp_path / "train_transitions.npz",
        normalize=True,
        log_v_mean=metadata["normalization"]["log_v_mean"],
        log_v_std=metadata["normalization"]["log_v_std"],
        num_actions=1,
    )
    item = ds[0]
    assert item["condition"].shape == (2,)
    assert int(item["action"]) == 0
    assert torch.isclose(item["condition"][1], torch.tensor(1.0))


def test_action_dropout_zeroes_action_slice(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path, n_train=3, n_val=1, n_test=1, n_steps=4,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=14, save_transitions=True,
    )
    num_actions = metadata["num_actions"]
    vol_ds = HestonVolTransitionDataset(
        tmp_path / "train_transitions.npz",
        normalize=True,
        log_v_mean=metadata["normalization"]["log_v_mean"],
        log_v_std=metadata["normalization"]["log_v_std"],
        num_actions=num_actions,
        action_dropout_prob=1.0,
    )
    assert torch.allclose(vol_ds[0]["condition"][1:], torch.zeros(num_actions))

    ret_ds = HestonRetTransitionDataset(
        tmp_path / "train_transitions.npz",
        normalize=True,
        log_v_mean=metadata["normalization"]["log_v_mean"],
        log_v_std=metadata["normalization"]["log_v_std"],
        return_mean=metadata["normalization"]["return_mean"],
        return_std=metadata["normalization"]["return_std"],
        num_actions=num_actions,
        action_dropout_prob=1.0,
    )
    assert torch.allclose(ret_ds[0]["condition"][3:], torch.zeros(num_actions))
