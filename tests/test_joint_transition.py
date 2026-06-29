import json
from pathlib import Path

import numpy as np
import torch

from finflow.data import (
    DEFAULT_REGIMES,
    DEFAULT_TRANSITION_MATRIX,
    HestonJointTransitionDataset,
    generate_heston_dataset,
)
from finflow.inference import joint_autoregressive_rollout
from finflow.training import (
    TransitionFMTrainConfig,
    TwoStageFMModelConfig,
    train_joint_trans_fm,
)
from scripts.rollout_calibration import calibrate_return_paths


class ConstantJointSampler:
    kind = "dummy"

    def __init__(self, num_actions: int, include_prev_return: bool = True) -> None:
        self.state_dim = 2
        self.condition_dim = (2 if include_prev_return else 1) + num_actions
        self.num_actions = num_actions
        self.device = torch.device("cpu")

    def sample(
        self,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        cfg_w: float = 0.0,
    ) -> torch.Tensor:
        return torch.zeros(condition.shape[0], 2, device=condition.device, dtype=condition.dtype)


def test_joint_dataset_shape_action_layout_and_cache(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path, n_train=4, n_val=1, n_test=1, n_steps=6,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=21, save_transitions=True,
    )
    num_actions = metadata["num_actions"]
    ds = HestonJointTransitionDataset(
        tmp_path / "train_transitions.npz",
        normalize=True,
        log_v_mean=metadata["normalization"]["log_v_mean"],
        log_v_std=metadata["normalization"]["log_v_std"],
        return_mean=metadata["normalization"]["return_mean"],
        return_std=metadata["normalization"]["return_std"],
        num_actions=num_actions,
    )

    assert len(ds) == 4 * 6
    item = ds[0]
    assert item["condition"].shape == (2 + num_actions,)
    assert item["target"].shape == (2,)
    assert torch.allclose(item["condition"][:2], torch.stack([item["log_v_t"], item["r_t"]]))
    action_slice = item["condition"][2:]
    assert torch.isclose(action_slice.sum(), torch.tensor(1.0))
    assert int(action_slice.argmax()) == int(item["action"])

    cached = ds.as_condition_target_tensors()
    assert cached["condition"].shape == (len(ds), 2 + num_actions)
    assert cached["target"].shape == (len(ds), 2)
    assert cached["action_start"] == 2

    markov_ds = HestonJointTransitionDataset(
        tmp_path / "train_transitions.npz",
        normalize=True,
        log_v_mean=metadata["normalization"]["log_v_mean"],
        log_v_std=metadata["normalization"]["log_v_std"],
        return_mean=metadata["normalization"]["return_mean"],
        return_std=metadata["normalization"]["return_std"],
        num_actions=num_actions,
        include_prev_return=False,
    )
    markov_item = markov_ds[0]
    assert markov_item["condition"].shape == (1 + num_actions,)
    assert torch.isclose(markov_item["condition"][0], markov_item["log_v_t"])
    assert int(markov_item["condition"][1:].argmax()) == int(markov_item["action"])
    markov_cached = markov_ds.as_condition_target_tensors()
    assert markov_cached["condition"].shape == (len(markov_ds), 1 + num_actions)
    assert markov_cached["target"].shape == (len(markov_ds), 2)
    assert markov_cached["action_start"] == 1


def test_joint_rollout_returns_compatible_paths() -> None:
    normalization = {
        "log_v_mean": 0.0,
        "log_v_std": 1.0,
        "return_mean": 0.0,
        "return_std": 1.0,
    }
    result = joint_autoregressive_rollout(
        ConstantJointSampler(num_actions=3),
        normalization=normalization,
        n_paths=5,
        n_steps=7,
        num_actions=3,
        initial_v=1.0,
        initial_s=100.0,
        initial_r_prev=0.0,
        actions=np.zeros((5, 7), dtype=np.int8),
        noise_seed=3,
    )

    assert result.r_paths.shape == (5, 7)
    assert result.s_paths.shape == (5, 8)
    assert result.v_paths.shape == (5, 8)
    assert np.allclose(result.r_paths, 0.0)
    assert np.allclose(result.s_paths, 100.0)
    assert np.all(result.v_paths > 0.0)


def test_markov_minimal_joint_rollout_returns_compatible_paths() -> None:
    normalization = {
        "log_v_mean": 0.0,
        "log_v_std": 1.0,
        "return_mean": 0.0,
        "return_std": 1.0,
    }
    result = joint_autoregressive_rollout(
        ConstantJointSampler(num_actions=3, include_prev_return=False),
        normalization=normalization,
        n_paths=5,
        n_steps=7,
        num_actions=3,
        initial_v=1.0,
        initial_s=100.0,
        initial_r_prev=0.0,
        actions=np.zeros((5, 7), dtype=np.int8),
        noise_seed=3,
    )

    assert result.r_paths.shape == (5, 7)
    assert result.s_paths.shape == (5, 8)
    assert result.v_paths.shape == (5, 8)
    assert np.allclose(result.r_paths, 0.0)
    assert np.allclose(result.s_paths, 100.0)
    assert np.all(result.v_paths > 0.0)


def test_joint_rollout_calibration_matches_target_return_moments() -> None:
    raw_returns = np.asarray(
        [
            [0.05, 0.07, 0.04],
            [0.02, 0.03, 0.06],
        ],
        dtype=np.float32,
    )
    return_mean = 0.01
    return_std = 0.2

    r_cal, s_paths, info = calibrate_return_paths(
        raw_returns,
        initial_s=100.0,
        return_mean=return_mean,
        return_std=return_std,
        eps=1e-6,
    )

    standardized = (r_cal - return_mean) / return_std
    assert abs(float(standardized.mean())) < 1e-6
    assert abs(float(standardized.std(ddof=0)) - 1.0) < 1e-6
    assert s_paths.shape == (2, 4)
    assert np.allclose(s_paths[:, 0], 100.0)
    assert np.allclose(s_paths[:, 1:], 100.0 * np.exp(np.cumsum(r_cal, axis=1)))
    assert info["return_mean"] == return_mean
    assert info["return_std"] == return_std


def test_train_joint_trans_fm_smoke(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path / "data", n_train=3, n_val=1, n_test=1, n_steps=4,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=22, save_transitions=True,
    )
    num_actions = metadata["num_actions"]
    summary = train_joint_trans_fm(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "runs",
        run_name="joint_smoke",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=2,
            condition_dim=2 + num_actions,
            hidden_dim=8,
            time_embedding_dim=4,
            num_blocks=1,
        ),
        train_config=TransitionFMTrainConfig(
            batch_size=4,
            epochs=1,
            max_train_batches=1,
            max_val_batches=1,
            cache_data_device=True,
            progress=False,
            ema_decay=0.5,
            target_loss_weights=(1.0, 2.0),
            seed=23,
        ),
    )

    assert summary["stage"] == "joint"
    assert Path(summary["checkpoints"]["last"]).exists()
    assert Path(summary["checkpoints"]["ema_last"]).exists()
    assert Path(summary["checkpoints"]["ema_best"]).exists()
    config = json.loads((Path(summary["run_dir"]) / "config.json").read_text())
    assert config["train_config"]["target_loss_weights"] == [1.0, 2.0]


def test_train_markov_minimal_joint_trans_fm_smoke(tmp_path: Path) -> None:
    metadata = generate_heston_dataset(
        tmp_path / "data", n_train=3, n_val=1, n_test=1, n_steps=4,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=24, save_transitions=True,
    )
    num_actions = metadata["num_actions"]
    summary = train_joint_trans_fm(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "runs",
        run_name="joint_markov_smoke",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=2,
            condition_dim=1 + num_actions,
            hidden_dim=8,
            time_embedding_dim=4,
            num_blocks=1,
        ),
        train_config=TransitionFMTrainConfig(
            batch_size=4,
            epochs=1,
            max_train_batches=1,
            max_val_batches=1,
            cache_data_device=True,
            progress=False,
            seed=25,
        ),
        include_prev_return=False,
    )

    assert summary["stage"] == "joint"
    assert Path(summary["checkpoints"]["last"]).exists()
    config = json.loads((Path(summary["run_dir"]) / "config.json").read_text())
    assert config["model_config"]["condition_dim"] == 1 + num_actions
    assert config["transition_type"] == "joint_vr_markov_minimal"
    assert config["include_prev_return"] is False
