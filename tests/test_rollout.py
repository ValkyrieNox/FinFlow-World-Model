from pathlib import Path

import numpy as np
import torch

from finflow.data import (
    DEFAULT_TRANSITION_MATRIX,
    DEFAULT_REGIMES,
    generate_heston_dataset,
)
from finflow.inference import (
    MeanFlowSampler,
    FMTeacherSampler,
    autoregressive_rollout,
    sample_action_schedule,
)
from finflow.models import MeanFlowStudent, TransitionFM


def test_sample_action_schedule_single_action_zeros():
    actions = sample_action_schedule(n_paths=4, n_steps=10, num_actions=1)
    assert actions.shape == (4, 10)
    assert (actions == 0).all()


def test_sample_action_schedule_respects_initial_regime_constant():
    actions = sample_action_schedule(
        n_paths=3, n_steps=5, num_actions=3,
        initial_regime=2, constant=True,
    )
    assert (actions == 2).all()


def test_sample_action_schedule_markov_chain():
    actions = sample_action_schedule(
        n_paths=5, n_steps=20, num_actions=3,
        transition_matrix=DEFAULT_TRANSITION_MATRIX, initial_regime=0, seed=0,
    )
    assert actions.shape == (5, 20)
    assert actions.min() >= 0 and actions.max() < 3
    assert (actions[:, 0] == 0).all()


def _build_samplers(num_actions):
    torch.manual_seed(0)
    vol_model = MeanFlowStudent(
        state_dim=1, condition_dim=1 + num_actions,
        hidden_dim=16, time_embedding_dim=8, num_blocks=2,
    )
    ret_model = TransitionFM(
        state_dim=1, condition_dim=3 + num_actions,
        hidden_dim=16, time_embedding_dim=8, num_blocks=2,
    )
    return MeanFlowSampler(vol_model), FMTeacherSampler(ret_model, n_steps=4)


def test_autoregressive_rollout_smoke(tmp_path: Path):
    metadata = generate_heston_dataset(
        tmp_path, n_train=4, n_val=2, n_test=2, n_steps=6,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=4, save_transitions=True,
    )
    num_actions = metadata["num_actions"]
    vol_sampler, ret_sampler = _build_samplers(num_actions)
    result = autoregressive_rollout(
        vol_sampler=vol_sampler,
        ret_sampler=ret_sampler,
        normalization=metadata["normalization"],
        n_paths=6, n_steps=12, num_actions=num_actions,
        initial_v=0.04, initial_s=100.0,
        initial_regime=0,
        constant_action=True,
        device="cpu",
        action_seed=1, noise_seed=2,
    )
    assert result.v_paths.shape == (6, 13)
    assert result.r_paths.shape == (6, 12)
    assert result.s_paths.shape == (6, 13)
    assert result.actions.shape == (6, 12)
    assert np.all(np.isfinite(result.r_paths))
    assert np.all(result.v_paths > 0)
    assert np.all(result.s_paths > 0)


def test_autoregressive_rollout_validates_action_shape():
    vol_sampler, ret_sampler = _build_samplers(num_actions=3)
    bad_actions = np.zeros((4, 3), dtype=np.int8)
    try:
        autoregressive_rollout(
            vol_sampler=vol_sampler, ret_sampler=ret_sampler,
            normalization={"log_v_mean": 0.0, "log_v_std": 1.0,
                          "return_mean": 0.0, "return_std": 1.0},
            n_paths=4, n_steps=5, num_actions=3, initial_v=0.04,
            actions=bad_actions,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong actions shape")
