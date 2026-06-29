"""Autoregressive rollout for the V3 two-stage transition kernel.

Given trained vol and ret samplers (FM teacher / Mean Flow / Consistency), this
module steps the latent market state forward in normalized space and returns
denormalized paths. The alignment matches
:mod:`finflow.data.heston.build_transition_arrays`::

    condition_t = (log_v_t_norm, r_{t-1}_norm, a_t)
    target_t    = (log_v_{t+1}_norm, r_t_norm)

The action-aware joint sampler also supports the Markov-minimal ablation
``condition_t = (log_v_t_norm, a_t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from finflow.data.heston import _sample_actions
from finflow.inference.samplers import Sampler


def _onehot(actions: torch.Tensor, num_actions: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(actions.long(), num_classes=num_actions).to(actions.device).float()


@dataclass
class RolloutResult:
    """Rollout output bundle.

    All arrays have batch dimension first.
    """

    log_v_paths_norm: np.ndarray  # [N, T+1]   normalized log-variance, including initial state
    r_paths_norm: np.ndarray      # [N, T]     normalized log-returns
    log_v_paths: np.ndarray       # [N, T+1]   denormalized log-variance
    v_paths: np.ndarray           # [N, T+1]
    r_paths: np.ndarray           # [N, T]     denormalized log-returns
    s_paths: np.ndarray           # [N, T+1]   price paths
    actions: np.ndarray           # [N, T]
    initial_v: float
    initial_s: float
    num_actions: int


def sample_action_schedule(
    n_paths: int,
    n_steps: int,
    *,
    num_actions: int,
    transition_matrix: np.ndarray | None = None,
    initial_regime: int = 0,
    seed: int | None = None,
    constant: bool = False,
) -> np.ndarray:
    """Sample a per-path action sequence.

    - ``num_actions == 1``: returns all zeros.
    - ``constant == True`` or ``transition_matrix is None``: stay in
      ``initial_regime`` for the whole rollout.
    - otherwise: Markov chain with the given transition matrix.
    """

    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    if num_actions == 1:
        return np.zeros((n_paths, n_steps), dtype=np.int8)
    if constant or transition_matrix is None:
        return np.full((n_paths, n_steps), initial_regime, dtype=np.int8)
    transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
    if transition_matrix.shape != (num_actions, num_actions):
        raise ValueError(
            f"transition_matrix shape must be ({num_actions}, {num_actions})"
        )
    rng = np.random.default_rng(seed)
    return _sample_actions(n_paths, n_steps, transition_matrix, initial_regime, rng)


def autoregressive_rollout(
    vol_sampler: Sampler,
    ret_sampler: Sampler,
    *,
    normalization: dict[str, float],
    n_paths: int,
    n_steps: int,
    num_actions: int,
    initial_v: float,
    initial_s: float = 100.0,
    initial_r_prev: float = 0.0,
    actions: np.ndarray | None = None,
    transition_matrix: np.ndarray | None = None,
    initial_regime: int = 0,
    action_seed: int | None = None,
    noise_seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    constant_action: bool = False,
    cfg_w: float = 0.0,
) -> RolloutResult:
    """Roll out ``n_paths`` autoregressive trajectories of length ``n_steps``.

    The vol and ret samplers must have matching ``num_actions`` and produce
    state values in normalized space; this function denormalizes using the
    provided ``normalization`` dict (typically loaded from ``metadata.json``).
    """

    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    if vol_sampler.state_dim != 1 or ret_sampler.state_dim != 1:
        raise ValueError("vol and ret samplers must have state_dim=1")
    expected_vol_cond = 1 + num_actions
    expected_ret_cond = 3 + num_actions
    if vol_sampler.condition_dim != expected_vol_cond:
        raise ValueError(
            f"vol sampler condition_dim {vol_sampler.condition_dim} != "
            f"1 + num_actions = {expected_vol_cond}"
        )
    if ret_sampler.condition_dim != expected_ret_cond:
        raise ValueError(
            f"ret sampler condition_dim {ret_sampler.condition_dim} != "
            f"3 + num_actions = {expected_ret_cond}"
        )
    if initial_v <= 0:
        raise ValueError("initial_v must be positive")
    if cfg_w < 0.0:
        raise ValueError("cfg_w must be non-negative")

    if actions is None:
        actions = sample_action_schedule(
            n_paths=n_paths, n_steps=n_steps, num_actions=num_actions,
            transition_matrix=transition_matrix, initial_regime=initial_regime,
            seed=action_seed, constant=constant_action,
        )
    actions = np.asarray(actions, dtype=np.int8)
    if actions.shape != (n_paths, n_steps):
        raise ValueError(f"actions shape must be ({n_paths}, {n_steps}), got {actions.shape}")
    if actions.min() < 0 or actions.max() >= num_actions:
        raise ValueError(f"action indices must be in [0, {num_actions})")

    device = torch.device(device) if isinstance(device, str) else device

    log_v_mean = float(normalization["log_v_mean"])
    log_v_std = float(normalization["log_v_std"])
    return_mean = float(normalization["return_mean"])
    return_std = float(normalization["return_std"])

    log_v_paths_norm = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    r_paths_norm = np.empty((n_paths, n_steps), dtype=np.float32)
    log_v_t_norm = (np.log(initial_v) - log_v_mean) / log_v_std
    log_v_paths_norm[:, 0] = log_v_t_norm
    r_prev_norm = (initial_r_prev - return_mean) / return_std

    log_v_t = torch.full((n_paths, 1), log_v_t_norm, device=device, dtype=dtype)
    r_prev_t = torch.full((n_paths, 1), r_prev_norm, device=device, dtype=dtype)
    actions_t = torch.from_numpy(actions).to(device=device)

    rng = torch.Generator(device="cpu")
    if noise_seed is not None:
        rng.manual_seed(noise_seed)

    for step in range(n_steps):
        a_step = actions_t[:, step]
        a_onehot = _onehot(a_step, num_actions).to(dtype=dtype)

        vol_cond = torch.cat([log_v_t, a_onehot], dim=-1)
        z_vol = torch.randn(n_paths, 1, generator=rng, dtype=dtype).to(device)
        log_v_next = vol_sampler.sample(vol_cond, noise=z_vol, cfg_w=cfg_w)  # [N, 1] normalized

        ret_cond = torch.cat([log_v_next, log_v_t, r_prev_t, a_onehot], dim=-1)
        z_ret = torch.randn(n_paths, 1, generator=rng, dtype=dtype).to(device)
        r_next = ret_sampler.sample(ret_cond, noise=z_ret, cfg_w=cfg_w)       # [N, 1] normalized

        log_v_paths_norm[:, step + 1] = log_v_next.detach().cpu().numpy().reshape(-1)
        r_paths_norm[:, step] = r_next.detach().cpu().numpy().reshape(-1)

        log_v_t = log_v_next
        r_prev_t = r_next

    # Denormalize.
    log_v_paths = log_v_paths_norm * log_v_std + log_v_mean
    v_paths = np.exp(log_v_paths)
    r_paths = r_paths_norm * return_std + return_mean

    log_s = np.log(initial_s) + np.cumsum(r_paths, axis=1)
    s_paths = np.concatenate(
        [np.full((n_paths, 1), initial_s, dtype=np.float64), np.exp(log_s)], axis=1,
    )

    return RolloutResult(
        log_v_paths_norm=log_v_paths_norm,
        r_paths_norm=r_paths_norm,
        log_v_paths=log_v_paths.astype(np.float32),
        v_paths=v_paths.astype(np.float32),
        r_paths=r_paths.astype(np.float32),
        s_paths=s_paths.astype(np.float32),
        actions=actions,
        initial_v=float(initial_v),
        initial_s=float(initial_s),
        num_actions=num_actions,
    )


def joint_autoregressive_rollout(
    joint_sampler: Sampler,
    *,
    normalization: dict[str, float],
    n_paths: int,
    n_steps: int,
    num_actions: int,
    initial_v: float,
    initial_s: float = 100.0,
    initial_r_prev: float = 0.0,
    actions: np.ndarray | None = None,
    transition_matrix: np.ndarray | None = None,
    initial_regime: int = 0,
    action_seed: int | None = None,
    noise_seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    constant_action: bool = False,
    cfg_w: float = 0.0,
) -> RolloutResult:
    """Roll out paths from an action-aware joint transition sampler.

    The sampler must emit ``[log_v_next_norm, r_next_norm]`` from condition
    ``[log_v_t_norm, r_t_norm, action_onehot]`` or, for the Markov-minimal
    ablation, ``[log_v_t_norm, action_onehot]``.
    """

    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    if joint_sampler.state_dim != 2:
        raise ValueError("joint sampler must have state_dim=2")
    full_condition_dim = 2 + num_actions
    markov_minimal_condition_dim = 1 + num_actions
    if joint_sampler.condition_dim == full_condition_dim:
        include_prev_return = True
    elif joint_sampler.condition_dim == markov_minimal_condition_dim:
        include_prev_return = False
    else:
        raise ValueError(
            f"joint sampler condition_dim {joint_sampler.condition_dim} != "
            f"{full_condition_dim} (full) or {markov_minimal_condition_dim} "
            "(Markov-minimal)"
        )
    if initial_v <= 0:
        raise ValueError("initial_v must be positive")
    if cfg_w < 0.0:
        raise ValueError("cfg_w must be non-negative")

    if actions is None:
        actions = sample_action_schedule(
            n_paths=n_paths, n_steps=n_steps, num_actions=num_actions,
            transition_matrix=transition_matrix, initial_regime=initial_regime,
            seed=action_seed, constant=constant_action,
        )
    actions = np.asarray(actions, dtype=np.int8)
    if actions.shape != (n_paths, n_steps):
        raise ValueError(f"actions shape must be ({n_paths}, {n_steps}), got {actions.shape}")
    if actions.min() < 0 or actions.max() >= num_actions:
        raise ValueError(f"action indices must be in [0, {num_actions})")

    device = torch.device(device) if isinstance(device, str) else device

    log_v_mean = float(normalization["log_v_mean"])
    log_v_std = float(normalization["log_v_std"])
    return_mean = float(normalization["return_mean"])
    return_std = float(normalization["return_std"])

    log_v_paths_norm = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    r_paths_norm = np.empty((n_paths, n_steps), dtype=np.float32)
    log_v_t_norm = (np.log(initial_v) - log_v_mean) / log_v_std
    log_v_paths_norm[:, 0] = log_v_t_norm
    r_prev_norm = (initial_r_prev - return_mean) / return_std

    log_v_t = torch.full((n_paths, 1), log_v_t_norm, device=device, dtype=dtype)
    r_prev_t = torch.full((n_paths, 1), r_prev_norm, device=device, dtype=dtype)
    actions_t = torch.from_numpy(actions).to(device=device)

    rng = torch.Generator(device="cpu")
    if noise_seed is not None:
        rng.manual_seed(noise_seed)

    for step in range(n_steps):
        a_step = actions_t[:, step]
        a_onehot = _onehot(a_step, num_actions).to(dtype=dtype)

        condition_parts = [log_v_t]
        if include_prev_return:
            condition_parts.append(r_prev_t)
        condition_parts.append(a_onehot)
        condition = torch.cat(condition_parts, dim=-1)
        z = torch.randn(n_paths, 2, generator=rng, dtype=dtype).to(device)
        next_state = joint_sampler.sample(condition, noise=z, cfg_w=cfg_w)
        log_v_next = next_state[:, 0:1]
        r_next = next_state[:, 1:2]

        log_v_paths_norm[:, step + 1] = log_v_next.detach().cpu().numpy().reshape(-1)
        r_paths_norm[:, step] = r_next.detach().cpu().numpy().reshape(-1)

        log_v_t = log_v_next
        r_prev_t = r_next

    log_v_paths = log_v_paths_norm * log_v_std + log_v_mean
    v_paths = np.exp(log_v_paths)
    r_paths = r_paths_norm * return_std + return_mean

    log_s = np.log(initial_s) + np.cumsum(r_paths, axis=1)
    s_paths = np.concatenate(
        [np.full((n_paths, 1), initial_s, dtype=np.float64), np.exp(log_s)], axis=1,
    )

    return RolloutResult(
        log_v_paths_norm=log_v_paths_norm,
        r_paths_norm=r_paths_norm,
        log_v_paths=log_v_paths.astype(np.float32),
        v_paths=v_paths.astype(np.float32),
        r_paths=r_paths.astype(np.float32),
        s_paths=s_paths.astype(np.float32),
        actions=actions,
        initial_v=float(initial_v),
        initial_s=float(initial_s),
        num_actions=num_actions,
    )
