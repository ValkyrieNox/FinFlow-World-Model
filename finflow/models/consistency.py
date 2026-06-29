"""Consistency Distillation student (Song et al. 2023).

The student ``f_theta(x_t, t, c)`` maps any point on a Flow-Matching ODE
trajectory back to its terminal data point (t = 1 in our convention). The
self-consistency constraint is::

    f(x_{t_{n+1}}, t_{n+1}) = f(x_{t_n}, t_n)

for adjacent steps on the same trajectory.

Boundary parameterization::

    f(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t, c)

with ``c_skip(t) = t`` and ``c_out(t) = 1 - t`` so that ``f(x, 1) = x``
(the data anchor) and ``f(x, 0) = F_theta(x, 0, c)`` (network output on pure
noise during inference).
"""

from __future__ import annotations

import torch
from torch import nn

from finflow.models.transition_fm import TransitionFM


def _broadcast(scalar: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    while scalar.ndim < like.ndim:
        scalar = scalar.unsqueeze(-1)
    return scalar


class ConsistencyStudent(nn.Module):
    """Consistency model with boundary-preserving parameterization.

    Wraps a :class:`TransitionFM`-style backbone and applies the
    ``c_skip(t) * x + c_out(t) * F(x, t, c)`` head.
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = TransitionFM(
            state_dim=state_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            time_embedding_dim=time_embedding_dim,
            num_blocks=num_blocks,
        )
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_blocks = num_blocks

    @staticmethod
    def c_skip(t: torch.Tensor) -> torch.Tensor:
        # 1 at t=1 (data anchor), 0 at t=0 (noise side)
        return t

    @staticmethod
    def c_out(t: torch.Tensor) -> torch.Tensor:
        # 0 at t=1, 1 at t=0
        return 1.0 - t

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2 or condition.ndim != 2:
            raise ValueError("x and condition must be 2D")
        f_net = self.backbone(x_tau=x, tau=t, condition=condition)
        c_skip = _broadcast(self.c_skip(t).to(dtype=x.dtype), x)
        c_out = _broadcast(self.c_out(t).to(dtype=x.dtype), x)
        return c_skip * x + c_out * f_net


def warm_start_consistency_from_fm(
    student: ConsistencyStudent,
    teacher: TransitionFM,
) -> int:
    """Copy teacher parameters into the student's backbone where shapes match."""

    teacher_state = teacher.state_dict()
    backbone_state = dict(student.backbone.state_dict())
    copied = 0
    for name, value in backbone_state.items():
        if name in teacher_state and teacher_state[name].shape == value.shape:
            backbone_state[name] = teacher_state[name].clone()
            copied += 1
    student.backbone.load_state_dict(backbone_state)
    return copied
