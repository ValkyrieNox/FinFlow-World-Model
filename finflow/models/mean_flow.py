"""Mean Flow student model (Geng et al. 2025, NeurIPS Oral).

The student predicts the average velocity ``u(x_t, r, t, c)`` over a Mean Flow
sub-interval ``[r, t]``. In this codebase the student is distilled in the
``data -> noise`` time convention so a one-step sample from noise ``z`` is
``z - u(z, 0, 1, c)``. Architecture mirrors :class:`TransitionFM` (FiLM
residual MLP with sinusoidal time embeddings), except both ``r`` and ``t`` are
embedded separately and concatenated into the FiLM context.

The Mean Flow identity that the loss enforces is::

    u(x_t, r, t) = v(x_t, t) - (t - r) * d/dt u(x_t, r, t)

so at the degenerate case ``r == t`` the student reduces to the instantaneous
velocity in that reversed convention. ``MeanFlowStudent`` is unaware of this
identity --- it is just a function approximator. The Mean Flow loss lives in
:mod:`finflow.distillation.mean_flow`.
"""

from __future__ import annotations

import torch
from torch import nn

from finflow.models.transition_fm import FiLMResidualBlock, SinusoidalTimeEmbedding


class MeanFlowStudent(nn.Module):
    """Mean Flow average-velocity network.

    Inputs:
        x:         noised state, shape ``[B, state_dim]``
        r:         lower FM time ``r in [0, 1]``, shape ``[B]`` or ``[B, 1]``
        t:         upper FM time ``t in [r, 1]``, shape ``[B]`` or ``[B, 1]``
        condition: external condition, shape ``[B, condition_dim]``

    Output: predicted average velocity, shape ``[B, state_dim]``.
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
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if condition_dim < 0:
            raise ValueError("condition_dim must be non-negative")
        if hidden_dim <= 0 or time_embedding_dim <= 0:
            raise ValueError("hidden_dim and time_embedding_dim must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_blocks = num_blocks

        self.time_embedding_t = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_embedding_r = SinusoidalTimeEmbedding(time_embedding_dim)
        context_in = condition_dim + 2 * time_embedding_dim
        self.context_net = nn.Sequential(
            nn.Linear(context_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim=hidden_dim, context_dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, state_dim]")
        if condition.ndim != 2:
            raise ValueError("condition must have shape [batch, condition_dim]")
        if x.shape[0] != condition.shape[0]:
            raise ValueError("x and condition batch sizes must match")
        if x.shape[1] != self.state_dim:
            raise ValueError(f"x must have state_dim={self.state_dim}")
        if condition.shape[1] != self.condition_dim:
            raise ValueError(f"condition must have condition_dim={self.condition_dim}")

        r_emb = self.time_embedding_r(r.to(device=x.device))
        t_emb = self.time_embedding_t(t.to(device=x.device))
        if r_emb.shape[0] != x.shape[0] or t_emb.shape[0] != x.shape[0]:
            raise ValueError("r and t batch sizes must match x")

        ctx_in = torch.cat(
            [
                condition,
                r_emb.to(dtype=condition.dtype),
                t_emb.to(dtype=condition.dtype),
            ],
            dim=-1,
        )
        context = self.context_net(ctx_in)
        hidden = self.input_proj(x)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output(hidden)


def warm_start_mean_flow_from_fm(
    student: MeanFlowStudent,
    teacher,
) -> int:
    """Copy any teacher parameters with matching shape into the student.

    Returns the number of tensors copied. Useful to bias the MF student
    toward the teacher's FM velocity at the start of distillation.
    """

    teacher_state = teacher.state_dict()
    student_state = dict(student.state_dict())
    copied = 0
    # Map the shared MLP backbone (input_proj, blocks, output) verbatim.
    for name, value in student_state.items():
        if name in teacher_state and teacher_state[name].shape == value.shape:
            student_state[name] = teacher_state[name].clone()
            copied += 1
    # Map TransitionFM's single time embedding into both MF time embeddings,
    # provided shapes line up.
    for src_key, dst_keys in (
        ("time_embedding", ("time_embedding_t", "time_embedding_r")),
    ):
        for tk, tv in teacher_state.items():
            if not tk.startswith(src_key + "."):
                continue
            tail = tk[len(src_key) + 1 :]
            for dst_key in dst_keys:
                full = f"{dst_key}.{tail}"
                if full in student_state and student_state[full].shape == tv.shape:
                    student_state[full] = tv.clone()
                    copied += 1
    student.load_state_dict(student_state)
    return copied
