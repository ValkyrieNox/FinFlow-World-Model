import torch
import torch.nn.functional as F

from finflow.models import MeanFlowStudent, TransitionFM
from finflow.models.transition_fm import euler_sample
from scripts.finetune_flow_map_onpolicy import (
    _student_one_step,
    sample_onpolicy_conditions,
)


def test_onpolicy_endpoint_loss_backpropagates_to_flowmap_student():
    torch.manual_seed(0)
    student = MeanFlowStudent(
        state_dim=2,
        condition_dim=3,
        hidden_dim=16,
        time_embedding_dim=8,
        num_blocks=2,
    )
    teacher = TransitionFM(
        state_dim=2,
        condition_dim=3,
        hidden_dim=16,
        time_embedding_dim=8,
        num_blocks=2,
    )
    normalization = {
        "log_v_mean": -3.2,
        "log_v_std": 0.4,
        "return_mean": 0.0,
        "return_std": 0.02,
    }
    metadata = {"regime_switching": False, "initial_regime": 0}
    cond = sample_onpolicy_conditions(
        student,
        normalization=normalization,
        metadata=metadata,
        num_actions=1,
        include_prev_return=True,
        n_paths=5,
        horizon=6,
        samples_per_path=2,
        initial_v=0.04,
        initial_r_prev=0.0,
        rng=__import__("numpy").random.default_rng(123),
    )
    assert cond.shape == (10, 3)
    assert torch.isfinite(cond).all()

    noise = torch.randn(cond.shape[0], 2)
    target = euler_sample(teacher, condition=cond, n_steps=2, noise=noise)
    pred = _student_one_step(student, cond, noise, cfg_w=0.0, num_actions=1)
    loss = F.mse_loss(pred, target)
    assert loss.ndim == 0
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in student.parameters()
    )
