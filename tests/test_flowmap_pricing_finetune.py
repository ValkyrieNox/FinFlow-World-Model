import torch

from finflow.models import MeanFlowStudent
from scripts.finetune_flow_map_pricing import (
    pricing_loss_from_grid,
    rollout_price_grid_torch,
)


def test_pricing_rollout_loss_backpropagates_to_flowmap_student():
    torch.manual_seed(0)
    student = MeanFlowStudent(
        state_dim=2,
        condition_dim=5,
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
    actions = torch.zeros(6, 8, dtype=torch.long)
    moneynesses = torch.tensor([0.9, 1.0], dtype=torch.float32)
    maturities = torch.tensor([0.5, 1.0], dtype=torch.float32)

    prices = rollout_price_grid_torch(
        student,
        normalization=normalization,
        actions=actions,
        moneynesses=moneynesses,
        maturities=maturities,
        dt=1.0 / 8,
        initial_v=0.04,
        initial_s=100.0,
        initial_r_prev=0.0,
        r=0.0,
        include_prev_return=True,
        cfg_w=0.0,
        price_chunk_paths=3,
    )
    assert prices.shape == (2, 2)
    assert torch.isfinite(prices).all()

    reference = torch.full_like(prices, 5.0)
    loss, rmse, mape = pricing_loss_from_grid(prices, reference, price_floor=1.0)
    assert loss.ndim == 0
    assert rmse.ndim == 0
    assert mape.ndim == 0
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in student.parameters()
    )
