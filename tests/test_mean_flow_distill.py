from pathlib import Path
import json

import torch

from finflow.data import (
    DEFAULT_REGIMES,
    DEFAULT_TRANSITION_MATRIX,
    generate_heston_dataset,
)
from finflow.distillation import (
    MeanFlowDistillConfig,
    mean_flow_loss,
    mean_flow_loss_components,
    train_mean_flow_distill,
)
from finflow.inference import load_sampler_from_checkpoint
from finflow.models import MeanFlowStudent, TransitionFM
from finflow.training import (
    TransitionFMTrainConfig,
    TwoStageFMModelConfig,
    load_checkpoint,
    train_joint_trans_fm,
    train_vol_trans_fm,
)
from scripts.distill_flow_map import flow_map_loss


def test_mean_flow_loss_runs_and_is_scalar():
    torch.manual_seed(0)
    teacher = TransitionFM(state_dim=1, condition_dim=4, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    student = MeanFlowStudent(state_dim=1, condition_dim=4, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    for p in teacher.parameters():
        p.requires_grad_(False)
    cond = torch.randn(8, 4)
    target = torch.randn(8, 1)
    loss = mean_flow_loss(student, teacher, cond, target, time_eps=1e-3, boundary_prob=0.25)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())


def test_mean_flow_loss_uses_reversed_teacher_velocity_at_boundary():
    class ConstantStudent(torch.nn.Module):
        state_dim = 1
        condition_dim = 4

        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(-2.0))

        def forward(self, x, r, t, condition):
            return torch.ones_like(x) * self.value

    class ConstantTeacher(torch.nn.Module):
        def forward(self, x_tau, tau, condition):
            return torch.full_like(x_tau, 2.0)

    cond = torch.randn(8, 4)
    target = torch.randn(8, 1)
    loss = mean_flow_loss(
        ConstantStudent(),
        ConstantTeacher(),
        cond,
        target,
        time_eps=1e-3,
        boundary_prob=1.0,
    )
    assert loss.item() < 1e-8


def test_mean_flow_loss_components_split_boundary_and_identity():
    torch.manual_seed(0)
    teacher = TransitionFM(state_dim=1, condition_dim=4, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    student = MeanFlowStudent(state_dim=1, condition_dim=4, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    cond = torch.randn(8, 4)
    target = torch.randn(8, 1)

    boundary = mean_flow_loss_components(
        student, teacher, cond, target, time_eps=1e-3, boundary_prob=1.0,
    )
    assert torch.isclose(boundary["boundary_fraction"], torch.tensor(1.0))
    assert boundary["identity_count"].item() == 0.0
    assert torch.isclose(boundary["loss"], boundary["boundary_loss"])

    identity = mean_flow_loss_components(
        student, teacher, cond, target, time_eps=1e-3, boundary_prob=0.0,
    )
    assert identity["boundary_count"].item() == 0.0
    assert torch.isclose(identity["loss"], identity["identity_loss"])


def test_joint_flow_map_loss_runs_and_is_scalar():
    torch.manual_seed(0)
    teacher = TransitionFM(state_dim=2, condition_dim=5, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    student = MeanFlowStudent(state_dim=2, condition_dim=5, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    for p in teacher.parameters():
        p.requires_grad_(False)
    cond = torch.randn(8, 5)
    target = torch.randn(8, 2)
    loss = flow_map_loss(student, teacher, cond, target, time_eps=1e-3, boundary_prob=0.25)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())


def _generate_smoke_data(tmp_path: Path):
    data_dir = tmp_path / "data"
    metadata = generate_heston_dataset(
        data_dir, n_train=4, n_val=2, n_test=2, n_steps=6,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=42, save_transitions=True,
    )
    return data_dir, metadata["num_actions"]


def test_train_mean_flow_distill_smoke(tmp_path: Path):
    data_dir, num_actions = _generate_smoke_data(tmp_path)
    teacher_summary = train_vol_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_vol",
        run_name="vol_smoke",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=1 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(
            batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
            seed=7, device="cpu", max_train_batches=2, max_val_batches=1, progress=False,
        ),
    )
    distill_summary = train_mean_flow_distill(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_mf",
        stage="vol",
        run_name="mf_smoke",
        distill_config=MeanFlowDistillConfig(
            teacher_checkpoint=teacher_summary["checkpoints"]["best"],
            batch_size=4, epochs=2, lr=1e-3, weight_decay=0.0,
            seed=8, device="cpu", max_train_batches=2, max_val_batches=1,
            boundary_prob_start=0.5, boundary_prob_end=0.1,
            identity_residual_eval=True, progress=False,
        ),
        student_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=1 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
    )
    assert Path(distill_summary["checkpoints"]["best"]).exists()
    assert distill_summary["stage"] == "mf_vol"
    assert distill_summary["num_actions"] == num_actions
    history = distill_summary["history"]
    assert history[0]["boundary_prob"] == 0.5
    assert history[-1]["boundary_prob"] == 0.1
    assert "train_boundary_loss" in history[-1]
    assert "train_identity_loss" in history[-1]
    assert "val_identity_residual" in history[-1]

    metrics_path = Path(distill_summary["run_dir"]) / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["boundary_prob"] == 0.1


def test_train_joint_mean_flow_distill_smoke(tmp_path: Path):
    data_dir, num_actions = _generate_smoke_data(tmp_path)
    teacher_summary = train_joint_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_joint",
        run_name="joint_teacher",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=2, condition_dim=2 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(
            batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
            seed=17, device="cpu", max_train_batches=2, max_val_batches=1, progress=False,
        ),
    )
    distill_summary = train_mean_flow_distill(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_mf_joint",
        stage="joint",
        run_name="mf_joint_smoke",
        distill_config=MeanFlowDistillConfig(
            teacher_checkpoint=teacher_summary["checkpoints"]["best"],
            batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0,
            seed=18, device="cpu", max_train_batches=2, max_val_batches=1,
            boundary_prob_start=0.5, boundary_prob_end=0.5, progress=False,
        ),
        student_config=TwoStageFMModelConfig(
            state_dim=2, condition_dim=2 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
    )
    ckpt_path = Path(distill_summary["checkpoints"]["best"])
    assert ckpt_path.exists()
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    assert ckpt["stage"] == "mf_joint"
    assert ckpt["model_config"]["state_dim"] == 2
    loaded = load_sampler_from_checkpoint(ckpt_path, device="cpu")
    assert loaded.stage == "joint"
    assert loaded.sampler.state_dim == 2
    assert loaded.sampler.condition_dim == 2 + num_actions
