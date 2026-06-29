from pathlib import Path

import torch

from finflow.data import (
    DEFAULT_REGIMES,
    DEFAULT_TRANSITION_MATRIX,
    generate_heston_dataset,
)
from finflow.distillation import (
    ConsistencyDistillConfig,
    consistency_distill_step,
    train_consistency_distill,
)
from finflow.distillation.consistency import _schedule
from finflow.distillation.consistency import _curriculum_ema_decay, _curriculum_n
from finflow.inference import load_sampler_from_checkpoint
from finflow.models import ConsistencyStudent, TransitionFM
from finflow.training import (
    TransitionFMTrainConfig,
    TwoStageFMModelConfig,
    load_checkpoint,
    train_joint_trans_fm,
    train_ret_trans_fm,
)


class _SpyConsistency(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_t: torch.Tensor | None = None

    def forward(self, x, t, condition):
        self.seen_t = t.detach().clone()
        return self.scale * x


class _ZeroTeacher(torch.nn.Module):
    def forward(self, x_tau, tau, condition):
        return torch.zeros_like(x_tau)


def test_consistency_distill_step_runs_and_is_scalar():
    torch.manual_seed(0)
    teacher = TransitionFM(state_dim=1, condition_dim=4, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    student = ConsistencyStudent(state_dim=1, condition_dim=4, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    target_net = ConsistencyStudent(state_dim=1, condition_dim=4, hidden_dim=16, time_embedding_dim=8, num_blocks=2)
    target_net.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in target_net.parameters():
        p.requires_grad_(False)

    schedule = _schedule(8, time_eps=1e-3, device=torch.device("cpu"), dtype=torch.float32)
    cond = torch.randn(8, 4)
    target = torch.randn(8, 1)
    loss = consistency_distill_step(student, target_net, teacher, cond, target, schedule)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())


def test_consistency_distill_step_matches_noisier_student_to_cleaner_target():
    torch.manual_seed(1)
    student = _SpyConsistency()
    target_net = _SpyConsistency()
    for p in target_net.parameters():
        p.requires_grad_(False)

    schedule = _schedule(8, time_eps=1e-3, device=torch.device("cpu"), dtype=torch.float32)
    cond = torch.randn(16, 4)
    target = torch.randn(16, 1)
    loss = consistency_distill_step(student, target_net, _ZeroTeacher(), cond, target, schedule)

    assert loss.ndim == 0
    assert student.seen_t is not None
    assert target_net.seen_t is not None
    assert torch.all(student.seen_t < target_net.seen_t)


def test_consistency_ict_curriculum_increases_n_and_ema():
    cfg = ConsistencyDistillConfig(teacher_checkpoint="unused", n_min=10, n_max=160)
    n1 = _curriculum_n(cfg, epoch=1, total_epochs=4)
    n4 = _curriculum_n(cfg, epoch=4, total_epochs=4)
    assert n1 == 10
    assert n4 == 160
    assert _curriculum_ema_decay(cfg, n1) < _curriculum_ema_decay(cfg, n4)


def _generate_smoke_data(tmp_path: Path):
    data_dir = tmp_path / "data"
    metadata = generate_heston_dataset(
        data_dir, n_train=4, n_val=2, n_test=2, n_steps=6,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=42, save_transitions=True,
    )
    return data_dir, metadata["num_actions"]


def test_train_consistency_distill_smoke(tmp_path: Path):
    data_dir, num_actions = _generate_smoke_data(tmp_path)
    teacher_summary = train_ret_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_ret",
        run_name="ret_smoke",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=3 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(
            batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
            seed=11, device="cpu", max_train_batches=2, max_val_batches=1, progress=False,
        ),
    )
    distill_summary = train_consistency_distill(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_cd",
        stage="ret",
        run_name="cd_smoke",
        distill_config=ConsistencyDistillConfig(
            teacher_checkpoint=teacher_summary["checkpoints"]["best"],
            batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0,
            seed=12, device="cpu", max_train_batches=2, max_val_batches=1,
            n_discretization=4, ema_decay=0.9, progress=False,
        ),
        student_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=3 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
    )
    assert Path(distill_summary["checkpoints"]["best"]).exists()
    ckpt = torch.load(distill_summary["checkpoints"]["best"], map_location="cpu", weights_only=False)
    assert ckpt["extra"]["model_state_kind"] == "ema"
    assert distill_summary["stage"] == "cd_ret"
    assert distill_summary["num_actions"] == num_actions


def test_train_joint_consistency_distill_smoke(tmp_path: Path):
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
            seed=19, device="cpu", max_train_batches=2, max_val_batches=1, progress=False,
        ),
    )
    distill_summary = train_consistency_distill(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_cd_joint",
        stage="joint",
        run_name="cd_joint_smoke",
        distill_config=ConsistencyDistillConfig(
            teacher_checkpoint=teacher_summary["checkpoints"]["best"],
            batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0,
            seed=20, device="cpu", max_train_batches=2, max_val_batches=1,
            n_discretization=4, ema_decay=0.9, curriculum_kind="fixed",
            progress=False,
        ),
        student_config=TwoStageFMModelConfig(
            state_dim=2, condition_dim=2 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
    )
    ckpt_path = Path(distill_summary["checkpoints"]["best"])
    assert ckpt_path.exists()
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    assert ckpt["stage"] == "cd_joint"
    assert ckpt["extra"]["model_state_kind"] == "ema"
    assert ckpt["model_config"]["state_dim"] == 2
    loaded = load_sampler_from_checkpoint(ckpt_path, device="cpu")
    assert loaded.stage == "joint"
    assert loaded.sampler.state_dim == 2
    assert loaded.sampler.condition_dim == 2 + num_actions
