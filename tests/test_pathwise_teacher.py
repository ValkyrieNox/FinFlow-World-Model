from pathlib import Path

import torch

from finflow.data import DEFAULT_REGIMES, DEFAULT_TRANSITION_MATRIX, generate_heston_dataset
from finflow.models import TransitionFM
from finflow.pathwise_teacher import (
    PathwiseTeacherFineTuneConfig,
    train_pathwise_teacher_finetune,
)
from finflow.training import load_normalization, save_checkpoint


def _write_teacher_checkpoint(
    path: Path,
    *,
    stage: str,
    condition_dim: int,
    num_actions: int,
    normalization: dict[str, float],
) -> None:
    model = TransitionFM(
        state_dim=1,
        condition_dim=condition_dim,
        hidden_dim=8,
        time_embedding_dim=4,
        num_blocks=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=0,
        global_step=0,
        best_val_loss=0.0,
        model_config={
            "state_dim": 1,
            "condition_dim": condition_dim,
            "hidden_dim": 8,
            "time_embedding_dim": 4,
            "num_blocks": 1,
        },
        train_config={},
        normalization=normalization,
        stage=stage,
        num_actions=num_actions,
        extra={"kind": "fm", "lambert_w_delta": 0.05 if stage == "vol" else 0.0},
    )


def test_pathwise_teacher_finetune_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    metadata = generate_heston_dataset(
        data_dir,
        n_train=6,
        n_val=2,
        n_test=2,
        n_steps=4,
        regimes=DEFAULT_REGIMES,
        transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0,
        seed=11,
        save_transitions=True,
    )
    normalization = load_normalization(data_dir)
    num_actions = metadata["num_actions"]
    vol_ckpt = tmp_path / "vol.pt"
    ret_ckpt = tmp_path / "ret.pt"
    _write_teacher_checkpoint(
        vol_ckpt,
        stage="vol",
        condition_dim=1 + num_actions,
        num_actions=num_actions,
        normalization=normalization,
    )
    _write_teacher_checkpoint(
        ret_ckpt,
        stage="ret",
        condition_dim=3 + num_actions,
        num_actions=num_actions,
        normalization=normalization,
    )

    summary = train_pathwise_teacher_finetune(
        vol_checkpoint=vol_ckpt,
        ret_checkpoint=ret_ckpt,
        data_dir=data_dir,
        output_dir=tmp_path / "runs",
        run_name="path_smoke",
        config=PathwiseTeacherFineTuneConfig(
            batch_size=2,
            epochs=1,
            steps_per_epoch=1,
            n_steps=4,
            fm_n_steps=1,
            critic_steps=1,
            critic_hidden_channels=4,
            critic_num_blocks=1,
            train_vol=False,
            train_ret=True,
            progress=False,
            device="cpu",
            seed=5,
        ),
    )

    assert Path(summary["checkpoints"]["vol_best"]).exists()
    assert Path(summary["checkpoints"]["ret_best"]).exists()
    assert summary["history"][0]["generator_loss"] == summary["history"][0]["generator_loss"]
