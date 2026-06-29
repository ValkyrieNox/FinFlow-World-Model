from pathlib import Path

from finflow.data import generate_heston_dataset
from finflow.training import (
    TransitionFMModelConfig,
    TransitionFMTrainConfig,
    evaluate_checkpoint,
    train_transition_fm,
)


def test_training_smoke_creates_checkpoints_and_evaluates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate_heston_dataset(
        data_dir,
        n_train=4,
        n_val=2,
        n_test=2,
        n_steps=6,
        seed=123,
        save_transitions=True,
    )

    summary = train_transition_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs",
        run_name="smoke",
        model_config=TransitionFMModelConfig(hidden_dim=32, time_embedding_dim=16, num_blocks=2),
        train_config=TransitionFMTrainConfig(
            batch_size=4,
            epochs=2,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_val_batches=1,
        ),
    )

    run_dir = Path(summary["run_dir"])
    best_ckpt = Path(summary["checkpoints"]["best"])
    last_ckpt = Path(summary["checkpoints"]["last"])

    assert run_dir.exists()
    assert best_ckpt.exists()
    assert last_ckpt.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()

    eval_result = evaluate_checkpoint(best_ckpt, data_dir=data_dir, split="val", batch_size=2, device="cpu")
    assert eval_result["loss"] >= 0.0
    assert eval_result["model_config"]["hidden_dim"] == 32

