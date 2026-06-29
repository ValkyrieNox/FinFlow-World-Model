#!/usr/bin/env python3
"""Train an action-aware joint transition kernel: p(v_{t+1}, r_t | v_t, r_{t-1}, a_t)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.training import (
    TransitionFMTrainConfig,
    TwoStageFMModelConfig,
    load_num_actions,
    train_joint_trans_fm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/heston_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/joint_trans_fm"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--num-actions", type=int, default=None,
        help="Override action dim; default auto-detected from metadata.",
    )

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--time-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-data-device", action="store_true",
        help="preload vectorized condition/target tensors onto the training device",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--action-dropout-prob", type=float, default=0.0,
        help="drop action one-hot during training for CFG support",
    )
    parser.add_argument(
        "--save-every-epochs", type=int, default=0,
        help=">0: also dump checkpoints/epoch_XXX.pt every N epochs",
    )
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--lr-min", type=float, default=0.0, help="eta_min for cosine schedule")
    parser.add_argument(
        "--ema-decay", type=float, default=0.0,
        help=">0: maintain EMA weights and save ema_last/ema_best checkpoints",
    )
    parser.add_argument(
        "--target-loss-weights", nargs="+", type=float, default=None,
        help="optional per-target FM loss weights, e.g. '1 2' for [log_v_next, r_next]",
    )
    parser.add_argument(
        "--markov-minimal",
        action="store_true",
        help=(
            "ablation: condition on [log_v_t, action] instead of "
            "[log_v_t, r_{t-1}, action]"
        ),
    )

    parser.add_argument(
        "--scheduled-sampling-max-prob", type=float, default=0.0,
        help=(
            ">0: self-scheduled-sampling. Replace a ramped fraction of each "
            "row's carried state with the model's own one-step sample from its "
            "autoregressive parent (closes the teacher-forcing/free-rollout gap)."
        ),
    )
    parser.add_argument(
        "--scheduled-sampling-start-epoch", type=int, default=1,
        help="epoch at which scheduled sampling begins ramping from 0",
    )
    parser.add_argument(
        "--scheduled-sampling-fm-steps", type=int, default=20,
        help="ODE steps for the in-training self-sampler",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_actions = args.num_actions
    if num_actions is None:
        num_actions = load_num_actions(args.data_dir)
    include_prev_return = not args.markov_minimal
    model_config = TwoStageFMModelConfig(
        state_dim=2,
        condition_dim=(2 if include_prev_return else 1) + num_actions,
        hidden_dim=args.hidden_dim,
        time_embedding_dim=args.time_embedding_dim,
        num_blocks=args.num_blocks,
    )
    train_config = TransitionFMTrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        time_eps=args.time_eps,
        num_workers=args.num_workers,
        cache_data_device=args.cache_data_device,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        action_dropout_prob=args.action_dropout_prob,
        scheduled_sampling_max_prob=args.scheduled_sampling_max_prob,
        scheduled_sampling_start_epoch=args.scheduled_sampling_start_epoch,
        scheduled_sampling_fm_steps=args.scheduled_sampling_fm_steps,
        save_every_epochs=args.save_every_epochs,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        ema_decay=args.ema_decay,
        target_loss_weights=(
            tuple(args.target_loss_weights) if args.target_loss_weights is not None else None
        ),
    )
    summary = train_joint_trans_fm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_actions=num_actions,
        model_config=model_config,
        train_config=train_config,
        include_prev_return=include_prev_return,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
