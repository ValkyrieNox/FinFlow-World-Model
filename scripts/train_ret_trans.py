#!/usr/bin/env python3
"""Train V3 Stage 1b return transition kernel: p(r_{t+1} | v_{t+1}, v_t, r_t, a_t).

By default training uses teacher-forced ground-truth v_{t+1} from the Heston
data. With --vol-sampler-checkpoint, training gradually replaces v_{t+1} with
Stage 1a sampler output to reduce the train/inference gap.
"""

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
    train_ret_trans_fm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ret_trans_fm"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-actions", type=int, default=None,
                        help="Override action dim; default auto-detected from metadata.")

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--time-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-data-device", action="store_true",
                        help="preload vectorized condition/target tensors onto the training device")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--action-dropout-prob", type=float, default=0.1,
                        help="drop action one-hot during training for CFG support")
    parser.add_argument("--vol-sampler-checkpoint", type=Path, default=None,
                        help="optional Stage 1a vol sampler checkpoint for scheduled sampling")
    parser.add_argument("--scheduled-sampling-max-prob", type=float, default=0.5,
                        help="max fraction of ret batches conditioned on sampled v_next")
    parser.add_argument("--scheduled-sampling-start-epoch", type=int, default=1,
                        help="first epoch where scheduled sampling can be active")
    parser.add_argument("--scheduled-sampling-fm-steps", type=int, default=20,
                        help="Euler steps when the vol sampler checkpoint is an FM teacher")
    parser.add_argument("--save-every-epochs", type=int, default=0,
                        help=">0: also dump checkpoints/epoch_XXX.pt every N epochs "
                             "(for pricing-aware checkpoint selection)")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--lr-min", type=float, default=0.0,
                        help="eta_min for cosine schedule")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_actions = args.num_actions
    if num_actions is None:
        num_actions = load_num_actions(args.data_dir)
    model_config = TwoStageFMModelConfig(
        state_dim=1,
        condition_dim=3 + num_actions,
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
        scheduled_sampling_max_prob=(
            args.scheduled_sampling_max_prob if args.vol_sampler_checkpoint is not None else 0.0
        ),
        scheduled_sampling_start_epoch=args.scheduled_sampling_start_epoch,
        scheduled_sampling_fm_steps=args.scheduled_sampling_fm_steps,
        save_every_epochs=args.save_every_epochs,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
    )
    summary = train_ret_trans_fm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_actions=num_actions,
        model_config=model_config,
        train_config=train_config,
        vol_sampler_checkpoint=args.vol_sampler_checkpoint,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
