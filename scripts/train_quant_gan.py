#!/usr/bin/env python3
"""Train the Quant GAN baseline on Heston log-return sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.baselines import QuantGANConfig, QuantGANTrainConfig, train_quant_gan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/heston_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/quant_gan"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--seq-len", type=int, default=252)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=3)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--d-steps-per-g", type=int, default=5)
    parser.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    parser.add_argument("--lambert-w-delta", type=float, default=0.1)
    parser.add_argument("--moment-penalty-weight", type=float, default=1.0)
    parser.add_argument("--moment-mean-weight", type=float, default=1.0)
    parser.add_argument("--moment-std-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = QuantGANConfig(
        latent_dim=args.latent_dim,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        seq_len=args.seq_len,
    )
    train_config = QuantGANTrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        d_steps_per_g=args.d_steps_per_g,
        gradient_penalty_weight=args.gradient_penalty_weight,
        lambert_w_delta=args.lambert_w_delta,
        moment_penalty_weight=args.moment_penalty_weight,
        moment_mean_weight=args.moment_mean_weight,
        moment_std_weight=args.moment_std_weight,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
    )
    summary = train_quant_gan(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_config=model_config,
        train_config=train_config,
        run_name=args.run_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
