#!/usr/bin/env python3
"""Fine-tune a two-stage FM teacher with a QGAN-style path critic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.pathwise_teacher import (  # noqa: E402
    PathwiseTeacherFineTuneConfig,
    train_pathwise_teacher_finetune,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vol-checkpoint", type=Path, required=True)
    parser.add_argument("--ret-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pathwise_teacher"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=252)
    parser.add_argument("--fm-n-steps", type=int, default=8)
    parser.add_argument("--lr-teacher", type=float, default=1e-5)
    parser.add_argument("--lr-critic", type=float, default=2e-4)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    parser.add_argument("--transform-delta", type=float, default=0.1)
    parser.add_argument("--moment-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--abs-sum-weight", type=float, default=0.25)
    parser.add_argument("--kurtosis-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=1e-6)
    parser.add_argument("--freeze-vol", action="store_true",
                        help="fine-tune only the return teacher")
    parser.add_argument("--freeze-ret", action="store_true",
                        help="fine-tune only the variance teacher")
    parser.add_argument("--critic-hidden-channels", type=int, default=32)
    parser.add_argument("--critic-num-blocks", type=int, default=5)
    parser.add_argument("--critic-kernel-size", type=int, default=3)
    parser.add_argument("--initial-v", type=float, default=0.04)
    parser.add_argument("--initial-s", type=float, default=100.0)
    parser.add_argument("--initial-r-prev", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--compile-models", action="store_true",
                        help="use torch.compile on the FM teachers")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_vol and args.freeze_ret:
        raise ValueError("cannot freeze both vol and ret teachers")
    config = PathwiseTeacherFineTuneConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        n_steps=args.n_steps,
        fm_n_steps=args.fm_n_steps,
        lr_teacher=args.lr_teacher,
        lr_critic=args.lr_critic,
        critic_steps=args.critic_steps,
        gradient_penalty_weight=args.gradient_penalty_weight,
        transform_delta=args.transform_delta,
        moment_weight=args.moment_weight,
        terminal_weight=args.terminal_weight,
        abs_sum_weight=args.abs_sum_weight,
        kurtosis_weight=args.kurtosis_weight,
        anchor_weight=args.anchor_weight,
        train_vol=not args.freeze_vol,
        train_ret=not args.freeze_ret,
        critic_hidden_channels=args.critic_hidden_channels,
        critic_num_blocks=args.critic_num_blocks,
        critic_kernel_size=args.critic_kernel_size,
        initial_v=args.initial_v,
        initial_s=args.initial_s,
        initial_r_prev=args.initial_r_prev,
        seed=args.seed,
        device=args.device,
        progress=not args.no_progress,
        compile_models=args.compile_models,
        compile_mode=args.compile_mode,
    )
    summary = train_pathwise_teacher_finetune(
        vol_checkpoint=args.vol_checkpoint,
        ret_checkpoint=args.ret_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        config=config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
