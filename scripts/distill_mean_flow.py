#!/usr/bin/env python3
"""Distill a Mean Flow (1-NFE) student from a trained Stage 1a/1b FM teacher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.distillation import MeanFlowDistillConfig, train_mean_flow_distill
from finflow.training import TwoStageFMModelConfig, load_checkpoint, load_num_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/heston_v3"))
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=("vol", "ret", "joint"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: runs/mf_<stage>_distill")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-actions", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--time-eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-data-device", action="store_true",
                        help="preload vectorized condition/target tensors onto the training device")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--boundary-prob", type=float, default=0.25,
                        help="fixed fraction of batch where r=t when no curriculum is set")
    parser.add_argument("--boundary-prob-start", type=float, default=0.5,
                        help="first-epoch r=t anchor fraction for P1 curriculum")
    parser.add_argument("--boundary-prob-end", type=float, default=0.1,
                        help="last-epoch r=t anchor fraction for P1 curriculum")
    parser.add_argument("--no-boundary-curriculum", action="store_true",
                        help="use --boundary-prob as a fixed anchor fraction")
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--identity-residual-eval", action="store_true",
                        help="also evaluate a pure identity-residual validation loss each epoch")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="train MF student from random init (no teacher copy)")

    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--time-embedding-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_actions = args.num_actions if args.num_actions is not None else load_num_actions(args.data_dir)

    if args.hidden_dim is not None and args.time_embedding_dim is not None and args.num_blocks is not None:
        if args.stage == "vol":
            state_dim = 1
            condition_dim = 1 + num_actions
        elif args.stage == "ret":
            state_dim = 1
            condition_dim = 3 + num_actions
        else:
            teacher_ckpt = load_checkpoint(args.teacher_checkpoint, map_location="cpu")
            teacher_model_config = teacher_ckpt["model_config"]
            state_dim = int(teacher_model_config["state_dim"])
            condition_dim = int(teacher_model_config["condition_dim"])
        student_config = TwoStageFMModelConfig(
            state_dim=state_dim,
            condition_dim=condition_dim,
            hidden_dim=args.hidden_dim,
            time_embedding_dim=args.time_embedding_dim,
            num_blocks=args.num_blocks,
        )
    else:
        student_config = None

    distill_config = MeanFlowDistillConfig(
        teacher_checkpoint=str(args.teacher_checkpoint),
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
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        boundary_prob=args.boundary_prob,
        boundary_prob_start=None if args.no_boundary_curriculum else args.boundary_prob_start,
        boundary_prob_end=None if args.no_boundary_curriculum else args.boundary_prob_end,
        identity_weight=args.identity_weight,
        identity_residual_eval=args.identity_residual_eval,
        warm_start=not args.no_warm_start,
    )

    output_dir = args.output_dir or Path("runs") / f"mf_{args.stage}_distill"
    summary = train_mean_flow_distill(
        data_dir=args.data_dir,
        output_dir=output_dir,
        stage=args.stage,
        distill_config=distill_config,
        student_config=student_config,
        run_name=args.run_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
