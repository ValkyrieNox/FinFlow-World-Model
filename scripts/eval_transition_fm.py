#!/usr/bin/env python3
"""Evaluate a trained transition FM checkpoint on a chosen Heston split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.training import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/heston_v3"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

