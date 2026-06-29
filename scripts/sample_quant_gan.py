#!/usr/bin/env python3
"""Sample paths from a trained Quant GAN generator and write a rollout-style npz."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.baselines import sample_quant_gan_paths
from finflow.baselines.quant_gan import load_quant_gan_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/quant_gan_paths.npz"))
    parser.add_argument("--n-paths", type=int, default=10_000)
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--lambert-w-delta", type=float, default=None,
                        help="override checkpoint transform delta")
    parser.add_argument("--no-calibrate-moments", action="store_true",
                        help="disable default affine mean/std calibration at sampling time")
    parser.add_argument("--calibration-eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator, ckpt = load_quant_gan_generator(args.checkpoint, map_location=args.device)
    normalization = ckpt.get("normalization", {})
    train_config = ckpt.get("train_config", {})
    lambert_w_delta = (
        float(args.lambert_w_delta)
        if args.lambert_w_delta is not None
        else float(train_config.get("lambert_w_delta", 0.0))
    )
    out = sample_quant_gan_paths(
        generator,
        n_paths=args.n_paths,
        s0=args.s0,
        return_mean=float(normalization.get("return_mean", 0.0)),
        return_std=float(normalization.get("return_std", 1.0)),
        lambert_w_delta=lambert_w_delta,
        calibrate_moments=not args.no_calibrate_moments,
        calibration_eps=args.calibration_eps,
        device=args.device,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, log_returns=out["log_returns"], s_paths=out["s_paths"])
    info = {
        "output": str(args.output),
        "n_paths": int(out["log_returns"].shape[0]),
        "n_steps": int(out["log_returns"].shape[1]),
        "checkpoint": str(args.checkpoint),
        "s0": args.s0,
        "lambert_w_delta": lambert_w_delta,
        "calibrate_moments": not args.no_calibrate_moments,
        "calibration": out.get("calibration", {}),
    }
    args.output.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
