#!/usr/bin/env python3
"""Autoregressive rollout from trained vol/ret samplers.

Loads a vol-stage and ret-stage checkpoint (each can be an FM teacher, a
Mean Flow student, or a Consistency student), rolls out N paths of length
T, and writes ``rollout.npz`` (returns / variance / price paths) + metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.data import DEFAULT_TRANSITION_MATRIX
from finflow.inference import autoregressive_rollout, load_sampler_from_checkpoint
from finflow.training import load_metadata
from scripts.rollout_calibration import calibrate_return_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vol-checkpoint", type=Path, required=True)
    parser.add_argument("--ret-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="used to read normalization + transition matrix (if --regime-actions)")
    parser.add_argument("--output", type=Path, default=Path("runs/rollout.npz"))
    parser.add_argument("--n-paths", type=int, default=10_000)
    parser.add_argument("--n-steps", type=int, default=252)
    parser.add_argument("--initial-v", type=float, default=0.04)
    parser.add_argument("--initial-s", type=float, default=100.0)
    parser.add_argument("--initial-r-prev", type=float, default=0.0)
    parser.add_argument("--initial-regime", type=int, default=0)
    parser.add_argument("--regime-actions", action="store_true",
                        help="sample actions from the Markov chain in metadata.json")
    parser.add_argument("--constant-action", action="store_true",
                        help="all paths stay in --initial-regime forever")
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--fm-n-steps", type=int, default=20,
                        help="ODE steps when a checkpoint is an FM teacher")
    parser.add_argument("--fm-solver", choices=("euler", "heun"), default="euler",
                        help="ODE solver when a checkpoint is an FM teacher")
    parser.add_argument("--cfg-w", type=float, default=0.0,
                        help="classifier-free guidance weight over action conditioning")
    parser.add_argument("--calibrate-moments", action="store_true",
                        help="affine-calibrate pooled generated log-returns to the data's "
                             "return_mean/return_std at sampling time (same correction the "
                             "Quant GAN baseline uses), then rebuild price paths. Default off.")
    parser.add_argument("--calibration-eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vol_loaded = load_sampler_from_checkpoint(
        args.vol_checkpoint, device=args.device, fm_n_steps=args.fm_n_steps,
        fm_solver=args.fm_solver,
    )
    ret_loaded = load_sampler_from_checkpoint(
        args.ret_checkpoint, device=args.device, fm_n_steps=args.fm_n_steps,
        fm_solver=args.fm_solver,
    )
    if vol_loaded.stage != "vol":
        raise ValueError(f"--vol-checkpoint stage is '{vol_loaded.stage}', expected 'vol'")
    if ret_loaded.stage != "ret":
        raise ValueError(f"--ret-checkpoint stage is '{ret_loaded.stage}', expected 'ret'")
    if vol_loaded.num_actions != ret_loaded.num_actions:
        raise ValueError(
            f"num_actions mismatch: vol={vol_loaded.num_actions}, ret={ret_loaded.num_actions}"
        )
    num_actions = vol_loaded.num_actions
    normalization = vol_loaded.normalization or ret_loaded.normalization
    if not normalization:
        raise ValueError("normalization stats missing from both checkpoints")

    metadata = load_metadata(args.data_dir) if args.data_dir.exists() else {}
    if args.regime_actions and num_actions > 1:
        if "transition_matrix" in metadata:
            transition_matrix = np.asarray(metadata["transition_matrix"], dtype=np.float64)
        else:
            transition_matrix = DEFAULT_TRANSITION_MATRIX
        if transition_matrix.shape != (num_actions, num_actions):
            raise ValueError(
                f"transition_matrix shape {transition_matrix.shape} != ({num_actions}, {num_actions})"
            )
    else:
        transition_matrix = None

    result = autoregressive_rollout(
        vol_sampler=vol_loaded.sampler,
        ret_sampler=ret_loaded.sampler,
        normalization=normalization,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        num_actions=num_actions,
        initial_v=args.initial_v,
        initial_s=args.initial_s,
        initial_r_prev=args.initial_r_prev,
        transition_matrix=transition_matrix,
        initial_regime=args.initial_regime,
        action_seed=args.action_seed,
        noise_seed=args.noise_seed,
        device=vol_loaded.sampler.device,
        constant_action=args.constant_action,
        cfg_w=args.cfg_w,
    )

    r_paths = result.r_paths
    s_paths = result.s_paths
    calibration_info: dict | None = None
    if args.calibrate_moments:
        r_paths, s_paths, calibration_info = calibrate_return_paths(
            result.r_paths, result.initial_s,
            return_mean=float(normalization["return_mean"]),
            return_std=float(normalization["return_std"]),
            eps=args.calibration_eps,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        log_v_paths=result.log_v_paths,
        v_paths=result.v_paths,
        r_paths=r_paths,
        s_paths=s_paths,
        actions=result.actions,
    )
    info = {
        "output": str(args.output),
        "n_paths": result.s_paths.shape[0],
        "n_steps": result.s_paths.shape[1] - 1,
        "num_actions": num_actions,
        "vol_checkpoint": str(args.vol_checkpoint),
        "ret_checkpoint": str(args.ret_checkpoint),
        "vol_kind": vol_loaded.sampler.kind,
        "ret_kind": ret_loaded.sampler.kind,
        "initial_v": result.initial_v,
        "initial_s": result.initial_s,
        "normalization": normalization,
        "regime_actions": bool(args.regime_actions and num_actions > 1),
        "constant_action": bool(args.constant_action),
        "cfg_w": args.cfg_w,
        "fm_n_steps": args.fm_n_steps,
        "fm_solver": args.fm_solver,
        "calibrate_moments": bool(args.calibrate_moments),
        "calibration": calibration_info or {},
    }
    info_path = args.output.with_suffix(".json")
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
