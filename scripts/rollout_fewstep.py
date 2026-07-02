#!/usr/bin/env python3
"""Few-step (NFE-sweep) rollout for distilled two-time students (flow-map / MeanFlow).

A flow-map / MeanFlow student parameterizes  X(x,s,t) = x - (t-s)*G(x,s,t).
K-step generation composes the map over K uniform sub-intervals of [0,1]:

    x = z;  for i in 0..K-1:  x <- x - (t_{i+1}-t_i) * G(x, t_i, t_{i+1}, c)

K=1 reproduces the standard 1-NFE sample (z - G(z,0,1)). Larger K trades compute
for quality. This lets us sweep NFE in {1,2,4,8,...} from the SAME checkpoint
(no retraining) and find the quality/speed knee.

Works for any checkpoint whose net is a two-time student (kind=mean_flow, incl.
the Lagrangian flow-map). Consistency (kind=consistency) uses a different
multistep procedure and is not handled here.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.data import DEFAULT_TRANSITION_MATRIX
from finflow.inference import autoregressive_rollout
from finflow.inference.samplers import Sampler, LambertWInverseSampler, _ensure_noise, _unconditional_condition
from finflow.models import MeanFlowStudent
from finflow.training import load_checkpoint, load_metadata, resolve_device


class FewStepFlowMapSampler(Sampler):
    """K-step composition of a two-time flow-map / MeanFlow student."""
    kind = "flow_map"

    def __init__(self, model: MeanFlowStudent, n_steps: int, num_actions=None):
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.model = model
        self.n_steps = int(n_steps)
        self.state_dim = model.state_dim
        self.condition_dim = model.condition_dim
        self.device = next(model.parameters()).device
        self.num_actions = num_actions

    @torch.no_grad()
    def sample(self, condition, *, noise=None, cfg_w: float = 0.0):
        condition = condition.to(self.device)
        uncond = _unconditional_condition(condition, self.num_actions, cfg_w)
        B = condition.shape[0]
        dtype = condition.dtype
        x = _ensure_noise(noise, B, self.state_dim, self.device, dtype)
        ts = torch.linspace(0.0, 1.0, self.n_steps + 1, device=self.device, dtype=dtype)
        for i in range(self.n_steps):
            s = ts[i].expand(B)
            t = ts[i + 1].expand(B)
            g = self.model(x, s, t, condition)
            if uncond is not None:
                g_u = self.model(x, s, t, uncond)
                g = (1.0 + cfg_w) * g - cfg_w * g_u
            x = x - (t - s).unsqueeze(-1) * g
        return x


def _load_student(path, device):
    ckpt = load_checkpoint(path, map_location=device)
    cfg = ckpt["model_config"]
    model = MeanFlowStudent(
        state_dim=int(cfg["state_dim"]), condition_dim=int(cfg["condition_dim"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        time_embedding_dim=int(cfg.get("time_embedding_dim", 64)),
        num_blocks=int(cfg.get("num_blocks", 4)),
    )
    model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
    extra = ckpt.get("extra", {})
    return model, ckpt, float(extra.get("lambert_w_delta", 0.0) or 0.0)


def _calibrate(r_paths, initial_s, return_mean, return_std, eps=1e-6):
    from finflow.baselines.quant_gan import calibrate_standardized_moments
    std, info = calibrate_standardized_moments(r_paths.reshape(-1), eps=eps)
    r = std.reshape(r_paths.shape).astype(np.float64) * return_std + return_mean
    s_tail = float(initial_s) * np.exp(np.cumsum(r, axis=1))
    s0 = np.full((s_tail.shape[0], 1), float(initial_s), dtype=s_tail.dtype)
    return r.astype(np.float32), np.concatenate([s0, s_tail], axis=1).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vol-checkpoint", type=Path, required=True)
    p.add_argument("--ret-checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--student-steps", type=int, default=1, help="K = NFE per transition")
    p.add_argument("--n-paths", type=int, default=5000)
    p.add_argument("--n-steps", type=int, default=252)
    p.add_argument("--initial-v", type=float, default=0.04)
    p.add_argument("--initial-s", type=float, default=100.0)
    p.add_argument("--regime-actions", action="store_true")
    p.add_argument("--calibrate-moments", action="store_true")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--action-seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    device = resolve_device(a.device)
    vol_model, vol_ckpt, vol_delta = _load_student(a.vol_checkpoint, device)
    ret_model, ret_ckpt, _ = _load_student(a.ret_checkpoint, device)
    num_actions = int(vol_ckpt.get("num_actions", 1))
    normalization = vol_ckpt.get("normalization") or ret_ckpt.get("normalization")

    vol_sampler: Sampler = FewStepFlowMapSampler(vol_model, a.student_steps, num_actions=num_actions)
    if vol_delta > 0.0:
        vol_sampler = LambertWInverseSampler(vol_sampler, delta=vol_delta)
    ret_sampler: Sampler = FewStepFlowMapSampler(ret_model, a.student_steps, num_actions=num_actions)

    metadata = load_metadata(a.data_dir) if a.data_dir.exists() else {}
    if a.regime_actions and num_actions > 1:
        tm = np.asarray(metadata.get("transition_matrix", DEFAULT_TRANSITION_MATRIX), dtype=np.float64)
    else:
        tm = None

    result = autoregressive_rollout(
        vol_sampler=vol_sampler, ret_sampler=ret_sampler, normalization=normalization,
        n_paths=a.n_paths, n_steps=a.n_steps, num_actions=num_actions,
        initial_v=a.initial_v, initial_s=a.initial_s, initial_r_prev=0.0,
        transition_matrix=tm, initial_regime=0, action_seed=a.action_seed,
        noise_seed=a.noise_seed, device=device, constant_action=False, cfg_w=0.0,
    )
    r_paths, s_paths = result.r_paths, result.s_paths
    if a.calibrate_moments:
        r_paths, s_paths = _calibrate(result.r_paths, result.initial_s,
                                      float(normalization["return_mean"]), float(normalization["return_std"]))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, log_v_paths=result.log_v_paths, v_paths=result.v_paths,
             r_paths=r_paths, s_paths=s_paths, actions=result.actions)
    info = {"output": str(a.output), "student_steps": a.student_steps, "n_paths": a.n_paths,
            "calibrate_moments": bool(a.calibrate_moments), "vol_lambert_w_delta": vol_delta}
    a.output.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
