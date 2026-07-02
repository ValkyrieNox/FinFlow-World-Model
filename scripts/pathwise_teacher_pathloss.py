#!/usr/bin/env python3
"""Ablation harness: pathwise teacher fine-tune with a strictly-proper path
distribution loss INSTEAD of the WGAN-GP critic. One --path-loss flag selects:

  sig_mmd : RBF-kernel MMD on truncated path signatures   (paper 1, Lu&Sester / sig-kernel score)
  sig_w1  : match expected signatures (linear sig kernel)  (paper 2, Conditional Sig-WGAN / Sig-W1)
  energy  : energy distance on standardized return paths    (paper 4, energy score / Pacchiardi)

Everything else (differentiable rollout, moment/terminal/abs/kurtosis/anchor
auxiliary losses, lr, freeze-vol) is identical to the 0.420 `pathwise_retonly`
baseline, so each run is a clean one-change ablation vs the WGAN-GP version.
"""
from __future__ import annotations

import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.pathwise_teacher import (
    PathwiseTeacherFineTuneConfig, ReturnPathBatcher, _differentiable_rollout_norm,
    _onehot_sequence, _path_moment_loss, _anchor_loss, _set_trainable, _save_finetuned_checkpoint,
)
from finflow.training import (
    build_run_dir, load_metadata, load_model_from_checkpoint,
    load_normalization, resolve_device, set_seed,
)


def signature_features(returns_norm: torch.Tensor, depth: int = 3) -> torch.Tensor:
    """[B,T] standardized returns -> [B,D] truncated signature of (time, cumret) path."""
    B, T = returns_norm.shape
    dev, dt_ = returns_norm.device, returns_norm.dtype
    dx = torch.cat([torch.full((B, T, 1), 1.0 / T, device=dev, dtype=dt_),
                    returns_norm.unsqueeze(-1)], dim=-1)
    S1 = torch.zeros(B, 2, device=dev, dtype=dt_)
    S2 = torch.zeros(B, 2, 2, device=dev, dtype=dt_)
    S3 = torch.zeros(B, 2, 2, 2, device=dev, dtype=dt_)
    for t in range(T):
        b1 = dx[:, t, :]
        b2 = 0.5 * torch.einsum('bi,bj->bij', b1, b1)
        if depth >= 3:
            b3 = (1.0 / 6.0) * torch.einsum('bi,bj,bk->bijk', b1, b1, b1)
            S3 = S3 + torch.einsum('bij,bk->bijk', S2, b1) + torch.einsum('bi,bjk->bijk', S1, b2) + b3
        S2 = S2 + torch.einsum('bi,bj->bij', S1, b1) + b2
        S1 = S1 + b1
    feats = [S1.reshape(B, -1), S2.reshape(B, -1)]
    if depth >= 3:
        feats.append(S3.reshape(B, -1))
    return torch.cat(feats, dim=1)


def _pdist2(a, b):
    return (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1).unsqueeze(0) - 2.0 * a @ b.t()).clamp_min(0.0)


def _standardize(real, fake):
    mu = real.mean(0, keepdim=True).detach()
    sd = real.std(0, keepdim=True).detach().clamp_min(1e-6)
    return (real - mu) / sd, (fake - mu) / sd


def loss_sig_mmd(real_feat, fake_feat):
    fr, ff = _standardize(real_feat, fake_feat)
    dxx, dyy, dxy = _pdist2(fr, fr), _pdist2(ff, ff), _pdist2(fr, ff)
    with torch.no_grad():
        med = dxy.flatten().median().clamp_min(1e-6)
    loss = fr.new_zeros(())
    for s in (0.5, 1.0, 2.0, 4.0, 8.0):
        g = 1.0 / (2.0 * s * med)
        loss = loss + torch.exp(-g * dxx).mean() + torch.exp(-g * dyy).mean() - 2.0 * torch.exp(-g * dxy).mean()
    return loss / 5.0


def loss_sig_w1(real_feat, fake_feat):
    fr, ff = _standardize(real_feat, fake_feat)
    return (fr.mean(0) - ff.mean(0)).pow(2).sum()


def loss_energy(real_path, fake_path):
    # energy distance on standardized return paths [B,T]
    def pdist(a, b):
        return _pdist2(a, b).clamp_min(1e-12).sqrt()
    return 2.0 * pdist(real_path, fake_path).mean() - pdist(real_path, real_path).mean() - pdist(fake_path, fake_path).mean()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vol-checkpoint", type=Path, required=True)
    p.add_argument("--ret-checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--path-loss", choices=["sig_mmd", "sig_w1", "energy", "none"], default="sig_mmd")
    p.add_argument("--path-loss-weight", type=float, default=10.0)
    p.add_argument("--sig-depth", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=240)
    p.add_argument("--fm-n-steps", type=int, default=4)
    p.add_argument("--lr-teacher", type=float, default=5e-6)
    p.add_argument("--moment-weight", type=float, default=1.0)
    p.add_argument("--terminal-weight", type=float, default=1.0)
    p.add_argument("--abs-sum-weight", type=float, default=0.25)
    p.add_argument("--kurtosis-weight", type=float, default=0.1)
    p.add_argument("--anchor-weight", type=float, default=1e-6)
    p.add_argument("--freeze-vol", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = PathwiseTeacherFineTuneConfig(
        batch_size=a.batch_size, epochs=a.epochs, steps_per_epoch=a.steps_per_epoch,
        fm_n_steps=a.fm_n_steps, lr_teacher=a.lr_teacher, moment_weight=a.moment_weight,
        terminal_weight=a.terminal_weight, abs_sum_weight=a.abs_sum_weight,
        kurtosis_weight=a.kurtosis_weight, anchor_weight=a.anchor_weight,
        train_vol=not a.freeze_vol, train_ret=True, seed=a.seed, device=a.device,
    )
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    vol_model, vol_ckpt = load_model_from_checkpoint(a.vol_checkpoint, map_location=device)
    ret_model, ret_ckpt = load_model_from_checkpoint(a.ret_checkpoint, map_location=device)
    num_actions = int(vol_ckpt.get("num_actions", 1))
    normalization = vol_ckpt.get("normalization") or ret_ckpt.get("normalization") or load_normalization(a.data_dir)
    load_metadata(a.data_dir)
    vol_delta = float(vol_ckpt.get("extra", {}).get("lambert_w_delta", 0.0) or 0.0)
    _set_trainable(vol_model, cfg.train_vol); vol_model.train(cfg.train_vol)
    _set_trainable(ret_model, cfg.train_ret); ret_model.train(cfg.train_ret)
    models = [vol_model, ret_model]
    anchors = {f"{i}:{n}": p.detach().clone() for i, m in enumerate(models)
               for n, p in m.named_parameters() if p.requires_grad}
    params = [p for m in models for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr_teacher)
    batcher = ReturnPathBatcher(a.data_dir, n_steps=cfg.n_steps, normalization=normalization, device=device)
    run_dir = build_run_dir(a.output_dir, run_name=a.run_name, prefix="pathloss")
    ckpt_dir = run_dir / "checkpoints"; metrics_path = run_dir / "metrics.jsonl"
    (run_dir / "config.json").write_text(json.dumps({
        "config": asdict(cfg), "path_loss": a.path_loss, "path_loss_weight": a.path_loss_weight,
        "sig_depth": a.sig_depth, "vol_lambert_w_delta": vol_delta,
        "vol_checkpoint": str(a.vol_checkpoint), "ret_checkpoint": str(a.ret_checkpoint)}, indent=2), encoding="utf-8")

    best = float("inf"); gstep = 0; t0 = time.monotonic()
    for epoch in range(1, cfg.epochs + 1):
        agg = {"generator_loss": 0.0, "path_loss": 0.0, "moment_loss": 0.0}
        for _ in range(cfg.steps_per_epoch):
            real_norm, actions = batcher.sample(cfg.batch_size)
            af = _onehot_sequence(actions, num_actions, next(ret_model.parameters()).dtype)
            fake_norm = _differentiable_rollout_norm(
                vol_model, ret_model, actions, normalization=normalization, num_actions=num_actions,
                initial_v=cfg.initial_v, initial_r_prev=cfg.initial_r_prev,
                fm_n_steps=cfg.fm_n_steps, vol_lambert_w_delta=vol_delta, action_features=af)
            if a.path_loss == "sig_mmd":
                pl = loss_sig_mmd(signature_features(real_norm, a.sig_depth), signature_features(fake_norm, a.sig_depth))
            elif a.path_loss == "sig_w1":
                pl = loss_sig_w1(signature_features(real_norm, a.sig_depth), signature_features(fake_norm, a.sig_depth))
            elif a.path_loss == "energy":
                pl = loss_energy(real_norm, fake_norm)
            else:
                pl = fake_norm.new_zeros(())
            moment = _path_moment_loss(fake_norm, real_norm, cfg)
            anchor = _anchor_loss(models, anchors) if cfg.anchor_weight > 0 else fake_norm.new_tensor(0.0)
            loss = a.path_loss_weight * pl + moment + cfg.anchor_weight * anchor
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            gstep += 1
            agg["generator_loss"] += float(loss.detach().cpu())
            agg["path_loss"] += float(pl.detach().cpu())
            agg["moment_loss"] += float(moment.detach().cpu())
        rec = {"epoch": epoch, "global_step": gstep,
               **{k: v / cfg.steps_per_epoch for k, v in agg.items()}, "elapsed_s": time.monotonic() - t0}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        score = rec["generator_loss"]; is_best = score < best
        if is_best: best = score
        for name, model, src, stage in (("vol", vol_model, vol_ckpt, "vol"), ("ret", ret_model, ret_ckpt, "ret")):
            extra = {"pathwise_finetuned": True, "loss": a.path_loss, "kind": "fm", "pathwise_score": score}
            _save_finetuned_checkpoint(ckpt_dir / f"{name}_last.pt", model, opt, source_checkpoint=src,
                stage=stage, num_actions=num_actions, config=cfg, epoch=epoch, global_step=gstep, score=score, extra=extra)
            if is_best:
                _save_finetuned_checkpoint(ckpt_dir / f"{name}_best.pt", model, opt, source_checkpoint=src,
                    stage=stage, num_actions=num_actions, config=cfg, epoch=epoch, global_step=gstep, score=score, extra=extra)
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"[{a.path_loss}] epoch {epoch}/{cfg.epochs} G={rec['generator_loss']:.5f} "
              f"PL={rec['path_loss']:.5f} Mom={rec['moment_loss']:.5f} peakGB={mem:.2f} "
              f"elapsed={rec['elapsed_s']:.0f}s", flush=True)
    (run_dir / "summary.json").write_text(json.dumps({"run_dir": str(run_dir), "best_score": best,
        "checkpoints": {k: str(ckpt_dir / f"{k}.pt") for k in ("vol_best", "vol_last", "ret_best", "ret_last")},
        "total_time_s": time.monotonic() - t0}, indent=2), encoding="utf-8")
    print(f"[{a.path_loss}] done {run_dir}", flush=True)


if __name__ == "__main__":
    main()
