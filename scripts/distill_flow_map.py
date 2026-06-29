#!/usr/bin/env python3
"""Lagrangian flow-map self-distillation (Boffi, Albergo, Vanden-Eijnden,
"How to build a consistency model: learning flow maps via self-distillation").

We learn a flow map X(x, s, t, c) that transports an FM-ODE state from time s to
time t. Parameterized as  X = x_s - (t-s) * G_theta(x_s, s, t, c)  so the
1-NFE sample  z - G_theta(z, 0, 1, c)  reuses the existing MeanFlow sampler.

LAGRANGIAN objective: enforce  d/dt X = v_teacher(X, t)  (velocity evaluated at
the map's OWN output X -- "following the particle"), which avoids spatial
derivatives and small-step bootstrapping:

    d/dt X = -G - (t-s) dG/dt   ==>   target_G = -v_teacher(X, t) - (t-s) sg(dG/dt)
    L = || G_theta - target_G ||^2

dG/dt is a forward-mode JVP w.r.t. t (torch.func.jvp). The teacher is the frozen
FM velocity; we evaluate it at X under no_grad (no backprop through teacher input
== the "no spatial derivative" property). A fraction of the batch uses s==t which
anchors  G(x,t,t,c) = -v_teacher(x,t)  (instantaneous velocity).

Checkpoint is saved with kind="mean_flow" (+ extra.method="lagrangian_flowmap")
so inference/rollout reuse MeanFlowSampler unchanged.
"""
from __future__ import annotations

import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.models import MeanFlowStudent, warm_start_mean_flow_from_fm
from finflow.training import (
    TwoStageFMModelConfig, build_vol_datasets, build_ret_datasets, build_joint_datasets,
    build_batch_loader,
    build_run_dir, load_model_from_checkpoint, load_normalization, load_num_actions,
    resolve_device, save_checkpoint, set_seed, _iterate_batches, _effective_num_batches,
)


def flow_map_loss(student, teacher, condition, target, *, time_eps, boundary_prob):
    B = target.shape[0]; dev = target.device; dt = target.dtype
    u = torch.rand(B, 2, device=dev, dtype=dt)
    s = u.min(dim=1).values.clamp(time_eps, 1.0 - time_eps)
    t = u.max(dim=1).values.clamp(time_eps, 1.0 - time_eps)
    t = torch.maximum(t, s)
    if boundary_prob > 0.0:
        mask = torch.rand(B, device=dev) < boundary_prob
        s = torch.where(mask, t, s)
    noise = torch.randn_like(target)
    s_view = s.reshape(B, *([1] * (target.ndim - 1)))
    x_s = (1.0 - s_view) * noise + s_view * target          # FM interpolant at tau=s (noise->data)

    def fn(t_in):
        return student(x_s, s, t_in, condition)             # G(x_s, s, t, c)
    G, dG_dt = torch.func.jvp(fn, (t,), (torch.ones_like(t),))

    delta = (t - s).reshape(B, *([1] * (target.ndim - 1)))
    X = x_s - delta * G                                      # flow map output
    with torch.no_grad():
        v_tea = teacher(x_tau=X, tau=t, condition=condition) # teacher velocity at the MAPPED point
    target_G = -v_tea - delta * dG_dt.detach()
    return F.mse_loss(G, target_G)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--stage", choices=("vol", "ret", "joint"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--time-eps", type=float, default=1e-3)
    p.add_argument("--boundary-prob", type=float, default=0.25)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--cache-data-device", action="store_true")
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--negate-warm-start", action="store_true",
                   help="after copying FM weights, negate the final output layer so G starts near -v_teacher")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    device = resolve_device(a.device)
    teacher, tck = load_model_from_checkpoint(a.teacher_checkpoint, map_location=device)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    tstage = tck.get("stage", "joint")
    if tstage != a.stage:
        raise ValueError(f"teacher stage={tstage} != requested {a.stage}")
    num_actions = int(tck.get("num_actions", load_num_actions(a.data_dir)))
    normalization = tck.get("normalization") or load_normalization(a.data_dir)
    delta = float(tck.get("extra", {}).get("lambert_w_delta", 0.0) or 0.0)

    include_prev_return = None
    if a.stage == "vol":
        datasets = build_vol_datasets(a.data_dir, normalization, num_actions, lambert_w_delta=delta)
    elif a.stage == "ret":
        datasets = build_ret_datasets(a.data_dir, normalization, num_actions)
    else:
        full_joint_cond = 2 + num_actions
        markov_minimal_cond = 1 + num_actions
        if teacher.condition_dim == full_joint_cond:
            include_prev_return = True
        elif teacher.condition_dim == markov_minimal_cond:
            include_prev_return = False
        else:
            raise ValueError(
                "joint teacher condition_dim must be "
                f"{full_joint_cond} or {markov_minimal_cond}; "
                f"got {teacher.condition_dim}"
            )
        datasets = build_joint_datasets(
            a.data_dir, normalization, num_actions,
            include_prev_return=include_prev_return,
        )

    student_config = TwoStageFMModelConfig(
        state_dim=teacher.state_dim, condition_dim=teacher.condition_dim,
        hidden_dim=teacher.hidden_dim, time_embedding_dim=teacher.time_embedding.embedding_dim,
        num_blocks=len(teacher.blocks),
    )
    student = MeanFlowStudent(**asdict(student_config)).to(device)
    warm = 0 if a.no_warm_start else warm_start_mean_flow_from_fm(student, teacher)
    if a.negate_warm_start and warm > 0:
        for module in reversed(student.output):
            if isinstance(module, torch.nn.Linear):
                module.weight.data.mul_(-1.0)
                if module.bias is not None:
                    module.bias.data.mul_(-1.0)
                break
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    train_loader = build_batch_loader(datasets["train"], batch_size=a.batch_size, shuffle=True,
                                      num_workers=0, device=device, cache_on_device=a.cache_data_device)
    val_loader = build_batch_loader(datasets["val"], batch_size=a.batch_size, shuffle=False,
                                    num_workers=0, device=device, cache_on_device=a.cache_data_device)
    run_dir = build_run_dir(a.output_dir, run_name=a.run_name, prefix=f"flowmap_{a.stage}_distill")
    ckpt_dir = run_dir / "checkpoints"; metrics_path = run_dir / "metrics.jsonl"
    (run_dir / "config.json").write_text(json.dumps({
        "stage": a.stage, "teacher_checkpoint": str(a.teacher_checkpoint), "method": "lagrangian_flowmap",
        "student_config": asdict(student_config), "lambert_w_delta": delta, "warm": warm,
        "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr, "boundary_prob": a.boundary_prob,
        "max_train_batches": a.max_train_batches, "negate_warm_start": a.negate_warm_start,
        "include_prev_return": include_prev_return}, indent=2), encoding="utf-8")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[flowmap] stage={a.stage} run={run_dir.name} params={n_params/1e3:.1f}k warm={warm} "
          f"cache={int(a.cache_data_device)} delta={delta} epochs={a.epochs}", flush=True)

    best = float("inf"); gstep = 0; t0 = time.monotonic()
    for epoch in range(1, a.epochs + 1):
        student.train()
        tr_sum, tr_n = 0.0, 0
        for batch in _iterate_batches(train_loader, a.max_train_batches):
            cond = batch["condition"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = flow_map_loss(student, teacher, cond, tgt, time_eps=a.time_eps, boundary_prob=a.boundary_prob)
            loss.backward()
            if a.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), a.grad_clip_norm)
            opt.step()
            tr_sum += float(loss.item()) * cond.shape[0]; tr_n += cond.shape[0]; gstep += 1
        student.eval()
        va_sum, va_n = 0.0, 0
        for batch in _iterate_batches(val_loader, a.max_val_batches):
            cond = batch["condition"].to(device); tgt = batch["target"].to(device)
            with torch.enable_grad():
                loss = flow_map_loss(student, teacher, cond, tgt, time_eps=a.time_eps, boundary_prob=0.0)
            va_sum += float(loss.item()) * cond.shape[0]; va_n += cond.shape[0]
        tr = tr_sum / max(tr_n, 1); va = va_sum / max(va_n, 1)
        rec = {"epoch": epoch, "train_loss": tr, "val_loss": va, "global_step": gstep,
               "elapsed_s": time.monotonic() - t0}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        is_best = va < best
        extra = {"kind": "mean_flow", "method": "lagrangian_flowmap", "lambert_w_delta": delta,
                 "teacher_checkpoint": str(a.teacher_checkpoint), "train_loss": tr, "val_loss": va}
        save_checkpoint(ckpt_dir / "last.pt", student, opt, epoch=epoch, global_step=gstep,
                        best_val_loss=best, model_config=asdict(student_config),
                        train_config={"method": "lagrangian_flowmap"}, normalization=normalization,
                        stage=f"mf_{a.stage}", num_actions=num_actions, extra=extra)
        if is_best:
            best = va
            save_checkpoint(ckpt_dir / "best.pt", student, opt, epoch=epoch, global_step=gstep,
                            best_val_loss=best, model_config=asdict(student_config),
                            train_config={"method": "lagrangian_flowmap"}, normalization=normalization,
                            stage=f"mf_{a.stage}", num_actions=num_actions, extra=extra)
        print(f"  epoch {epoch}/{a.epochs} train={tr:.5f} val={va:.5f} best={best:.5f}"
              f"{' *' if is_best else ''} elapsed={rec['elapsed_s']:.0f}s", flush=True)
    (run_dir / "summary.json").write_text(json.dumps({"run_dir": str(run_dir), "best_val_loss": best,
        "checkpoints": {"best": str(ckpt_dir / "best.pt"), "last": str(ckpt_dir / "last.pt")},
        "total_time_s": time.monotonic() - t0}, indent=2), encoding="utf-8")
    print(f"[flowmap] done {run_dir}", flush=True)


if __name__ == "__main__":
    main()
