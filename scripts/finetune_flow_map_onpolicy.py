#!/usr/bin/env python3
"""On-policy one-step correction for a joint flow-map distilled student.

The student remains a single-step MeanFlow/flow-map checkpoint.  Each update:

1. rolls out the current student for a short horizon with gradients disabled;
2. samples conditions from the student's own state distribution;
3. asks the frozen joint-FM teacher for the endpoint from the same noise;
4. trains the student one-step endpoint to match that teacher endpoint.

This targets autoregressive distribution shift without introducing a multi-step
student or a progressive distillation chain.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.inference import sample_action_schedule
from finflow.models import MeanFlowStudent
from finflow.models.transition_fm import euler_sample
from finflow.training import (
    build_batch_loader,
    build_joint_datasets,
    build_run_dir,
    load_checkpoint,
    load_metadata,
    load_model_from_checkpoint,
    load_normalization,
    load_num_actions,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from scripts.distill_flow_map import flow_map_loss
from scripts.finetune_flow_map_pricing import (
    _reference_price_grid,
    pricing_loss_from_grid,
    rollout_price_grid_torch,
)


def _strip_stage(stage: str) -> str:
    if stage.startswith("mf_") or stage.startswith("cd_"):
        return stage[3:]
    return stage


def _metadata_dt(metadata: dict[str, Any]) -> float:
    n_steps = int(metadata.get("n_steps", 252))
    return float(metadata.get("dt", 1.0 / n_steps))


def _student_one_step(
    student: MeanFlowStudent,
    condition: torch.Tensor,
    noise: torch.Tensor,
    *,
    cfg_w: float,
    num_actions: int,
) -> torch.Tensor:
    batch = condition.shape[0]
    r = torch.zeros(batch, device=condition.device, dtype=condition.dtype)
    t = torch.ones(batch, device=condition.device, dtype=condition.dtype)
    u = student(noise, r, t, condition)
    if cfg_w > 0.0:
        uncond = condition.clone()
        uncond[:, -num_actions:] = 0.0
        u_uncond = student(noise, r, t, uncond)
        u = (1.0 + cfg_w) * u - cfg_w * u_uncond
    return noise - u


def _make_actions(
    *,
    n_paths: int,
    n_steps: int,
    num_actions: int,
    metadata: dict[str, Any],
    rng: np.random.Generator,
    constant_action: bool,
) -> np.ndarray:
    transition_matrix = None
    if metadata.get("regime_switching") and not constant_action and num_actions > 1:
        transition_matrix = np.asarray(metadata["transition_matrix"], dtype=np.float64)
    return sample_action_schedule(
        n_paths=n_paths,
        n_steps=n_steps,
        num_actions=num_actions,
        transition_matrix=transition_matrix,
        initial_regime=int(metadata.get("initial_regime", 0)),
        seed=int(rng.integers(0, 2**31 - 1)),
        constant=constant_action,
    )


@torch.no_grad()
def sample_onpolicy_conditions(
    student: MeanFlowStudent,
    *,
    normalization: dict[str, float],
    metadata: dict[str, Any],
    num_actions: int,
    include_prev_return: bool,
    n_paths: int,
    horizon: int,
    samples_per_path: int,
    initial_v: float,
    initial_r_prev: float,
    rng: np.random.Generator,
    cfg_w: float = 0.0,
    constant_action: bool = False,
    state_clamp: float = 0.0,
) -> torch.Tensor:
    """Sample conditions from a detached rollout of the current student."""

    if n_paths <= 0 or horizon <= 0 or samples_per_path <= 0:
        raise ValueError("n_paths, horizon, and samples_per_path must be positive")
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    actions_np = _make_actions(
        n_paths=n_paths,
        n_steps=horizon,
        num_actions=num_actions,
        metadata=metadata,
        rng=rng,
        constant_action=constant_action,
    )
    actions = torch.as_tensor(actions_np, device=device, dtype=torch.long)
    capture = torch.as_tensor(
        rng.integers(0, horizon, size=(n_paths, samples_per_path)),
        device=device,
        dtype=torch.long,
    )

    log_v_mean = float(normalization["log_v_mean"])
    log_v_std = float(normalization["log_v_std"])
    return_mean = float(normalization["return_mean"])
    return_std = float(normalization["return_std"])
    log_v0 = (math.log(float(initial_v)) - log_v_mean) / log_v_std
    r_prev0 = (float(initial_r_prev) - return_mean) / return_std
    log_v_t = torch.full((n_paths, 1), log_v0, device=device, dtype=dtype)
    r_prev_t = torch.full((n_paths, 1), r_prev0, device=device, dtype=dtype)

    out: list[torch.Tensor] = []
    for step in range(horizon):
        a_onehot = F.one_hot(actions[:, step], num_classes=num_actions).to(dtype=dtype)
        condition_parts = [log_v_t]
        if include_prev_return:
            condition_parts.append(r_prev_t)
        condition_parts.append(a_onehot)
        condition = torch.cat(condition_parts, dim=-1)
        mask = capture == step
        if mask.any():
            rows = mask.nonzero(as_tuple=False)[:, 0]
            out.append(condition.index_select(0, rows).detach())
        noise = torch.randn(n_paths, 2, device=device, dtype=dtype)
        next_state = _student_one_step(
            student, condition, noise, cfg_w=cfg_w, num_actions=num_actions,
        )
        log_v_t = next_state[:, 0:1]
        r_prev_t = next_state[:, 1:2]
        if state_clamp > 0:
            log_v_t = log_v_t.clamp(-state_clamp, state_clamp)
            r_prev_t = r_prev_t.clamp(-state_clamp, state_clamp)

    if not out:
        raise RuntimeError("no on-policy conditions were sampled")
    return torch.cat(out, dim=0)


def _next_batch(loader, state: dict[str, Any]) -> dict[str, torch.Tensor]:
    iterator = state.get("iterator")
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    state["iterator"] = iterator
    return batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=80)
    p.add_argument("--path-batch-size", type=int, default=1024)
    p.add_argument("--rollout-horizon", type=int, default=64)
    p.add_argument("--samples-per-path", type=int, default=1)
    p.add_argument("--teacher-n-steps", type=int, default=30)
    p.add_argument("--transition-batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--endpoint-weight", type=float, default=1.0)
    p.add_argument("--flowmap-weight", type=float, default=0.2)
    p.add_argument("--pricing-weight", type=float, default=0.0)
    p.add_argument("--pricing-every", type=int, default=0)
    p.add_argument("--pricing-paths", type=int, default=512)
    p.add_argument("--pricing-val-paths", type=int, default=4096)
    p.add_argument("--mc-oracle", type=Path, default=None)
    p.add_argument("--moneynesses", nargs="+", type=float, default=[0.85, 0.90, 0.95, 1.00, 1.05])
    p.add_argument("--maturities", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    p.add_argument("--pricing-r", type=float, default=0.0)
    p.add_argument("--price-floor", type=float, default=1.0)
    p.add_argument("--time-eps", type=float, default=1e-3)
    p.add_argument("--boundary-prob", type=float, default=0.10)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--state-clamp", type=float, default=0.0)
    p.add_argument("--initial-v", type=float, default=None)
    p.add_argument("--initial-s", type=float, default=None)
    p.add_argument("--initial-r-prev", type=float, default=0.0)
    p.add_argument("--constant-action", action="store_true")
    p.add_argument("--cfg-w", type=float, default=0.0)
    p.add_argument("--cache-data-device", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.endpoint_weight <= 0:
        raise ValueError("--endpoint-weight must be positive")
    if args.flowmap_weight < 0 or args.pricing_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if args.pricing_weight > 0 and args.pricing_every <= 0:
        raise ValueError("--pricing-every must be positive when --pricing-weight > 0")

    set_seed(args.seed)
    device = resolve_device(args.device)
    metadata = load_metadata(args.data_dir)
    normalization = load_normalization(args.data_dir)
    num_actions = load_num_actions(args.data_dir)
    dt = _metadata_dt(metadata)
    n_steps = int(metadata.get("n_steps", 252))
    initial_v = float(args.initial_v if args.initial_v is not None else metadata.get("v0", 0.04))
    initial_s = float(args.initial_s if args.initial_s is not None else metadata.get("s0", 100.0))

    init_ckpt = load_checkpoint(args.init_checkpoint, map_location=device)
    if _strip_stage(str(init_ckpt.get("stage", ""))) != "joint":
        raise ValueError(f"init checkpoint must be joint, got stage={init_ckpt.get('stage')}")
    config = init_ckpt["model_config"]
    student = MeanFlowStudent(
        state_dim=int(config["state_dim"]),
        condition_dim=int(config["condition_dim"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        time_embedding_dim=int(config.get("time_embedding_dim", 64)),
        num_blocks=int(config.get("num_blocks", 4)),
    ).to(device)
    student.load_state_dict(init_ckpt["model_state"])
    if student.state_dim != 2:
        raise ValueError("this script currently expects joint state_dim=2")

    full_cond = 2 + num_actions
    minimal_cond = 1 + num_actions
    if student.condition_dim == full_cond:
        include_prev_return = True
    elif student.condition_dim == minimal_cond:
        include_prev_return = False
    else:
        raise ValueError(f"student condition_dim={student.condition_dim}, expected {full_cond} or {minimal_cond}")

    teacher, teacher_ckpt = load_model_from_checkpoint(args.teacher_checkpoint, map_location=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if _strip_stage(str(teacher_ckpt.get("stage", ""))) != "joint":
        raise ValueError(f"teacher checkpoint must be joint, got stage={teacher_ckpt.get('stage')}")
    if teacher.condition_dim != student.condition_dim or teacher.state_dim != student.state_dim:
        raise ValueError("teacher and student dimensions must match")

    train_loader = None
    loader_state: dict[str, Any] = {}
    if args.flowmap_weight > 0:
        datasets = build_joint_datasets(
            args.data_dir,
            normalization,
            num_actions,
            include_prev_return=include_prev_return,
        )
        train_loader = build_batch_loader(
            datasets["train"],
            batch_size=args.transition_batch_size,
            shuffle=True,
            num_workers=0,
            device=device,
            cache_on_device=args.cache_data_device,
        )

    reference_prices = None
    moneynesses = None
    maturities = None
    if args.pricing_weight > 0:
        oracle = args.mc_oracle or (args.data_dir / "mc_oracle.npz")
        reference = _reference_price_grid(
            oracle,
            dt=dt,
            moneynesses=args.moneynesses,
            maturities=args.maturities,
            s0=initial_s,
            r=args.pricing_r,
            limit=None,
        )
        reference_prices = torch.as_tensor(reference["prices"], device=device, dtype=torch.float32)
        moneynesses = torch.as_tensor(args.moneynesses, device=device, dtype=torch.float32)
        maturities = torch.as_tensor(args.maturities, device=device, dtype=torch.float32)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_dir = build_run_dir(args.output_dir, run_name=args.run_name, prefix="flowmap_onpolicy")
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    train_config = {
        "method": "onpolicy_flowmap_teacher_endpoint",
        "data_dir": str(args.data_dir),
        "init_checkpoint": str(args.init_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "model_config": config,
        "num_actions": num_actions,
        "include_prev_return": include_prev_return,
        "normalization": normalization,
        "dt": dt,
        "n_steps": n_steps,
        "initial_v": initial_v,
        "initial_s": initial_s,
        "args": vars(args) | {
            "data_dir": str(args.data_dir),
            "init_checkpoint": str(args.init_checkpoint),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "output_dir": str(args.output_dir),
            "mc_oracle": str(args.mc_oracle) if args.mc_oracle else None,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    print(
        f"[flowmap-onpolicy] run={run_dir.name} paths={args.path_batch_size} "
        f"horizon={args.rollout_horizon} teacher_nfe={args.teacher_n_steps} "
        f"flow_w={args.flowmap_weight} price_w={args.pricing_weight}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 31415)
    best = float("inf")
    global_step = 0
    t0 = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        student.train()
        totals = {
            "loss": 0.0,
            "endpoint_loss": 0.0,
            "flowmap_loss": 0.0,
            "pricing_loss": 0.0,
        }
        for local_step in range(1, args.steps_per_epoch + 1):
            with torch.no_grad():
                onpolicy_condition = sample_onpolicy_conditions(
                    student,
                    normalization=normalization,
                    metadata=metadata,
                    num_actions=num_actions,
                    include_prev_return=include_prev_return,
                    n_paths=args.path_batch_size,
                    horizon=args.rollout_horizon,
                    samples_per_path=args.samples_per_path,
                    initial_v=initial_v,
                    initial_r_prev=args.initial_r_prev,
                    rng=rng,
                    cfg_w=args.cfg_w,
                    constant_action=args.constant_action,
                    state_clamp=args.state_clamp,
                )
                noise = torch.randn(
                    onpolicy_condition.shape[0],
                    student.state_dim,
                    device=device,
                    dtype=onpolicy_condition.dtype,
                )
                teacher_target = euler_sample(
                    teacher,
                    condition=onpolicy_condition,
                    n_steps=args.teacher_n_steps,
                    noise=noise,
                )

            opt.zero_grad(set_to_none=True)
            pred = _student_one_step(
                student,
                onpolicy_condition,
                noise,
                cfg_w=args.cfg_w,
                num_actions=num_actions,
            )
            endpoint_loss = F.mse_loss(pred, teacher_target)
            fm_loss = pred.new_tensor(0.0)
            if args.flowmap_weight > 0 and train_loader is not None:
                batch = _next_batch(train_loader, loader_state)
                cond = batch["condition"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)
                fm_loss = flow_map_loss(
                    student,
                    teacher,
                    cond,
                    tgt,
                    time_eps=args.time_eps,
                    boundary_prob=args.boundary_prob,
                )
            pricing_loss = pred.new_tensor(0.0)
            if (
                args.pricing_weight > 0
                and reference_prices is not None
                and moneynesses is not None
                and maturities is not None
                and global_step % args.pricing_every == 0
            ):
                actions_np = _make_actions(
                    n_paths=args.pricing_paths,
                    n_steps=n_steps,
                    num_actions=num_actions,
                    metadata=metadata,
                    rng=rng,
                    constant_action=args.constant_action,
                )
                fake_prices = rollout_price_grid_torch(
                    student,
                    normalization=normalization,
                    actions=torch.as_tensor(actions_np, device=device),
                    moneynesses=moneynesses,
                    maturities=maturities,
                    dt=dt,
                    initial_v=initial_v,
                    initial_s=initial_s,
                    initial_r_prev=args.initial_r_prev,
                    r=args.pricing_r,
                    include_prev_return=include_prev_return,
                    cfg_w=args.cfg_w,
                )
                pricing_loss, _, _ = pricing_loss_from_grid(
                    fake_prices,
                    reference_prices,
                    price_floor=args.price_floor,
                )

            loss = (
                args.endpoint_weight * endpoint_loss
                + args.flowmap_weight * fm_loss
                + args.pricing_weight * pricing_loss
            )
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip_norm)
            opt.step()
            global_step += 1
            totals["loss"] += float(loss.detach().item())
            totals["endpoint_loss"] += float(endpoint_loss.detach().item())
            totals["flowmap_loss"] += float(fm_loss.detach().item())
            totals["pricing_loss"] += float(pricing_loss.detach().item())

        student.eval()
        with torch.no_grad():
            val_condition = sample_onpolicy_conditions(
                student,
                normalization=normalization,
                metadata=metadata,
                num_actions=num_actions,
                include_prev_return=include_prev_return,
                n_paths=args.path_batch_size,
                horizon=args.rollout_horizon,
                samples_per_path=args.samples_per_path,
                initial_v=initial_v,
                initial_r_prev=args.initial_r_prev,
                rng=np.random.default_rng(args.seed + 100_000 + epoch),
                cfg_w=args.cfg_w,
                constant_action=args.constant_action,
                state_clamp=args.state_clamp,
            )
            val_noise = torch.randn(
                val_condition.shape[0],
                student.state_dim,
                device=device,
                dtype=val_condition.dtype,
            )
            val_target = euler_sample(
                teacher,
                condition=val_condition,
                n_steps=args.teacher_n_steps,
                noise=val_noise,
            )
            val_pred = _student_one_step(
                student,
                val_condition,
                val_noise,
                cfg_w=args.cfg_w,
                num_actions=num_actions,
            )
            val_endpoint = F.mse_loss(val_pred, val_target)

        denom = max(args.steps_per_epoch, 1)
        rec = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": totals["loss"] / denom,
            "train_endpoint_loss": totals["endpoint_loss"] / denom,
            "train_flowmap_loss": totals["flowmap_loss"] / denom,
            "train_pricing_loss": totals["pricing_loss"] / denom,
            "val_endpoint_loss": float(val_endpoint.item()),
            "elapsed_s": time.monotonic() - t0,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        is_best = rec["val_endpoint_loss"] < best
        if is_best:
            best = rec["val_endpoint_loss"]
        extra = {
            "kind": "mean_flow",
            "method": "onpolicy_flowmap_teacher_endpoint",
            "init_checkpoint": str(args.init_checkpoint),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "teacher_n_steps": args.teacher_n_steps,
            "rollout_horizon": args.rollout_horizon,
            "endpoint_weight": args.endpoint_weight,
            "flowmap_weight": args.flowmap_weight,
            "pricing_weight": args.pricing_weight,
            "val_endpoint_loss": rec["val_endpoint_loss"],
        }
        save_checkpoint(
            ckpt_dir / "last.pt",
            student,
            opt,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best,
            model_config=config,
            train_config=train_config,
            normalization=normalization,
            stage="mf_joint",
            num_actions=num_actions,
            extra=extra,
        )
        if is_best:
            save_checkpoint(
                ckpt_dir / "best.pt",
                student,
                opt,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best,
                model_config=config,
                train_config=train_config,
                normalization=normalization,
                stage="mf_joint",
                num_actions=num_actions,
                extra=extra,
            )
        print(
            f"  epoch {epoch}/{args.epochs} endpoint={rec['train_endpoint_loss']:.6f} "
            f"flowmap={rec['train_flowmap_loss']:.6f} val={rec['val_endpoint_loss']:.6f} "
            f"best={best:.6f}{' *' if is_best else ''} elapsed={rec['elapsed_s']:.0f}s",
            flush=True,
        )

    summary = {
        "run_dir": str(run_dir),
        "stage": "mf_joint",
        "method": "onpolicy_flowmap_teacher_endpoint",
        "best_val_endpoint_loss": best,
        "checkpoints": {
            "best": str(ckpt_dir / "best.pt"),
            "last": str(ckpt_dir / "last.pt"),
        },
        "total_time_s": time.monotonic() - t0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[flowmap-onpolicy] done {run_dir}", flush=True)


if __name__ == "__main__":
    main()
