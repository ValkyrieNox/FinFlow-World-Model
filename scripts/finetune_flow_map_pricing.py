#!/usr/bin/env python3
"""Pricing-aware fine-tuning for a joint flow-map distilled student.

This script starts from an existing flow-map/MeanFlow checkpoint and optimizes a
small differentiable MC option-pricing loss, while optionally keeping the
Lagrangian flow-map distillation loss as a regularizer.  The saved checkpoint
keeps kind="mean_flow" and stage="mf_joint" so rollout_joint.py can evaluate it
unchanged.
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

from finflow.eval.pricing import mc_call_prices_grid
from finflow.inference import sample_action_schedule
from finflow.models import MeanFlowStudent
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


def _strip_stage(stage: str) -> str:
    if stage.startswith("mf_") or stage.startswith("cd_"):
        return stage[3:]
    return stage


def _metadata_dt(metadata: dict[str, Any]) -> float:
    n_steps = int(metadata.get("n_steps", 252))
    return float(metadata.get("dt", 1.0 / n_steps))


def _reference_price_grid(
    oracle_path: Path,
    *,
    dt: float,
    moneynesses: list[float],
    maturities: list[float],
    s0: float,
    r: float,
    limit: int | None,
) -> dict[str, np.ndarray]:
    arr = np.load(oracle_path)
    if "s_paths" not in arr.files:
        raise ValueError(f"{oracle_path} must contain s_paths")
    s_paths = np.asarray(arr["s_paths"], dtype=np.float64)
    if limit is not None:
        s_paths = s_paths[:limit]
    out = mc_call_prices_grid(
        s_paths,
        dt=dt,
        moneynesses=moneynesses,
        maturities=maturities,
        s0=s0,
        r=r,
    )
    arr.close()
    return out


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


def _sample_flowmap(
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
        unconditional = condition.clone()
        unconditional[:, -num_actions:] = 0.0
        u_uncond = student(noise, r, t, unconditional)
        u = (1.0 + cfg_w) * u - cfg_w * u_uncond
    return noise - u


def rollout_price_grid_torch(
    student: MeanFlowStudent,
    *,
    normalization: dict[str, float],
    actions: torch.Tensor,
    moneynesses: torch.Tensor,
    maturities: torch.Tensor,
    dt: float,
    initial_v: float,
    initial_s: float,
    initial_r_prev: float,
    r: float,
    include_prev_return: bool,
    cfg_w: float,
    price_chunk_paths: int | None = None,
) -> torch.Tensor:
    """Return differentiable MC call prices with shape [n_maturities, n_strikes]."""

    if actions.ndim != 2:
        raise ValueError("actions must have shape [n_paths, n_steps]")
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    actions = actions.to(device=device)
    n_paths, n_steps = actions.shape
    num_actions = int(student.condition_dim - (2 if include_prev_return else 1))
    if num_actions <= 0:
        raise ValueError("could not infer positive num_actions")

    maturity_indices = torch.round(maturities.to(device=device, dtype=dtype) / float(dt)).long()
    if torch.any(maturity_indices < 1) or torch.any(maturity_indices > n_steps):
        raise ValueError("maturity indices are outside rollout horizon")
    unique_indices = sorted({int(x) for x in maturity_indices.detach().cpu().tolist()})
    idx_to_pos = {idx: pos for pos, idx in enumerate(unique_indices)}

    log_v_mean = float(normalization["log_v_mean"])
    log_v_std = float(normalization["log_v_std"])
    return_mean = float(normalization["return_mean"])
    return_std = float(normalization["return_std"])
    log_v0 = (math.log(float(initial_v)) - log_v_mean) / log_v_std
    r_prev0 = (float(initial_r_prev) - return_mean) / return_std

    strikes = float(initial_s) * moneynesses.to(device=device, dtype=dtype)
    chunk = int(price_chunk_paths or n_paths)
    payoff_sums = [
        torch.zeros(strikes.numel(), device=device, dtype=dtype)
        for _ in unique_indices
    ]
    total_paths = 0
    for start in range(0, n_paths, chunk):
        end = min(start + chunk, n_paths)
        batch_actions = actions[start:end]
        bsz = end - start
        log_v_t = torch.full((bsz, 1), log_v0, device=device, dtype=dtype)
        r_prev_t = torch.full((bsz, 1), r_prev0, device=device, dtype=dtype)
        log_s_t = torch.full((bsz, 1), math.log(float(initial_s)), device=device, dtype=dtype)
        s_at_maturities: dict[int, torch.Tensor] = {}
        for step in range(n_steps):
            a_onehot = F.one_hot(batch_actions[:, step].long(), num_classes=num_actions).to(dtype=dtype)
            condition_parts = [log_v_t]
            if include_prev_return:
                condition_parts.append(r_prev_t)
            condition_parts.append(a_onehot)
            condition = torch.cat(condition_parts, dim=-1)
            noise = torch.randn(bsz, 2, device=device, dtype=dtype)
            next_state = _sample_flowmap(
                student, condition, noise, cfg_w=cfg_w, num_actions=num_actions,
            )
            log_v_t = next_state[:, 0:1]
            r_prev_t = next_state[:, 1:2]
            r_next = r_prev_t * return_std + return_mean
            log_s_t = log_s_t + r_next
            step_index = step + 1
            if step_index in idx_to_pos:
                s_at_maturities[step_index] = torch.exp(log_s_t)

        for idx, s_t in s_at_maturities.items():
            payoff = torch.clamp(s_t - strikes.reshape(1, -1), min=0.0)
            pos = idx_to_pos[idx]
            payoff_sums[pos] = payoff_sums[pos] + payoff.sum(dim=0)
        total_paths += bsz

    prices_unique = torch.stack(payoff_sums, dim=0) / max(total_paths, 1)
    prices_unique = prices_unique * prices_unique.new_tensor(
        [math.exp(-float(r) * unique_indices[i] * float(dt)) for i in range(len(unique_indices))]
    ).reshape(-1, 1)
    rows = [idx_to_pos[int(i)] for i in maturity_indices.detach().cpu().tolist()]
    return prices_unique[rows]


def pricing_loss_from_grid(
    fake_prices: torch.Tensor,
    reference_prices: torch.Tensor,
    *,
    price_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = torch.clamp(reference_prices.abs(), min=float(price_floor))
    rel = (fake_prices - reference_prices) / denom
    loss = torch.mean(rel.square())
    rmse = torch.sqrt(torch.mean((fake_prices.detach() - reference_prices).square()))
    mape = torch.mean((fake_prices.detach() - reference_prices).abs() / torch.clamp(reference_prices.abs(), min=1e-8))
    return loss, rmse, mape


def _next_batch(loader, state: dict[str, Any]):
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
    p.add_argument("--teacher-checkpoint", type=Path, default=None)
    p.add_argument("--mc-oracle", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--steps-per-epoch", type=int, default=20)
    p.add_argument("--transition-batch-size", type=int, default=4096)
    p.add_argument("--path-batch-size", type=int, default=512)
    p.add_argument("--val-paths", type=int, default=4096)
    p.add_argument("--price-chunk-paths", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--pricing-weight", type=float, default=10.0)
    p.add_argument("--distill-weight", type=float, default=0.2)
    p.add_argument("--time-eps", type=float, default=1e-3)
    p.add_argument("--boundary-prob", type=float, default=0.10)
    p.add_argument("--moneynesses", nargs="+", type=float, default=[0.85, 0.90, 0.95, 1.00, 1.05])
    p.add_argument("--maturities", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    p.add_argument("--pricing-r", type=float, default=0.0)
    p.add_argument("--price-floor", type=float, default=1.0)
    p.add_argument("--oracle-limit", type=int, default=None)
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
    if args.pricing_weight <= 0:
        raise ValueError("--pricing-weight must be positive")
    if args.distill_weight > 0 and args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required when --distill-weight > 0")

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
    init_extra = init_ckpt.get("extra", {})
    if init_extra.get("kind") not in (None, "mean_flow"):
        raise ValueError(f"init checkpoint must be mean_flow kind, got {init_extra.get('kind')}")
    model_config = init_ckpt["model_config"]
    student = MeanFlowStudent(
        state_dim=int(model_config["state_dim"]),
        condition_dim=int(model_config["condition_dim"]),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        time_embedding_dim=int(model_config.get("time_embedding_dim", 64)),
        num_blocks=int(model_config.get("num_blocks", 4)),
    ).to(device)
    student.load_state_dict(init_ckpt["model_state"])

    full_joint_cond = 2 + num_actions
    markov_minimal_cond = 1 + num_actions
    if student.condition_dim == full_joint_cond:
        include_prev_return = True
    elif student.condition_dim == markov_minimal_cond:
        include_prev_return = False
    else:
        raise ValueError(
            f"student condition_dim={student.condition_dim}, expected {full_joint_cond} or {markov_minimal_cond}"
        )

    teacher = None
    train_loader = None
    loader_state: dict[str, Any] = {}
    if args.distill_weight > 0:
        teacher, teacher_ckpt = load_model_from_checkpoint(args.teacher_checkpoint, map_location=device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if _strip_stage(str(teacher_ckpt.get("stage", ""))) != "joint":
            raise ValueError(f"teacher checkpoint must be joint, got stage={teacher_ckpt.get('stage')}")
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

    oracle_path = args.mc_oracle or (args.data_dir / "mc_oracle.npz")
    reference = _reference_price_grid(
        oracle_path,
        dt=dt,
        moneynesses=args.moneynesses,
        maturities=args.maturities,
        s0=initial_s,
        r=args.pricing_r,
        limit=args.oracle_limit,
    )
    reference_prices = torch.as_tensor(reference["prices"], device=device, dtype=torch.float32)
    moneynesses = torch.as_tensor(args.moneynesses, device=device, dtype=torch.float32)
    maturities = torch.as_tensor(args.maturities, device=device, dtype=torch.float32)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_dir = build_run_dir(args.output_dir, run_name=args.run_name, prefix="flowmap_pricing")
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    config = {
        "data_dir": str(args.data_dir),
        "init_checkpoint": str(args.init_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
        "mc_oracle": str(oracle_path),
        "method": "pricing_aware_flowmap",
        "model_config": model_config,
        "num_actions": num_actions,
        "include_prev_return": include_prev_return,
        "normalization": normalization,
        "dt": dt,
        "n_steps": n_steps,
        "initial_v": initial_v,
        "initial_s": initial_s,
        "moneynesses": args.moneynesses,
        "maturities": args.maturities,
        "reference_prices": reference["prices"].tolist(),
        "pricing_weight": args.pricing_weight,
        "distill_weight": args.distill_weight,
        "price_floor": args.price_floor,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "path_batch_size": args.path_batch_size,
        "val_paths": args.val_paths,
        "lr": args.lr,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"[flowmap-pricing] run={run_dir.name} init={args.init_checkpoint} "
        f"paths/batch={args.path_batch_size} steps/epoch={args.steps_per_epoch} "
        f"pricing_w={args.pricing_weight} distill_w={args.distill_weight}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 997)
    best = float("inf")
    global_step = 0
    t0 = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        student.train()
        totals = {"loss": 0.0, "pricing_loss": 0.0, "distill_loss": 0.0, "pricing_rmse": 0.0, "pricing_mape": 0.0}
        for _ in range(args.steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            actions_np = _make_actions(
                n_paths=args.path_batch_size,
                n_steps=n_steps,
                num_actions=num_actions,
                metadata=metadata,
                rng=rng,
                constant_action=args.constant_action,
            )
            actions = torch.as_tensor(actions_np, device=device)
            fake_prices = rollout_price_grid_torch(
                student,
                normalization=normalization,
                actions=actions,
                moneynesses=moneynesses,
                maturities=maturities,
                dt=dt,
                initial_v=initial_v,
                initial_s=initial_s,
                initial_r_prev=args.initial_r_prev,
                r=args.pricing_r,
                include_prev_return=include_prev_return,
                cfg_w=args.cfg_w,
                price_chunk_paths=args.price_chunk_paths,
            )
            pricing_loss, pricing_rmse, pricing_mape = pricing_loss_from_grid(
                fake_prices, reference_prices, price_floor=args.price_floor,
            )
            distill_loss = reference_prices.new_tensor(0.0)
            if args.distill_weight > 0 and teacher is not None and train_loader is not None:
                batch = _next_batch(train_loader, loader_state)
                cond = batch["condition"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)
                distill_loss = flow_map_loss(
                    student, teacher, cond, tgt,
                    time_eps=args.time_eps,
                    boundary_prob=args.boundary_prob,
                )
            loss = args.pricing_weight * pricing_loss + args.distill_weight * distill_loss
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip_norm)
            opt.step()
            global_step += 1
            totals["loss"] += float(loss.detach().item())
            totals["pricing_loss"] += float(pricing_loss.detach().item())
            totals["distill_loss"] += float(distill_loss.detach().item())
            totals["pricing_rmse"] += float(pricing_rmse.item())
            totals["pricing_mape"] += float(pricing_mape.item())

        student.eval()
        with torch.no_grad():
            val_actions_np = _make_actions(
                n_paths=args.val_paths,
                n_steps=n_steps,
                num_actions=num_actions,
                metadata=metadata,
                rng=np.random.default_rng(args.seed + 100_000 + epoch),
                constant_action=args.constant_action,
            )
            val_prices = rollout_price_grid_torch(
                student,
                normalization=normalization,
                actions=torch.as_tensor(val_actions_np, device=device),
                moneynesses=moneynesses,
                maturities=maturities,
                dt=dt,
                initial_v=initial_v,
                initial_s=initial_s,
                initial_r_prev=args.initial_r_prev,
                r=args.pricing_r,
                include_prev_return=include_prev_return,
                cfg_w=args.cfg_w,
                price_chunk_paths=args.price_chunk_paths,
            )
            val_loss, val_rmse, val_mape = pricing_loss_from_grid(
                val_prices, reference_prices, price_floor=args.price_floor,
            )

        rec = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": totals["loss"] / args.steps_per_epoch,
            "train_pricing_loss": totals["pricing_loss"] / args.steps_per_epoch,
            "train_distill_loss": totals["distill_loss"] / args.steps_per_epoch,
            "train_pricing_rmse": totals["pricing_rmse"] / args.steps_per_epoch,
            "train_pricing_mape": totals["pricing_mape"] / args.steps_per_epoch,
            "val_pricing_loss": float(val_loss.item()),
            "val_pricing_rmse": float(val_rmse.item()),
            "val_pricing_mape": float(val_mape.item()),
            "elapsed_s": time.monotonic() - t0,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        is_best = rec["val_pricing_rmse"] < best
        if is_best:
            best = rec["val_pricing_rmse"]
        extra = {
            "kind": "mean_flow",
            "method": "pricing_aware_flowmap",
            "base_method": init_extra.get("method", "lagrangian_flowmap"),
            "init_checkpoint": str(args.init_checkpoint),
            "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
            "mc_oracle": str(oracle_path),
            "pricing_weight": args.pricing_weight,
            "distill_weight": args.distill_weight,
            "val_pricing_rmse": rec["val_pricing_rmse"],
            "val_pricing_mape": rec["val_pricing_mape"],
        }
        save_checkpoint(
            ckpt_dir / "last.pt",
            student,
            opt,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best,
            model_config=model_config,
            train_config=config,
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
                model_config=model_config,
                train_config=config,
                normalization=normalization,
                stage="mf_joint",
                num_actions=num_actions,
                extra=extra,
            )
        print(
            f"  epoch {epoch}/{args.epochs} "
            f"train_price_rmse={rec['train_pricing_rmse']:.4f} "
            f"val_price_rmse={rec['val_pricing_rmse']:.4f} "
            f"val_mape={rec['val_pricing_mape']:.4f} "
            f"distill={rec['train_distill_loss']:.5f} best={best:.4f}"
            f"{' *' if is_best else ''} elapsed={rec['elapsed_s']:.0f}s",
            flush=True,
        )

    summary = {
        "run_dir": str(run_dir),
        "stage": "mf_joint",
        "method": "pricing_aware_flowmap",
        "best_val_pricing_rmse": best,
        "checkpoints": {
            "best": str(ckpt_dir / "best.pt"),
            "last": str(ckpt_dir / "last.pt"),
        },
        "total_time_s": time.monotonic() - t0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[flowmap-pricing] done {run_dir}", flush=True)


if __name__ == "__main__":
    main()
