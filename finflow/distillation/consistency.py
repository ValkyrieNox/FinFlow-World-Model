"""Consistency Distillation (Song et al. 2023) on a trained FM teacher.

For adjacent points on the same FM ODE trajectory we enforce::

    f_student(x_{t_n}, t_n, c) == sg( f_target(x_{t_{n+1}}, t_{n+1}, c) )

where ``f_target`` is the EMA of ``f_student`` and ``x_{t_n}`` is obtained by
one teacher Euler step from ``x_{t_{n+1}}`` toward ``t_n``::

    x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) * v_teacher(x_{t_{n+1}}, t_{n+1}, c)

The project uses the FM convention ``t=0`` = noise and ``t=1`` = data, so the
cleaner endpoint is ``t_{n+1}``. The noisy-side student must therefore match
the cleaner-side EMA target, not the other way around.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finflow.models import (
    ConsistencyStudent,
    TransitionFM,
    warm_start_consistency_from_fm,
)
from finflow.training import (
    TwoStageFMModelConfig,
    _effective_num_batches,
    _fmt_time,
    _iterate_batches,
    _make_progress,
    TensorBatchLoader,
    build_batch_loader,
    build_joint_datasets,
    build_ret_datasets,
    build_run_dir,
    build_vol_datasets,
    load_model_from_checkpoint,
    load_normalization,
    load_num_actions,
    resolve_device,
    save_checkpoint,
    set_seed,
)


@dataclass(frozen=True)
class ConsistencyDistillConfig:
    """Configuration for Consistency Distillation."""

    teacher_checkpoint: str
    batch_size: int = 512
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    time_eps: float = 1e-3
    num_workers: int = 0
    cache_data_device: bool = False
    seed: int = 1234
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    # CD-specific
    n_discretization: int = 18
    ema_decay: float = 0.999
    warm_start: bool = True
    progress: bool = True
    progress_min_interval: float = 0.2
    curriculum_kind: Literal["fixed", "ict"] = "ict"
    n_min: int = 10
    n_max: int = 160
    huber_c: float = 0.03
    karras_s0: float = -2.0
    time_sampling: Literal["uniform", "lognormal"] = "lognormal"


DistillStage = Literal["vol", "ret", "joint"]


def _schedule(n_discretization: int, time_eps: float, device, dtype) -> torch.Tensor:
    """Uniform schedule on ``[time_eps, 1 - time_eps]`` with ``n_discretization+1`` points."""

    return torch.linspace(time_eps, 1.0 - time_eps, n_discretization + 1, device=device, dtype=dtype)


def _curriculum_n(config: ConsistencyDistillConfig, epoch: int, total_epochs: int) -> int:
    """Resolve the iCT discretization count for the current epoch."""

    if config.curriculum_kind == "fixed":
        return int(config.n_discretization)
    if config.n_min <= 0 or config.n_max <= 0:
        raise ValueError("n_min and n_max must be positive")
    if config.n_max < config.n_min:
        raise ValueError("n_max must be >= n_min")
    if total_epochs <= 1:
        progress = 0.0
    else:
        progress = (epoch - 1) / max(total_epochs - 1, 1)
    ratio = config.n_max / config.n_min
    n = round(config.n_min * (ratio ** progress))
    return int(min(max(n, config.n_min), config.n_max))


def _curriculum_ema_decay(config: ConsistencyDistillConfig, n_discretization: int) -> float:
    """Resolve EMA decay; iCT uses a Karras-style decay that grows with N."""

    if config.curriculum_kind == "fixed":
        return float(config.ema_decay)
    decay = math.exp(config.karras_s0 * math.log(2.0) / max(n_discretization, 1))
    return float(min(max(decay, 0.0), 0.999999))


def _pseudo_huber_loss(pred: torch.Tensor, target: torch.Tensor, c: float) -> torch.Tensor:
    if c <= 0:
        raise ValueError("huber_c must be positive")
    diff = pred - target
    return (torch.sqrt(diff.square() + c * c) - c).mean()


def _sample_interval_indices(
    batch_size: int,
    n_segments: int,
    device: torch.device,
    time_sampling: Literal["uniform", "lognormal"],
) -> torch.Tensor:
    if time_sampling == "uniform":
        return torch.randint(0, n_segments, (batch_size,), device=device)
    if time_sampling == "lognormal":
        sigma = torch.exp(torch.randn(batch_size, device=device))
        unit_t = sigma / (1.0 + sigma)
        return torch.clamp((unit_t * n_segments).long(), min=0, max=n_segments - 1)
    raise ValueError(f"unknown time_sampling '{time_sampling}'")


def consistency_distill_step(
    student: ConsistencyStudent,
    target_net: ConsistencyStudent,
    teacher: TransitionFM,
    condition: torch.Tensor,
    target: torch.Tensor,
    schedule: torch.Tensor,
    *,
    huber_c: float | None = None,
    time_sampling: Literal["uniform", "lognormal"] = "uniform",
) -> torch.Tensor:
    """Single Consistency Distillation training step (loss only)."""

    if target.ndim != 2 or condition.ndim != 2:
        raise ValueError("target and condition must be 2D")

    batch_size = target.shape[0]
    device = target.device
    dtype = target.dtype
    n_segments = schedule.shape[0] - 1  # number of intervals = n_discretization

    noise = torch.randn_like(target)

    # Pick a random index n in {0, ..., n_segments - 1} per sample.
    n_idx = _sample_interval_indices(batch_size, n_segments, device, time_sampling)
    t_next = schedule[n_idx + 1]  # shape [B]
    t_curr = schedule[n_idx]      # shape [B]

    t_next_view = t_next.reshape(batch_size, *([1] * (target.ndim - 1)))
    t_curr_view = t_curr.reshape(batch_size, *([1] * (target.ndim - 1)))

    # Construct x at t_next via straight-line CFM interpolation.
    x_next = (1.0 - t_next_view) * noise + t_next_view * target

    # Teacher Euler step toward t_curr (one step backwards in t). Since this
    # repo's FM time runs noise->data, t_next is the cleaner endpoint. Train
    # the noisier student call at t_curr to match the cleaner EMA target.
    with torch.no_grad():
        v_teacher = teacher(x_tau=x_next, tau=t_next, condition=condition)
        x_curr_hat = x_next - (t_next_view - t_curr_view) * v_teacher
        target_val = target_net(x_next, t_next, condition)

    pred = student(x_curr_hat, t_curr, condition)
    if huber_c is None:
        return F.mse_loss(pred, target_val)
    return _pseudo_huber_loss(pred, target_val, huber_c)


def _ema_update(target_net: torch.nn.Module, source_net: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target_net.parameters(), source_net.parameters()):
            p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


def _train_one_epoch_consistency(
    student: ConsistencyStudent,
    target_net: ConsistencyStudent,
    teacher: TransitionFM,
    loader: DataLoader | TensorBatchLoader,
    optimizer: torch.optim.Optimizer,
    schedule: torch.Tensor,
    device: torch.device,
    *,
    ema_decay: float,
    huber_c: float | None,
    time_sampling: Literal["uniform", "lognormal"],
    grad_clip_norm: float,
    max_batches: int | None,
    disable_progress: bool,
    progress_min_interval: float,
    desc: str,
) -> float:
    student.train()
    target_net.eval()
    teacher.eval()
    total = 0.0
    seen = 0
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=_effective_num_batches(loader, max_batches),
        desc=desc, disable=disable_progress, min_interval=progress_min_interval,
    )
    for batch in bar:
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = consistency_distill_step(
            student, target_net, teacher, condition, target, schedule,
            huber_c=huber_c, time_sampling=time_sampling,
        )
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
        optimizer.step()
        _ema_update(target_net, student, ema_decay)
        bs = condition.shape[0]
        total += float(loss.item()) * bs
        seen += bs
        if not disable_progress:
            bar.set_postfix(loss=f"{total / max(seen, 1):.4f}", refresh=False)
    bar.close()
    return total / max(seen, 1)


def _evaluate_consistency(
    student: ConsistencyStudent,
    target_net: ConsistencyStudent,
    teacher: TransitionFM,
    loader: DataLoader | TensorBatchLoader,
    schedule: torch.Tensor,
    device: torch.device,
    *,
    huber_c: float | None,
    time_sampling: Literal["uniform", "lognormal"],
    max_batches: int | None,
    disable_progress: bool,
    progress_min_interval: float,
    desc: str,
) -> float:
    student.eval()
    target_net.eval()
    teacher.eval()
    total = 0.0
    seen = 0
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=_effective_num_batches(loader, max_batches),
        desc=desc, disable=disable_progress, min_interval=progress_min_interval,
    )
    with torch.no_grad():
        for batch in bar:
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            loss = consistency_distill_step(
                student, target_net, teacher, condition, target, schedule,
                huber_c=huber_c, time_sampling=time_sampling,
            )
            bs = condition.shape[0]
            total += float(loss.item()) * bs
            seen += bs
            if not disable_progress:
                bar.set_postfix(loss=f"{total / max(seen, 1):.4f}", refresh=False)
    bar.close()
    return total / max(seen, 1)


def train_consistency_distill(
    data_dir: str | Path,
    output_dir: str | Path,
    stage: DistillStage,
    distill_config: ConsistencyDistillConfig,
    student_config: TwoStageFMModelConfig | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Distill a Consistency student from a trained FM teacher."""

    if stage not in ("vol", "ret", "joint"):
        raise ValueError("stage must be 'vol', 'ret', or 'joint'")
    set_seed(distill_config.seed)

    device = resolve_device(distill_config.device)
    teacher, teacher_ckpt = load_model_from_checkpoint(
        distill_config.teacher_checkpoint, map_location=device,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    teacher_stage = teacher_ckpt.get("stage", "joint")
    if teacher_stage != stage:
        raise ValueError(
            f"teacher checkpoint stage='{teacher_stage}' does not match requested stage='{stage}'"
        )
    num_actions = int(teacher_ckpt.get("num_actions", load_num_actions(data_dir)))
    normalization = teacher_ckpt.get("normalization") or load_normalization(data_dir)
    teacher_extra = teacher_ckpt.get("extra", {})
    teacher_lambert_w_delta = float(teacher_extra.get("lambert_w_delta", 0.0) or 0.0)

    if stage == "vol":
        datasets = build_vol_datasets(
            data_dir, normalization, num_actions,
            lambert_w_delta=teacher_lambert_w_delta,
        )
        expected_state, expected_cond = 1, 1 + num_actions
    elif stage == "ret":
        datasets = build_ret_datasets(data_dir, normalization, num_actions)
        expected_state, expected_cond = 1, 3 + num_actions
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
            data_dir, normalization, num_actions,
            include_prev_return=include_prev_return,
        )
        expected_state = 2
        expected_cond = full_joint_cond if include_prev_return else markov_minimal_cond

    if student_config is None:
        student_config = TwoStageFMModelConfig(
            state_dim=teacher.state_dim,
            condition_dim=teacher.condition_dim,
            hidden_dim=teacher.hidden_dim,
            time_embedding_dim=teacher.time_embedding.embedding_dim,
            num_blocks=len(teacher.blocks),
        )

    if student_config.state_dim != expected_state or student_config.condition_dim != expected_cond:
        raise ValueError(
            f"{stage}-stage student must have state_dim={expected_state}, "
            f"condition_dim={expected_cond}; got "
            f"({student_config.state_dim}, {student_config.condition_dim})"
        )

    student = ConsistencyStudent(**asdict(student_config)).to(device)
    warm_copied = 0
    if distill_config.warm_start:
        warm_copied = warm_start_consistency_from_fm(student, teacher)
    target_net = copy.deepcopy(student)
    for p in target_net.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=distill_config.lr,
        weight_decay=distill_config.weight_decay,
    )

    train_loader = build_batch_loader(
        datasets["train"], batch_size=distill_config.batch_size, shuffle=True,
        num_workers=distill_config.num_workers, device=device,
        cache_on_device=distill_config.cache_data_device,
    )
    val_loader = build_batch_loader(
        datasets["val"], batch_size=distill_config.batch_size, shuffle=False,
        num_workers=distill_config.num_workers, device=device,
        cache_on_device=distill_config.cache_data_device,
    )

    initial_n = _curriculum_n(distill_config, 1, distill_config.epochs)
    initial_ema_decay = _curriculum_ema_decay(distill_config, initial_n)
    initial_schedule = _schedule(
        initial_n, distill_config.time_eps,
        device=device, dtype=torch.float32,
    )

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix=f"cd_{stage}_distill")
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    config_blob = {
        "run_dir": str(run_dir.resolve()),
        "stage": stage,
        "num_actions": num_actions,
        "student_config": asdict(student_config),
        "distill_config": asdict(distill_config),
        "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
        "teacher_lambert_w_delta": teacher_lambert_w_delta,
        "warm_started_params": warm_copied,
        "normalization": normalization,
        "data_dir": str(Path(data_dir).resolve()),
        "initial_schedule": initial_schedule.cpu().tolist(),
    }
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    disable_progress = not distill_config.progress
    train_batches = _effective_num_batches(train_loader, distill_config.max_train_batches)
    val_batches = _effective_num_batches(val_loader, distill_config.max_val_batches)
    n_params = sum(p.numel() for p in student.parameters())
    if distill_config.progress:
        header = (
            f"[finflow] cd_distill stage={stage} | run={run_dir.name} | device={device} | "
            f"cache_data_device={int(distill_config.cache_data_device)} | "
            f"params={n_params/1e3:.1f}k | warm_start={warm_copied} tensors | "
            f"curriculum={distill_config.curriculum_kind} N={initial_n}"
            f"{'' if distill_config.curriculum_kind == 'fixed' else f'->{distill_config.n_max}'} "
            f"ema={initial_ema_decay:.6f} | "
            f"train={len(datasets['train'])} ({train_batches} batch x {distill_config.batch_size}) | "
            f"val={len(datasets['val'])} ({val_batches} batch) | epochs={distill_config.epochs}"
        )
        print(header, file=sys.stderr, flush=True)

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    run_start = time.monotonic()
    global_step = 0

    for epoch in range(1, distill_config.epochs + 1):
        epoch_start = time.monotonic()
        n_discretization = _curriculum_n(distill_config, epoch, distill_config.epochs)
        ema_decay = _curriculum_ema_decay(distill_config, n_discretization)
        schedule = _schedule(
            n_discretization, distill_config.time_eps,
            device=device, dtype=torch.float32,
        )
        huber_c = distill_config.huber_c if distill_config.curriculum_kind == "ict" else None
        time_sampling = (
            distill_config.time_sampling
            if distill_config.curriculum_kind == "ict"
            else "uniform"
        )
        desc_train = f"epoch {epoch:>3}/{distill_config.epochs} train"
        desc_val = f"epoch {epoch:>3}/{distill_config.epochs} val  "
        train_loss = _train_one_epoch_consistency(
            student, target_net, teacher, train_loader, optimizer, schedule, device,
            ema_decay=ema_decay,
            huber_c=huber_c,
            time_sampling=time_sampling,
            grad_clip_norm=distill_config.grad_clip_norm,
            max_batches=distill_config.max_train_batches,
            disable_progress=disable_progress,
            progress_min_interval=distill_config.progress_min_interval,
            desc=desc_train,
        )
        val_loss = _evaluate_consistency(
            student, target_net, teacher, val_loader, schedule, device,
            huber_c=huber_c,
            time_sampling=time_sampling,
            max_batches=distill_config.max_val_batches,
            disable_progress=disable_progress,
            progress_min_interval=distill_config.progress_min_interval,
            desc=desc_val,
        )
        epoch_time = time.monotonic() - epoch_start
        global_step += train_batches

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "n_discretization": n_discretization,
            "ema_decay": ema_decay,
            "huber_c": huber_c,
            "time_sampling": time_sampling,
            "global_step": global_step,
            "epoch_time_s": epoch_time,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        is_best = val_loss < best_val
        save_checkpoint(
            ckpt_dir / "last.pt", target_net, optimizer,
            epoch=epoch, global_step=global_step, best_val_loss=best_val,
            model_config=asdict(student_config),
            train_config=asdict(distill_config),
            normalization=normalization, stage=f"cd_{stage}", num_actions=num_actions,
            extra={
                "train_loss": train_loss, "val_loss": val_loss,
                "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
                "lambert_w_delta": teacher_lambert_w_delta,
                "warm_started_params": warm_copied,
                "kind": "consistency",
                "model_state_kind": "ema",
                "ema_decay": ema_decay,
                "n_discretization": n_discretization,
                "curriculum_kind": distill_config.curriculum_kind,
                "huber_c": huber_c,
                "time_sampling": time_sampling,
            },
        )
        if is_best:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt", target_net, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_val,
                model_config=asdict(student_config),
                train_config=asdict(distill_config),
                normalization=normalization, stage=f"cd_{stage}", num_actions=num_actions,
                extra={
                    "train_loss": train_loss, "val_loss": val_loss,
                    "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
                    "lambert_w_delta": teacher_lambert_w_delta,
                    "warm_started_params": warm_copied,
                    "kind": "consistency",
                    "model_state_kind": "ema",
                    "ema_decay": ema_decay,
                    "n_discretization": n_discretization,
                    "curriculum_kind": distill_config.curriculum_kind,
                    "huber_c": huber_c,
                    "time_sampling": time_sampling,
                },
            )

        if distill_config.progress:
            elapsed = time.monotonic() - run_start
            eta_s = (elapsed / epoch) * (distill_config.epochs - epoch)
            print(
                f"  epoch {epoch:>3}/{distill_config.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"N={n_discretization} ema={ema_decay:.6f} | "
                f"best={best_val:.4f}{' *' if is_best else '  '} | "
                f"epoch={_fmt_time(epoch_time)} | elapsed={_fmt_time(elapsed)} | "
                f"eta={_fmt_time(eta_s)}",
                file=sys.stderr, flush=True,
            )

    summary = {
        "run_dir": str(run_dir),
        "stage": f"cd_{stage}",
        "num_actions": num_actions,
        "checkpoints": {
            "best": str(ckpt_dir / "best.pt"),
            "last": str(ckpt_dir / "last.pt"),
        },
        "best_val_loss": best_val,
        "history": history,
        "device": str(device),
        "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
        "warm_started_params": warm_copied,
        "total_time_s": time.monotonic() - run_start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
