"""Mean Flow distillation: turn a multi-step FM teacher into a 1-NFE student.

Loss (Geng et al. 2025 + standard CFM straight-line probability path)::

    u_target(x_t, r, t) = v_reversed_teacher(x_t, t) - (t - r) * sg(du/dt)
    L = mean( || u_student(x_t, r, t) - u_target ||^2 )

The FM teacher is trained with the CFM convention ``noise -> data``. Mean Flow
1-NFE sampling starts from known noise, so this distillation flips the time
axis to ``data -> noise``:

    ``x_t = (1 - t) * x_data + t * eps``

The instantaneous velocity in this reversed convention is
``-v_teacher(x_t, 1 - t)``. The derivative ``du/dt`` is the total derivative
w.r.t. ``t`` along the reversed interpolation. We compute it with forward-mode
autodiff (``torch.func.jvp``) so it stays cheap and the backward graph can flow
through the primal output ``u_student``.

A fraction ``boundary_prob`` of the batch is sampled with ``r == t``: in that
regime the loss reduces to standard FM regression and anchors the model.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finflow.models import MeanFlowStudent, TransitionFM, warm_start_mean_flow_from_fm
from finflow.training import (
    TransitionFMTrainConfig,
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
class MeanFlowDistillConfig:
    """Configuration for Mean Flow distillation."""

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
    # MF-specific
    boundary_prob: float = 0.25
    boundary_prob_start: float | None = None
    boundary_prob_end: float | None = None
    identity_weight: float = 1.0
    identity_residual_eval: bool = False
    warm_start: bool = True
    progress: bool = True
    progress_min_interval: float = 0.2


DistillStage = Literal["vol", "ret", "joint"]


def _sample_r_t(
    batch_size: int,
    time_eps: float,
    boundary_prob: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(r, t)`` in ``[time_eps, 1 - time_eps]`` with ``r <= t``.

    A fraction ``boundary_prob`` of the batch is forced to ``r == t``.
    """

    u = torch.rand(batch_size, 2, device=device, dtype=dtype)
    r = u.min(dim=1).values
    t = u.max(dim=1).values
    r = r.clamp(min=time_eps, max=1.0 - time_eps)
    t = t.clamp(min=time_eps, max=1.0 - time_eps)
    t = torch.maximum(t, r)
    if boundary_prob > 0.0:
        mask = torch.rand(batch_size, device=device) < boundary_prob
        r = torch.where(mask, t, r)
    return r, t


def _sample_r_t_with_mask(
    batch_size: int,
    time_eps: float,
    boundary_prob: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``(r, t)`` and return the exact ``r == t`` boundary mask."""

    u = torch.rand(batch_size, 2, device=device, dtype=dtype)
    r = u.min(dim=1).values
    t = u.max(dim=1).values
    r = r.clamp(min=time_eps, max=1.0 - time_eps)
    t = t.clamp(min=time_eps, max=1.0 - time_eps)
    t = torch.maximum(t, r)
    if boundary_prob > 0.0:
        boundary_mask = torch.rand(batch_size, device=device) < boundary_prob
        r = torch.where(boundary_mask, t, r)
    else:
        boundary_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    return r, t, boundary_mask


def _mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.new_zeros(())


def mean_flow_loss_components(
    student: MeanFlowStudent,
    teacher: TransitionFM,
    condition: torch.Tensor,
    target: torch.Tensor,
    time_eps: float = 1e-3,
    boundary_prob: float = 0.25,
    identity_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute total, boundary, and identity Mean Flow losses for one batch."""

    if not 0.0 <= boundary_prob <= 1.0:
        raise ValueError("boundary_prob must be in [0, 1]")

    if target.ndim != 2:
        raise ValueError("target must have shape [batch, state_dim]")
    if condition.ndim != 2 or condition.shape[0] != target.shape[0]:
        raise ValueError("condition must have shape [batch, condition_dim]")

    batch_size = target.shape[0]
    device = target.device
    dtype = target.dtype

    r, t, boundary_mask = _sample_r_t_with_mask(batch_size, time_eps, boundary_prob, device, dtype)
    identity_mask = ~boundary_mask
    noise = torch.randn_like(target)

    def fn(t_in: torch.Tensor) -> torch.Tensor:
        t_view = t_in.reshape(batch_size, *([1] * (target.ndim - 1)))
        x_t_local = (1.0 - t_view) * target + t_view * noise
        return student(x_t_local, r, t_in, condition)

    pred_u, du_dt = torch.func.jvp(fn, (t,), (torch.ones_like(t),))

    t_view = t.reshape(batch_size, *([1] * (target.ndim - 1)))
    x_t = (1.0 - t_view) * target + t_view * noise
    with torch.no_grad():
        # Convert the FM teacher's noise->data vector field into the reversed
        # data->noise convention used by Mean Flow sampling.
        v_teacher = -teacher(x_tau=x_t, tau=1.0 - t, condition=condition)

    delta = (t - r).reshape(batch_size, *([1] * (target.ndim - 1)))
    u_target = v_teacher - identity_weight * delta * du_dt.detach()

    per_item_loss = F.mse_loss(pred_u, u_target, reduction="none").reshape(batch_size, -1).mean(dim=1)
    boundary_count = boundary_mask.sum()
    identity_count = identity_mask.sum()
    return {
        "loss": per_item_loss.mean(),
        "boundary_loss": _mean_or_zero(per_item_loss, boundary_mask),
        "identity_loss": _mean_or_zero(per_item_loss, identity_mask),
        "boundary_count": boundary_count.to(dtype=per_item_loss.dtype),
        "identity_count": identity_count.to(dtype=per_item_loss.dtype),
        "boundary_fraction": boundary_count.to(dtype=per_item_loss.dtype) / float(batch_size),
    }


def mean_flow_loss(
    student: MeanFlowStudent,
    teacher: TransitionFM,
    condition: torch.Tensor,
    target: torch.Tensor,
    time_eps: float = 1e-3,
    boundary_prob: float = 0.25,
    identity_weight: float = 1.0,
) -> torch.Tensor:
    """Compute the scalar Mean Flow distillation loss for a single batch."""

    return mean_flow_loss_components(
        student, teacher, condition, target,
        time_eps=time_eps, boundary_prob=boundary_prob,
        identity_weight=identity_weight,
    )["loss"]


def _empty_mean_flow_stats() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "items": 0.0,
        "boundary_loss_sum": 0.0,
        "boundary_items": 0.0,
        "identity_loss_sum": 0.0,
        "identity_items": 0.0,
    }


def _accumulate_mean_flow_stats(
    stats: dict[str, float],
    components: dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    boundary_items = float(components["boundary_count"].item())
    identity_items = float(components["identity_count"].item())
    stats["loss_sum"] += float(components["loss"].item()) * batch_size
    stats["items"] += float(batch_size)
    stats["boundary_loss_sum"] += float(components["boundary_loss"].item()) * boundary_items
    stats["boundary_items"] += boundary_items
    stats["identity_loss_sum"] += float(components["identity_loss"].item()) * identity_items
    stats["identity_items"] += identity_items


def _finalize_mean_flow_stats(stats: dict[str, float]) -> dict[str, float]:
    items = max(stats["items"], 1.0)
    boundary_items = stats["boundary_items"]
    identity_items = stats["identity_items"]
    return {
        "loss": stats["loss_sum"] / items,
        "boundary_loss": (
            stats["boundary_loss_sum"] / boundary_items if boundary_items > 0 else 0.0
        ),
        "identity_loss": (
            stats["identity_loss_sum"] / identity_items if identity_items > 0 else 0.0
        ),
        "boundary_fraction": boundary_items / items,
    }


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _boundary_prob_for_epoch(config: MeanFlowDistillConfig, epoch: int) -> float:
    if (config.boundary_prob_start is None) != (config.boundary_prob_end is None):
        raise ValueError("boundary_prob_start and boundary_prob_end must be set together")
    if config.boundary_prob_start is None:
        return float(config.boundary_prob)

    start = float(config.boundary_prob_start)
    end = float(config.boundary_prob_end)
    if config.epochs <= 1:
        return start
    if epoch >= config.epochs:
        return end
    fraction = (epoch - 1) / max(config.epochs - 1, 1)
    return start + (end - start) * fraction


def _evaluate_mean_flow(
    student: MeanFlowStudent,
    teacher: TransitionFM,
    loader: DataLoader | TensorBatchLoader,
    device: torch.device,
    *,
    time_eps: float,
    boundary_prob: float,
    identity_weight: float,
    max_batches: int | None,
    disable_progress: bool,
    progress_min_interval: float,
    desc: str,
) -> dict[str, float]:
    student.eval()
    teacher.eval()
    stats = _empty_mean_flow_stats()
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=_effective_num_batches(loader, max_batches),
        desc=desc, disable=disable_progress, min_interval=progress_min_interval,
    )
    with torch.no_grad():
        # The JVP needs a fresh forward, but we can still do a deterministic
        # eval by averaging the MSE over a single sampled tau-batch per data
        # batch -- noise level reused per batch to keep eval cheap.
        for batch in bar:
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            # No grad path -- need to wrap loss in enable_grad for jvp? No,
            # jvp works under no_grad as long as we don't backward. To avoid
            # extra cost, evaluate with random (r, t) and a single noise.
            with torch.enable_grad():
                components = mean_flow_loss_components(
                    student, teacher, condition, target,
                    time_eps=time_eps, boundary_prob=boundary_prob,
                    identity_weight=identity_weight,
                )
            bs = condition.shape[0]
            _accumulate_mean_flow_stats(stats, components, bs)
            if not disable_progress:
                current = _finalize_mean_flow_stats(stats)
                bar.set_postfix(
                    loss=f"{current['loss']:.4f}",
                    ident=f"{current['identity_loss']:.4f}",
                    refresh=False,
                )
    bar.close()
    return _finalize_mean_flow_stats(stats)


def _train_one_epoch_mean_flow(
    student: MeanFlowStudent,
    teacher: TransitionFM,
    loader: DataLoader | TensorBatchLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    time_eps: float,
    boundary_prob: float,
    identity_weight: float,
    grad_clip_norm: float,
    max_batches: int | None,
    disable_progress: bool,
    progress_min_interval: float,
    desc: str,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    stats = _empty_mean_flow_stats()
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=_effective_num_batches(loader, max_batches),
        desc=desc, disable=disable_progress, min_interval=progress_min_interval,
    )
    for batch in bar:
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        components = mean_flow_loss_components(
            student, teacher, condition, target,
            time_eps=time_eps, boundary_prob=boundary_prob,
            identity_weight=identity_weight,
        )
        loss = components["loss"]
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
        optimizer.step()
        bs = condition.shape[0]
        _accumulate_mean_flow_stats(stats, components, bs)
        if not disable_progress:
            current = _finalize_mean_flow_stats(stats)
            bar.set_postfix(
                loss=f"{current['loss']:.4f}",
                ident=f"{current['identity_loss']:.4f}",
                refresh=False,
            )
    bar.close()
    return _finalize_mean_flow_stats(stats)


def train_mean_flow_distill(
    data_dir: str | Path,
    output_dir: str | Path,
    stage: DistillStage,
    distill_config: MeanFlowDistillConfig,
    student_config: TwoStageFMModelConfig | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Distill a Mean Flow student from a trained FM teacher."""

    if stage not in ("vol", "ret", "joint"):
        raise ValueError("stage must be 'vol', 'ret', or 'joint'")
    _validate_probability("boundary_prob", float(distill_config.boundary_prob))
    if distill_config.boundary_prob_start is not None:
        _validate_probability("boundary_prob_start", float(distill_config.boundary_prob_start))
    if distill_config.boundary_prob_end is not None:
        _validate_probability("boundary_prob_end", float(distill_config.boundary_prob_end))
    _boundary_prob_for_epoch(distill_config, 1)
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
        expected_state = 1
        expected_cond = 1 + num_actions
    elif stage == "ret":
        datasets = build_ret_datasets(data_dir, normalization, num_actions)
        expected_state = 1
        expected_cond = 3 + num_actions
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

    student = MeanFlowStudent(**asdict(student_config)).to(device)
    warm_copied = 0
    if distill_config.warm_start:
        warm_copied = warm_start_mean_flow_from_fm(student, teacher)

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

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix=f"mf_{stage}_distill")
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
    }
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    disable_progress = not distill_config.progress
    train_batches = _effective_num_batches(train_loader, distill_config.max_train_batches)
    val_batches = _effective_num_batches(val_loader, distill_config.max_val_batches)
    n_params = sum(p.numel() for p in student.parameters())
    if distill_config.progress:
        header = (
            f"[finflow] mf_distill stage={stage} | run={run_dir.name} | device={device} | "
            f"cache_data_device={int(distill_config.cache_data_device)} | "
            f"params={n_params/1e3:.1f}k | warm_start={warm_copied} tensors | "
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
        desc_train = f"epoch {epoch:>3}/{distill_config.epochs} train"
        desc_val = f"epoch {epoch:>3}/{distill_config.epochs} val  "
        boundary_prob = _boundary_prob_for_epoch(distill_config, epoch)
        train_stats = _train_one_epoch_mean_flow(
            student, teacher, train_loader, optimizer, device,
            time_eps=distill_config.time_eps,
            boundary_prob=boundary_prob,
            identity_weight=distill_config.identity_weight,
            grad_clip_norm=distill_config.grad_clip_norm,
            max_batches=distill_config.max_train_batches,
            disable_progress=disable_progress,
            progress_min_interval=distill_config.progress_min_interval,
            desc=desc_train,
        )
        val_stats = _evaluate_mean_flow(
            student, teacher, val_loader, device,
            time_eps=distill_config.time_eps,
            boundary_prob=boundary_prob,
            identity_weight=distill_config.identity_weight,
            max_batches=distill_config.max_val_batches,
            disable_progress=disable_progress,
            progress_min_interval=distill_config.progress_min_interval,
            desc=desc_val,
        )
        identity_residual = None
        if distill_config.identity_residual_eval:
            identity_stats = _evaluate_mean_flow(
                student, teacher, val_loader, device,
                time_eps=distill_config.time_eps,
                boundary_prob=0.0,
                identity_weight=distill_config.identity_weight,
                max_batches=distill_config.max_val_batches,
                disable_progress=disable_progress,
                progress_min_interval=distill_config.progress_min_interval,
                desc=f"epoch {epoch:>3}/{distill_config.epochs} ident",
            )
            identity_residual = identity_stats["identity_loss"]
        train_loss = train_stats["loss"]
        val_loss = val_stats["loss"]
        epoch_time = time.monotonic() - epoch_start
        global_step += train_batches

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_boundary_loss": train_stats["boundary_loss"],
            "train_identity_loss": train_stats["identity_loss"],
            "train_boundary_fraction": train_stats["boundary_fraction"],
            "val_loss": val_loss,
            "val_boundary_loss": val_stats["boundary_loss"],
            "val_identity_loss": val_stats["identity_loss"],
            "val_boundary_fraction": val_stats["boundary_fraction"],
            "boundary_prob": boundary_prob,
            "global_step": global_step,
            "epoch_time_s": epoch_time,
        }
        if identity_residual is not None:
            record["val_identity_residual"] = identity_residual
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        is_best = val_loss < best_val
        save_checkpoint(
            ckpt_dir / "last.pt", student, optimizer,
            epoch=epoch, global_step=global_step, best_val_loss=best_val,
            model_config=asdict(student_config),
            train_config=asdict(distill_config),
            normalization=normalization, stage=f"mf_{stage}", num_actions=num_actions,
            extra={
                "train_loss": train_loss, "val_loss": val_loss,
                "train_boundary_loss": train_stats["boundary_loss"],
                "train_identity_loss": train_stats["identity_loss"],
                "val_boundary_loss": val_stats["boundary_loss"],
                "val_identity_loss": val_stats["identity_loss"],
                "boundary_prob": boundary_prob,
                "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
                "lambert_w_delta": teacher_lambert_w_delta,
                "warm_started_params": warm_copied,
                "kind": "mean_flow",
            },
        )
        if is_best:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt", student, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_val,
                model_config=asdict(student_config),
                train_config=asdict(distill_config),
                normalization=normalization, stage=f"mf_{stage}", num_actions=num_actions,
                extra={
                    "train_loss": train_loss, "val_loss": val_loss,
                    "train_boundary_loss": train_stats["boundary_loss"],
                    "train_identity_loss": train_stats["identity_loss"],
                    "val_boundary_loss": val_stats["boundary_loss"],
                    "val_identity_loss": val_stats["identity_loss"],
                    "boundary_prob": boundary_prob,
                    "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
                    "lambert_w_delta": teacher_lambert_w_delta,
                    "warm_started_params": warm_copied,
                    "kind": "mean_flow",
                },
            )

        if distill_config.progress:
            elapsed = time.monotonic() - run_start
            eta_s = (elapsed / epoch) * (distill_config.epochs - epoch)
            print(
                f"  epoch {epoch:>3}/{distill_config.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"ident={val_stats['identity_loss']:.4f} | boundary_p={boundary_prob:.3f} | "
                f"best={best_val:.4f}{' *' if is_best else '  '} | "
                f"epoch={_fmt_time(epoch_time)} | elapsed={_fmt_time(elapsed)} | "
                f"eta={_fmt_time(eta_s)}",
                file=sys.stderr, flush=True,
            )

    summary = {
        "run_dir": str(run_dir),
        "stage": f"mf_{stage}",
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
