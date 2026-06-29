"""Training and evaluation helpers for the V3 FM stages.

Two public entry families:

1. Single-stage joint model (legacy, kept for back compat with V3 baseline):
   ``TransitionFMModelConfig`` / ``TransitionFMTrainConfig`` / ``train_transition_fm``.

2. Action-aware joint model:
   ``train_joint_trans_fm`` learns
   ``p(log_v_next, r_next | log_v_t, r_t, action)`` in one FM teacher, or the
   Markov-minimal ablation ``p(log_v_next, r_next | log_v_t, action)``.

3. V3 two-stage models:
   - Stage 1a (variance kernel): ``train_vol_trans_fm``
   - Stage 1b (return kernel):   ``train_ret_trans_fm``
   Both share ``TwoStageFMModelConfig`` and ``TransitionFMTrainConfig`` and
   reuse the same ``TransitionFM`` backbone with different
   ``(state_dim, condition_dim)`` shapes.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from finflow.data import (
    HestonJointTransitionDataset,
    HestonRetTransitionDataset,
    HestonTransitionDataset,
    HestonVolTransitionDataset,
)
from finflow.models import TransitionFM, conditional_flow_matching_loss
from finflow.models.transition_fm import euler_sample


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionFMModelConfig:
    """Single-stage joint model config (legacy)."""

    state_dim: int = 2
    condition_dim: int = 2
    hidden_dim: int = 128
    time_embedding_dim: int = 64
    num_blocks: int = 4


@dataclass(frozen=True)
class TwoStageFMModelConfig:
    """V3 two-stage model config. Used by both vol and ret stages."""

    state_dim: int
    condition_dim: int
    hidden_dim: int = 128
    time_embedding_dim: int = 64
    num_blocks: int = 4


@dataclass(frozen=True)
class TransitionFMTrainConfig:
    """Shared training hyperparameters."""

    batch_size: int = 512
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    time_eps: float = 1e-4
    num_workers: int = 0
    cache_data_device: bool = False
    seed: int = 1234
    device: str = "auto"
    log_every: int = 50
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    progress: bool = True
    progress_min_interval: float = 0.2
    action_dropout_prob: float = 0.0
    scheduled_sampling_max_prob: float = 0.0
    scheduled_sampling_start_epoch: int = 1
    scheduled_sampling_fm_steps: int = 20
    # P3 tuning hooks
    save_every_epochs: int = 0  # >0: also dump checkpoints/epoch_XXX.pt every N epochs
    lr_schedule: Literal["constant", "cosine"] = "constant"
    lr_min: float = 0.0  # eta_min for cosine schedule
    ema_decay: float = 0.0  # >0: maintain EMA weights and save ema_* checkpoints
    target_loss_weights: tuple[float, ...] | None = None


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_metadata(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_normalization(data_dir: str | Path) -> dict[str, float]:
    metadata = load_metadata(data_dir)
    normalization = metadata.get("normalization")
    if not normalization:
        raise ValueError("metadata.json is missing normalization stats")
    return {
        "log_v_mean": float(normalization["log_v_mean"]),
        "log_v_std": float(normalization["log_v_std"]),
        "return_mean": float(normalization["return_mean"]),
        "return_std": float(normalization["return_std"]),
    }


def load_num_actions(data_dir: str | Path) -> int:
    metadata = load_metadata(data_dir)
    return int(metadata.get("num_actions", 1))


class TensorBatchLoader:
    """Minimal batch loader over prebuilt condition/target tensors.

    This is intended for very small per-sample tabular tensors where PyTorch's
    map-style DataLoader spends most of its time constructing Python objects.
    """

    def __init__(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
        *,
        batch_size: int,
        shuffle: bool,
        device: torch.device,
        action_start: int | None = None,
        action_dropout_prob: float = 0.0,
        parent_condition: torch.Tensor | None = None,
        has_parent: torch.Tensor | None = None,
    ) -> None:
        if condition.shape[0] != target.shape[0]:
            raise ValueError("condition and target must have the same length")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.condition = condition.to(device, non_blocking=True).contiguous()
        self.target = target.to(device, non_blocking=True).contiguous()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = device
        self.action_start = action_start
        self.action_dropout_prob = float(action_dropout_prob)
        if parent_condition is not None:
            if parent_condition.shape != condition.shape:
                raise ValueError("parent_condition must match condition shape")
            parent_condition = parent_condition.to(device, non_blocking=True).contiguous()
        if has_parent is not None:
            has_parent = has_parent.to(device, non_blocking=True).reshape(-1, 1).contiguous()
        self.parent_condition = parent_condition
        self.has_parent = has_parent

    def __len__(self) -> int:
        n = int(self.condition.shape[0])
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = int(self.condition.shape[0])
        if self.shuffle:
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                yield self._make_batch(idx=idx)
        else:
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = slice(start, end)
                yield self._make_batch(idx=idx)

    def _select(self, tensor: torch.Tensor, idx) -> torch.Tensor:
        if isinstance(idx, slice):
            return tensor[idx]
        return tensor.index_select(0, idx)

    def _make_batch(self, idx) -> dict[str, torch.Tensor]:
        condition = self._select(self.condition, idx)
        target = self._select(self.target, idx)
        if self.action_start is not None and self.action_dropout_prob > 0.0:
            condition = condition.clone()
            keep = (
                torch.rand(condition.shape[0], 1, device=condition.device)
                >= self.action_dropout_prob
            ).to(dtype=condition.dtype)
            condition[:, self.action_start:] *= keep
        batch = {"condition": condition, "target": target}
        if self.parent_condition is not None:
            batch["parent_condition"] = self._select(self.parent_condition, idx)
        if self.has_parent is not None:
            batch["has_parent"] = self._select(self.has_parent, idx)
        return batch


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader | TensorBatchLoader:
    return build_batch_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        device=device,
        cache_on_device=False,
    )


def build_batch_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    cache_on_device: bool,
) -> DataLoader | TensorBatchLoader:
    if cache_on_device:
        if not hasattr(dataset, "as_condition_target_tensors"):
            raise TypeError(
                f"{type(dataset).__name__} does not support cache_data_device"
            )
        tensors = dataset.as_condition_target_tensors()
        return TensorBatchLoader(
            tensors["condition"],
            tensors["target"],
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
            action_start=tensors.get("action_start"),
            action_dropout_prob=float(tensors.get("action_dropout_prob", 0.0)),
            parent_condition=tensors.get("parent_condition"),
            has_parent=tensors.get("has_parent"),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _iterate_batches(loader: DataLoader, max_batches: int | None):
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        yield batch


def _effective_num_batches(loader: DataLoader, max_batches: int | None) -> int:
    if max_batches is None:
        return len(loader)
    return min(len(loader), max_batches)


def _make_progress(
    iterable,
    total: int,
    desc: str,
    disable: bool,
    min_interval: float = 0.2,
):
    # When stderr is not a real TTY (piped to file, captured by pytest, etc.)
    # tqdm cannot rewrite a single line, so crank up the refresh interval so
    # the log file gets one progress line every few seconds instead of one
    # per micro-batch.
    if not disable and not sys.stderr.isatty():
        min_interval = max(min_interval, 10.0)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        disable=disable,
        leave=False,
        dynamic_ncols=True,
        mininterval=min_interval,
        file=sys.stderr,
    )


def evaluate_model(
    model: TransitionFM,
    loader: DataLoader | TensorBatchLoader,
    device: torch.device,
    time_eps: float,
    target_loss_weights: torch.Tensor | None = None,
    max_batches: int | None = None,
    desc: str = "val",
    disable_progress: bool = False,
    progress_min_interval: float = 0.2,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    total = _effective_num_batches(loader, max_batches)
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=total, desc=desc,
        disable=disable_progress, min_interval=progress_min_interval,
    )
    with torch.no_grad():
        for batch in bar:
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            loss = conditional_flow_matching_loss(
                model, condition=condition, target=target, time_eps=time_eps,
                target_weights=target_loss_weights,
            )
            batch_size = condition.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            if not disable_progress:
                bar.set_postfix(loss=f"{total_loss / max(total_items, 1):.4f}", refresh=False)
    bar.close()
    return total_loss / max(total_items, 1)


def train_one_epoch(
    model: TransitionFM,
    loader: DataLoader | TensorBatchLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    time_eps: float,
    grad_clip_norm: float,
    max_batches: int | None = None,
    desc: str = "train",
    disable_progress: bool = False,
    progress_min_interval: float = 0.2,
    condition_transform: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor] | None = None,
    post_step_callback: Callable[[], None] | None = None,
    target_loss_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    total = _effective_num_batches(loader, max_batches)
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=total, desc=desc,
        disable=disable_progress, min_interval=progress_min_interval,
    )
    for batch in bar:
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if condition_transform is not None:
            condition = condition_transform(batch, condition)
        optimizer.zero_grad(set_to_none=True)
        loss = conditional_flow_matching_loss(
            model, condition=condition, target=target, time_eps=time_eps,
            target_weights=target_loss_weights,
        )
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if post_step_callback is not None:
            post_step_callback()
        batch_size = condition.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        if not disable_progress:
            bar.set_postfix(loss=f"{total_loss / max(total_items, 1):.4f}", refresh=False)
    bar.close()
    return total_loss / max(total_items, 1)


def save_checkpoint(
    path: str | Path,
    model: TransitionFM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    normalization: dict[str, float],
    stage: str = "joint",
    num_actions: int = 1,
    extra: dict[str, Any] | None = None,
    model_state: dict[str, torch.Tensor] | None = None,
) -> None:
    if model_state is None:
        saved_model_state = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
    else:
        saved_model_state = {
            name: tensor.detach().cpu() for name, tensor in model_state.items()
        }
    checkpoint = {
        "model_state": saved_model_state,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_config": model_config,
        "train_config": train_config,
        "normalization": normalization,
        "stage": stage,
        "num_actions": int(num_actions),
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _validate_ema_decay(decay: float) -> float:
    decay = float(decay)
    if decay < 0.0 or decay >= 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    return decay


def _init_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


@torch.no_grad()
def _update_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
) -> None:
    for name, tensor in model.state_dict().items():
        current = tensor.detach()
        ema = ema_state[name]
        if torch.is_floating_point(ema):
            ema.mul_(decay).add_(current, alpha=1.0 - decay)
        else:
            ema.copy_(current)


def _target_loss_weights_tensor(
    weights: tuple[float, ...] | None,
    state_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if len(weights) != state_dim:
        raise ValueError(f"target_loss_weights must have length {state_dim}")
    tensor = torch.as_tensor(weights, device=device, dtype=torch.float32)
    if torch.any(tensor <= 0):
        raise ValueError("target_loss_weights must be positive")
    return tensor


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def build_run_dir(output_dir: str | Path, run_name: str | None = None, prefix: str = "transition_fm") -> Path:
    output_dir = Path(output_dir)
    if run_name is None:
        run_name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------


def _run_fm_training(
    model: TransitionFM,
    datasets: dict[str, Dataset],
    train_config: TransitionFMTrainConfig,
    run_dir: Path,
    normalization: dict[str, float],
    model_config_dict: dict[str, Any],
    stage: str,
    num_actions: int,
    config_blob_extra: dict[str, Any] | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
    train_condition_transform_factory: (
        Callable[[int], tuple[Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor] | None, dict[str, Any]]]
        | None
    ) = None,
    joint_self_scheduled_sampling: bool = False,
) -> dict[str, Any]:
    device = resolve_device(train_config.device)
    model = model.to(device)
    target_loss_weights = _target_loss_weights_tensor(
        train_config.target_loss_weights, model.state_dim, device,
    )
    ema_decay = _validate_ema_decay(train_config.ema_decay)
    ema_enabled = ema_decay > 0.0
    ema_state = _init_ema_state(model) if ema_enabled else None
    ema_model = TransitionFM(**model_config_dict).to(device) if ema_enabled else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = None
    if train_config.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(train_config.epochs, 1), eta_min=train_config.lr_min,
        )

    train_loader = build_batch_loader(
        datasets["train"], batch_size=train_config.batch_size, shuffle=True,
        num_workers=train_config.num_workers, device=device,
        cache_on_device=train_config.cache_data_device,
    )
    val_loader = build_batch_loader(
        datasets["val"], batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, device=device,
        cache_on_device=train_config.cache_data_device,
    )

    config_blob = {
        "run_dir": str(run_dir.resolve()),
        "stage": stage,
        "num_actions": num_actions,
        "model_config": model_config_dict,
        "train_config": asdict(train_config),
        "normalization": normalization,
    }
    if config_blob_extra:
        config_blob.update(config_blob_extra)
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    best_val_loss = float("inf")
    best_ema_val_loss = float("inf")
    global_step = 0
    history: list[dict[str, Any]] = []
    epoch_ckpts: list[str] = []
    ema_epoch_ckpts: list[str] = []
    extra_base = dict(checkpoint_extra or {})

    disable_progress = not train_config.progress
    train_batches = _effective_num_batches(train_loader, train_config.max_train_batches)
    val_batches = _effective_num_batches(val_loader, train_config.max_val_batches)
    n_params = sum(p.numel() for p in model.parameters())

    if train_config.progress:
        header = (
            f"[finflow] stage={stage} | run={run_dir.name} | device={device} | "
            f"cache_data_device={int(train_config.cache_data_device)} | "
            f"ema_decay={ema_decay:g} | "
            f"target_loss_weights={train_config.target_loss_weights} | "
            f"params={n_params/1e3:.1f}k | train={len(datasets['train'])} samples "
            f"({train_batches} batch/epoch x {train_config.batch_size}) | "
            f"val={len(datasets['val'])} samples ({val_batches} batch) | epochs={train_config.epochs}"
        )
        print(header, file=sys.stderr, flush=True)

    run_start = time.monotonic()

    if joint_self_scheduled_sampling and train_config.scheduled_sampling_max_prob > 0.0:
        # Self-scheduled-sampling for the joint kernel: with a ramped probability
        # replace a row's carried state (log_v_t[, r_{t-1}]) with the model's own
        # one-step sample drawn from that row's autoregressive parent condition.
        # This exposes the kernel to its own marginal at train time, closing the
        # teacher-forcing / free-rollout gap that drives raw error accumulation.
        ss_state_cols = model.condition_dim - num_actions
        ss_fm_steps = max(int(train_config.scheduled_sampling_fm_steps), 1)

        def _joint_ss_factory(epoch: int):
            prob = _scheduled_sampling_prob_for_epoch(
                epoch, train_config.epochs,
                train_config.scheduled_sampling_max_prob,
                train_config.scheduled_sampling_start_epoch,
            )
            base_extra = {
                "scheduled_sampling_prob": prob,
                "scheduled_sampling_mode": "joint_self",
                "scheduled_sampling_fm_steps": ss_fm_steps,
            }
            if prob <= 0.0:
                return None, base_extra
            # Prefer the EMA weights as a stable target-network sampler; sync once
            # per epoch (cheap) rather than per batch.
            if ema_enabled and ema_state is not None and ema_model is not None:
                ema_model.load_state_dict(ema_state)
                sampler_model = ema_model
            else:
                sampler_model = model

            def transform(
                batch: dict[str, torch.Tensor],
                condition: torch.Tensor,
            ) -> torch.Tensor:
                parent = batch.get("parent_condition")
                if parent is None:
                    return condition
                has_parent = batch.get("has_parent")
                was_training = sampler_model.training
                sampler_model.eval()
                with torch.no_grad():
                    sampled = euler_sample(
                        sampler_model,
                        parent.to(device=condition.device, dtype=condition.dtype),
                        n_steps=ss_fm_steps,
                    )
                if was_training:
                    sampler_model.train()
                mask = torch.rand(condition.shape[0], 1, device=condition.device) < prob
                if has_parent is not None:
                    mask = mask & has_parent.to(device=condition.device, dtype=torch.bool)
                updated = condition.clone()
                updated[:, :ss_state_cols] = torch.where(
                    mask, sampled[:, :ss_state_cols].to(condition.dtype),
                    updated[:, :ss_state_cols],
                )
                return updated

            return transform, base_extra

        train_condition_transform_factory = _joint_ss_factory

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.monotonic()
        desc_train = f"epoch {epoch:>3}/{train_config.epochs} train"
        desc_val = f"epoch {epoch:>3}/{train_config.epochs} val  "
        condition_transform = None
        epoch_extra: dict[str, Any] = {}
        if train_condition_transform_factory is not None:
            condition_transform, epoch_extra = train_condition_transform_factory(epoch)
        post_step_callback = (
            (lambda: _update_ema_state(ema_state, model, ema_decay))
            if ema_enabled and ema_state is not None else None
        )
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device=device,
            time_eps=train_config.time_eps, grad_clip_norm=train_config.grad_clip_norm,
            max_batches=train_config.max_train_batches,
            desc=desc_train, disable_progress=disable_progress,
            progress_min_interval=train_config.progress_min_interval,
            condition_transform=condition_transform,
            post_step_callback=post_step_callback,
            target_loss_weights=target_loss_weights,
        )
        val_loss = evaluate_model(
            model, val_loader, device=device, time_eps=train_config.time_eps,
            target_loss_weights=target_loss_weights,
            max_batches=train_config.max_val_batches,
            desc=desc_val, disable_progress=disable_progress,
            progress_min_interval=train_config.progress_min_interval,
        )
        ema_val_loss = None
        if ema_enabled and ema_state is not None and ema_model is not None:
            ema_model.load_state_dict(ema_state)
            ema_val_loss = evaluate_model(
                ema_model, val_loader, device=device, time_eps=train_config.time_eps,
                target_loss_weights=target_loss_weights,
                max_batches=train_config.max_val_batches,
                desc=f"epoch {epoch:>3}/{train_config.epochs} ema  ",
                disable_progress=disable_progress,
                progress_min_interval=train_config.progress_min_interval,
            )
        epoch_time = time.monotonic() - epoch_start
        global_step += train_batches

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "global_step": global_step,
            "epoch_time_s": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if ema_val_loss is not None:
            record["ema_val_loss"] = ema_val_loss
        record.update(epoch_extra)
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer,
            epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
            model_config=model_config_dict, train_config=asdict(train_config),
            normalization=normalization, stage=stage, num_actions=num_actions,
            extra={"train_loss": train_loss, "val_loss": val_loss, **extra_base, **epoch_extra},
        )
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
                model_config=model_config_dict, train_config=asdict(train_config),
                normalization=normalization, stage=stage, num_actions=num_actions,
                extra={"train_loss": train_loss, "val_loss": val_loss, **extra_base, **epoch_extra},
            )

        if ema_enabled and ema_state is not None and ema_val_loss is not None:
            is_ema_best = ema_val_loss < best_ema_val_loss
            if is_ema_best:
                best_ema_val_loss = ema_val_loss
            ema_extra = {
                "train_loss": train_loss,
                "val_loss": ema_val_loss,
                "base_val_loss": val_loss,
                "ema": True,
                "ema_decay": ema_decay,
                **extra_base,
                **epoch_extra,
            }
            save_checkpoint(
                ckpt_dir / "ema_last.pt", model, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_ema_val_loss,
                model_config=model_config_dict, train_config=asdict(train_config),
                normalization=normalization, stage=stage, num_actions=num_actions,
                extra=ema_extra, model_state=ema_state,
            )
            if is_ema_best:
                save_checkpoint(
                    ckpt_dir / "ema_best.pt", model, optimizer,
                    epoch=epoch, global_step=global_step, best_val_loss=best_ema_val_loss,
                    model_config=model_config_dict, train_config=asdict(train_config),
                    normalization=normalization, stage=stage, num_actions=num_actions,
                    extra=ema_extra, model_state=ema_state,
                )

        # Periodic snapshots enable pricing-aware checkpoint selection downstream
        # (val_loss is known to be a poor proxy for path/pricing quality here).
        if train_config.save_every_epochs and (epoch % train_config.save_every_epochs == 0):
            epoch_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                epoch_path, model, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
                model_config=model_config_dict, train_config=asdict(train_config),
                normalization=normalization, stage=stage, num_actions=num_actions,
                extra={"train_loss": train_loss, "val_loss": val_loss, **extra_base, **epoch_extra},
            )
            epoch_ckpts.append(str(epoch_path))
            if ema_enabled and ema_state is not None and ema_val_loss is not None:
                ema_epoch_path = ckpt_dir / f"ema_epoch_{epoch:03d}.pt"
                save_checkpoint(
                    ema_epoch_path, model, optimizer,
                    epoch=epoch, global_step=global_step, best_val_loss=best_ema_val_loss,
                    model_config=model_config_dict, train_config=asdict(train_config),
                    normalization=normalization, stage=stage, num_actions=num_actions,
                    extra={
                        "train_loss": train_loss,
                        "val_loss": ema_val_loss,
                        "base_val_loss": val_loss,
                        "ema": True,
                        "ema_decay": ema_decay,
                        **extra_base,
                        **epoch_extra,
                    },
                    model_state=ema_state,
                )
                ema_epoch_ckpts.append(str(ema_epoch_path))

        if scheduler is not None:
            scheduler.step()

        if train_config.progress:
            elapsed = time.monotonic() - run_start
            eta_s = (elapsed / epoch) * (train_config.epochs - epoch)
            ema_text = ""
            if ema_val_loss is not None:
                ema_text = f"ema_val={ema_val_loss:.4f} | ema_best={best_ema_val_loss:.4f} | "
            line = (
                f"  epoch {epoch:>3}/{train_config.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"{_format_epoch_extra(epoch_extra)}"
                f"{ema_text}"
                f"best={best_val_loss:.4f}{' *' if is_best else '  '} | "
                f"epoch={_fmt_time(epoch_time)} | elapsed={_fmt_time(elapsed)} | "
                f"eta={_fmt_time(eta_s)}"
            )
            print(line, file=sys.stderr, flush=True)

    summary = {
        "run_dir": str(run_dir),
        "stage": stage,
        "num_actions": num_actions,
        "checkpoints": {
            "best": str(ckpt_dir / "best.pt"),
            "last": str(ckpt_dir / "last.pt"),
            "epochs": epoch_ckpts,
        },
        "best_val_loss": best_val_loss,
        "history": history,
        "device": str(device),
        "total_time_s": time.monotonic() - run_start,
    }
    if ema_enabled:
        summary["checkpoints"].update({
            "ema_best": str(ckpt_dir / "ema_best.pt"),
            "ema_last": str(ckpt_dir / "ema_last.pt"),
            "ema_epochs": ema_epoch_ckpts,
        })
        summary["best_ema_val_loss"] = best_ema_val_loss
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:5.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes:d}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m"


def _format_epoch_extra(epoch_extra: dict[str, Any]) -> str:
    if "scheduled_sampling_prob" not in epoch_extra:
        return ""
    return f"sched={float(epoch_extra['scheduled_sampling_prob']):.3f} | "


# ---------------------------------------------------------------------------
# Stage: single-stage joint (legacy)
# ---------------------------------------------------------------------------


def build_transition_datasets(
    data_dir: str | Path,
    normalization: dict[str, float],
) -> dict[str, HestonTransitionDataset]:
    data_dir = Path(data_dir)
    return {
        split: HestonTransitionDataset(
            data_dir / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            return_mean=normalization["return_mean"],
            return_std=normalization["return_std"],
        )
        for split in ("train", "val", "test")
    }


def build_model(config: TransitionFMModelConfig) -> TransitionFM:
    return TransitionFM(**asdict(config))


def train_transition_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    model_config: TransitionFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
) -> dict[str, Any]:
    """Legacy: single-stage joint FM. Predicts (v_{t+1}, r_{t+1}) given (v_t, r_t)."""

    model_config = model_config or TransitionFMModelConfig()
    train_config = train_config or TransitionFMTrainConfig()

    set_seed(train_config.seed)
    normalization = load_normalization(data_dir)
    datasets = build_transition_datasets(data_dir, normalization)
    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="transition_fm")
    model = build_model(model_config)
    return _run_fm_training(
        model=model,
        datasets=datasets,
        train_config=train_config,
        run_dir=run_dir,
        normalization=normalization,
        model_config_dict=asdict(model_config),
        stage="joint",
        num_actions=load_num_actions(data_dir),
        config_blob_extra={"data_dir": str(Path(data_dir).resolve())},
    )


# ---------------------------------------------------------------------------
# Stage: action-aware joint transition kernel
# ---------------------------------------------------------------------------


def build_joint_datasets(
    data_dir: str | Path,
    normalization: dict[str, float],
    num_actions: int,
    train_action_dropout_prob: float = 0.0,
    include_prev_return: bool = True,
) -> dict[str, HestonJointTransitionDataset]:
    data_dir = Path(data_dir)
    return {
        split: HestonJointTransitionDataset(
            data_dir / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            return_mean=normalization["return_mean"],
            return_std=normalization["return_std"],
            num_actions=num_actions,
            action_dropout_prob=train_action_dropout_prob if split == "train" else 0.0,
            include_prev_return=include_prev_return,
        )
        for split in ("train", "val", "test")
    }


def train_joint_trans_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    num_actions: int | None = None,
    model_config: TwoStageFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
    include_prev_return: bool = True,
) -> dict[str, Any]:
    """Train an action-aware joint FM.

    The model samples ``[log_v_next_norm, r_next_norm]`` jointly from
    ``[log_v_t_norm, r_t_norm, action_onehot]`` by default. With
    ``include_prev_return=False`` it runs the Markov-minimal ablation and
    samples from ``[log_v_t_norm, action_onehot]``. The joint target keeps the
    immediate return/volatility dependence inside one teacher instead of
    splitting it across Stage 1a/1b.
    """

    train_config = train_config or TransitionFMTrainConfig()
    set_seed(train_config.seed)

    if num_actions is None:
        num_actions = load_num_actions(data_dir)
    normalization = load_normalization(data_dir)
    include_prev_return = bool(include_prev_return)
    datasets = build_joint_datasets(
        data_dir, normalization, num_actions,
        train_action_dropout_prob=train_config.action_dropout_prob,
        include_prev_return=include_prev_return,
    )

    condition_fields = (
        ["log_v_t", "r_prev", "action"]
        if include_prev_return
        else ["log_v_t", "action"]
    )
    expected = (2 if include_prev_return else 1) + num_actions
    if model_config is None:
        model_config = TwoStageFMModelConfig(
            state_dim=2, condition_dim=expected,
        )
    else:
        if model_config.state_dim != 2:
            raise ValueError("joint-stage model_config.state_dim must be 2")
        if model_config.condition_dim != expected:
            raise ValueError(
                f"joint-stage condition_dim must be {expected},"
                f" got {model_config.condition_dim}"
            )

    self_scheduled_sampling = train_config.scheduled_sampling_max_prob > 0.0
    transition_extra = {
        "kind": "fm",
        "transition_type": "joint_vr" if include_prev_return else "joint_vr_markov_minimal",
        "condition": condition_fields,
        "target": ["log_v_next", "r_next"],
        "include_prev_return": include_prev_return,
    }
    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="joint_trans_fm")
    model = TransitionFM(**asdict(model_config))
    return _run_fm_training(
        model=model,
        datasets=datasets,
        train_config=train_config,
        run_dir=run_dir,
        normalization=normalization,
        model_config_dict=asdict(model_config),
        stage="joint",
        num_actions=num_actions,
        config_blob_extra={
            "data_dir": str(Path(data_dir).resolve()),
            "self_scheduled_sampling": self_scheduled_sampling,
            **transition_extra,
        },
        checkpoint_extra=transition_extra,
        joint_self_scheduled_sampling=self_scheduled_sampling,
    )


def load_model_from_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu",
) -> tuple[TransitionFM, dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    config = checkpoint["model_config"]
    if "state_dim" not in config or "condition_dim" not in config:
        raise ValueError("checkpoint model_config missing state_dim/condition_dim")
    model = TransitionFM(
        state_dim=int(config["state_dim"]),
        condition_dim=int(config["condition_dim"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        time_embedding_dim=int(config.get("time_embedding_dim", 64)),
        num_blocks=int(config.get("num_blocks", 4)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    split: str = "val",
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a single-stage joint FM checkpoint on Heston transitions."""

    resolved_device = resolve_device(device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, map_location=resolved_device)
    normalization = checkpoint["normalization"]
    dataset = HestonTransitionDataset(
        Path(data_dir) / f"{split}_transitions.npz",
        normalize=True,
        log_v_mean=normalization["log_v_mean"],
        log_v_std=normalization["log_v_std"],
        return_mean=normalization["return_mean"],
        return_std=normalization["return_std"],
    )
    loader = build_dataloader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, device=resolved_device,
    )
    loss = evaluate_model(
        model, loader, device=resolved_device,
        time_eps=float(checkpoint["train_config"]["time_eps"]), max_batches=max_batches,
    )
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "split": split,
        "loss": loss,
        "device": str(resolved_device),
        "normalization": normalization,
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint["train_config"],
    }


# ---------------------------------------------------------------------------
# Stage 1a: variance transition kernel
# ---------------------------------------------------------------------------


def build_vol_datasets(
    data_dir: str | Path,
    normalization: dict[str, float],
    num_actions: int,
    train_action_dropout_prob: float = 0.0,
    lambert_w_delta: float = 0.0,
) -> dict[str, HestonVolTransitionDataset]:
    data_dir = Path(data_dir)
    return {
        split: HestonVolTransitionDataset(
            data_dir / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            num_actions=num_actions,
            action_dropout_prob=train_action_dropout_prob if split == "train" else 0.0,
            lambert_w_delta=lambert_w_delta,
        )
        for split in ("train", "val", "test")
    }


def train_vol_trans_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    num_actions: int | None = None,
    model_config: TwoStageFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
    lambert_w_delta: float = 0.0,
) -> dict[str, Any]:
    """Train Stage 1a: ``p(v_{t+1} | v_t, a_t)``.

    If ``num_actions`` is None it is read from the data ``metadata.json``.
    ``model_config`` defaults to ``state_dim=1, condition_dim=1+num_actions``.

    ``lambert_w_delta > 0`` trains the variance kernel in a Lambert-W
    Gaussianized target domain (heavy-tail trick borrowed from the Quant GAN
    baseline). The delta is stored in the checkpoint so the sampler inverts it
    automatically at rollout. Default 0 -> identity (unchanged behavior).
    """

    train_config = train_config or TransitionFMTrainConfig()
    set_seed(train_config.seed)

    if num_actions is None:
        num_actions = load_num_actions(data_dir)
    normalization = load_normalization(data_dir)
    datasets = build_vol_datasets(
        data_dir, normalization, num_actions,
        train_action_dropout_prob=train_config.action_dropout_prob,
        lambert_w_delta=lambert_w_delta,
    )

    if model_config is None:
        model_config = TwoStageFMModelConfig(
            state_dim=1, condition_dim=1 + num_actions,
        )
    else:
        expected = 1 + num_actions
        if model_config.state_dim != 1:
            raise ValueError("vol-stage model_config.state_dim must be 1")
        if model_config.condition_dim != expected:
            raise ValueError(
                f"vol-stage condition_dim must be 1 + num_actions = {expected},"
                f" got {model_config.condition_dim}"
            )

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="vol_trans_fm")
    model = TransitionFM(**asdict(model_config))
    return _run_fm_training(
        model=model,
        datasets=datasets,
        train_config=train_config,
        run_dir=run_dir,
        normalization=normalization,
        model_config_dict=asdict(model_config),
        stage="vol",
        num_actions=num_actions,
        config_blob_extra={
            "data_dir": str(Path(data_dir).resolve()),
            "lambert_w_delta": float(lambert_w_delta),
        },
        checkpoint_extra={"lambert_w_delta": float(lambert_w_delta)},
    )


# ---------------------------------------------------------------------------
# Stage 1b: return transition kernel
# ---------------------------------------------------------------------------


def build_ret_datasets(
    data_dir: str | Path,
    normalization: dict[str, float],
    num_actions: int,
    train_action_dropout_prob: float = 0.0,
) -> dict[str, HestonRetTransitionDataset]:
    data_dir = Path(data_dir)
    return {
        split: HestonRetTransitionDataset(
            data_dir / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            return_mean=normalization["return_mean"],
            return_std=normalization["return_std"],
            num_actions=num_actions,
            action_dropout_prob=train_action_dropout_prob if split == "train" else 0.0,
        )
        for split in ("train", "val", "test")
    }


def _scheduled_sampling_prob_for_epoch(
    epoch: int,
    total_epochs: int,
    max_prob: float,
    start_epoch: int,
) -> float:
    if max_prob <= 0.0:
        return 0.0
    if not 0.0 <= max_prob <= 1.0:
        raise ValueError("scheduled_sampling_max_prob must be in [0, 1]")
    if start_epoch <= 0:
        raise ValueError("scheduled_sampling_start_epoch must be positive")
    if epoch < start_epoch:
        return 0.0
    active_epochs = max(total_epochs - start_epoch + 1, 1)
    progress = (epoch - start_epoch + 1) / active_epochs
    return min(max_prob, max_prob * progress)


def _make_ret_scheduled_sampling_factory(
    *,
    vol_sampler_checkpoint: str | Path,
    device: torch.device,
    num_actions: int,
    train_config: TransitionFMTrainConfig,
):
    from finflow.inference.samplers import load_sampler_from_checkpoint

    loaded = load_sampler_from_checkpoint(
        vol_sampler_checkpoint,
        device=device,
        fm_n_steps=train_config.scheduled_sampling_fm_steps,
    )
    if loaded.stage != "vol":
        raise ValueError(
            f"vol_sampler_checkpoint must load a vol-stage sampler, got stage='{loaded.stage}'"
        )
    if loaded.num_actions != num_actions:
        raise ValueError(
            f"vol sampler num_actions={loaded.num_actions} does not match ret data num_actions={num_actions}"
        )
    expected_condition_dim = 1 + num_actions
    if loaded.sampler.condition_dim != expected_condition_dim:
        raise ValueError(
            f"vol sampler condition_dim must be {expected_condition_dim}, "
            f"got {loaded.sampler.condition_dim}"
        )

    def factory(epoch: int):
        prob = _scheduled_sampling_prob_for_epoch(
            epoch,
            train_config.epochs,
            train_config.scheduled_sampling_max_prob,
            train_config.scheduled_sampling_start_epoch,
        )

        def transform(
            batch: dict[str, torch.Tensor],
            condition: torch.Tensor,
        ) -> torch.Tensor:
            if prob <= 0.0:
                return condition
            vol_condition = torch.cat([condition[:, 1:2], condition[:, 3:]], dim=1)
            with torch.no_grad():
                sampled_log_v_next = loaded.sampler.sample(vol_condition).to(
                    device=condition.device,
                    dtype=condition.dtype,
                )
            mask = (torch.rand(condition.shape[0], 1, device=condition.device) < prob)
            updated = condition.clone()
            updated[:, 0:1] = torch.where(mask, sampled_log_v_next, updated[:, 0:1])
            return updated

        return transform, {
            "scheduled_sampling_prob": prob,
            "scheduled_sampling_checkpoint": str(Path(vol_sampler_checkpoint).resolve()),
        }

    return factory


def train_ret_trans_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    num_actions: int | None = None,
    model_config: TwoStageFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
    vol_sampler_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train Stage 1b: ``p(r_{t+1} | v_{t+1}, v_t, r_t, a_t)``.

    By default training uses teacher-forced ground-truth ``v_{t+1}`` from the
    data. If ``vol_sampler_checkpoint`` is provided, scheduled sampling replaces
    a linearly increasing fraction of each training batch's ``v_{t+1}`` with
    Stage 1a sampler output.
    """

    train_config = train_config or TransitionFMTrainConfig()
    set_seed(train_config.seed)

    if num_actions is None:
        num_actions = load_num_actions(data_dir)
    normalization = load_normalization(data_dir)
    datasets = build_ret_datasets(
        data_dir, normalization, num_actions,
        train_action_dropout_prob=train_config.action_dropout_prob,
    )

    if model_config is None:
        model_config = TwoStageFMModelConfig(
            state_dim=1, condition_dim=3 + num_actions,
        )
    else:
        expected = 3 + num_actions
        if model_config.state_dim != 1:
            raise ValueError("ret-stage model_config.state_dim must be 1")
        if model_config.condition_dim != expected:
            raise ValueError(
                f"ret-stage condition_dim must be 3 + num_actions = {expected},"
                f" got {model_config.condition_dim}"
            )

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="ret_trans_fm")
    model = TransitionFM(**asdict(model_config))
    resolved_device = resolve_device(train_config.device)
    train_condition_transform_factory = None
    scheduled_extra: dict[str, Any] = {"data_dir": str(Path(data_dir).resolve())}
    if vol_sampler_checkpoint is not None:
        train_condition_transform_factory = _make_ret_scheduled_sampling_factory(
            vol_sampler_checkpoint=vol_sampler_checkpoint,
            device=resolved_device,
            num_actions=num_actions,
            train_config=train_config,
        )
        scheduled_extra.update({
            "vol_sampler_checkpoint": str(Path(vol_sampler_checkpoint).resolve()),
            "scheduled_sampling_enabled": train_config.scheduled_sampling_max_prob > 0.0,
        })
    return _run_fm_training(
        model=model,
        datasets=datasets,
        train_config=train_config,
        run_dir=run_dir,
        normalization=normalization,
        model_config_dict=asdict(model_config),
        stage="ret",
        num_actions=num_actions,
        config_blob_extra=scheduled_extra,
        train_condition_transform_factory=train_condition_transform_factory,
    )


# ---------------------------------------------------------------------------
# Shared evaluation entry for two-stage checkpoints
# ---------------------------------------------------------------------------


def evaluate_two_stage_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    stage: Literal["vol", "ret"],
    split: str = "val",
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a vol- or ret-stage FM checkpoint."""

    resolved_device = resolve_device(device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, map_location=resolved_device)
    normalization = checkpoint["normalization"]
    num_actions = int(checkpoint.get("num_actions", 1))
    ckpt_stage = checkpoint.get("stage", "joint")
    if ckpt_stage != stage:
        raise ValueError(f"checkpoint stage='{ckpt_stage}' does not match requested stage='{stage}'")

    if stage == "vol":
        dataset = HestonVolTransitionDataset(
            Path(data_dir) / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            num_actions=num_actions,
        )
    elif stage == "ret":
        dataset = HestonRetTransitionDataset(
            Path(data_dir) / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            return_mean=normalization["return_mean"],
            return_std=normalization["return_std"],
            num_actions=num_actions,
        )
    else:
        raise ValueError("stage must be 'vol' or 'ret'")

    loader = build_dataloader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, device=resolved_device,
    )
    loss = evaluate_model(
        model, loader, device=resolved_device,
        time_eps=float(checkpoint["train_config"]["time_eps"]), max_batches=max_batches,
    )
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "stage": stage,
        "split": split,
        "loss": loss,
        "device": str(resolved_device),
        "normalization": normalization,
        "num_actions": num_actions,
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint["train_config"],
    }
