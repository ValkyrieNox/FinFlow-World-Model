"""Quant GAN baseline (Wiese et al. 2020).

A small Temporal Convolutional Network (TCN) generator and discriminator
trained with WGAN-GP on Lambert-W Gaussianized Heston log-return sequences.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from finflow.training import (
    _effective_num_batches,
    _fmt_time,
    _iterate_batches,
    _make_progress,
    build_run_dir,
    load_metadata,
    load_normalization,
    resolve_device,
    set_seed,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _lambertw_principal_nonnegative(z: np.ndarray, *, max_iter: int = 20) -> np.ndarray:
    """Principal Lambert W branch for non-negative real inputs."""

    z = np.asarray(z, dtype=np.float64)
    if np.any(z < 0):
        raise ValueError("Lambert W fallback expects non-negative inputs")
    w = np.where(z < 1.0, z / (1.0 + z), np.log1p(z))
    w = np.maximum(w, 0.0)
    for _ in range(max_iter):
        ew = np.exp(w)
        f = w * ew - z
        denom = ew * (w + 1.0) - ((w + 2.0) * f) / np.maximum(2.0 * w + 2.0, 1e-12)
        step = f / np.maximum(denom, 1e-12)
        w_next = np.maximum(w - step, 0.0)
        if np.max(np.abs(w_next - w)) < 1e-12:
            return w_next
        w = w_next
    return w


def lambert_w_transform(values: np.ndarray, delta: float = 0.1) -> np.ndarray:
    """Gaussianize heavy-tailed standardized returns with the Lambert-W inverse.

    The input is expected to already be centered and scaled. ``delta=0`` keeps
    the legacy identity transform.
    """

    x = np.asarray(values, dtype=np.float64)
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if delta == 0:
        return x.astype(np.float32, copy=False)
    z = delta * np.square(x)
    w = _lambertw_principal_nonnegative(z)
    y = np.sign(x) * np.sqrt(np.maximum(w, 0.0) / delta)
    return y.astype(np.float32, copy=False)


def inverse_lambert_w_transform(values: np.ndarray, delta: float = 0.1) -> np.ndarray:
    """Map Lambert-W Gaussianized returns back to the standardized return domain."""

    y = np.asarray(values, dtype=np.float64)
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if delta == 0:
        return y.astype(np.float32, copy=False)
    exponent = np.clip(0.5 * delta * np.square(y), 0.0, 20.0)
    x = y * np.exp(exponent)
    return x.astype(np.float32, copy=False)


def calibrate_standardized_moments(
    values: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float]]:
    """Affine-calibrate generated standardized returns to zero mean / unit std."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    x = np.asarray(values, dtype=np.float64)
    before_mean = float(x.mean())
    before_std = float(x.std(ddof=0))
    scale = max(before_std, eps)
    y = (x - before_mean) / scale
    info = {
        "before_mean": before_mean,
        "before_std": before_std,
        "after_mean": float(y.mean()),
        "after_std": float(y.std(ddof=0)),
    }
    return y.astype(np.float32, copy=False), info


class HestonLogReturnSequenceDataset(Dataset[torch.Tensor]):
    """Yield transformed log-return sequences ``[n_steps]`` from a split ``.npz``."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        return_mean: float = 0.0,
        return_std: float = 1.0,
        lambert_w_delta: float = 0.1,
    ) -> None:
        self.npz_path = Path(npz_path)
        if return_std <= 0:
            raise ValueError("return_std must be positive")
        if lambert_w_delta < 0:
            raise ValueError("lambert_w_delta must be non-negative")
        npz = np.load(self.npz_path)
        if "log_returns" not in npz.files:
            raise ValueError(f"{self.npz_path} missing 'log_returns'")
        self.returns = np.asarray(npz["log_returns"], dtype=np.float32)
        npz.close()
        self.return_mean = float(return_mean)
        self.return_std = float(return_std)
        self.lambert_w_delta = float(lambert_w_delta)
        returns_norm = (self.returns - self.return_mean) / self.return_std
        self.transformed_returns = lambert_w_transform(
            returns_norm, delta=self.lambert_w_delta,
        )

    def __len__(self) -> int:
        return int(self.returns.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        r_trans = self.transformed_returns[index]
        return torch.from_numpy(np.ascontiguousarray(r_trans)).float()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class _DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
        )
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(x + self.net(x))


class QuantGANGenerator(nn.Module):
    """1D TCN generator: ``z [B, latent_dim, L] -> returns [B, 1, L]``."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_channels: int = 32,
        num_blocks: int = 5,
        kernel_size: int = 3,
        seq_len: int = 252,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_proj = nn.Conv1d(latent_dim, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _DilatedConvBlock(hidden_channels, kernel_size, dilation=2 ** i)
                for i in range(num_blocks)
            ]
        )
        self.output = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        # Affine head only: no LayerNorm-over-time (would erase per-path heterogeneity)
        # and no tanh (would bound outputs and crush heavy tails). Wiese 2020 §3.4
        # uses a plain linear final layer; the Lambert-W Gaussianization makes the
        # target distribution near-Gaussian, so heavy tails are recovered by the
        # inverse Lambert-W transform at sampling time.
        self.output_scale = nn.Parameter(torch.ones(1, 1, 1))
        self.output_shift = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        raw = self.output(h)
        return self.output_scale * raw + self.output_shift


class QuantGANDiscriminator(nn.Module):
    """1D TCN discriminator: ``returns [B, 1, L] -> realism score [B]``."""

    def __init__(
        self,
        hidden_channels: int = 32,
        num_blocks: int = 5,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(1, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _DilatedConvBlock(hidden_channels, kernel_size, dilation=2 ** i)
                for i in range(num_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.pool(h).squeeze(-1)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantGANConfig:
    latent_dim: int = 8
    hidden_channels: int = 32
    num_blocks: int = 5
    kernel_size: int = 3
    seq_len: int = 252


@dataclass(frozen=True)
class QuantGANTrainConfig:
    batch_size: int = 128
    epochs: int = 30
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple[float, float] = (0.0, 0.9)
    d_steps_per_g: int = 5
    gradient_penalty_weight: float = 10.0
    lambert_w_delta: float = 0.1
    moment_penalty_weight: float = 1.0
    moment_mean_weight: float = 1.0
    moment_std_weight: float = 1.0
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    seed: int = 1234
    device: str = "auto"
    max_train_batches: int | None = None
    progress: bool = True
    progress_min_interval: float = 0.2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _gradient_penalty(
    discriminator: QuantGANDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, device=real.device, dtype=real.dtype)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = discriminator(interpolated)
    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(batch_size, -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def _moment_penalty(
    fake: torch.Tensor,
    real: torch.Tensor,
    *,
    mean_weight: float = 1.0,
    std_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match global first/second moments in the transformed training domain."""

    if mean_weight < 0 or std_weight < 0:
        raise ValueError("moment penalty weights must be non-negative")
    fake_mean = fake.mean()
    real_mean = real.mean()
    fake_std = fake.std(unbiased=False).clamp_min(eps)
    real_std = real.std(unbiased=False).clamp_min(eps)
    return (
        mean_weight * (fake_mean - real_mean).square()
        + std_weight * (fake_std - real_std).square()
    )


def train_quant_gan(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model_config: QuantGANConfig | None = None,
    train_config: QuantGANTrainConfig | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train a Quant GAN baseline on Heston log-return sequences."""

    model_config = model_config or QuantGANConfig()
    train_config = train_config or QuantGANTrainConfig()
    if train_config.d_steps_per_g <= 0:
        raise ValueError("d_steps_per_g must be positive")
    if train_config.gradient_penalty_weight < 0:
        raise ValueError("gradient_penalty_weight must be non-negative")
    if train_config.lambert_w_delta < 0:
        raise ValueError("lambert_w_delta must be non-negative")
    if train_config.moment_penalty_weight < 0:
        raise ValueError("moment_penalty_weight must be non-negative")
    if train_config.moment_mean_weight < 0 or train_config.moment_std_weight < 0:
        raise ValueError("moment mean/std weights must be non-negative")
    set_seed(train_config.seed)

    device = resolve_device(train_config.device)
    metadata = load_metadata(data_dir)
    normalization = load_normalization(data_dir)
    n_steps_meta = int(metadata.get("n_steps", model_config.seq_len))
    if n_steps_meta != model_config.seq_len:
        raise ValueError(
            f"metadata n_steps={n_steps_meta} does not match config seq_len={model_config.seq_len}"
        )

    data_dir = Path(data_dir)
    train_dataset = HestonLogReturnSequenceDataset(
        data_dir / "train.npz",
        return_mean=normalization["return_mean"],
        return_std=normalization["return_std"],
        lambert_w_delta=train_config.lambert_w_delta,
    )
    val_dataset = HestonLogReturnSequenceDataset(
        data_dir / "val.npz",
        return_mean=normalization["return_mean"],
        return_std=normalization["return_std"],
        lambert_w_delta=train_config.lambert_w_delta,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True,
        num_workers=train_config.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, pin_memory=device.type == "cuda",
    )

    generator = QuantGANGenerator(
        latent_dim=model_config.latent_dim,
        hidden_channels=model_config.hidden_channels,
        num_blocks=model_config.num_blocks,
        kernel_size=model_config.kernel_size,
        seq_len=model_config.seq_len,
    ).to(device)
    discriminator = QuantGANDiscriminator(
        hidden_channels=model_config.hidden_channels,
        num_blocks=model_config.num_blocks,
        kernel_size=model_config.kernel_size,
    ).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=train_config.lr_g, betas=train_config.betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=train_config.lr_d, betas=train_config.betas)

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="quant_gan")
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    config_blob = {
        "run_dir": str(run_dir.resolve()),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "normalization": normalization,
        "data_dir": str(Path(data_dir).resolve()),
    }
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    disable_progress = not train_config.progress
    train_batches = _effective_num_batches(train_loader, train_config.max_train_batches)
    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())
    if train_config.progress:
        header = (
            f"[finflow] quant_gan | run={run_dir.name} | device={device} | "
            f"params_G={n_params_g/1e3:.1f}k params_D={n_params_d/1e3:.1f}k | "
            f"wgan_gp={train_config.gradient_penalty_weight:g} "
            f"moment={train_config.moment_penalty_weight:g} "
            f"lambert_delta={train_config.lambert_w_delta:g} | "
            f"train={len(train_dataset)} samples ({train_batches} batch x {train_config.batch_size}) | "
            f"epochs={train_config.epochs}"
        )
        print(header, file=sys.stderr, flush=True)

    history: list[dict[str, Any]] = []
    run_start = time.monotonic()
    global_step = 0
    best_val = float("inf")

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.monotonic()
        bar = _make_progress(
            _iterate_batches(train_loader, train_config.max_train_batches),
            total=train_batches,
            desc=f"epoch {epoch:>3}/{train_config.epochs} train",
            disable=disable_progress,
            min_interval=train_config.progress_min_interval,
        )
        running_g, running_g_adv, running_d = 0.0, 0.0, 0.0
        running_gp, running_w, running_moment, running_n = 0.0, 0.0, 0.0, 0
        generator.train(); discriminator.train()
        for batch in bar:
            real = batch.to(device).unsqueeze(1)  # [B, 1, L]
            B = real.shape[0]

            # Train D
            for _ in range(train_config.d_steps_per_g):
                z = torch.randn(B, model_config.latent_dim, model_config.seq_len, device=device)
                with torch.no_grad():
                    fake = generator(z)
                d_real = discriminator(real)
                d_fake = discriminator(fake)
                gp = _gradient_penalty(discriminator, real, fake)
                wasserstein_est = d_real.mean() - d_fake.mean()
                loss_d = d_fake.mean() - d_real.mean() + train_config.gradient_penalty_weight * gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if train_config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), train_config.grad_clip_norm)
                opt_d.step()

            # Train G
            z = torch.randn(B, model_config.latent_dim, model_config.seq_len, device=device)
            fake = generator(z)
            d_fake = discriminator(fake)
            loss_g_adv = -d_fake.mean()
            if train_config.moment_penalty_weight > 0:
                moment_loss = _moment_penalty(
                    fake,
                    real,
                    mean_weight=train_config.moment_mean_weight,
                    std_weight=train_config.moment_std_weight,
                )
            else:
                moment_loss = fake.new_tensor(0.0)
            loss_g = loss_g_adv + train_config.moment_penalty_weight * moment_loss
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if train_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), train_config.grad_clip_norm)
            opt_g.step()

            running_g += float(loss_g.item()) * B
            running_g_adv += float(loss_g_adv.item()) * B
            running_d += float(loss_d.item()) * B
            running_gp += float(gp.item()) * B
            running_w += float(wasserstein_est.item()) * B
            running_moment += float(moment_loss.item()) * B
            running_n += B
            if not disable_progress:
                bar.set_postfix(
                    g=f"{running_g / max(running_n, 1):.4f}",
                    d=f"{running_d / max(running_n, 1):.4f}",
                    gp=f"{running_gp / max(running_n, 1):.3f}",
                    mom=f"{running_moment / max(running_n, 1):.3f}",
                    refresh=False,
                )
        bar.close()
        train_g = running_g / max(running_n, 1)
        train_g_adv = running_g_adv / max(running_n, 1)
        train_d = running_d / max(running_n, 1)
        train_gp = running_gp / max(running_n, 1)
        train_w = running_w / max(running_n, 1)
        train_moment = running_moment / max(running_n, 1)

        # Validation: critic gap on real validation batch + fake.
        generator.eval(); discriminator.eval()
        v_running, v_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                real = batch.to(device).unsqueeze(1)
                B = real.shape[0]
                z = torch.randn(B, model_config.latent_dim, model_config.seq_len, device=device)
                fake = generator(z)
                d_real = discriminator(real)
                d_fake = discriminator(fake)
                critic_loss = d_fake.mean() - d_real.mean()
                v_running += float(critic_loss.item()) * B
                v_n += B
        val_d = v_running / max(v_n, 1)
        epoch_time = time.monotonic() - epoch_start
        global_step += train_batches

        record = {
            "epoch": epoch,
            "train_g_loss": train_g,
            "train_g_adv_loss": train_g_adv,
            "train_d_loss": train_d,
            "train_gradient_penalty": train_gp,
            "train_wasserstein_estimate": train_w,
            "train_moment_loss": train_moment,
            "val_d_loss": val_d,
            "epoch_time_s": epoch_time,
            "global_step": global_step,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        is_best = val_d < best_val
        if is_best:
            best_val = val_d
        ckpt = {
            "generator_state": {n: t.detach().cpu() for n, t in generator.state_dict().items()},
            "discriminator_state": {n: t.detach().cpu() for n, t in discriminator.state_dict().items()},
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "normalization": normalization,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_d_loss": best_val,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")

        if train_config.progress:
            elapsed = time.monotonic() - run_start
            eta_s = (elapsed / epoch) * (train_config.epochs - epoch)
            print(
                f"  epoch {epoch:>3}/{train_config.epochs} | "
                f"G={train_g:.4f} Gadv={train_g_adv:.4f} D={train_d:.4f} "
                f"GP={train_gp:.4f} W={train_w:.4f} Mom={train_moment:.4f} | "
                f"val_D={val_d:.4f}{' *' if is_best else '  '} | "
                f"epoch={_fmt_time(epoch_time)} | elapsed={_fmt_time(elapsed)} | "
                f"eta={_fmt_time(eta_s)}",
                file=sys.stderr, flush=True,
            )

    summary = {
        "run_dir": str(run_dir),
        "checkpoints": {
            "best": str(ckpt_dir / "best.pt"),
            "last": str(ckpt_dir / "last.pt"),
        },
        "best_val_d_loss": best_val,
        "history": history,
        "device": str(device),
        "total_time_s": time.monotonic() - run_start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_quant_gan_paths(
    generator: QuantGANGenerator,
    n_paths: int,
    *,
    s0: float = 100.0,
    return_mean: float = 0.0,
    return_std: float = 1.0,
    lambert_w_delta: float = 0.1,
    calibrate_moments: bool = True,
    calibration_eps: float = 1e-6,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate ``n_paths`` log-return + price sequences from a trained Quant GAN."""

    device = torch.device(device) if isinstance(device, str) else device
    generator = generator.to(device).eval()
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)
    z = torch.randn(
        n_paths, generator.latent_dim, generator.seq_len,
        generator=rng,
    ).to(device)
    fake_trans = generator(z).squeeze(1).cpu().numpy()  # [B, L]
    fake_norm = inverse_lambert_w_transform(fake_trans, delta=lambert_w_delta)
    calibration: dict[str, float | bool]
    if calibrate_moments:
        fake_norm, calibration_info = calibrate_standardized_moments(
            fake_norm,
            eps=calibration_eps,
        )
        calibration = {"enabled": True, **calibration_info}
    else:
        calibration = {
            "enabled": False,
            "before_mean": float(fake_norm.mean()),
            "before_std": float(fake_norm.std(ddof=0)),
            "after_mean": float(fake_norm.mean()),
            "after_std": float(fake_norm.std(ddof=0)),
        }
    fake = fake_norm * return_std + return_mean
    s = np.empty((n_paths, generator.seq_len + 1), dtype=np.float32)
    s[:, 0] = s0
    log_s = np.log(s0) + np.cumsum(fake, axis=1)
    s[:, 1:] = np.exp(log_s).astype(np.float32)
    return {
        "log_returns": fake.astype(np.float32),
        "s_paths": s,
        "calibration": calibration,
    }


def load_quant_gan_generator(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[QuantGANGenerator, dict[str, Any]]:
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    cfg = ckpt["model_config"]
    generator = QuantGANGenerator(
        latent_dim=int(cfg["latent_dim"]),
        hidden_channels=int(cfg["hidden_channels"]),
        num_blocks=int(cfg["num_blocks"]),
        kernel_size=int(cfg["kernel_size"]),
        seq_len=int(cfg["seq_len"]),
    )
    generator.load_state_dict(ckpt["generator_state"], strict=False)
    generator.to(map_location).eval()
    return generator, ckpt
