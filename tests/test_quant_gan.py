from pathlib import Path

import numpy as np
import torch

from finflow.baselines import (
    QuantGANConfig,
    QuantGANDiscriminator,
    QuantGANGenerator,
    QuantGANTrainConfig,
    calibrate_standardized_moments,
    inverse_lambert_w_transform,
    lambert_w_transform,
    sample_quant_gan_paths,
    train_quant_gan,
)
from finflow.baselines.quant_gan import load_quant_gan_generator
from finflow.data import generate_heston_dataset


def test_quant_gan_generator_forward_shape():
    g = QuantGANGenerator(latent_dim=4, hidden_channels=8, num_blocks=2, kernel_size=3, seq_len=24)
    z = torch.randn(3, 4, 24)
    out = g(z)
    assert out.shape == (3, 1, 24)


def test_quant_gan_discriminator_forward_shape():
    d = QuantGANDiscriminator(hidden_channels=8, num_blocks=2, kernel_size=3)
    x = torch.randn(3, 1, 24)
    out = d(x)
    assert out.shape == (3,)


def test_lambert_w_transform_round_trip():
    x = torch.linspace(-2.0, 2.0, steps=25).numpy()
    y = lambert_w_transform(x, delta=0.1)
    restored = inverse_lambert_w_transform(y, delta=0.1)
    assert torch.allclose(torch.from_numpy(restored), torch.from_numpy(x).float(), atol=1e-5)


def test_calibrate_standardized_moments_matches_zero_one():
    x = np.linspace(-2.0, 3.0, num=64, dtype=np.float32).reshape(8, 8)
    y, info = calibrate_standardized_moments(x)
    assert abs(float(y.mean())) < 1e-6
    assert abs(float(y.std(ddof=0)) - 1.0) < 1e-6
    assert info["before_mean"] != info["after_mean"]


def test_sample_quant_gan_moment_calibration_matches_requested_stats():
    g = QuantGANGenerator(latent_dim=4, hidden_channels=8, num_blocks=2, kernel_size=3, seq_len=16)
    out = sample_quant_gan_paths(
        g,
        n_paths=32,
        s0=100.0,
        return_mean=0.001,
        return_std=0.02,
        lambert_w_delta=0.1,
        calibrate_moments=True,
        device="cpu",
        seed=7,
    )
    returns = out["log_returns"]
    assert abs(float(returns.mean()) - 0.001) < 1e-6
    assert abs(float(returns.std(ddof=0)) - 0.02) < 1e-6
    assert out["calibration"]["enabled"] is True


def test_train_quant_gan_smoke(tmp_path: Path):
    data_dir = tmp_path / "data"
    generate_heston_dataset(
        data_dir, n_train=8, n_val=4, n_test=4, n_steps=16, seed=21, save_transitions=False,
    )
    summary = train_quant_gan(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_gan",
        run_name="smoke",
        model_config=QuantGANConfig(latent_dim=4, hidden_channels=8, num_blocks=2,
                                    kernel_size=3, seq_len=16),
        train_config=QuantGANTrainConfig(
            batch_size=4, epochs=1, lr_g=1e-3, lr_d=1e-3,
            d_steps_per_g=1, seed=22, device="cpu", max_train_batches=2, progress=False,
        ),
    )
    best = Path(summary["checkpoints"]["best"])
    assert best.exists()

    generator, ckpt = load_quant_gan_generator(best, map_location="cpu")
    out = sample_quant_gan_paths(
        generator, n_paths=5, s0=100.0,
        return_mean=ckpt["normalization"]["return_mean"],
        return_std=ckpt["normalization"]["return_std"],
        lambert_w_delta=ckpt["train_config"]["lambert_w_delta"],
        calibrate_moments=True,
        device="cpu", seed=0,
    )
    assert out["log_returns"].shape == (5, 16)
    assert out["s_paths"].shape == (5, 17)
