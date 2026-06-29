import numpy as np

from finflow.eval import marginal_wasserstein_curve, path_wasserstein, wasserstein_1d


def test_wasserstein_1d_same_distribution_is_small():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024)
    y = rng.standard_normal(1024)
    d = wasserstein_1d(x, y)
    assert d < 0.1


def test_wasserstein_1d_separated_distributions_large():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(512)
    y = rng.standard_normal(512) + 5.0
    d = wasserstein_1d(x, y)
    assert d > 4.0


def test_wasserstein_1d_different_sizes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(100)
    y = rng.standard_normal(250)
    d = wasserstein_1d(x, y)
    assert d >= 0.0


def test_marginal_wasserstein_curve_shape():
    rng = np.random.default_rng(3)
    real = rng.standard_normal((128, 16))
    fake = rng.standard_normal((128, 16))
    curve = marginal_wasserstein_curve(real, fake)
    assert curve.shape == (16,)
    assert np.all(curve >= 0)


def test_path_wasserstein_reducer_sum_finite():
    rng = np.random.default_rng(4)
    real = rng.standard_normal((64, 32))
    fake = rng.standard_normal((64, 32))
    d = path_wasserstein(real, fake, reducer="sum")
    assert np.isfinite(d) and d >= 0
