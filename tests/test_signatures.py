import numpy as np

from finflow.eval import (
    returns_to_time_cumsum_paths,
    signature_features,
    signature_wasserstein,
)


def test_signature_features_shape_and_identity_on_identical_paths():
    returns = np.array([[0.01, -0.02, 0.03], [0.0, 0.01, -0.01]], dtype=np.float64)
    paths = returns_to_time_cumsum_paths(returns)
    features = signature_features(paths, depth=3)
    assert paths.shape == (2, 4, 2)
    assert features.shape == (2, 14)

    distance = signature_wasserstein(returns, returns.copy(), depth=3)
    assert distance["depth"] == 3
    assert distance["mean"] == 0.0
    assert distance["max"] == 0.0
    assert len(distance["per_coordinate"]) == 14


def test_signature_wasserstein_detects_path_distribution_shift():
    real = np.zeros((8, 4), dtype=np.float64)
    fake = np.full((8, 4), 0.01, dtype=np.float64)
    distance = signature_wasserstein(real, fake, depth=2)
    assert distance["mean"] > 0.0
    assert distance["max"] > 0.0
