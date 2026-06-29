import numpy as np

from finflow.eval import (
    StylizedFactReport,
    aggregational_kurtosis,
    autocorrelation,
    compare_stylized_facts,
    stylized_fact_report,
)
from finflow.eval.stylized_facts import (
    absolute_return_acf,
    kurtosis,
    leverage_correlation,
    tail_index_hill,
)


def test_kurtosis_of_gaussian_is_near_three():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 252))
    k = kurtosis(x)
    assert 2.5 < k < 3.5


def test_autocorrelation_of_iid_noise_is_small():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((128, 252))
    acf = autocorrelation(x, lags=5)
    assert acf.shape == (5,)
    assert np.all(np.abs(acf) < 0.15)


def test_absolute_return_acf_shape():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((64, 100))
    out = absolute_return_acf(x, lags=10)
    assert out.shape == (10,)


def test_leverage_correlation_shape_and_finite():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((64, 100))
    out = leverage_correlation(x, lags=5)
    assert out.shape == (5,)
    assert np.isfinite(out).all()


def test_aggregational_kurtosis_returns_finite_dict():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((64, 252))
    out = aggregational_kurtosis(x, scales=(1, 5, 21))
    assert set(out.keys()) == {1, 5, 21}
    for v in out.values():
        assert np.isfinite(v)


def test_tail_index_hill_positive_for_gaussian():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((256, 252))
    idx = tail_index_hill(x, frac=0.05)
    assert np.isfinite(idx)
    assert idx > 0


def test_stylized_fact_report_to_dict_serializable():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((32, 64))
    report = stylized_fact_report(x, return_acf_lags=4, absolute_acf_lags=4, leverage_lags=3,
                                  aggregation_scales=(1, 4))
    assert isinstance(report, StylizedFactReport)
    d = report.to_dict()
    assert "kurtosis" in d and "return_acf" in d and "tail_index" in d


def test_compare_stylized_facts_self_diff_is_zero():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((32, 64))
    rep = stylized_fact_report(x, return_acf_lags=4, absolute_acf_lags=4, leverage_lags=3,
                               aggregation_scales=(1, 4))
    cmp = compare_stylized_facts(rep, rep)
    assert cmp["kurtosis_abs_diff"] == 0.0
    assert cmp["return_acf_l1"] == 0.0
