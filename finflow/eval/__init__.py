"""Evaluation suite for V3: stylized facts, distances, pricing, reports."""

from finflow.eval.distances import (
    marginal_wasserstein_curve,
    path_wasserstein,
    wasserstein_1d,
)
from finflow.eval.pricing import (
    asian_pricing_rmse_vs_mc_oracle,
    mc_arithmetic_asian_call_prices_grid,
    mc_call_prices_grid,
    pricing_rmse_vs_carr_madan,
    pricing_rmse_vs_mc_oracle,
    pricing_rmse_vs_reference,
)
from finflow.eval.reports import build_full_report
from finflow.eval.signatures import (
    returns_to_time_cumsum_paths,
    signature_features,
    signature_wasserstein,
)
from finflow.eval.stylized_facts import (
    StylizedFactReport,
    aggregational_kurtosis,
    autocorrelation,
    compare_stylized_facts,
    leverage_correlation,
    stylized_fact_report,
    tail_index_hill,
)

__all__ = [
    "StylizedFactReport",
    "aggregational_kurtosis",
    "autocorrelation",
    "build_full_report",
    "compare_stylized_facts",
    "leverage_correlation",
    "asian_pricing_rmse_vs_mc_oracle",
    "mc_arithmetic_asian_call_prices_grid",
    "marginal_wasserstein_curve",
    "mc_call_prices_grid",
    "path_wasserstein",
    "pricing_rmse_vs_carr_madan",
    "pricing_rmse_vs_mc_oracle",
    "pricing_rmse_vs_reference",
    "returns_to_time_cumsum_paths",
    "signature_features",
    "signature_wasserstein",
    "stylized_fact_report",
    "tail_index_hill",
    "wasserstein_1d",
]
