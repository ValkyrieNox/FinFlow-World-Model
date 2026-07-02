"""End-to-end report builder: stylized facts + distances + pricing."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from finflow.data import HestonParams
from finflow.eval.distances import marginal_wasserstein_curve, path_wasserstein
from finflow.eval.pricing import (
    PricingComparison,
    asian_pricing_rmse_vs_mc_oracle,
    pricing_rmse_vs_carr_madan,
    pricing_rmse_vs_mc_oracle,
)
from finflow.eval.signatures import signature_wasserstein
from finflow.eval.stylized_facts import (
    StylizedFactReport,
    compare_stylized_facts,
    stylized_fact_report,
)


def build_full_report(
    *,
    real_returns: np.ndarray,
    fake_returns: np.ndarray,
    real_s_paths: np.ndarray | None = None,
    fake_s_paths: np.ndarray | None = None,
    oracle_s_paths: np.ndarray | None = None,
    params: HestonParams | None = None,
    moneynesses: Sequence[float] = (
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.95,
        1.0,
        1.05,
        1.10,
        1.20,
        1.30,
        1.50,
    ),
    maturities: Sequence[float] = (0.25, 0.5, 1.0),
    asian_moneynesses: Sequence[float] | None = None,
    asian_maturities: Sequence[float] | None = None,
    dt: float = 1.0 / 252.0,
    pricing_r: float = 0.0,
    signature_depth: int | None = 3,
) -> dict[str, Any]:
    """Compute the full V3 evaluation matrix.

    Always returns the stylized-fact comparison and the per-step / total
    Wasserstein distances on log-returns. If ``params`` and ``*_s_paths`` are
    supplied, also returns the pricing RMSE vs Carr-Madan reference for the
    generated paths. If ``oracle_s_paths`` is supplied, additionally compares
    generated and real paths against that independent MC oracle.
    """

    real_returns = np.asarray(real_returns, dtype=np.float64)
    fake_returns = np.asarray(fake_returns, dtype=np.float64)

    real_facts: StylizedFactReport = stylized_fact_report(real_returns)
    fake_facts: StylizedFactReport = stylized_fact_report(fake_returns)
    comparison = compare_stylized_facts(real_facts, fake_facts)

    marginal_w = marginal_wasserstein_curve(real_returns, fake_returns)
    total_return_w = path_wasserstein(real_returns, fake_returns, reducer="sum")
    abs_total_w = path_wasserstein(real_returns, fake_returns, reducer="abs_sum")
    sig_w = (
        signature_wasserstein(real_returns, fake_returns, depth=signature_depth)
        if signature_depth is not None
        else None
    )

    out: dict[str, Any] = {
        "real_facts": real_facts.to_dict(),
        "fake_facts": fake_facts.to_dict(),
        "stylized_facts_comparison": comparison,
        "distances": {
            "marginal_wasserstein_mean": float(marginal_w.mean()),
            "marginal_wasserstein_max": float(marginal_w.max()),
            "marginal_wasserstein_curve": marginal_w.tolist(),
            "total_return_wasserstein": float(total_return_w),
            "abs_total_return_wasserstein": float(abs_total_w),
        },
    }
    if sig_w is not None:
        out["distances"]["signature_wasserstein"] = sig_w

    if params is not None and fake_s_paths is not None:
        pricing: PricingComparison = pricing_rmse_vs_carr_madan(
            fake_s_paths, dt=dt,
            moneynesses=moneynesses, maturities=maturities,
            params=params, r=pricing_r,
        )
        out["pricing_fake_vs_carr_madan"] = pricing.to_dict()
        if real_s_paths is not None:
            real_pricing: PricingComparison = pricing_rmse_vs_carr_madan(
                real_s_paths, dt=dt,
                moneynesses=moneynesses, maturities=maturities,
                params=params, r=pricing_r,
            )
            out["pricing_real_vs_carr_madan"] = real_pricing.to_dict()

    if oracle_s_paths is not None and fake_s_paths is not None:
        oracle_pricing: PricingComparison = pricing_rmse_vs_mc_oracle(
            fake_s_paths,
            oracle_s_paths,
            dt=dt,
            moneynesses=moneynesses,
            maturities=maturities,
            r=pricing_r,
        )
        out["pricing_fake_vs_mc_oracle"] = oracle_pricing.to_dict()
        asian_moneynesses_resolved = asian_moneynesses if asian_moneynesses is not None else moneynesses
        asian_maturities_resolved = asian_maturities if asian_maturities is not None else maturities
        asian_oracle_pricing: PricingComparison = asian_pricing_rmse_vs_mc_oracle(
            fake_s_paths,
            oracle_s_paths,
            dt=dt,
            moneynesses=asian_moneynesses_resolved,
            maturities=asian_maturities_resolved,
            r=pricing_r,
        )
        out["asian_pricing_fake_vs_mc_oracle"] = asian_oracle_pricing.to_dict()
        if real_s_paths is not None:
            real_oracle_pricing: PricingComparison = pricing_rmse_vs_mc_oracle(
                real_s_paths,
                oracle_s_paths,
                dt=dt,
                moneynesses=moneynesses,
                maturities=maturities,
                r=pricing_r,
            )
            out["pricing_real_vs_mc_oracle"] = real_oracle_pricing.to_dict()
            real_asian_oracle_pricing: PricingComparison = asian_pricing_rmse_vs_mc_oracle(
                real_s_paths,
                oracle_s_paths,
                dt=dt,
                moneynesses=asian_moneynesses_resolved,
                maturities=asian_maturities_resolved,
                r=pricing_r,
            )
            out["asian_pricing_real_vs_mc_oracle"] = real_asian_oracle_pricing.to_dict()

    return out
