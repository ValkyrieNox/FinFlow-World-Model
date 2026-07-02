#!/usr/bin/env python3
"""Run the V3 evaluation suite (stylized facts + Wasserstein + pricing)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.data import HestonParams
from finflow.eval import build_full_report
from finflow.training import load_metadata


def _load_returns_and_paths(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(npz_path)
    files = set(arr.files)
    if {"log_returns", "s_paths"}.issubset(files):
        returns = np.asarray(arr["log_returns"], dtype=np.float64)
        s_paths = np.asarray(arr["s_paths"], dtype=np.float64)
    elif {"r_paths", "s_paths"}.issubset(files):
        returns = np.asarray(arr["r_paths"], dtype=np.float64)
        s_paths = np.asarray(arr["s_paths"], dtype=np.float64)
    else:
        raise ValueError(
            f"{npz_path} must contain (log_returns, s_paths) or (r_paths, s_paths)"
        )
    arr.close()
    return returns, s_paths


def _load_s_paths(npz_path: Path) -> np.ndarray:
    arr = np.load(npz_path)
    if "s_paths" not in arr.files:
        raise ValueError(f"{npz_path} must contain s_paths")
    s_paths = np.asarray(arr["s_paths"], dtype=np.float64)
    arr.close()
    return s_paths


def _metadata_dt(metadata: dict) -> float:
    n_steps = int(metadata.get("n_steps", 252))
    fallback = 1.0 / n_steps
    if metadata.get("regime_switching"):
        return float(metadata.get("dt", fallback))
    params = metadata.get("params", {})
    return float(metadata.get("dt", params.get("dt", fallback)))


def _normal_params_from_metadata(metadata: dict, dt: float) -> HestonParams:
    if metadata.get("regime_switching"):
        regimes = metadata.get("regimes", [])
        normal = regimes[0] if regimes else {}
        return HestonParams(
            kappa=float(normal.get("kappa", 2.0)),
            theta=float(normal.get("theta", 0.04)),
            xi=float(normal.get("xi", 0.3)),
            rho=float(normal.get("rho", -0.7)),
            mu=float(normal.get("mu", 0.05)),
            v0=float(metadata.get("v0", 0.04)),
            s0=float(metadata.get("s0", 100.0)),
            dt=dt,
        )
    p = metadata.get("params", {})
    return HestonParams(
        kappa=float(p.get("kappa", 2.0)),
        theta=float(p.get("theta", 0.04)),
        xi=float(p.get("xi", 0.3)),
        rho=float(p.get("rho", -0.7)),
        mu=float(p.get("mu", 0.05)),
        v0=float(p.get("v0", 0.04)),
        s0=float(p.get("s0", 100.0)),
        dt=dt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True,
                        help="path to a Heston {split}.npz file (real reference)")
    parser.add_argument("--fake", type=Path, required=True,
                        help="path to a rollout npz or another Heston-format npz")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="metadata source (defaults to --real's parent)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mc-oracle", type=Path, default=None,
                        help="independent oracle npz containing s_paths for MC price reference")
    parser.add_argument("--moneynesses", nargs="+", type=float,
                        default=[0.60, 0.70, 0.80, 0.85, 0.90, 0.95,
                                 1.00, 1.05, 1.10, 1.20, 1.30, 1.50],
                        help="European-call moneyness grid K/S0")
    parser.add_argument("--maturities", nargs="+", type=float,
                        default=[0.25, 0.5, 1.0])
    parser.add_argument("--asian-moneynesses", nargs="+", type=float, default=None,
                        help="arithmetic-Asian call moneyness grid; default mirrors --moneynesses")
    parser.add_argument("--asian-maturities", nargs="+", type=float, default=None,
                        help="arithmetic-Asian call maturities; default mirrors --maturities")
    parser.add_argument("--pricing-r", type=float, default=None,
                        help="discount rate for MC pricing; default = mu from metadata")
    parser.add_argument("--skip-pricing", action="store_true")
    parser.add_argument(
        "--force-regime-pricing",
        action="store_true",
        help=(
            "For regime-switching data, still compare MC prices against the "
            "normal-regime Heston Carr-Madan reference. By default pricing is "
            "skipped because the Markov mixture has no single-Heston closed form."
        ),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="limit number of paths used from each side")
    parser.add_argument("--signature-depth", type=int, default=3,
                        help="truncated signature depth for Sig-Wasserstein; set 0 to disable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or args.real.parent
    metadata = load_metadata(data_dir)
    dt = _metadata_dt(metadata)

    pricing_skipped_reason = None
    if args.skip_pricing:
        params = None
        pricing_skipped_reason = "disabled by --skip-pricing"
    elif metadata.get("regime_switching") and args.mc_oracle is None and not args.force_regime_pricing:
        params = None
        pricing_skipped_reason = (
            "regime-switching data has no single-Heston Carr-Madan reference; "
            "use --mc-oracle for MC reference or --force-regime-pricing to "
            "compare against normal regime only"
        )
    else:
        params = None if metadata.get("regime_switching") else _normal_params_from_metadata(metadata, dt)
        if metadata.get("regime_switching") and args.force_regime_pricing:
            params = _normal_params_from_metadata(metadata, dt)
    pricing_r = args.pricing_r
    if pricing_r is None and params is not None:
        pricing_r = float(params.mu)

    real_returns, real_s = _load_returns_and_paths(args.real)
    fake_returns, fake_s = _load_returns_and_paths(args.fake)
    oracle_s = (
        _load_s_paths(args.mc_oracle)
        if args.mc_oracle is not None and not args.skip_pricing
        else None
    )

    if args.limit is not None:
        real_returns = real_returns[: args.limit]
        real_s = real_s[: args.limit]
        fake_returns = fake_returns[: args.limit]
        fake_s = fake_s[: args.limit]
        if oracle_s is not None:
            oracle_s = oracle_s[: args.limit]

    report = build_full_report(
        real_returns=real_returns,
        fake_returns=fake_returns,
        real_s_paths=real_s,
        fake_s_paths=fake_s,
        oracle_s_paths=oracle_s,
        params=params,
        moneynesses=args.moneynesses,
        maturities=args.maturities,
        asian_moneynesses=args.asian_moneynesses,
        asian_maturities=args.asian_maturities,
        dt=dt,
        pricing_r=pricing_r or 0.0,
        signature_depth=args.signature_depth or None,
    )
    if pricing_skipped_reason is not None:
        report["pricing_skipped"] = pricing_skipped_reason

    out_path = args.output or args.fake.with_suffix(".eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
