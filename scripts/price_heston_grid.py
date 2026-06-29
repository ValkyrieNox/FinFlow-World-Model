#!/usr/bin/env python3
"""Price Heston European calls on the V3 15-point (moneyness, maturity) grid
via the Carr-Madan FFT. Writes a JSON file with prices and grid metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.data import HestonParams, price_heston_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/heston_v3/option_grid.json"))
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--theta", type=float, default=0.04)
    parser.add_argument("--xi", type=float, default=0.3)
    parser.add_argument("--rho", type=float, default=-0.7)
    parser.add_argument("--v0", type=float, default=0.04)
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--r", type=float, default=0.0, help="risk-free rate")
    parser.add_argument("--q", type=float, default=0.0, help="dividend yield")
    parser.add_argument("--alpha", type=float, default=1.5, help="Carr-Madan damping factor")
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument(
        "--moneynesses", nargs="+", type=float,
        default=[0.85, 0.90, 0.95, 1.00, 1.05],
    )
    parser.add_argument(
        "--maturities", nargs="+", type=float,
        default=[0.25, 0.5, 1.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = HestonParams(
        kappa=args.kappa, theta=args.theta, xi=args.xi, rho=args.rho,
        v0=args.v0, s0=args.s0, mu=args.mu,
    )
    grid = price_heston_grid(
        params=params,
        moneynesses=args.moneynesses,
        maturities=args.maturities,
        r=args.r, q=args.q, alpha=args.alpha,
        n_fft=args.n_fft, eta=args.eta,
    )
    payload = {
        "params": {
            "kappa": params.kappa, "theta": params.theta, "xi": params.xi,
            "rho": params.rho, "v0": params.v0, "s0": params.s0, "mu": params.mu,
        },
        "fft": {"alpha": args.alpha, "n_fft": args.n_fft, "eta": args.eta},
        **grid.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
