#!/usr/bin/env python3
"""Generate Heston QE data for the V3 transition-kernel pipeline.

Two modes:
- Single regime (default): one fixed set of Heston parameters.
- Regime switching: pass ``--regimes`` to enable the default 3-regime mix
  (normal / high_vol / crash) with a Markov chain over per-step regimes.
  Each step's regime is stored as the action label ``a_t``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.data import (
    DEFAULT_REGIMES,
    DEFAULT_TRANSITION_MATRIX,
    HestonParams,
    generate_heston_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/heston_v3"))
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=5_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=252)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--no-transitions", action="store_true",
        help="Only save full paths, not flattened transitions.",
    )

    # single-regime params (also used to seed v0/s0/dt under regime switching)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--theta", type=float, default=0.04)
    parser.add_argument("--xi", type=float, default=0.3)
    parser.add_argument("--rho", type=float, default=-0.7)
    parser.add_argument("--v0", type=float, default=0.04)
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1.0 / 252.0)

    # regime switching
    parser.add_argument(
        "--regimes", action="store_true",
        help="Use the default 3-regime Markov mix (normal / high_vol / crash).",
    )
    parser.add_argument(
        "--initial-regime", type=int, default=0,
        help="Index of the starting regime when --regimes is set (default 0=normal).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = HestonParams(
        kappa=args.kappa, theta=args.theta, xi=args.xi, rho=args.rho,
        v0=args.v0, s0=args.s0, mu=args.mu, dt=args.dt,
    )
    if args.regimes:
        metadata = generate_heston_dataset(
            output_dir=args.output,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            n_steps=args.steps,
            params=params,
            regimes=DEFAULT_REGIMES,
            transition_matrix=DEFAULT_TRANSITION_MATRIX,
            initial_regime=args.initial_regime,
            seed=args.seed,
            save_transitions=not args.no_transitions,
        )
    else:
        metadata = generate_heston_dataset(
            output_dir=args.output,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            n_steps=args.steps,
            params=params,
            seed=args.seed,
            save_transitions=not args.no_transitions,
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
