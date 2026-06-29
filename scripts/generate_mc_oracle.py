#!/usr/bin/env python3
"""Generate an independent MC oracle path set from a dataset metadata.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.data import (  # noqa: E402
    HestonParams,
    RegimeParams,
    simulate_heston_qe,
    simulate_regime_switching_heston,
)
from finflow.training import load_metadata  # noqa: E402


def _dtype_from_name(name: str) -> np.dtype[Any]:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError("--dtype must be float32 or float64")


def _single_params_from_metadata(metadata: dict[str, Any]) -> HestonParams:
    p = metadata.get("params", {})
    return HestonParams(
        kappa=float(p.get("kappa", 2.0)),
        theta=float(p.get("theta", 0.04)),
        xi=float(p.get("xi", 0.3)),
        rho=float(p.get("rho", -0.7)),
        v0=float(p.get("v0", metadata.get("v0", 0.04))),
        s0=float(p.get("s0", metadata.get("s0", 100.0))),
        mu=float(p.get("mu", 0.05)),
        dt=float(p.get("dt", metadata.get("dt", 1.0 / float(metadata.get("n_steps", 252))))),
    )


def _regimes_from_metadata(metadata: dict[str, Any]) -> tuple[RegimeParams, ...]:
    regimes = metadata.get("regimes")
    if not regimes:
        raise ValueError("regime-switching metadata is missing regimes")
    return tuple(
        RegimeParams(
            name=str(r.get("name", f"regime_{idx}")),
            kappa=float(r["kappa"]),
            theta=float(r["theta"]),
            xi=float(r["xi"]),
            rho=float(r.get("rho", -0.7)),
            mu=float(r.get("mu", 0.05)),
        )
        for idx, r in enumerate(regimes)
    )


def generate_mc_oracle(
    *,
    data_dir: str | Path,
    output: str | Path,
    n_paths: int = 100_000,
    n_steps: int | None = None,
    seed: int = 20260519,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> dict[str, Any]:
    """Generate an independent oracle npz using the simulation config in metadata."""

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    data_dir = Path(data_dir)
    output = Path(output)
    metadata = load_metadata(data_dir)
    steps = int(n_steps if n_steps is not None else metadata.get("n_steps", 252))
    if steps <= 0:
        raise ValueError("n_steps must be positive")

    if metadata.get("regime_switching"):
        regimes = _regimes_from_metadata(metadata)
        transition_matrix = np.asarray(metadata.get("transition_matrix"), dtype=np.float64)
        arrays = simulate_regime_switching_heston(
            n_paths=n_paths,
            n_steps=steps,
            regimes=regimes,
            transition_matrix=transition_matrix,
            v0=float(metadata.get("v0", 0.04)),
            s0=float(metadata.get("s0", 100.0)),
            dt=float(metadata.get("dt", 1.0 / steps)),
            initial_regime=int(metadata.get("initial_regime", 0)),
            seed=seed,
            dtype=dtype,
        )
        mode = "regime_switching"
    else:
        params = _single_params_from_metadata(metadata)
        if n_steps is not None and steps != int(metadata.get("n_steps", steps)):
            params = HestonParams(
                kappa=params.kappa,
                theta=params.theta,
                xi=params.xi,
                rho=params.rho,
                v0=params.v0,
                s0=params.s0,
                mu=params.mu,
                dt=params.dt,
            )
        arrays = simulate_heston_qe(
            n_paths=n_paths,
            n_steps=steps,
            params=params,
            seed=seed,
            dtype=dtype,
        )
        mode = "single_regime"

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    info = {
        "output": str(output),
        "data_dir": str(data_dir),
        "mode": mode,
        "n_paths": int(n_paths),
        "n_steps": int(steps),
        "seed": int(seed),
        "dtype": str(dtype),
        "has_actions": "actions" in arrays,
        "s0": float(metadata.get("s0", arrays["s_paths"][0, 0])),
        "dt": float(metadata.get("dt", 1.0 / steps)),
    }
    output.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/heston_v3"))
    parser.add_argument("--output", type=Path, default=None,
                        help="default: DATA_DIR/mc_oracle.npz")
    parser.add_argument("--n-paths", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=None,
                        help="override metadata n_steps")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (args.data_dir / "mc_oracle.npz")
    info = generate_mc_oracle(
        data_dir=args.data_dir,
        output=output,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=args.seed,
        dtype=_dtype_from_name(args.dtype),
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
