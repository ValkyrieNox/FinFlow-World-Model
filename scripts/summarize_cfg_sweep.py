#!/usr/bin/env python3
"""Summarize MF CFG sweep JSON files into a compact text table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _fmt(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{x:.6f}" if math.isfinite(x) else "nan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="directory containing eval_mf_cfg*.json files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cfg-weights", type=str, default="0 0.5 1 2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines: list[str] = []
    for weight in args.cfg_weights.split():
        report = json.loads((args.run_dir / f"eval_mf_cfg{weight}.json").read_text(encoding="utf-8"))
        distances = report["distances"]
        signature = distances.get("signature_wasserstein", {})
        pricing = report.get("pricing_fake_vs_mc_oracle", {})
        lines.append(
            f"cfg_w={weight:>3} | "
            f"margW={_fmt(distances.get('marginal_wasserstein_mean'))} | "
            f"totalW={_fmt(distances.get('total_return_wasserstein'))} | "
            f"sigW={_fmt(signature.get('mean'))} | "
            f"priceRMSE={_fmt(pricing.get('rmse_overall'))} | "
            f"priceMAPE={_fmt(pricing.get('mape_overall'))}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
