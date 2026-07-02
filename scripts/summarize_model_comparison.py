#!/usr/bin/env python3
"""Build markdown comparison tables from model evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _fmt(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--qgan-ablation-output",
        type=Path,
        default=None,
        help="default: EVAL_DIR/qgan_checkpoint_ablation.md",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(report: dict[str, object]) -> dict[str, object]:
    distances = report.get("distances", {})
    if not isinstance(distances, dict):
        distances = {}
    signature = distances.get("signature_wasserstein") or {}
    if not isinstance(signature, dict):
        signature = {}
    pricing = report.get("pricing_fake_vs_mc_oracle") or {}
    if not isinstance(pricing, dict):
        pricing = {}
    fake_facts = report.get("fake_facts") or {}
    if not isinstance(fake_facts, dict):
        fake_facts = {}
    stylized = report.get("stylized_facts_comparison") or {}
    if not isinstance(stylized, dict):
        stylized = {}
    return {
        "kurtosis": fake_facts.get("kurtosis"),
        "kurtosis_abs_diff": stylized.get("kurtosis_abs_diff"),
        "marginal_wasserstein_mean": distances.get("marginal_wasserstein_mean"),
        "total_return_wasserstein": distances.get("total_return_wasserstein"),
        "signature_wasserstein_mean": signature.get("mean"),
        "pricing_rmse": pricing.get("rmse_overall"),
        "pricing_mape": pricing.get("mape_overall"),
    }


def _qgan_primary(eval_dir: Path) -> tuple[str, Path]:
    preferred = eval_dir / "eval_quant_gan_last_calibrated.json"
    if preferred.exists():
        return "Quant GAN last calibrated", preferred
    return "Quant GAN", eval_dir / "eval_quant_gan.json"


def _table_row(label: str, path: Path) -> str:
    report = _load(path)
    metrics = _metrics(report)
    return (
        "| "
        + " | ".join([
            label,
            _fmt(metrics.get("kurtosis")),
            _fmt(metrics.get("kurtosis_abs_diff")),
            _fmt(metrics.get("marginal_wasserstein_mean")),
            _fmt(metrics.get("total_return_wasserstein")),
            _fmt(metrics.get("signature_wasserstein_mean")),
            _fmt(metrics.get("pricing_rmse")),
            _fmt(metrics.get("pricing_mape")),
            str(path),
        ])
        + " |"
    )


def _write_qgan_ablation(eval_dir: Path, output: Path) -> None:
    variants = [
        ("QGAN legacy/default", eval_dir / "eval_quant_gan.json"),
        ("QGAN best raw", eval_dir / "eval_quant_gan_best_raw.json"),
        ("QGAN best calibrated", eval_dir / "eval_quant_gan_best_calibrated.json"),
        ("QGAN last raw", eval_dir / "eval_quant_gan_last_raw.json"),
        ("QGAN last calibrated", eval_dir / "eval_quant_gan_last_calibrated.json"),
    ]
    present = [(label, path) for label, path in variants if path.exists()]
    if len(present) <= 1:
        return

    lines = [
        "# Quant GAN Checkpoint Ablation",
        "",
        "This table separates checkpoint choice (`best.pt` vs `last.pt`) from "
        "sampling-time moment calibration.",
        "",
        "| Variant | Kurtosis | Kurt diff | Marginal W1 mean | Total-return W1 | "
        "Sig-W1 mean | Pricing RMSE | Pricing MAPE | JSON |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, path in present:
        lines.append(_table_row(label, path))
    lines += [
        "",
        "Primary model summaries use `QGAN last calibrated` when available. "
        "`QGAN legacy/default` is retained for backward compatibility and "
        "traceability.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    qgan_label, qgan_path = _qgan_primary(args.eval_dir)
    models = [
        ("FM teacher", args.eval_dir / "eval_fm.json"),
        ("Mean Flow best CFG", args.eval_dir / "eval_mf.json"),
        ("Consistency", args.eval_dir / "eval_cd.json"),
        (qgan_label, qgan_path),
    ]

    lines = [
        "# Model Comparison",
        "",
        "| Model | Kurtosis | Kurt diff | Marginal W1 mean | Total-return W1 | "
        "Sig-W1 mean | Pricing RMSE | Pricing MAPE | JSON |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, path in models:
        lines.append(_table_row(label, path))
    if qgan_path.name != "eval_quant_gan.json":
        lines += [
            "",
            "Note: Quant GAN primary result uses `last.pt` with sampling-time "
            "moment calibration. `eval_quant_gan.json` is retained as a legacy "
            "alias or older/default result depending on how the run was produced.",
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ablation_output = args.qgan_ablation_output or (
        args.eval_dir / "qgan_checkpoint_ablation.md"
    )
    _write_qgan_ablation(args.eval_dir, ablation_output)
    print(args.output)


if __name__ == "__main__":
    main()
