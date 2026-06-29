#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/run_full_evaluation.sh ROLLOUT_DIR [REAL_NPZ] [OUT_DIR]

Environment:
  DATA_DIR                 metadata directory; defaults to REAL_NPZ parent
  MC_ORACLE                optional npz containing oracle s_paths
  MONEYNESS                space-separated grid, default "0.85 0.90 0.95 1.00 1.05"
  MATURITIES               space-separated grid, default "0.25 0.5 1.0"
  LIMIT                    optional path limit passed to evaluate_rollout.py
  FORCE_REGIME_PRICING     set to 1 to compare regime data to normal Carr-Madan

Expected rollout filenames inside ROLLOUT_DIR:
  rollout_fm.npz
  rollout_mf.npz
  rollout_cd.npz
  quant_gan_paths.npz or rollout_quant_gan.npz
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_DIR="$1"
REAL_NPZ="${2:-${REAL_NPZ:-data/heston_v3/test.npz}}"
OUT_DIR="${3:-${OUT_DIR:-${ROLLOUT_DIR%/}/evaluation}}"
DATA_DIR="${DATA_DIR:-$(dirname "$REAL_NPZ")}"
MONEYNESS="${MONEYNESS:-0.85 0.90 0.95 1.00 1.05}"
MATURITIES="${MATURITIES:-0.25 0.5 1.0}"

mkdir -p "$OUT_DIR"

read -r -a MONEYNESS_ARGS <<< "$MONEYNESS"
read -r -a MATURITY_ARGS <<< "$MATURITIES"

base_args=(
  --real "$REAL_NPZ"
  --data-dir "$DATA_DIR"
  --moneynesses "${MONEYNESS_ARGS[@]}"
  --maturities "${MATURITY_ARGS[@]}"
)
if [[ -n "${MC_ORACLE:-}" ]]; then
  base_args+=(--mc-oracle "$MC_ORACLE")
fi
if [[ -n "${LIMIT:-}" ]]; then
  base_args+=(--limit "$LIMIT")
fi
if [[ "${FORCE_REGIME_PRICING:-0}" == "1" ]]; then
  base_args+=(--force-regime-pricing)
fi

json_specs=()
specs=(
  "fm|FM teacher|rollout_fm.npz|"
  "mf|Mean Flow|rollout_mf.npz|"
  "cd|Consistency|rollout_cd.npz|"
  "quant_gan|Quant GAN|quant_gan_paths.npz|rollout_quant_gan.npz"
)

for spec in "${specs[@]}"; do
  IFS='|' read -r key label primary fallback <<< "$spec"
  fake="${ROLLOUT_DIR%/}/$primary"
  if [[ ! -f "$fake" && -n "$fallback" ]]; then
    fake="${ROLLOUT_DIR%/}/$fallback"
  fi
  if [[ ! -f "$fake" ]]; then
    continue
  fi
  out_json="$OUT_DIR/eval_${key}.json"
  python3 "$SCRIPT_DIR/evaluate_rollout.py" \
    "${base_args[@]}" \
    --fake "$fake" \
    --output "$out_json" >/dev/null
  json_specs+=("$key|$label|$out_json")
done

if [[ ${#json_specs[@]} -eq 0 ]]; then
  echo "No known rollout files found in $ROLLOUT_DIR" >&2
  exit 1
fi

summary_md="$OUT_DIR/summary.md"
python3 - "$summary_md" "${json_specs[@]}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


summary_path = Path(sys.argv[1])
rows: list[list[str]] = []
for spec in sys.argv[2:]:
    key, label, json_path = spec.split("|", 2)
    report = json.loads(Path(json_path).read_text(encoding="utf-8"))
    distances = report.get("distances", {})
    pricing = (
        report.get("pricing_fake_vs_mc_oracle")
        or report.get("pricing_fake_vs_carr_madan")
        or {}
    )
    pricing_ref = "MC oracle" if "pricing_fake_vs_mc_oracle" in report else (
        "Carr-Madan" if "pricing_fake_vs_carr_madan" in report else "skipped"
    )
    rows.append([
        label,
        fmt(distances.get("marginal_wasserstein_mean")),
        fmt(distances.get("marginal_wasserstein_max")),
        fmt(distances.get("total_return_wasserstein")),
        fmt((distances.get("signature_wasserstein") or {}).get("mean")),
        pricing_ref,
        fmt(pricing.get("rmse_overall")),
        fmt(pricing.get("mape_overall")),
        json_path,
    ])

lines = [
    "# Full Evaluation Summary",
    "",
    "| Model | Marginal W1 mean | Marginal W1 max | Total-return W1 | Sig-W1 mean | Pricing ref | Pricing RMSE | Pricing MAPE | JSON |",
    "|---|---:|---:|---:|---:|---|---:|---:|---|",
]
for row in rows:
    lines.append("| " + " | ".join(row) + " |")
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_path)
PY
