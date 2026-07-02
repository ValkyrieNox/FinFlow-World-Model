#!/usr/bin/env bash
set -euo pipefail

# Batch rollout + evaluation for all joint best.pt checkpoints.
# Works in Git Bash / WSL-style shells. On native Windows cmd, run via:
#   bash scripts/evaluate_all_checkpoints_full_surface.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/heston_v3"
CHECKPOINT_ROOT="$REPO_ROOT/release/checkpoints"
OUT_DIR="$REPO_ROOT/runs/full_surface_eval"
PYTHON_BIN="${PYTHON:-python}"
N_PATHS="${N_PATHS:-10000}"
N_STEPS="${N_STEPS:-252}"
DEVICE="${DEVICE:-auto}"
FM_N_STEPS="${FM_N_STEPS:-20}"
FM_SOLVER="${FM_SOLVER:-euler}"
SIGNATURE_DEPTH="${SIGNATURE_DEPTH:-3}"
LIMIT="${LIMIT:-}"
FORCE="${FORCE:-0}"
CALIBRATE_MOMENTS="${CALIBRATE_MOMENTS:-0}"

REAL="$DATA_DIR/test.npz"
ORACLE="$DATA_DIR/mc_oracle.npz"
METADATA="$DATA_DIR/metadata.json"
ROLLOUT_DIR="$OUT_DIR/rollouts"
EVAL_DIR="$OUT_DIR/evals"
SUMMARY_CSV="$OUT_DIR/summary_full_surface.csv"
SUMMARY_JSON="$OUT_DIR/summary_full_surface.json"

MONEYNESS=(
  0.50 0.60 0.70 0.80 0.85 0.90 0.95
  1.00 1.05 1.10 1.15 1.20 1.30 1.40 1.50 1.75 2.00
)
MATURITIES=(0.25 0.5 1.0)
ASIAN_MONEYNESS=("${MONEYNESS[@]}")
ASIAN_MATURITIES=("${MATURITIES[@]}")

usage() {
  cat <<USAGE
Usage:
  bash scripts/evaluate_all_checkpoints_full_surface.sh

Environment overrides:
  PYTHON=/path/to/python
  N_PATHS=10000
  DEVICE=auto|cpu|cuda
  FM_N_STEPS=20
  FM_SOLVER=euler|heun
  SIGNATURE_DEPTH=3
  LIMIT=10000          optional; if empty, no limit
  FORCE=1             regenerate existing rollouts/evals
  CALIBRATE_MOMENTS=1 optional rollout moment calibration

Outputs:
  $OUT_DIR
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for required in "$DATA_DIR" "$CHECKPOINT_ROOT" "$REAL" "$ORACLE" "$METADATA" \
                "$REPO_ROOT/scripts/rollout_joint.py" "$REPO_ROOT/scripts/evaluate_rollout.py"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$ROLLOUT_DIR" "$EVAL_DIR"

mapfile -t CHECKPOINTS < <(
  find "$CHECKPOINT_ROOT" -type f -name 'best.pt' \
    ! -path "$CHECKPOINT_ROOT/checkpoints/*" | sort
)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No best.pt checkpoints found under $CHECKPOINT_ROOT" >&2
  exit 1
fi

printf 'RepoRoot       : %s\n' "$REPO_ROOT"
printf 'DataDir        : %s\n' "$DATA_DIR"
printf 'CheckpointRoot : %s\n' "$CHECKPOINT_ROOT"
printf 'OutDir         : %s\n' "$OUT_DIR"
printf 'N checkpoints  : %s\n' "${#CHECKPOINTS[@]}"
printf 'Moneynesses    : %s\n' "${MONEYNESS[*]}"
printf 'Maturities     : %s\n' "${MATURITIES[*]}"

JSON_SPECS=()
idx=0
for ckpt in "${CHECKPOINTS[@]}"; do
  idx=$((idx + 1))
  ckpt_dir="$(dirname "$ckpt")"
  rel="${ckpt_dir#$CHECKPOINT_ROOT/}"
  safe="$(printf '%s' "$rel" | sed -E 's#[\\/:*?"<>| ]+#_#g; s#^_+##; s#_+$##')"
  [[ -n "$safe" ]] || safe="checkpoint_$idx"

  rollout="$ROLLOUT_DIR/$safe.npz"
  eval_json="$EVAL_DIR/$safe.json"

  printf '[%d/%d] Rollout: %s\n' "$idx" "${#CHECKPOINTS[@]}" "$rel"
  if [[ "$FORCE" == "1" || ! -f "$rollout" ]]; then
    rollout_args=(
      "$REPO_ROOT/scripts/rollout_joint.py"
      --checkpoint "$ckpt"
      --data-dir "$DATA_DIR"
      --output "$rollout"
      --n-paths "$N_PATHS"
      --n-steps "$N_STEPS"
      --regime-actions
      --action-seed 20260701
      --noise-seed 20260701
      --fm-n-steps "$FM_N_STEPS"
      --fm-solver "$FM_SOLVER"
      --device "$DEVICE"
    )
    if [[ "$CALIBRATE_MOMENTS" == "1" ]]; then
      rollout_args+=(--calibrate-moments)
    fi
    "$PYTHON_BIN" "${rollout_args[@]}"
  else
    echo "  existing rollout found; set FORCE=1 to regenerate"
  fi

  printf '[%d/%d] Evaluate: %s\n' "$idx" "${#CHECKPOINTS[@]}" "$rel"
  if [[ "$FORCE" == "1" || ! -f "$eval_json" ]]; then
    eval_args=(
      "$REPO_ROOT/scripts/evaluate_rollout.py"
      --real "$REAL"
      --fake "$rollout"
      --data-dir "$DATA_DIR"
      --mc-oracle "$ORACLE"
      --output "$eval_json"
      --moneynesses "${MONEYNESS[@]}"
      --maturities "${MATURITIES[@]}"
      --asian-moneynesses "${ASIAN_MONEYNESS[@]}"
      --asian-maturities "${ASIAN_MATURITIES[@]}"
      --signature-depth "$SIGNATURE_DEPTH"
    )
    if [[ -n "$LIMIT" ]]; then
      eval_args+=(--limit "$LIMIT")
    fi
    "$PYTHON_BIN" "${eval_args[@]}" >/dev/null
  else
    echo "  existing evaluation found; set FORCE=1 to recompute"
  fi

  JSON_SPECS+=("$rel|$ckpt|$rollout|$eval_json")
done

"$PYTHON_BIN" - "$SUMMARY_CSV" "$SUMMARY_JSON" "${JSON_SPECS[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
rows = []
for spec in sys.argv[3:]:
    model, checkpoint, rollout, eval_json = spec.split('|', 3)
    report = json.loads(Path(eval_json).read_text(encoding='utf-8'))
    dist = report.get('distances', {})
    pricing = report.get('pricing_fake_vs_mc_oracle') or {}
    asian = report.get('asian_pricing_fake_vs_mc_oracle') or {}
    sig = dist.get('signature_wasserstein') or {}
    rows.append({
        'model': model,
        'checkpoint': checkpoint,
        'rollout': rollout,
        'eval_json': eval_json,
        'vanilla_rmse': pricing.get('rmse_overall'),
        'vanilla_mape': pricing.get('mape_overall'),
        'asian_rmse': asian.get('rmse_overall'),
        'asian_mape': asian.get('mape_overall'),
        'marginal_w1_mean': dist.get('marginal_wasserstein_mean'),
        'marginal_w1_max': dist.get('marginal_wasserstein_max'),
        'total_return_w1': dist.get('total_return_wasserstein'),
        'abs_total_return_w1': dist.get('abs_total_return_wasserstein'),
        'sig_w1_mean': sig.get('mean'),
    })

csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(f'Done.\nSummary CSV : {csv_path}\nSummary JSON: {json_path}')
PY
