#!/usr/bin/env bash
set -euo pipefail

# Batch rollout + evaluation for all release best.pt checkpoints under the
# full-surface protocol used by Table 4.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/release/checkpoints}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/full_surface_eval}"
PYTHON_BIN="${PYTHON:-python}"
N_PATHS="${N_PATHS:-10000}"
N_STEPS="${N_STEPS:-252}"
DEVICE="${DEVICE:-auto}"
FM_N_STEPS="${FM_N_STEPS:-20}"
FM_SOLVER="${FM_SOLVER:-euler}"
SIGNATURE_DEPTH="${SIGNATURE_DEPTH:-3}"
ACTION_SEED="${ACTION_SEED:-20260701}"
NOISE_SEED="${NOISE_SEED:-20260701}"
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

usage() {
  cat <<USAGE
Usage:
  bash scripts/evaluate_all_checkpoints_full_surface.sh

Environment overrides:
  PYTHON=/path/to/python
  DATA_DIR=$DATA_DIR
  CHECKPOINT_ROOT=$CHECKPOINT_ROOT
  OUT_DIR=$OUT_DIR
  N_PATHS=10000
  DEVICE=auto|cpu|cuda
  FM_N_STEPS=20
  FM_SOLVER=euler|heun
  SIGNATURE_DEPTH=3
  ACTION_SEED=20260701
  NOISE_SEED=20260701
  LIMIT=10000          optional; if empty, no limit
  FORCE=1             regenerate existing rollouts/evals
  CALIBRATE_MOMENTS=1 optional rollout moment calibration

Outputs:
  $SUMMARY_CSV
  $SUMMARY_JSON
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

printf 'model,checkpoint,rollout,eval_json,vanilla_rmse,vanilla_mape,asian_rmse,asian_mape,marginal_w1_mean,marginal_w1_max,total_return_w1,abs_total_return_w1,sig_w1_mean\n' > "$SUMMARY_CSV"

idx=0
for ckpt in "${CHECKPOINTS[@]}"; do
  idx=$((idx + 1))
  rel="${ckpt#$CHECKPOINT_ROOT/}"
  model="${rel%/best.pt}"
  safe="${model//\//_}"
  rollout="$ROLLOUT_DIR/$safe.npz"
  eval_json="$EVAL_DIR/$safe.json"

  printf '[%s/%s] %s\n' "$idx" "${#CHECKPOINTS[@]}" "$model"
  if [[ "$FORCE" == "1" || ! -e "$rollout" ]]; then
    rollout_args=(
      "$REPO_ROOT/scripts/rollout_joint.py"
      --checkpoint "$ckpt"
      --data-dir "$DATA_DIR"
      --output "$rollout"
      --n-paths "$N_PATHS"
      --n-steps "$N_STEPS"
      --regime-actions
      --action-seed "$ACTION_SEED"
      --noise-seed "$NOISE_SEED"
      --fm-n-steps "$FM_N_STEPS"
      --fm-solver "$FM_SOLVER"
      --device "$DEVICE"
    )
    if [[ "$CALIBRATE_MOMENTS" == "1" ]]; then
      rollout_args+=(--calibrate-moments)
    fi
    "$PYTHON_BIN" "${rollout_args[@]}" >/dev/null
  fi

  if [[ "$FORCE" == "1" || ! -e "$eval_json" ]]; then
    eval_args=(
      "$REPO_ROOT/scripts/evaluate_rollout.py"
      --real "$REAL"
      --fake "$rollout"
      --data-dir "$DATA_DIR"
      --mc-oracle "$ORACLE"
      --output "$eval_json"
      --moneynesses "${MONEYNESS[@]}"
      --maturities "${MATURITIES[@]}"
      --asian-moneynesses "${MONEYNESS[@]}"
      --asian-maturities "${MATURITIES[@]}"
      --signature-depth "$SIGNATURE_DEPTH"
    )
    if [[ -n "$LIMIT" ]]; then
      eval_args+=(--limit "$LIMIT")
    fi
    "$PYTHON_BIN" "${eval_args[@]}" >/dev/null
  fi

  "$PYTHON_BIN" - "$model" "$ckpt" "$rollout" "$eval_json" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

model, ckpt, rollout, eval_json, summary_csv = sys.argv[1:]
report = json.loads(Path(eval_json).read_text(encoding="utf-8"))
pricing = report["pricing_fake_vs_mc_oracle"]
asian = report["asian_pricing_fake_vs_mc_oracle"]
dist = report["distances"]
sig = dist.get("signature_wasserstein") or {}
row = {
    "model": model,
    "checkpoint": ckpt,
    "rollout": rollout,
    "eval_json": eval_json,
    "vanilla_rmse": pricing.get("rmse_overall"),
    "vanilla_mape": pricing.get("mape_overall"),
    "asian_rmse": asian.get("rmse_overall"),
    "asian_mape": asian.get("mape_overall"),
    "marginal_w1_mean": dist.get("marginal_wasserstein_mean"),
    "marginal_w1_max": dist.get("marginal_wasserstein_max"),
    "total_return_w1": dist.get("total_return_wasserstein"),
    "abs_total_return_w1": dist.get("abs_total_return_wasserstein"),
    "sig_w1_mean": sig.get("mean"),
}
with Path(summary_csv).open("a", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row))
    writer.writerow(row)
PY
done

"$PYTHON_BIN" - "$SUMMARY_CSV" "$SUMMARY_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8")))
numeric_fields = {
    "vanilla_rmse",
    "vanilla_mape",
    "asian_rmse",
    "asian_mape",
    "marginal_w1_mean",
    "marginal_w1_max",
    "total_return_w1",
    "abs_total_return_w1",
    "sig_w1_mean",
}
for row in rows:
    for key in numeric_fields:
        if row.get(key) not in (None, ""):
            row[key] = float(row[key])
Path(sys.argv[2]).write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY

printf 'Summary CSV : %s\n' "$SUMMARY_CSV"
printf 'Summary JSON: %s\n' "$SUMMARY_JSON"
