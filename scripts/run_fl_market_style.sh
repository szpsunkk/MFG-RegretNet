#!/usr/bin/env bash
# FL-market.pdf–style figures (privacy-paper bids + RQ4 accuracy).
# Usage: from repo root, after checkpoints exist under result/
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND="${MPLBACKEND:-Agg}"

OUT="${OUT:-run/privacy_paper/fl_market_style}"
RQ4_RAW="${RQ4_RAW:-run/privacy_paper/rq4/raw}"

python exp_rq/fl_market_style_figures.py \
  --out-dir "$OUT" \
  --rq4-dir "$RQ4_RAW" \
  --n-agents 10 \
  --n-items 1 \
  --n-profiles 4000 \
  --nb-budget 20 \
  --budget-step 0.1 \
  "$@"

echo "Figures: $OUT/figures/"
