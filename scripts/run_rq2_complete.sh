#!/usr/bin/env bash
# =============================================================================
# run_rq2_complete.sh  —  RQ2: Scalability Analysis (one-click)
#
# Generates:
#   run/privacy_paper/rq2/rq2_paper_data.json
#   run/privacy_paper/rq2/figures/figure_rq2_1_time_vs_N_loglog.png
#   run/privacy_paper/rq2/figures/figure_rq2_2_memory_comm.png
#   run/privacy_paper/rq2/figures/figure_rq2_3_stacked_latency.png
#
# Required: trained MFG-RegretNet for N=10,50,100 in result/
#   result/mfg_regretnet_n10_5_checkpoint.pt
#   result/mfg_regretnet_n50_5_checkpoint.pt
#   result/mfg_regretnet_n100_5_checkpoint.pt
#
# The script also uses RegretNet and DM-RegretNet checkpoints if available.
# =============================================================================
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT_DIR="${OUT_DIR:-run/privacy_paper/rq2}"
N_LIST="${N_LIST:-10,50,100}"
BUDGET="${BUDGET:-50.0}"
REPEAT="${REPEAT:-20}"
WARMUP="${WARMUP:-3}"
BATCH_SIZE="${BATCH_SIZE:-100}"

echo "========================================================"
echo "RQ2: Scalability Benchmark"
echo "  N list   : $N_LIST"
echo "  Out dir  : $OUT_DIR"
echo "  Repeat   : $REPEAT"
echo "========================================================"

mkdir -p "$OUT_DIR"

# Step 1: Run timing benchmark
python3 exp_rq/rq2_paper_benchmark.py \
  --n-list "$N_LIST" \
  --budget "$BUDGET" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --batch-size "$BATCH_SIZE" \
  --out-dir "$OUT_DIR"

# Step 2: Plot figures
python3 exp_rq/rq2_plot_paper_figures.py \
  --input "$OUT_DIR/rq2_paper_data.json" \
  --out-dir "$OUT_DIR/figures"

echo ""
echo "========================================================"
echo "RQ2 complete. Outputs in: $OUT_DIR"
echo "  Figures: $OUT_DIR/figures/"
echo "========================================================"
