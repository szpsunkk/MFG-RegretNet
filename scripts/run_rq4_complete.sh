#!/usr/bin/env bash
# =============================================================================
# run_rq4_complete.sh  —  RQ4: FL Final Accuracy (one-click)
#
# Generates:
#   run/privacy_paper/rq4/raw/{DATASET}_a{alpha}_s{seed}.json
#   run/privacy_paper/rq4/rq4_aggregated.json
#   run/privacy_paper/rq4/figures/ (Figs A-D)
#   run/privacy_paper/rq4/table_rq4_paper.md   ← LaTeX/Markdown table
#   run/privacy_paper/rq4/table_rq4_paper.tex
#
# Methods: MFG-RegretNet(Ours), RegretNet, DM-RegretNet, PAC, VCG,
#          MFG-Pricing, CSRA, No-DP FL (upper bound)
# =============================================================================
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT_DIR="${OUT_DIR:-run/privacy_paper/rq4}"
ROUNDS="${ROUNDS:-100}"
RND_STEP="${RND_STEP:-5}"
SEEDS="${SEEDS:-0,1,2}"
N_AGENTS="${N_AGENTS:-10}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
LOCAL_BS="${LOCAL_BS:-64}"
FIXED_BUDGET="${FIXED_BUDGET:-50.0}"

# Datasets & alphas: MNIST alpha=0.5, MNIST alpha=0.1, CIFAR-10 alpha=0.5
declare -a DATASETS=("MNIST" "MNIST" "CIFAR10")
declare -a ALPHAS=("0.5" "0.1" "0.5")

echo "========================================================"
echo "RQ4: FL Final Accuracy Benchmark"
echo "  Methods : Ours, RegretNet, DM-RegretNet, PAC, VCG,"
echo "            MFG-Pricing, CSRA, No-DP FL"
echo "  Rounds  : $ROUNDS  |  Seeds: $SEEDS"
echo "  Budget  : $FIXED_BUDGET (fixed per round)"
echo "  Out dir : $OUT_DIR"
echo "========================================================"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/figures"

# Convert comma-separated seeds to array
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

TOTAL=$(( ${#DATASETS[@]} * ${#SEED_LIST[@]} ))
RUN=0

for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"
    ALPHA="${ALPHAS[$i]}"
    for SEED in "${SEED_LIST[@]}"; do
        RUN=$((RUN + 1))
        echo ""
        echo "--- [$RUN/$TOTAL] Dataset=$DS alpha=$ALPHA seed=$SEED ---"
        python3 exp_rq/rq4_fl_benchmark.py \
            --dataset "$DS" \
            --alpha "$ALPHA" \
            --seed "$SEED" \
            --n-agents "$N_AGENTS" \
            --rounds "$ROUNDS" \
            --rnd-step "$RND_STEP" \
            --local-epochs "$LOCAL_EPOCHS" \
            --local-batch-size "$LOCAL_BS" \
            --fixed-budget "$FIXED_BUDGET" \
            --pac \
            --out-dir "$OUT_DIR"
    done
done

echo ""
echo "--- Aggregating results and generating figures ---"
python3 exp_rq/rq4_plot_paper_figures.py \
    --rq4-dir "$OUT_DIR" \
    --fig-a-alpha 0.5

echo ""
echo "--- Generating final accuracy table ---"
python3 exp_rq/rq4_final_table.py \
    --rq4-dir "$OUT_DIR" \
    --out-dir "$OUT_DIR"

echo ""
echo "========================================================"
echo "RQ4 complete. Outputs in: $OUT_DIR"
echo "  Raw data  : $OUT_DIR/raw/"
echo "  Figures   : $OUT_DIR/figures/"
echo "  Table (MD): $OUT_DIR/table_rq4_paper.md"
echo "  Table (TeX): $OUT_DIR/table_rq4_paper.tex"
echo "========================================================"
