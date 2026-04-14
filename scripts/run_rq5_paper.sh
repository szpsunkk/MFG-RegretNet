#!/usr/bin/env bash
# RQ5 一键：多档预算 B × 多方法 ×（可选多种子）→ 图 A–E
# 依赖：同 RQ4（torch、torchvision、RegretNet/MFG checkpoint）
#
#   bash scripts/run_rq5_paper.sh
#   QUICK=1 bash scripts/run_rq5_paper.sh
#   SEEDS=3 DATASETS="MNIST CIFAR10" BUDGET_RATES="0.3,0.6,1.0,1.4" bash scripts/run_rq5_paper.sh
#   bash scripts/run_rq5_paper.sh -- --pac --rounds 40
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-run/privacy_paper/rq5}"
SEEDS="${SEEDS:-3}"
ROUNDS="${ROUNDS:-60}"
ALPHA="${ALPHA:-0.5}"
BUDGET_RATES="${BUDGET_RATES:-0.35,0.7,1.05,1.4}"
DATASETS="${DATASETS:-MNIST}"
EXTRA=()
if [[ "${1:-}" == "--" ]]; then shift; EXTRA=("$@"); fi
if [[ "${QUICK:-0}" == "1" ]]; then
  SEEDS=1
  ROUNDS=15
  BUDGET_RATES="0.6,1.0"
  EXTRA+=(--quick)
fi

mkdir -p "$OUT/raw"
echo "[RQ5] OUT=$OUT SEEDS=$SEEDS ROUNDS=$ROUNDS B=$BUDGET_RATES datasets=$DATASETS"
MFG_ARGS=()
[[ -n "${MFG_CKPT:-}" ]] && MFG_ARGS+=(--mfg-ckpt "$MFG_CKPT")
[[ -n "${FL_LR:-}" ]] && MFG_ARGS+=(--fl-lr "$FL_LR")

for ds in $DATASETS; do
  for s in $(seq 0 $((SEEDS - 1))); do
    echo ">>> $ds seed=$s"
    python exp_rq/rq5_fl_benchmark.py \
      --dataset "$ds" \
      --alpha "$ALPHA" \
      --seed "$s" \
      --rounds "$ROUNDS" \
      --budget-rates "$BUDGET_RATES" \
      --out-dir "$OUT" \
      "${MFG_ARGS[@]}" \
      "${EXTRA[@]}"
  done
done

# 与上方训练循环同一套 $DATASETS 词切分（勿用 read <<<，否则只读第一行，且与 for ds in $DATASETS 不一致）
_PLOT_DS="${DATASETS:-MNIST}"
for _ds in $_PLOT_DS; do
  [[ -n "${_ds// }" ]] || continue
  python exp_rq/rq5_plot_paper_figures.py --rq5-dir "$OUT" --dataset-filter "$_ds"
done
echo "Done. Figures: $OUT/figures/figure_rq5_*.png"
