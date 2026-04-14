#!/usr/bin/env bash
# RQ4 一键：MNIST + CIFAR-10，Dirichlet α∈{0.1,0.5}，多种子 → 图 A/B/C/D
# 依赖：torch、torchvision；机制权重见 RQ1（RegretNet / MFG-RegretNet checkpoint）
#
# 用法：
#   bash scripts/run_rq4_paper.sh
#   QUICK=1 bash scripts/run_rq4_paper.sh          # 快速试跑（20 轮、1 个 seed）
#   SEEDS=5 ROUNDS=100 bash scripts/run_rq4_paper.sh
#   bash scripts/run_rq4_paper.sh -- --budget-rate 0.8 --pac
# 注意：额外参数必须写在单独一行「--」后面；勿把多行命令粘成一行（如 ...shbash...）。
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-run/privacy_paper/rq4}"
SEEDS="${SEEDS:-3}"
ROUNDS="${ROUNDS:-80}"
RND_STEP="${RND_STEP:-5}"
FIG_A_ALPHA="${FIG_A_ALPHA:-0.5}"
EXTRA=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA=("$@")
elif [[ -n "${1:-}" ]]; then
  echo "[RQ4] ERROR: 不识别的参数 '$1'。Python 选项必须放在 -- 之后，例如："
  echo "  bash scripts/run_rq4_paper.sh -- --budget-rate 0.8 --pac"
  exit 1
fi
if [[ "${QUICK:-0}" == "1" ]]; then
  SEEDS=1
  ROUNDS=20
  RND_STEP=4
  EXTRA+=(--quick)
fi

mkdir -p "$OUT/raw" "$OUT/figures"
echo "[RQ4] OUT=$OUT SEEDS=$SEEDS ROUNDS=$ROUNDS datasets=MNIST,CIFAR10 alphas=0.1,0.5"

# 用算术循环避免 SEEDS=0 时 GNU seq 报错退出；且不依赖 seq（macOS 友好）
[[ "$SEEDS" =~ ^[0-9]+$ ]] || SEEDS=3

for ds in MNIST CIFAR10; do
  for a in 0.1 0.5; do
    for ((s = 0; s < SEEDS; s++)); do
      echo ">>> $ds alpha=$a seed=$s"
      python exp_rq/rq4_fl_benchmark.py \
        --dataset "$ds" --alpha "$a" --seed "$s" \
        --rounds "$ROUNDS" --rnd-step "$RND_STEP" \
        --out-dir "$OUT" \
        "${EXTRA[@]}"
    done
  done
done

echo "[RQ4] Plotting..."
python exp_rq/rq4_plot_paper_figures.py --rq4-dir "$OUT" --fig-a-alpha "$FIG_A_ALPHA"
echo "Done. Figures: $OUT/figures/figure_rq4_*.png"
