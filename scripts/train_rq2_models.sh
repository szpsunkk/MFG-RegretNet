#!/usr/bin/env bash
# 为 RQ2 可扩展性实验训练所有需要的 MFG-RegretNet 模型
# 
# 用法：
#   bash scripts/train_rq2_models.sh              # 完整训练（200 epochs，16-20 小时）
#   QUICK=1 bash scripts/train_rq2_models.sh      # 快速模式（20 epochs，2-3 小时）
#   N_LIST="10 50 100" bash scripts/train_rq2_models.sh  # 自定义 N 列表

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 配置
N_LIST="${N_LIST:-10 50 100 200}"
EPOCHS="${EPOCHS:-200}"
EXAMPLES="${EXAMPLES:-102400}"

if [[ "${QUICK:-0}" == "1" ]]; then
  EPOCHS=20
  EXAMPLES=10240
  echo "[Quick mode] epochs=$EPOCHS, examples=$EXAMPLES"
fi

echo "=========================================="
echo "  Training MFG-RegretNet for RQ2"
echo "=========================================="
echo "N values: $N_LIST"
echo "Epochs: $EPOCHS"
echo "Examples: $EXAMPLES"
echo "=========================================="
echo ""

START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAILED_COUNT=0

for N in $N_LIST; do
  echo ">>> Training MFG-RegretNet for N=$N"
  echo "    Command: python train_mfg_regretnet.py --num-epochs $EPOCHS --num-examples $EXAMPLES --n-agents $N --n-items 1"
  
  if python train_mfg_regretnet.py \
    --num-epochs "$EPOCHS" \
    --num-examples "$EXAMPLES" \
    --n-agents "$N" \
    --n-items 1; then
    echo "✓ N=$N completed successfully"
    ((SUCCESS_COUNT++))
  else
    echo "✗ N=$N failed!"
    ((FAILED_COUNT++))
  fi
  echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=========================================="
echo "  Training Summary"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Success: $SUCCESS_COUNT"
echo "Failed:  $FAILED_COUNT"
echo ""

if [[ $SUCCESS_COUNT -gt 0 ]]; then
  echo "Trained checkpoints:"
  ls -lh result/mfg_regretnet_privacy_*_checkpoint.pt 2>/dev/null | tail -n "$SUCCESS_COUNT" || true
  echo ""
fi

echo "Next steps:"
echo "  1. python exp_rq/rq2_paper_benchmark.py --n-list \"$(echo $N_LIST | tr ' ' ',')\""
echo "  2. python exp_rq/rq2_plot_paper_figures.py"
echo ""

if [[ $FAILED_COUNT -gt 0 ]]; then
  exit 1
fi
