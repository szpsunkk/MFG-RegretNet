#!/usr/bin/env bash
# 一键：RQ1 论文主表 + 图 A（遗憾）+ 图 B（IR%）
# 默认 5 seeds；需 cvxpy + scipy + matplotlib；神经方法需与 N=10,n_items=1 一致的 checkpoint。
#
# 用法：
#   export MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt
#   export REGRETNET_CKPT=path/to/regretnet.pt   # 可选
#   export DM_REGRETNET_CKPT=path/to/dm.pt       # 可选，需架构匹配
#   ./scripts/run_rq1_paper.sh
#
# 子表按数据集（MNIST/CIFAR/…）：当前脚本为合成 bid；扩展时复制本脚本改数据源即可。

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-run/privacy_paper/rq1}"
SEEDS="${SEEDS:-42,43,44,45,46}"
MFG_CKPT="${MFG_CKPT:-}"
if [[ -z "$MFG_CKPT" ]]; then
  for c in result/mfg_regretnet_privacy_200_checkpoint.pt \
           $(ls -1 result/mfg_regretnet_privacy_*_checkpoint.pt 2>/dev/null | sort -V | tail -n1); do
    [[ -f "$c" ]] && MFG_CKPT="$c" && break
  done
fi

EXTRA=()
[[ -f "${REGRETNET_CKPT:-}" ]] && EXTRA+=(--regretnet-ckpt "$REGRETNET_CKPT")
[[ -f "${DM_REGRETNET_CKPT:-}" ]] && EXTRA+=(--dm-regretnet-ckpt "$DM_REGRETNET_CKPT")
[[ -f "$MFG_CKPT" ]] && EXTRA+=(--mfg-regretnet-ckpt "$MFG_CKPT")

IR_ARG=()
[[ "${IR_LOG:-}" == "1" ]] && IR_ARG=(--ir-log-scale)

python exp_rq/rq1_paper_table_figures.py \
  --out-dir "$OUT" \
  --seeds "$SEEDS" \
  --num-profiles "${NUM_PROFILES:-1000}" \
  --batch-size "${BATCH_SIZE:-256}" \
  "${IR_ARG[@]}" \
  "${EXTRA[@]}"

echo ""
echo "输出："
echo "  $OUT/table_rq1_paper.md / .csv"
echo "  $OUT/rq1_paper.json"
echo "  $OUT/figure_rq1_paper_regret.png"
echo "  $OUT/figure_rq1_paper_ir.png"
echo "（IR 对数纵轴：IR_LOG=1 ./scripts/run_rq1_paper.sh）"
