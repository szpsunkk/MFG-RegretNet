#!/usr/bin/env bash
# 一键完成 RQ1：数据 v~U[0,1], ε~U[0.1,5]；基线 PAC/VCG/CSRA + RegretNet + MFG-RegretNet；
# 表格、柱状图、遗憾随 PGA 步数收敛曲线、t 检验（需 scipy）。
# MFG 默认保存路径（train_mfg_regretnet.py --name 默认 mfg_regretnet_privacy）：
#   result/mfg_regretnet_privacy_<epoch>_checkpoint.pt
# RegretNet 无统一默认文件名，需自行指定 REGRETNET_CKPT=...

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-run/privacy_paper/rq1}"
# MFG：优先 200 轮最终权重；若无则自动选 result/mfg_regretnet_privacy_* 中 epoch 最大的
MFG_CKPT="${MFG_CKPT:-result/mfg_regretnet_privacy_200_checkpoint.pt}"
if [[ ! -f "$MFG_CKPT" ]]; then
  _mfg_auto=$(ls -1 result/mfg_regretnet_privacy_*_checkpoint.pt 2>/dev/null | sort -V | tail -n1)
  if [[ -n "$_mfg_auto" ]]; then
    echo "[INFO] MFG-RegretNet: using $_mfg_auto"
    MFG_CKPT="$_mfg_auto"
  fi
fi
REGRETNET_CKPT="${REGRETNET_CKPT:-}"

RN_ARG=""
MFG_ARG=""
if [[ -n "$REGRETNET_CKPT" && -f "$REGRETNET_CKPT" ]]; then
  RN_ARG="--regretnet-ckpt $REGRETNET_CKPT"
elif [[ -n "${REGRETNET_CKPT}" ]]; then
  echo "[WARN] RegretNet ckpt not found: $REGRETNET_CKPT"
else
  echo "[INFO] RegretNet: 未设置 REGRETNET_CKPT，跳过（仅 MFG 也可画 PGA 收敛图）"
fi
if [[ -f "$MFG_CKPT" ]]; then
  MFG_ARG="--mfg-regretnet-ckpt $MFG_CKPT"
else
  echo "[WARN] MFG-RegretNet ckpt not found（仓库里尚无 .pt 时需先训练，见文末）"
fi

python exp_rq/rq1_incentive_compatibility.py \
  --out-dir "$OUT" \
  --num-profiles "${NUM_PROFILES:-1000}" \
  --seeds "${SEEDS:-42,43,44}" \
  --n-agents "${N_AGENTS:-10}" \
  --budget "${BUDGET:-50}" \
  $RN_ARG $MFG_ARG \
  --convergence-curve

echo ""
echo "RQ1 输出目录: $OUT"
echo "  - table_rq1.csv / table_rq1.md"
echo "  - rq1_statistics.json（神经两行都有时含 t-test）"
echo "  - figure_rq1_regret_bar.png"
if [[ -f "$OUT/figure_rq1_regret_vs_pga_rounds.png" ]]; then
  echo "  - figure_rq1_regret_vs_pga_rounds.png"
  echo "  - rq1_convergence_curve.json"
else
  echo "  - （未生成）PGA 收敛图：需至少一个有效的 RegretNet 或 MFG-RegretNet checkpoint"
  echo ""
  echo "── 生成 MFG 权重示例（几分钟试跑）──"
  echo "  python train_mfg_regretnet.py --num-epochs 10 --num-examples 10240 --n-agents 10 --n-items 1"
  echo "  然后: ./scripts/run_rq1_full.sh"
  echo "RegretNet：设置 REGRETNET_CKPT=你的 regretnet 权重（须与 N=10,n_items=1 及隐私 bid 一致）"
fi
