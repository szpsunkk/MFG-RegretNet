#!/usr/bin/env bash
# 消融研究一键：MFG 组件（RegretNet vs MFG-RegretNet）、增强拉格朗日影响
# 依赖：torch、cvxpy
#
# 用法：
#   bash scripts/run_ablation_study.sh
#   QUICK=1 bash scripts/run_ablation_study.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-run/privacy_paper/ablation}"
N_AGENTS="${N_AGENTS:-10}"
BUDGET="${BUDGET:-50.0}"
NUM_PROFILES="${NUM_PROFILES:-1000}"

EXTRA=()
if [[ "${1:-}" == "--" ]]; then shift; EXTRA=("$@"); fi
if [[ "${QUICK:-0}" == "1" ]]; then
  NUM_PROFILES=300
  EXTRA+=(--quick)
fi

mkdir -p "$OUT"
echo "[Ablation] OUT=$OUT N=$N_AGENTS"

# 构建参数
ARGS=(--out-dir "$OUT" --n-agents "$N_AGENTS" --budget "$BUDGET" --num-profiles "$NUM_PROFILES")

# RegretNet checkpoint（无 b_MFG）
if [[ -n "${REGRETNET_CKPT:-}" ]] && [[ -f "$REGRETNET_CKPT" ]]; then
  ARGS+=(--regretnet-ckpt "$REGRETNET_CKPT")
else
  echo "[Warning] REGRETNET_CKPT not set or file not found. Will skip RegretNet comparison."
fi

# MFG-RegretNet checkpoint（有 b_MFG）
if [[ -n "${MFG_CKPT:-}" ]] && [[ -f "$MFG_CKPT" ]]; then
  ARGS+=(--mfg-regretnet-ckpt "$MFG_CKPT")
elif ls result/mfg_regretnet_privacy_*_checkpoint.pt 1> /dev/null 2>&1; then
  # 自动查找最新的 MFG checkpoint
  MFG_AUTO=$(ls -t result/mfg_regretnet_privacy_*_checkpoint.pt | head -1)
  echo "[Info] Auto-detected MFG checkpoint: $MFG_AUTO"
  ARGS+=(--mfg-regretnet-ckpt "$MFG_AUTO")
else
  echo "[Warning] MFG_CKPT not found. Will skip MFG-RegretNet comparison."
fi

python exp_rq/ablation_study.py "${ARGS[@]}" "${EXTRA[@]}"

echo "Done. Results: $OUT/ablation_table.csv"
