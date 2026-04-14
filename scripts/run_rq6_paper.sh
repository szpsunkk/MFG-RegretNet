#!/usr/bin/env bash
# RQ6 一键：鲁棒性评估（虚假报价、勾结攻击）
# 依赖：torch、cvxpy（与 Phase4 相同）
#
# 用法：
#   bash scripts/run_rq6_paper.sh
#   QUICK=1 bash scripts/run_rq6_paper.sh
#   FALSE_RATIOS="0.1,0.2,0.3" bash scripts/run_rq6_paper.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-run/privacy_paper/rq6}"
N_AGENTS="${N_AGENTS:-10}"
BUDGET="${BUDGET:-50.0}"
NUM_PROFILES="${NUM_PROFILES:-1000}"
FALSE_RATIOS="${FALSE_RATIOS:-0.1,0.3,0.5}"
SEEDS="${SEEDS:-3}"

EXTRA=()
if [[ "${1:-}" == "--" ]]; then shift; EXTRA=("$@"); fi
if [[ "${QUICK:-0}" == "1" ]]; then
  NUM_PROFILES=300
  FALSE_RATIOS="0.1,0.3"
  SEEDS=1
  EXTRA+=(--quick)
fi

mkdir -p "$OUT"
echo "[RQ6] OUT=$OUT N=$N_AGENTS FALSE_RATIOS=$FALSE_RATIOS SEEDS=$SEEDS"

# 若指定了 MFG checkpoint，传递给脚本
MFG_ARGS=()
if [[ -n "${MFG_CKPT:-}" ]] && [[ -f "$MFG_CKPT" ]]; then
  MFG_ARGS+=(--mfg-regretnet-ckpt "$MFG_CKPT")
fi

# 循环运行不同的虚假报价比例
IFS=',' read -ra RATIOS <<< "$FALSE_RATIOS"
for ratio in "${RATIOS[@]}"; do
  for s in $(seq 0 $((SEEDS - 1))); do
    echo ">>> false_ratio=$ratio seed=$s"
    python exp_rq/rq6_robustness.py \
      --out-dir "$OUT" \
      --n-agents "$N_AGENTS" \
      --budget "$BUDGET" \
      --num-profiles "$NUM_PROFILES" \
      --false-ratio "$ratio" \
      --seed "$s" \
      "${MFG_ARGS[@]}" \
      "${EXTRA[@]}"
  done
done

echo "Done. Results: $OUT/rq6_results.json"
