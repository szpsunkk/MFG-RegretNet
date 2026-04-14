#!/usr/bin/env bash
# 面向 RQ3（更高 η_rev / W̄）的 MFG-RegretNet 训练：在 regret/IR 主目标上叠加温和辅助项。
# 用法：bash scripts/train_mfg_for_rq3.sh
# 调参：LAMBDA_R（默认 0.06）、LAMBDA_W（默认 0.002）、EPOCHS 等
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EPOCHS="${EPOCHS:-50}"
LAMBDA_R="${LAMBDA_R:-0.06}"
LAMBDA_W="${LAMBDA_W:-0.002}"
NAME="${NAME:-mfg_regretnet_privacy_rq3tune}"
echo ">>> train_mfg_regretnet: lambda_revenue_util=$LAMBDA_R lambda_participant_welfare=$LAMBDA_W name=$NAME"
python3 train_mfg_regretnet.py \
  --num-epochs "$EPOCHS" \
  --num-examples "${NUM_EXAMPLES:-10240}" \
  --n-agents "${N_AGENTS:-10}" --n-items 1 \
  --lambda-revenue-util "$LAMBDA_R" \
  --lambda-participant-welfare "$LAMBDA_W" \
  --name "$NAME"
echo ">>> 将 MFG_CKPT 指向上方 result/${NAME}_*_checkpoint.pt 后运行 ./run_rq3.sh"
