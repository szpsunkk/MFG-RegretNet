#!/usr/bin/env bash
# 面向 RQ5（隐私–精度 FL）：略增 participant welfare，减轻「部分客户端 ε≈0 → 噪声爆炸拖垮精度」。
# 在 RQ1 IR 仍可接受的前提下网格搜索 LAMBDA_W / LAMBDA_R 更佳。
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EPOCHS="${EPOCHS:-20}"
# 默认：比 RQ3 稍强调 welfare（可按验证集调）
LAMBDA_W="${LAMBDA_W:-0.008}"
LAMBDA_R="${LAMBDA_R:-0.05}"
NAME="${NAME:-mfg_regretnet_privacy_rq5tune}"
echo ">>> RQ5-oriented MFG: lambda_revenue_util=$LAMBDA_R lambda_participant_welfare=$LAMBDA_W -> $NAME"
python3 train_mfg_regretnet.py \
  --num-epochs "$EPOCHS" \
  --num-examples "${NUM_EXAMPLES:-10240}" \
  --n-agents "${N_AGENTS:-10}" --n-items 1 \
  --lambda-revenue-util "$LAMBDA_R" \
  --lambda-participant-welfare "$LAMBDA_W" \
  --name "$NAME"
echo ">>> 运行 RQ5: MFG_CKPT=result/${NAME}_*_checkpoint.pt bash scripts/run_rq5_paper.sh"
echo ">>> 说明见 exp_rq/RQ5_IMPROVING_OURS.md"
