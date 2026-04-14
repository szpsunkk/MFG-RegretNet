#!/usr/bin/env bash
# 一键运行所有 RQ1-RQ6 + 消融研究（按照 main_m.pdf 论文实验）
# 
# 用法：
#   bash run_all_rq.sh                    # 完整运行（需要预先训练好的 checkpoints）
#   QUICK=1 bash run_all_rq.sh            # 快速测试模式
#   SKIP_RQ1=1 bash run_all_rq.sh         # 跳过 RQ1
#   bash run_all_rq.sh --only rq1,rq2     # 只运行指定的实验
#
# 环境变量：
#   MFG_CKPT          - MFG-RegretNet checkpoint 路径（必需）
#   REGRETNET_CKPT    - RegretNet checkpoint 路径（RQ1/消融需要）
#   DM_REGRETNET_CKPT - DM-RegretNet checkpoint 路径（RQ1 需要）
#   OUT               - 输出目录（默认：run/privacy_paper）
#   QUICK             - 快速测试模式（1=开启）
#   SKIP_RQ1-6        - 跳过对应的实验（1=跳过）
#   SKIP_ABLATION     - 跳过消融研究（1=跳过）
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ============= 配置区 =============
OUT="${OUT:-run/privacy_paper}"
QUICK="${QUICK:-0}"

# 默认运行所有实验
SKIP_RQ1="${SKIP_RQ1:-0}"
SKIP_RQ2="${SKIP_RQ2:-0}"
SKIP_RQ3="${SKIP_RQ3:-0}"
SKIP_RQ4="${SKIP_RQ4:-0}"
SKIP_RQ5="${SKIP_RQ5:-0}"
SKIP_RQ6="${SKIP_RQ6:-0}"
SKIP_ABLATION="${SKIP_ABLATION:-0}"

# 处理 --only 参数
ONLY_RQ=""
if [[ "${1:-}" == "--only" ]] && [[ -n "${2:-}" ]]; then
  ONLY_RQ="$2"
  echo "[Info] Only running: $ONLY_RQ"
  # 先全部跳过
  SKIP_RQ1=1; SKIP_RQ2=1; SKIP_RQ3=1; SKIP_RQ4=1; SKIP_RQ5=1; SKIP_RQ6=1; SKIP_ABLATION=1
  # 然后打开指定的
  [[ "$ONLY_RQ" =~ rq1 ]] && SKIP_RQ1=0
  [[ "$ONLY_RQ" =~ rq2 ]] && SKIP_RQ2=0
  [[ "$ONLY_RQ" =~ rq3 ]] && SKIP_RQ3=0
  [[ "$ONLY_RQ" =~ rq4 ]] && SKIP_RQ4=0
  [[ "$ONLY_RQ" =~ rq5 ]] && SKIP_RQ5=0
  [[ "$ONLY_RQ" =~ rq6 ]] && SKIP_RQ6=0
  [[ "$ONLY_RQ" =~ ablation ]] && SKIP_ABLATION=0
fi

# ============= 检查依赖 =============
echo "=========================================="
echo "  运行 main_m.pdf 论文完整实验（RQ1-RQ6）"
echo "=========================================="
echo ""
echo "输出目录: $OUT"
echo "快速模式: $QUICK"
echo ""

# 检查 MFG checkpoint（大部分实验需要）
if [[ -z "${MFG_CKPT:-}" ]]; then
  # 尝试自动查找
  if ls result/mfg_regretnet_privacy_*_checkpoint.pt 1> /dev/null 2>&1; then
    MFG_CKPT=$(ls -t result/mfg_regretnet_privacy_*_checkpoint.pt | head -1)
    echo "[Info] Auto-detected MFG_CKPT: $MFG_CKPT"
  else
    echo "[Warning] MFG_CKPT not set and not found in result/. Some experiments may fail."
    echo "          Please train first: python train_mfg_regretnet.py --num-epochs 200"
  fi
else
  if [[ ! -f "$MFG_CKPT" ]]; then
    echo "[Error] MFG_CKPT file not found: $MFG_CKPT"
    exit 1
  fi
  echo "[Info] Using MFG_CKPT: $MFG_CKPT"
fi

# 导出环境变量供子脚本使用
export MFG_CKPT
export REGRETNET_CKPT="${REGRETNET_CKPT:-}"
export DM_REGRETNET_CKPT="${DM_REGRETNET_CKPT:-}"
export OUT
export QUICK

mkdir -p "$OUT/logs"

# ============= 开始运行实验 =============
START_TIME=$(date +%s)

# RQ1: 激励相容性（Incentive Compatibility）
if [[ "$SKIP_RQ1" == "0" ]]; then
  echo ""
  echo "================================================"
  echo "  RQ1: 激励相容性（Regret + IR Violation）"
  echo "================================================"
  echo "Running: bash scripts/run_rq1_complete.sh"
  bash scripts/run_rq1_complete.sh 2>&1 | tee "$OUT/logs/rq1.log"
  echo "✓ RQ1 完成。结果: $OUT/rq1/"
else
  echo "⊘ 跳过 RQ1"
fi

# RQ2: 可扩展性（Scalability）
if [[ "$SKIP_RQ2" == "0" ]]; then
  echo ""
  echo "================================================"
  echo "  RQ2: 可扩展性（Time vs N）"
  echo "================================================"
  echo "Running: bash scripts/run_rq2_paper.sh"
  bash scripts/run_rq2_paper.sh 2>&1 | tee "$OUT/logs/rq2.log"
  echo "✓ RQ2 完成。结果: $OUT/rq2/figures/"
else
  echo "⊘ 跳过 RQ2"
fi

# RQ3: 拍卖效率（Auction Efficiency）
if [[ "$SKIP_RQ3" == "0" ]]; then
  echo ""
  echo "================================================"
  echo "  RQ3: 拍卖效率（Revenue + Social Welfare）"
  echo "================================================"
  echo "Running: bash scripts/run_rq3_complete.sh"
  bash scripts/run_rq3_complete.sh 2>&1 | tee "$OUT/logs/rq3.log"
  echo "✓ RQ3 完成。结果: $OUT/rq3/"
else
  echo "⊘ 跳过 RQ3"
fi

# RQ4: FL 收敛性（FL Convergence）
if [[ "$SKIP_RQ4" == "0" ]]; then
  echo ""
  echo "================================================"
  echo "  RQ4: FL 收敛性（MNIST + CIFAR-10）"
  echo "================================================"
  echo "Running: bash scripts/run_rq4_paper.sh"
  bash scripts/run_rq4_paper.sh 2>&1 | tee "$OUT/logs/rq4.log"
  echo "✓ RQ4 完成。结果: $OUT/rq4/figures/"
else
  echo "⊘ 跳过 RQ4"
fi

# RQ5: 隐私-效用权衡（Privacy-Utility Tradeoff）
if [[ "$SKIP_RQ5" == "0" ]]; then
  echo ""
  echo "================================================"
  echo "  RQ5: 隐私-效用权衡（Pareto + 负担分布）"
  echo "================================================"
  echo "Running: bash scripts/run_rq5_paper.sh"
  bash scripts/run_rq5_paper.sh 2>&1 | tee "$OUT/logs/rq5.log"
  echo "✓ RQ5 完成。结果: $OUT/rq5/figures/"
else
  echo "⊘ 跳过 RQ5"
fi

# RQ6: 鲁棒性（Robustness）
if [[ "$SKIP_RQ6" == "0" ]]; then
  echo ""
  echo "================================================"
  echo "  RQ6: 鲁棒性（虚假报价 + 勾结攻击）"
  echo "================================================"
  echo "Running: bash scripts/run_rq6_paper.sh"
  bash scripts/run_rq6_paper.sh 2>&1 | tee "$OUT/logs/rq6.log"
  echo "✓ RQ6 完成。结果: $OUT/rq6/"
else
  echo "⊘ 跳过 RQ6"
fi

# 消融研究（Ablation Study）
if [[ "$SKIP_ABLATION" == "0" ]]; then
  echo ""
  echo "================================================"
  echo "  消融研究（MFG Component + Lagrangian）"
  echo "================================================"
  echo "Running: bash scripts/run_ablation_study.sh"
  bash scripts/run_ablation_study.sh 2>&1 | tee "$OUT/logs/ablation.log"
  echo "✓ 消融研究完成。结果: $OUT/ablation/"
else
  echo "⊘ 跳过消融研究"
fi

# ============= 完成汇总 =============
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "  所有实验完成！"
echo "=========================================="
echo ""
echo "总用时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "结果目录结构:"
echo "  $OUT/"
echo "  ├── rq1/         # RQ1: 激励相容性"
echo "  ├── rq2/         # RQ2: 可扩展性"
echo "  ├── rq3/         # RQ3: 拍卖效率"
echo "  ├── rq4/         # RQ4: FL 收敛性"
echo "  ├── rq5/         # RQ5: 隐私-效用"
echo "  ├── rq6/         # RQ6: 鲁棒性"
echo "  ├── ablation/    # 消融研究"
echo "  └── logs/        # 运行日志"
echo ""
echo "关键图表:"
[[ "$SKIP_RQ1" == "0" ]] && echo "  - RQ1 遗憾与 IR: $OUT/rq1/figure_rq1_paper_*.png"
[[ "$SKIP_RQ2" == "0" ]] && echo "  - RQ2 可扩展性: $OUT/rq2/figures/figure_rq2_*.png"
[[ "$SKIP_RQ3" == "0" ]] && echo "  - RQ3 收益福利: $OUT/rq3/figure_rq3_*.png"
[[ "$SKIP_RQ4" == "0" ]] && echo "  - RQ4 FL 精度: $OUT/rq4/figures/figure_rq4_*.png"
[[ "$SKIP_RQ5" == "0" ]] && echo "  - RQ5 Pareto: $OUT/rq5/figures/figure_rq5_*.png"
[[ "$SKIP_RQ6" == "0" ]] && echo "  - RQ6 鲁棒性: $OUT/rq6/rq6_results.json"
[[ "$SKIP_ABLATION" == "0" ]] && echo "  - 消融表格: $OUT/ablation/ablation_table.csv"
echo ""
echo "详细日志见: $OUT/logs/*.log"
echo ""
