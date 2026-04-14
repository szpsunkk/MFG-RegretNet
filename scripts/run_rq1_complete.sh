#!/usr/bin/env bash
# =============================================================================
# RQ1 完整一键：实验过程 + 全部表图
# =============================================================================
# 顺序：环境检查 → 阶段A（激励相容表+柱图+PGA收敛+Welch t-test）
#      → 阶段B（论文主表+图A/B+paired t-test）→ 写出本次运行记录
#
# 依赖：Python3、torch、numpy、cvxpy、scipy、matplotlib
#
# 常用环境变量：
#   OUT=run/privacy_paper/rq1
#   SEEDS=42,43,44,45,46
#   NUM_PROFILES=1000
#   MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt
#   REGRETNET_CKPT=...        # 可选；不设则自动探测 result/regretnet_privacy_*.pt
#   REGRETNET_AUTO_TRAIN=0    # 不自动训练 RegretNet（默认 1：缺则 train，默认 10 epoch）
#   REGRETNET_TRAIN_EPOCHS=10 # 自动训练轮数（可改大以提升效果）
#   DM_REGRETNET_CKPT=...     # 可选；不设则自动探测 dm_regretnet_privacy_*.pt
#   DM_REGRETNET_AUTO_TRAIN=1 # 缺 DM 权重时自动 train_dm_regretnet_privacy（默认 1）
#   DM_REGRETNET_TRAIN_EPOCHS=10
#   DM_REGRETNET_TRAIN_EXAMPLES=32768  # 不设则同 REGRETNET_TRAIN_EXAMPLES
#   IR_LOG=1                  # 论文图B 纵轴对数
#   SKIP_PAPER=1              # 仅跑阶段A
#   SKIP_INCENTIVE=1          # 仅跑阶段B
#   RQ1_FIG_CD=1              # 阶段 2b：图C（遗憾+IR% vs 训练epoch）+ 图D（遗憾分布）（默认 1）
#   RQ1_FIG_CD=0              # 跳过图C/D（checkpoint 多时图C较久）
#   RQ1_FIG_C_PROFILES=800    # 图 C 用 profile 数（略少可加速）
#   RQ1_FIG_D_PROFILES=1200   # 图 D 用 profile 数
#   RQ1_FIG_C_MAX_CKPTS=12    # 图 C 每类神经机制最多评估的 epoch 点数
# =============================================================================

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-run/privacy_paper/rq1}"
SEEDS="${SEEDS:-42,43,44,45,46}"
NUM_PROFILES="${NUM_PROFILES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUDGET="${BUDGET:-50}"
N_AGENTS="${N_AGENTS:-10}"

mkdir -p "$OUT"
RUN_TS="$(date -Iseconds)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

# ---------- Checkpoint 解析 ----------
# 未设置 MFG_CKPT 时直接自动探测，不假定一定有 _200_（短训只有 _10_/_50_ 等）
MFG_CKPT="${MFG_CKPT:-}"
if [[ -n "$MFG_CKPT" && ! -f "$MFG_CKPT" ]]; then
  echo "[INFO] MFG_CKPT 文件不存在（常见：只训了少量 epoch，没有 _200_）："
  echo "       $MFG_CKPT → 将自动选用 result/mfg_regretnet_privacy_* 中已有权重"
  MFG_CKPT=""
fi
if [[ -z "$MFG_CKPT" ]]; then
  if [[ -f result/mfg_regretnet_privacy_200_checkpoint.pt ]]; then
    MFG_CKPT="result/mfg_regretnet_privacy_200_checkpoint.pt"
  else
    _auto=$(ls -1 result/mfg_regretnet_privacy_*_checkpoint.pt 2>/dev/null | sort -V | tail -n1)
    [[ -n "$_auto" ]] && MFG_CKPT="$_auto"
  fi
fi
[[ -n "$MFG_CKPT" ]] && echo "[INFO] MFG-RegretNet 使用: $MFG_CKPT"
REGRETNET_CKPT="${REGRETNET_CKPT:-}"
DM_REGRETNET_CKPT="${DM_REGRETNET_CKPT:-}"

if [[ -n "$REGRETNET_CKPT" && ! -f "$REGRETNET_CKPT" ]]; then
  echo "[INFO] REGRETNET_CKPT 无效，将自动探测或训练: $REGRETNET_CKPT"
  REGRETNET_CKPT=""
fi
if [[ -z "$REGRETNET_CKPT" ]]; then
  REGRETNET_CKPT="$(python3 -c "from exp_rq.rq1_ckpt_resolve import resolve_regretnet_ckpt; print(resolve_regretnet_ckpt(${N_AGENTS},1) or '')" 2>/dev/null || true)"
fi
if [[ -z "$REGRETNET_CKPT" && "${REGRETNET_AUTO_TRAIN:-1}" == "1" ]]; then
  echo "[INFO] 未找到 RegretNet 权重，正在训练（约数分钟；跳过请设 REGRETNET_AUTO_TRAIN=0）"
  python3 train_regretnet_privacy.py \
    --num-epochs "${REGRETNET_TRAIN_EPOCHS:-10}" \
    --num-examples "${REGRETNET_TRAIN_EXAMPLES:-32768}" \
    --n-agents "$N_AGENTS" --n-items 1
  REGRETNET_CKPT="$(python3 -c "from exp_rq.rq1_ckpt_resolve import resolve_regretnet_ckpt; print(resolve_regretnet_ckpt(${N_AGENTS},1) or '')" 2>/dev/null || true)"
fi

RN_ARG=()
if [[ -n "$REGRETNET_CKPT" && -f "$REGRETNET_CKPT" ]]; then
  RN_ARG=(--regretnet-ckpt "$REGRETNET_CKPT")
  echo "[INFO] RegretNet: $REGRETNET_CKPT"
else
  echo "[WARN] 无 RegretNet 权重：设 REGRETNET_AUTO_TRAIN=1（默认）或先运行 python3 train_regretnet_privacy.py"
fi

if [[ -n "${DM_REGRETNET_CKPT:-}" && ! -f "$DM_REGRETNET_CKPT" ]]; then
  echo "[INFO] DM_REGRETNET_CKPT 文件不存在，将自动探测或训练: $DM_REGRETNET_CKPT"
  DM_REGRETNET_CKPT=""
fi
if [[ -z "$DM_REGRETNET_CKPT" ]]; then
  DM_REGRETNET_CKPT="$(python3 -c "from exp_rq.rq1_ckpt_resolve import resolve_dm_regretnet_ckpt; print(resolve_dm_regretnet_ckpt(${N_AGENTS},1) or '')" 2>/dev/null || true)"
fi
# 直接按文件名选最新（不依赖 Python 解析）
_dm_pick() {
  ls -1 result/dm_regretnet_privacy_*_checkpoint.pt 2>/dev/null | sort -V | tail -n1
}
if [[ -z "$DM_REGRETNET_CKPT" || ! -f "$DM_REGRETNET_CKPT" ]]; then
  _dmf="$(_dm_pick)"
  [[ -n "$_dmf" && -f "$_dmf" ]] && DM_REGRETNET_CKPT="$_dmf"
fi
if [[ -z "$DM_REGRETNET_CKPT" && "${DM_REGRETNET_AUTO_TRAIN:-1}" == "1" ]]; then
  echo "[INFO] 未找到 DM-RegretNet 权重，正在训练（跳过请 export DM_REGRETNET_AUTO_TRAIN=0）"
  _DM_NE="${DM_REGRETNET_TRAIN_EXAMPLES:-}"
  [[ -z "$_DM_NE" ]] && _DM_NE="${REGRETNET_TRAIN_EXAMPLES:-32768}"
  python3 train_dm_regretnet_privacy.py \
    --num-epochs "${DM_REGRETNET_TRAIN_EPOCHS:-10}" \
    --num-examples "$_DM_NE" \
    --n-agents "$N_AGENTS" --n-items 1
  DM_REGRETNET_CKPT="$(python3 -c "from exp_rq.rq1_ckpt_resolve import resolve_dm_regretnet_ckpt; print(resolve_dm_regretnet_ckpt(${N_AGENTS},1) or '')" 2>/dev/null || true)"
  if [[ -z "$DM_REGRETNET_CKPT" || ! -f "$DM_REGRETNET_CKPT" ]]; then
    _dmf="$(_dm_pick)"
    [[ -n "$_dmf" && -f "$_dmf" ]] && DM_REGRETNET_CKPT="$_dmf"
  fi
fi
if [[ -n "$DM_REGRETNET_CKPT" && -f "$DM_REGRETNET_CKPT" ]]; then
  echo "[INFO] DM-RegretNet: $DM_REGRETNET_CKPT"
else
  if [[ "${DM_REGRETNET_AUTO_TRAIN:-1}" != "1" ]]; then
    echo "[INFO] 已跳过 DM-RegretNet（DM_REGRETNET_AUTO_TRAIN=0）。需要时请运行: python3 train_dm_regretnet_privacy.py"
  else
    echo "[WARN] DM-RegretNet 仍未就绪。请在项目根目录执行:"
    echo "       python3 train_dm_regretnet_privacy.py --n-agents $N_AGENTS --n-items 1"
    echo "       并确认生成 result/dm_regretnet_privacy_*_checkpoint.pt"
  fi
fi

DM_ARG=()
[[ -f "${DM_REGRETNET_CKPT:-}" ]] && DM_ARG=(--dm-regretnet-ckpt "$DM_REGRETNET_CKPT")

MFG_ARG=()
[[ -f "$MFG_CKPT" ]] && MFG_ARG=(--mfg-regretnet-ckpt "$MFG_CKPT") || echo "[WARN] 未找到 MFG checkpoint，阶段A/B 中 Ours 可能缺失"

PAPER_EXTRA=()
[[ -f "${REGRETNET_CKPT:-}" ]] && PAPER_EXTRA+=(--regretnet-ckpt "$REGRETNET_CKPT")
[[ -f "${DM_REGRETNET_CKPT:-}" ]] && PAPER_EXTRA+=(--dm-regretnet-ckpt "$DM_REGRETNET_CKPT")
[[ -f "$MFG_CKPT" ]] && PAPER_EXTRA+=(--mfg-regretnet-ckpt "$MFG_CKPT")

IR_ARG=()
[[ "${IR_LOG:-}" == "1" ]] && IR_ARG=(--ir-log-scale)

echo "=============================================="
echo " RQ1 完整实验  |  OUT=$OUT"
echo " SEEDS=$SEEDS  |  profiles=$NUM_PROFILES"
echo " MFG_CKPT=${MFG_CKPT:-（无）}"
echo "=============================================="

# ---------- 0) 环境 ----------
echo ""
echo ">>> [0/3] 环境检查"
if ! python3 -c "import torch, numpy, cvxpy, scipy, matplotlib" 2>/dev/null; then
  echo "[ERROR] 缺少依赖。请安装: pip install torch numpy cvxpy scipy matplotlib"
  exit 1
fi
echo "    cvxpy / scipy / matplotlib / torch — OK"

# ---------- 1) 阶段 A ----------
if [[ "${SKIP_INCENTIVE:-}" != "1" ]]; then
  echo ""
  echo ">>> [1/3] 阶段 A：rq1_incentive_compatibility（表+遗憾柱图+Welch t-test+PGA收敛）"
  python3 exp_rq/rq1_incentive_compatibility.py \
    --out-dir "$OUT" \
    --seeds "$SEEDS" \
    --num-profiles "$NUM_PROFILES" \
    --batch-size "$BATCH_SIZE" \
    --n-agents "$N_AGENTS" \
    --budget "$BUDGET" \
    --baseline-v-grid 15 \
    --baseline-eps-grid 7 \
    "${RN_ARG[@]}" \
    "${DM_ARG[@]}" \
    "${MFG_ARG[@]}" \
    --convergence-curve
else
  echo ""
  echo ">>> [1/3] 跳过阶段 A（SKIP_INCENTIVE=1）"
fi

# ---------- 2) 阶段 B ----------
if [[ "${SKIP_PAPER:-}" != "1" ]]; then
  echo ""
  echo ">>> [2/3] 阶段 B：rq1_paper_table_figures（论文主表+图A/B+paired t-test）"
  python3 exp_rq/rq1_paper_table_figures.py \
    --out-dir "$OUT" \
    --seeds "$SEEDS" \
    --num-profiles "$NUM_PROFILES" \
    --batch-size "$BATCH_SIZE" \
    --n-agents "$N_AGENTS" \
    --budget "$BUDGET" \
    --baseline-v-grid 15 \
    --baseline-eps-grid 7 \
    "${IR_ARG[@]}" \
    "${PAPER_EXTRA[@]}"
else
  echo ""
  echo ">>> [2/3] 跳过阶段 B（SKIP_PAPER=1）"
fi

# ---------- 2b) 可选图 C / D ----------
if [[ "${RQ1_FIG_CD:-1}" == "1" ]]; then
  echo ""
  echo ">>> [2b] 图 C/D：regret & IR vs training epoch + regret distribution"
  _NFC="${RQ1_FIG_C_PROFILES:-800}"
  _NFD="${RQ1_FIG_D_PROFILES:-1200}"
  _MCK="${RQ1_FIG_C_MAX_CKPTS:-18}"
  python3 exp_rq/rq1_figure_c_training_rounds.py \
    --out-dir "$OUT" --n-agents "$N_AGENTS" --budget "$BUDGET" \
    --num-profiles "$_NFC" --batch-size "$BATCH_SIZE" \
    --max-ckpts-per-method "$_MCK" || echo "[WARN] figure C failed (see log above)"
  FIGD_X=()
  [[ -f "${REGRETNET_CKPT:-}" ]] && FIGD_X+=(--regretnet-ckpt "$REGRETNET_CKPT")
  [[ -f "${DM_REGRETNET_CKPT:-}" ]] && FIGD_X+=(--dm-regretnet-ckpt "$DM_REGRETNET_CKPT")
  [[ -f "${MFG_CKPT:-}" ]] && FIGD_X+=(--mfg-regretnet-ckpt "$MFG_CKPT")
  python3 exp_rq/rq1_figure_d_regret_distribution.py \
    --out-dir "$OUT" --n-agents "$N_AGENTS" --budget "$BUDGET" \
    --num-profiles "$_NFD" --batch-size "$BATCH_SIZE" "${FIGD_X[@]}" \
    || echo "[WARN] figure D failed (see log above)"
fi

# ---------- 3) 运行记录 ----------
LOG="$OUT/RQ1_last_run.md"
{
  echo "# RQ1 本次运行记录"
  echo ""
  echo "- **时间**: $RUN_TS"
  echo "- **目录**: \`$OUT\`"
  echo "- **Seeds**: $SEEDS"
  echo "- **Profiles/seed**: $NUM_PROFILES"
  echo "- **N, B**: $N_AGENTS, $BUDGET"
  echo "- **MFG_CKPT**: ${MFG_CKPT:-无}"
  echo "- **REGRETNET_CKPT**: ${REGRETNET_CKPT:-未设置}"
  echo "- **DM_REGRETNET_CKPT**: ${DM_REGRETNET_CKPT:-未设置}"
  echo ""
  echo "## 已执行阶段"
  echo ""
  echo "### 阶段 A（激励相容 + PGA 收敛）"
  echo '```bash'
  echo "python3 exp_rq/rq1_incentive_compatibility.py --out-dir \"$OUT\" --seeds \"$SEEDS\" \\"
  echo "  --num-profiles $NUM_PROFILES --n-agents $N_AGENTS --budget $BUDGET \\"
  echo "  --convergence-curve [+ regretnet/mfg ckpt]"
  echo '```'
  echo ""
  echo "### 阶段 B（论文表 + 图 A/B）"
  echo '```bash'
  echo "python3 exp_rq/rq1_paper_table_figures.py --out-dir \"$OUT\" --seeds \"$SEEDS\" \\"
  echo "  --num-profiles $NUM_PROFILES [+ ckpt 与可选 --ir-log-scale]"
  echo '```'
  echo ""
  echo "### 可选 图 C / 图 D（RQ1_FIG_CD=1）"
  echo "- figure_rq1_paper_regret_vs_epoch.png, rq1_figure_c.json"
  echo "- figure_rq1_paper_regret_distribution.png, rq1_figure_d.json"
  echo ""
  echo "## 产出清单"
  ls -la "$OUT" 2>/dev/null | grep -E '\.(png|csv|md|json)$' || true
  echo ""
  echo "## 完整实验过程说明"
  echo "见仓库 \`exp_rq/RQ1_EXPERIMENT_PROCESS.md\`"
  echo ""
  echo "## 常见错误"
  echo "- 勿把 \`result/....pt\`、\`../..\` 等占位符当作真实参数执行（会报 unrecognized arguments）。"
  echo "- 阶段 A 若显示 \`No valid neural checkpoints\`：请在项目根目录运行，并保证 \`result/mfg_regretnet_privacy_*_checkpoint.pt\` 存在或设置正确的 \`MFG_CKPT\`。"
} > "$LOG"

echo ""
echo ">>> [3/3] 已写入运行记录: $LOG"
echo ""
echo "=============================================="
echo " RQ1 完成"
echo "  过程文档: exp_rq/RQ1_EXPERIMENT_PROCESS.md"
echo "  本次记录: $OUT/RQ1_last_run.md"
echo "  表图: $OUT/"
echo "=============================================="
