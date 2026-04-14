#!/usr/bin/env bash
# =============================================================================
# RQ3 一键：收益 η_rev + 社会福利 W̄（每种子对 T 轮时间平均，再 mean±std 跨种子）
#    定义见 exp_rq/RQ3_PROCESS.md
# =============================================================================
# 依赖：项目根目录、torch、cvxpy、matplotlib
# 可选环境变量（与 RQ1 类似）：
#   OUT=run/privacy_paper/rq3
#   BUDGET=50  N_AGENTS=10  NUM_PROFILES=1000  SEEDS=42,43,44
#   REGRETNET_CKPT / DM_REGRETNET_CKPT / MFG_CKPT（不设则自动探测 result/*.pt）
#   RQ3_NO_FIG2=1  RQ3_NO_FIG3=1  # 跳过较耗时的图2/图3
# =============================================================================
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-run/privacy_paper/rq3}"
N_AGENTS="${N_AGENTS:-10}"
BUDGET="${BUDGET:-50}"
NUM_PROFILES="${NUM_PROFILES:-1000}"
SEEDS="${SEEDS:-42,43,44,45,46}"

MFG_CKPT="${MFG_CKPT:-${MFG_REGRETNET_CKPT:-}}"
if [[ -z "$MFG_CKPT" ]]; then
  MFG_CKPT="$(python3 -c "from exp_rq.rq1_ckpt_resolve import resolve_mfg_regretnet_ckpt; print(resolve_mfg_regretnet_ckpt(${N_AGENTS},1) or '')" 2>/dev/null || true)"
fi
RN="$(python3 -c "from exp_rq.rq1_ckpt_resolve import resolve_regretnet_ckpt; print(resolve_regretnet_ckpt(${N_AGENTS},1) or '')" 2>/dev/null || true)"
DM="$(python3 -c "from exp_rq.rq1_ckpt_resolve import resolve_dm_regretnet_ckpt; print(resolve_dm_regretnet_ckpt(${N_AGENTS},1) or '')" 2>/dev/null || true)"

EXTRA=(--out-dir "$OUT" --n-agents "$N_AGENTS" --budget "$BUDGET" --num-profiles "$NUM_PROFILES" --seeds "$SEEDS")
[[ -n "$RN" && -f "$RN" ]] && EXTRA+=(--regretnet-ckpt "$RN")
[[ -n "$DM" && -f "$DM" ]] && EXTRA+=(--dm-regretnet-ckpt "$DM")
[[ -n "$MFG_CKPT" && -f "$MFG_CKPT" ]] && EXTRA+=(--mfg-regretnet-ckpt "$MFG_CKPT")

[[ "${RQ3_NO_FIG2:-}" == "1" ]] && EXTRA+=(--no-figure2)
[[ "${RQ3_NO_FIG3:-}" == "1" ]] && EXTRA+=(--no-figure3)

echo ">>> RQ3 输出目录: $OUT"
python3 exp_rq/rq3_paper_complete.py "${EXTRA[@]}" "$@"
echo ">>> 完成：figure_rq3_revenue_welfare_bars.png, figure_rq3_R_W_vs_epoch.png, figure_rq3_budget_sensitivity.png"
