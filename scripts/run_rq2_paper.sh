#!/usr/bin/env bash
# RQ2 一键：可扩展性 → 图1 log-log 时间、图2 内存/通信、图3 堆叠延迟
# 依赖：torch、cvxpy（与 Phase4 相同）
#
#   bash scripts/run_rq2_paper.sh
#   QUICK=1 bash scripts/run_rq2_paper.sh    # N=10,50,100 仅
#   N_LIST="10,50,100,200" bash scripts/run_rq2_paper.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-run/privacy_paper/rq2}"
if [[ "${QUICK:-0}" == "1" ]]; then
  N_LIST="${N_LIST:-10,50,100}"
  EXTRA=(--skip-large-n)
else
  N_LIST="${N_LIST:-10,50,100,200,400}"
  EXTRA=()
fi
mkdir -p "$OUT/figures"
echo "[RQ2] N_LIST=$N_LIST OUT=$OUT"
python exp_rq/rq2_paper_benchmark.py --n-list "$N_LIST" --out-dir "$OUT" "${EXTRA[@]}" "$@"
python exp_rq/rq2_plot_paper_figures.py --input "$OUT/rq2_paper_data.json" --out-dir "$OUT/figures"
echo "Done: $OUT/figures/figure_rq2_1_time_vs_N_loglog.png"
echo "      $OUT/figures/figure_rq2_2_memory_comm.png"
echo "      $OUT/figures/figure_rq2_3_stacked_latency.png"
