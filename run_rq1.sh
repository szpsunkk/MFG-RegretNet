#!/usr/bin/env bash
# RQ1 一键：基线(PAC/VCG/CSRA/MFG-Pricing) + RegretNet/DM/MFG-RegretNet + 全部表与图
# 缺权重时可自动训练 RegretNet/DM（见脚本内说明）。用法：在项目根目录执行
#   ./run_rq1.sh
# 或
#   bash run_rq1.sh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/scripts/run_rq1_complete.sh" "$@"
