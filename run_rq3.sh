#!/usr/bin/env bash
# RQ3 一键：图1 收益+福利柱图、图2 随训练轮次、图3 预算扫描
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/scripts/run_rq3_complete.sh" "$@"
