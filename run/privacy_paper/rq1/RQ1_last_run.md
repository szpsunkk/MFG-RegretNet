# RQ1 本次运行记录

- **时间**: 2026-03-21T14:37:24+08:00
- **目录**: `run/privacy_paper/rq1`
- **Seeds**: 42,43,44,45,46
- **Profiles/seed**: 1000
- **N, B**: 10, 50
- **MFG_CKPT**: result/mfg_regretnet_privacy_200_checkpoint.pt
- **REGRETNET_CKPT**: result/regretnet_privacy_200_checkpoint.pt
- **DM_REGRETNET_CKPT**: result/dm_regretnet_privacy_200_checkpoint.pt

## 已执行阶段

### 阶段 A（激励相容 + PGA 收敛）
```bash
python3 exp_rq/rq1_incentive_compatibility.py --out-dir "run/privacy_paper/rq1" --seeds "42,43,44,45,46" \
  --num-profiles 1000 --n-agents 10 --budget 50 \
  --convergence-curve [+ regretnet/mfg ckpt]
```

### 阶段 B（论文表 + 图 A/B）
```bash
python3 exp_rq/rq1_paper_table_figures.py --out-dir "run/privacy_paper/rq1" --seeds "42,43,44,45,46" \
  --num-profiles 1000 [+ ckpt 与可选 --ir-log-scale]
```

### 可选 图 C / 图 D（RQ1_FIG_CD=1）
- figure_rq1_paper_regret_vs_epoch.png, rq1_figure_c.json
- figure_rq1_paper_regret_distribution.png, rq1_figure_d.json

## 产出清单
-rw-rw-r--  1 skk skk  59886  3月 21 14:41 figure_rq1_paper_ir.png
-rw-rw-r--  1 skk skk  74810  3月 21 14:42 figure_rq1_paper_regret_distribution.png
-rw-rw-r--  1 skk skk  92055  3月 21 14:41 figure_rq1_paper_regret.png
-rw-rw-r--  1 skk skk 118931  3月 21 14:42 figure_rq1_paper_regret_vs_epoch.png
-rw-rw-r--  1 skk skk  53961  3月 21 14:39 figure_rq1_regret_bar.png
-rw-rw-r--  1 skk skk  73691  3月 21 14:39 figure_rq1_regret_vs_pga_rounds.png
-rw-rw-r--  1 skk skk    660  3月 21 14:39 rq1_convergence_curve.json
-rw-rw-r--  1 skk skk   4522  3月 21 14:42 rq1_figure_c.json
-rw-rw-r--  1 skk skk   1305  3月 21 14:42 rq1_figure_d.json
-rw-rw-r--  1 skk skk   1046  3月 21 14:42 RQ1_last_run.md
-rw-rw-r--  1 skk skk  13896  3月 21 14:41 rq1_paper.json
-rw-rw-r--  1 skk skk   4702  3月 21 14:39 rq1_statistics.json
-rw-rw-r--  1 skk skk    899  3月 21 14:39 table_rq1.csv
-rw-rw-r--  1 skk skk    991  3月 21 14:39 table_rq1.md
-rw-rw-r--  1 skk skk    561  3月 21 14:41 table_rq1_paper.csv
-rw-rw-r--  1 skk skk   1387  3月 21 14:41 table_rq1_paper.md
-rw-rw-r--  1 skk skk    609  3月 21 14:42 table_rq1_regret_distribution.csv
-rw-rw-r--  1 skk skk    585  3月 21 14:42 table_rq1_regret_distribution.md

## 完整实验过程说明
见仓库 `exp_rq/RQ1_EXPERIMENT_PROCESS.md`

## 常见错误
- 勿把 `result/....pt`、`../..` 等占位符当作真实参数执行（会报 unrecognized arguments）。
- 阶段 A 若显示 `No valid neural checkpoints`：请在项目根目录运行，并保证 `result/mfg_regretnet_privacy_*_checkpoint.pt` 存在或设置正确的 `MFG_CKPT`。
