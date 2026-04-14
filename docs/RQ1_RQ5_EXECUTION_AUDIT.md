# RQ1–RQ5 执行链检查说明

## 已修复 / 加固


| RQ            | 问题                                                                                    | 处理                                                                                        |
| ------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **RQ5**       | 同上；另：本次只跑 MNIST 但 `raw/` 残留 CIFAR JSON 时，单次全量作图会误报多数据集                                | `run_rq5_paper.sh` **对每个 `DATASETS` 中的数据集** 均带 `--dataset-filter` 作图，与本次跑法一致且忽略其它数据集的残留文件 |
| **RQ5**       | 聚合后无预算点导致 `Bs[0]` 越界                                                                  | 空 `Bs` 时退出并提示                                                                             |
| **RQ1/RQ3 等** | `rq1_ckpt_resolve` 使用 `torch.load(..., weights_only=False)`，PyTorch<2.0 会 `TypeError` | `_torch_load_meta()` 回退到无 `weights_only` 的 `load`                                         |


## 各 RQ 执行入口（核对用）


| RQ                  | 一键脚本                          | 核心 Python                                                     |
| ------------------- | ----------------------------- | ------------------------------------------------------------- |
| RQ1 主表/图            | `scripts/run_rq1_paper.sh`    | `exp_rq/rq1_paper_table_figures.py`                           |
| RQ1 完整（含 IC、t-test） | `scripts/run_rq1_complete.sh` | `rq1_incentive_compatibility.py` 等                            |
| RQ2                 | `scripts/run_rq2_paper.sh`    | `exp_rq/rq2_paper_benchmark.py` → `rq2_plot_paper_figures.py` |
| RQ3                 | `scripts/run_rq3_complete.sh` | `exp_rq/rq3_paper_complete.py`                                |
| RQ4                 | `scripts/run_rq4_paper.sh`    | `exp_rq/rq4_fl_benchmark.py` → `rq4_plot_paper_figures.py`    |
| RQ5                 | `scripts/run_rq5_paper.sh`    | `exp_rq/rq5_fl_benchmark.py` → `rq5_plot_paper_figures.py`    |


## 使用注意（非 bug，需知）

- **RQ4 / RQ5**：`bash scripts/run_rq4_paper.sh -- --pac` 需 `**--` 再跟参数**；`run_rq5_paper.sh` 同理。
- **RQ4**：`raw/` 混放 `*_pagalg2.json` 与标准 JSON 时，画图默认只用一类；见 `docs/MAIN_PDF_ALGORITHM_MAP.md`。
- **RQ1**：`Net` 使用 `log_softmax` + `CrossEntropyLoss` 为历史写法；与机制评估无直接关系。
- **RQ2**：`DEVICE` 字符串已规范为 `torch.device`，避免 `.type` 报错。
- **Checkpoint**：`N_AGENTS` 须与训练时 `n_agents` 一致，否则 RQ3/RQ4 可能跳过或 shape 错误。

