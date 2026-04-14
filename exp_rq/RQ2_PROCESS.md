# RQ2：可扩展性（墙钟时间 / 内存 / 通信）

## 一键

```bash
bash scripts/run_rq2_paper.sh
QUICK=1 bash scripts/run_rq2_paper.sh
N_LIST="10,50,100,200,500,1000" bash scripts/run_rq2_paper.sh   # 大 N 时 CSRA 很慢，可去掉 1000
```

## 输出图

| 图 | 文件 | 内容 |
|----|------|------|
| **1** | `figure_rq2_1_time_vs_N_loglog.png` | 单次拍卖墙钟时间 vs **N**（log–log），Ours / RegretNet（若有 ckpt）/ CSRA 等 + **O(N)、O(N²) 参考线** |
| **2** | `figure_rq2_2_memory_comm.png` | (a) 峰值 GPU 显存：CSRA vs Ours；(b) 每轮通信量粗估（float32 bid+下发） |
| **3** | `figure_rq2_3_stacked_latency.png` | 一轮 FL：**本地训练 proxy** \| **服务器梯度聚合** \| **拍卖+机制侧聚合** |

## 数据

- `rq2_paper_data.json`：含 `rq2_time_rows`、`per_n_detail`（各 N 下分段时间与显存）。
- **计时**：每个 **batch=1 个 profile**（对应一轮 FL 一次拍卖），与论文「单次拍卖」一致。

## Ours / RegretNet 多 N 曲线

神经网 **输入维随 N 变**，需 **每个 N 一份 checkpoint**。脚本会扫描 `result/mfg_regretnet_privacy_*_checkpoint.pt`、`result/regretnet_privacy_*_checkpoint.pt` 的 `arch.n_agents`，仅对 **匹配的 N** 画 Ours/RegretNet。若只有 **N=10** 权重，则 Ours 仅一个点；要满曲线需训练 `n_agents ∈ {50,100,…}` 的 MFG。

## 与旧 Phase4 RQ2 的关系

- `run_phase4_eval.py --n-list` 仍写 `phase4_summary.json` 的 `rq2`；本管线 **独立 JSON + 三张论文图**，指标定义一致时可对照。
