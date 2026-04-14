# main.pdf ↔ 代码对应（简要）

## Algorithm 2（PAG-FL 部署）

| 论文步骤 | 实现 |
|----------|------|
| 本地训练得到 \(w_i\) | `FL.pag_fl_alg2_round`：多 epoch 小批量 SGD（`local_epochs` / `local_batch_size`），再对权重加噪 |
| \(\varepsilon_{\mathrm{out},i}\) 来自分配 | `plosses[i]`（拍卖后 `allocs_to_plosses`，与旧 FL 一致） |
| \(\sigma_i \propto \Delta_f / \max(\varepsilon_{\mathrm{out},i}, \varepsilon_{\min})\) | \(\sigma_i = \sqrt{2\ln(1.25/\delta)}\cdot L / \max(\varepsilon_i,\varepsilon_{\min})\)，\(L=\texttt{args.L}\) |
| \( \tilde w_i = w_i + \mathcal{N}(0,\sigma_i^2 I)\) | 对 `state_dict` 逐张量加高斯噪声 |
| \(w_G = \sum_i \alpha_i \tilde w_i,\ \alpha_i \propto \varepsilon_{\mathrm{out},i}\) | \(\alpha_i = \varepsilon_i / \sum_j \varepsilon_j\)（仅参与者） |

**启用方式：** RQ4 / RQ5 加 `--pag-fl-alg2`；可选 `--delta-dp`、`--eps-min-alg2`。  
原始 JSON 会带后缀 `_pagalg2`，避免与 Laplace+ConvlAggr 跑法互相覆盖。

**实现注意：** \(\Delta_f\) 取 `args.L`（与梯度裁剪一致）；严格意义上「一步 SGD 后整模型权重」的 \(\ell_2\) 敏感度上界约为 \(\eta L\) 量级，若要与某 DP 定义完全对齐可再缩放 \(\sigma\)（当前与正文「\(\sigma \propto \Delta_f/\varepsilon\)」形式一致）。  
\(\delta\) 会钳到 \((0,1.24]\)，防止 \(\ln(1.25/\delta)<0\) 导致 NaN。非浮点 `state_dict` 项（如整型 buffer）不加噪声；聚合时浮点做 ε 加权平均，整型 buffer取首个客户端副本（避免 `α·Long→float`）。

用于 \(\alpha_i\) 与 \(\sigma_i\) 的有效 \(\varepsilon\) 为 \(\max(\varepsilon_{\mathrm{out}}, \max(\varepsilon_{\min},10^{-8}))\)，避免极小 \(\varepsilon\) 导致权重噪声数值爆炸。

**画图：** 同一 `raw/` 下若同时存在普通 JSON 与 `*_pagalg2.json`，默认只读一类（混合时默认标准，Alg.2 加 **`--prefer-pagalg2`**）。RQ4 会跳过无 `methods`/`meta` 的 JSON（避免误读聚合文件）。**`--prefer-pagalg2`** 时写出 `rq4_aggregated_pagalg2.json` / `rq5_summary_pagalg2.json`，避免覆盖标准汇总；图仍写入 `--out-dir`（Alg.2 建议另设输出目录以免覆盖 PNG）。

**默认（与旧实验一致）：** Laplace 梯度噪声 + ConvlAggr 权重 → `ldp_fed_sgd` / `_fed_round`。

## MFG-RegretNet 与式 (51)

Misreport 平铺时，一行报告里仅偏离客户端换为 \(b'_i\)，其余为真报告；`forward` 里对 agents 维求均值得到的 \(b_{\mathrm{MFG}}\) 等于  
\(b_{\mathrm{MFG}} + (b'_i - b_i)/N\)，与论文 MFG-aware 后悔定义一致。

## 训练侧（LMFG / HJB / ζ）

当前 `train_mfg_regretnet.py` 仍以 `lambda_revenue_util`、`lambda_participant_welfare` 等为主；正文式 (56)–(61) 的 Monte Carlo 平均场支付对齐、Huber 等可作为后续扩展项。
