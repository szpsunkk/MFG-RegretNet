# Privacy as Commodity：复现代码与在线平台

本仓库提供论文《Dependable and Secure Privacy Trading in Federated Learning via MFG-RegretNet》的最新 P2-v2／QV-sel 复现材料，以及可实际多人协作的拍卖演示平台。

- 在线平台：https://privacy-as-commodity.pages.dev/
- 算法比较：https://privacy-as-commodity.pages.dev/results
- 多人拍卖：https://privacy-as-commodity.pages.dev/auction
- 论文与补充材料：https://privacy-as-commodity.pages.dev/paper

最新入口是 [reproducibility/](reproducibility/README.md)，根目录原有实验脚本保留用于历史版本。新目录包含冻结检查点、阈值表、机制与基准代码，以及 880 次运行的原始结果、分配配置和累计隐私账本。哈希校验和分析程序可以重新生成论文表格及主要实验图。

```bash
python -m pip install -r reproducibility/requirements.txt
python reproducibility/reproduce.py verify
python reproducibility/reproduce.py tables
python reproducibility/reproduce.py figures
```

这些命令核验并重算已有结果；重新训练、运行 FL、攻击实验和安全计算的命令见[详细复现说明](reproducibility/README.md)。本次发布已实际完成主机制 12,000 步重训、一次真实 MNIST DP-SGD 轮次，以及全部保存精度与隐私统计的重算，具体证据见[验证记录](reproducibility/VALIDATION.md)。

平台使用 Cloudflare Pages 和 D1，支持 2–10 位用户创建房间、邀请参与、私密报价、神经机制分配、合同接受或拒绝、演示积分结算和收益凭证下载。它展示了论文机制与持续存储、多用户协作、合同执行的工程衔接，源码和独立部署说明见 [platform/](platform/README.md)。

当前平台的收益是演示积分，贡献是合成数据确认；接入真实机构仍需身份与数据量核验、客户端 DP/FL 代理、发布证据、跨轮隐私账本和支付服务。网页执行与论文独立的 SMPC 测量分开说明。准确率来自 MNIST/CIFAR-10，网络入侵检测是展示场景。

结果完整保留胜负：CIFAR-10 的 8 个设置均领先最佳可行外部基准，平均领先 1.701 个百分点；MNIST 的 10 个设置中有 5 个领先，平均差值为 +0.037 个百分点。固定购买隐私的比较与每个客户端累计隐私分布均可查看。
