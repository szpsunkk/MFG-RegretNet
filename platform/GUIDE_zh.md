# Privacy as Commodity 使用说明

正式地址：https://privacy-as-commodity.pages.dev/

## 展示研究成果

- 算法对比：https://privacy-as-commodity.pages.dev/results
  可以切换 MNIST / CIFAR-10、固定预算 / 相同购买隐私、数据异质性和预算档位。
  展示全部 18 个确认设置、可达的基准算法、三种子均值和标准差，可下载 CSV。
- 方法介绍：https://privacy-as-commodity.pages.dev/method
  按注册、人口特征、分配、支付、结算五步说明实际模型；各基准有独立介绍页。
- 论文与补充材料：https://privacy-as-commodity.pages.dev/paper
  来源是 `main_final2/p2v2_version/`。数据来源及文件哈希可公开下载。

本次论文附件固定为 2026-10-25 11:59 JST 同步的完整 PDF。目录之后的修改
不会自动上线；需要重新导出并部署，具体文件哈希记录在 `DEPLOYMENT.md`。

## 两台设备演示一次拍卖

1. 主持人在 https://privacy-as-commodity.pages.dev/auction 创建房间，选择预算和 κ。
2. 点击 **Copy invite link**，由其他设备或独立浏览器打开。主持人也可以注册成为参与者。
3. 至少两位参与者填写机构名称和示例记录数（10–500）。每个浏览器保留自己的私人访问密钥。
4. 主持人逐一点击 **Verify count**，再点击 **Open sealed bidding**。
5. 每人提交单位隐私成本（0–1）和容量（0.1–5）；结算前可修改自己的报价。
6. 全员提交后，主持人点击 **Clear auction & issue contracts**。
7. 每人查看隐私分配、补偿、数据红利和按自己报价计算的净效用，选择接受或拒绝。
8. 接受合同后记录演示积分，可下载收益凭证；拒绝时支付为零，不重新分配。

房间最多支持 10 位参与者，每五秒同步，保存七天。相同浏览器可返回原房间；
点击 **Save private access key** 可保存私人访问文件，在另一设备的拍卖首页
选择 **Restore your saved private access** 恢复身份。邀请链接可以分享，私人密钥请自行保管。

## 部署与适用范围

平台已使用 Cloudflare Pages、服务端计算和 D1 持久化存储，访问不依赖本机、
Tailscale 或 101 服务器在线。本次未开通付费套餐，使用受 Cloudflare 账号免费配额约束。

当前是可实际多人协作的研究演示：奖励是演示积分，接受合同表示确认一份合成贡献。
没有真实转账、私有数据上传或在线 FL 训练，也不将网页计算说成 SMPC。
页面内报价只向本人显示；Cloudflare 服务端处理明文报价。

接入真实机构时，需要补充身份与数据量验证、客户端 DP/FL 代理、累计隐私账本、
发布验证和支付服务，再将结算条件从演示确认改为可验证的真实发布记录。
模型沿用已冻结检查点；网页交互本身不构成新的精度或纳什均衡实验。

源码及部署命令见 `README.md`，版本记录见 `DEPLOYMENT.md`。
