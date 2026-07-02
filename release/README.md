# Release — 终版权重与实验结果

本目录汇集论文 [`../paper/Report.pdf`](../paper/Report.pdf) 报告的终版模型权重、Heston 数据归档与对应的原始（raw，未校准）评测结果。基础 RMSE / MAPE 为生成路径上欧式期权价格相对 10 万条 MC oracle 的误差；基础 15 点欧式协议下，真实测试集相对同一 MC oracle 的有限样本参照为 RMSE `0.1646`、MAPE `0.0112`。新增的 Asian RMSE 使用同一 oracle 的算术平均亚式看涨 payoff；峰度目标 4.60。

- `checkpoints/` — 一步学生权重（`best.pt` + `config.json` + `summary.json`）。每个 `best.pt` 约 112MB，**超过 GitHub 100MB/文件上限，故未纳入 git，改用网盘分发**；其 `config.json` / `summary.json` 已随仓库提交。
- `../data/`    — Heston 训练、验证、测试、transition 与 MC oracle 数据。GitHub 只跟踪 `metadata.json` / `mc_oracle.json`；完整 `.npz` 数据由 `data.tar.gz` 在同一网盘链接分发。
- `results/`     — 对应评测 JSON（随仓库提交）。`*_metrics.json` 为精简指标，无 `_metrics` 后缀者含完整逐点定价与风格化事实；表 4 的完整价值度曲面汇总位于 `results/onpolicy/summary_full_surface.*`。

### 数据与权重下载

> 📦 **北大网盘**：<https://disk.pku.edu.cn/link/AA5FFF5F5BD0AF446AA0BC210401887C2D>

从网盘下载 `checkpoints.tar.gz` 与 `data.tar.gz`，放到仓库根目录的 `release/` 下，然后运行：

```bash
sha256sum -c release/checkpoints.tar.gz.sha256
sha256sum -c release/data.tar.gz.sha256

tar -xzf release/checkpoints.tar.gz -C release/   # 还原 release/checkpoints/
tar -xzf release/data.tar.gz -C .                 # 还原 data/*.npz 与 data/*.json
```

打包文件 `checkpoints.tar.gz` 解压后即还原下表的 `checkpoints/` 结构；`data.tar.gz` 解压后还原 `data/train.npz`、`data/val.npz`、`data/test.npz`、`data/*_transitions.npz` 与 `data/mc_oracle.npz`。两个压缩包均未纳入 git，只提交对应 `.sha256` 校验文件。

下表中 RMSE 默认沿用基础 15 点欧式协议。表 4 使用 17×3 完整价值度曲面与亚式期权重新选择 checkpoint；该协议下最终 checkpoint 为 `onpolicy_flowmap/nfe120_h64_s30_e4/`，欧式 RMSE 为 `0.0917`，Asian RMSE 为 `0.0525`。

## checkpoints/ 索引

| 目录 | 方法 | NFE | RMSE | 峰度 | 论文 |
|------|------|-----|------|------|------|
| `joint_fm_teacher/` | 联合转移 FM teacher | 120 | 0.094 | 4.371 | 表 1/2，主方法 |
| `distill_flowmap_naive/` | flow-map 朴素一步蒸馏 | 1 | 0.179 | 5.974 | 表 2，蒸馏起点 |
| `distill_cd/` | Consistency Distillation | 1 | 0.447 | 4.028 | 表 2 |
| `distill_meanflow/` | Mean-Flow | 1 | 2.292 | 4.624 | 表 2 |
| `onpolicy_flowmap/nfe120_h128_s30_e1_BEST/` | on-policy 端点修正（基础 15 点最优） | 1 | **0.101** | 4.356 | 表 1 |
| `onpolicy_flowmap/nfe120_h64_s30_e1/` | on-policy h64/s30/e1 | 1 | 0.103 | 4.338 | 表 4 |
| `onpolicy_flowmap/nfe120_h64_s15_e1/` | on-policy h64/s15/e1 | 1 | 0.111 | 4.413 | 表 4 |
| `onpolicy_flowmap/nfe120_h64_s30_e2/` | on-policy h64/s30/e2 | 1 | 0.115 | 4.356 | 表 4 |
| `onpolicy_flowmap/nfe120_h64_s30_e4/` | **on-policy h64/s30/e4（完整曲面最优）** | 1 | 0.133 / **0.0917** | 4.402 | 表 4，终版 checkpoint |
| `onpolicy_flowmap/nfe120_h252_s30_e1/` | on-policy h252/s30/e1 | 1 | 0.105 | 4.414 | 表 4 |
| `onpolicy_flowmap/nfe30_h64_s60_e4/` | on-policy（弱 teacher NFE30） | 1 | 0.347 | 4.244 | 表 4 |
| `pricing_flowmap/w10_dw02/` | 可微定价微调（最优 RMSE） | 1 | 0.158 | 3.350 | 表 1/3，价格敏感性消融 |
| `pricing_flowmap/w10_dw1/` | 可微定价微调 w10/λ1.0 | 1 | 0.365 | 3.487 | 表 3 |
| `pricing_flowmap/w30_dw01/` | 可微定价微调 w30/λ0.1 | 1 | 0.747 | 3.349 | 表 3 |

## results/ 索引

| 子目录 | 内容 |
|--------|------|
| `teacher/` | joint-FM teacher NFE120 评测（raw + cal）+ EMA/NFE 选择结果 |
| `distill/` | flow-map / CD / Mean-Flow 原始评测（表 2）；`*_metrics.json` 已补充 Asian RMSE 块 |
| `onpolicy/` | on-policy 7 个配置的 raw 评测（表 1/4），最优 h128 另含 cal 评测；`summary_full_surface.*` 为表 4 的 17×3 汇总 |
| `pricing/` | 可微定价微调 3 个配置（表 1/3）；`*_metrics.json` 已补充 Asian RMSE 块 |
| `baselines/` | GARCH(1,1)-t、移动块自助、Quant-GAN、DDPM 的 raw/cal 评测 |
| `two_stage_and_finetune/` | 两阶段 FM + 调度采样、SIGMA 路径分布微调及其 CD 蒸馏（表 5） |
