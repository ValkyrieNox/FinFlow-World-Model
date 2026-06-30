# Release — 终版权重与实验结果

本目录汇集论文 [`../paper/main_zh.pdf`](../paper/main_zh.pdf) 报告的终版模型权重与对应的原始（raw，未校准）评测结果。所有 RMSE / MAPE 为生成路径上欧式期权价格相对 10 万条 MC oracle 的误差；峰度目标 4.60。

- `checkpoints/` — 一步学生权重（`best.pt` + `config.json` + `summary.json`）。每个 `best.pt` 约 112MB，**超过 GitHub 100MB/文件上限，故未纳入 git，改用网盘分发**；其 `config.json` / `summary.json` 已随仓库提交。
- `results/`     — 对应评测 JSON（随仓库提交）。`*_metrics.json` 为精简指标，无 `_metrics` 后缀者含完整逐点定价与风格化事实。

### 权重下载与解压

> 📦 **北大网盘**：<https://disk.pku.edu.cn/link/AA5FFF5F5BD0AF446AA0BC210401887C2D>

```bash
tar -xzf release/checkpoints.tar.gz -C release/   # 在仓库根目录解压
sha256sum -c release/checkpoints.tar.gz.sha256    # 可选：校验完整性
```

打包文件 `checkpoints.tar.gz` 解压后即还原下表的 `checkpoints/` 结构。

> teacher（`ema_epoch_060.pt`）与 Quant-GAN 权重在本地副本缺失，仅保留其 `config.json` / `summary.json` 与评测 JSON；用 `scripts/train_joint_trans.py` / `scripts/train_quant_gan.py` 可重新生成。

## checkpoints/ 索引

| 目录 | 方法 | NFE | RMSE | 峰度 | 论文 |
|------|------|-----|------|------|------|
| `joint_fm_teacher/` | 联合转移 FM teacher（权重缺失，仅配置） | 120 | 0.094 | 4.371 | 表 1/2，主方法 |
| `distill_flowmap_naive/` | flow-map 朴素一步蒸馏 | 1 | 0.179 | 5.974 | 表 2，蒸馏起点 |
| `distill_cd/` | Consistency Distillation | 1 | 0.447 | 4.028 | 表 2 |
| `distill_meanflow/` | Mean-Flow | 1 | 2.292 | 4.624 | 表 2 |
| `onpolicy_flowmap/nfe120_h128_s30_e1_BEST/` | **on-policy 端点修正（最优）** | 1 | **0.101** | 4.356 | 表 1，终版交付 |
| `onpolicy_flowmap/nfe120_h64_s30_e1/` | on-policy h64/s30/e1 | 1 | 0.103 | 4.338 | 表 4 |
| `onpolicy_flowmap/nfe120_h64_s15_e1/` | on-policy h64/s15/e1 | 1 | 0.111 | 4.413 | 表 4 |
| `onpolicy_flowmap/nfe120_h64_s30_e2/` | on-policy h64/s30/e2 | 1 | 0.115 | 4.356 | 表 4 |
| `onpolicy_flowmap/nfe120_h64_s30_e4/` | on-policy h64/s30/e4 | 1 | 0.133 | 4.402 | 表 4 |
| `onpolicy_flowmap/nfe120_h252_s30_e1/` | on-policy h252/s30/e1 | 1 | 0.105 | 4.414 | 表 4 |
| `onpolicy_flowmap/nfe30_h64_s60_e4/` | on-policy（弱 teacher NFE30） | 1 | 0.347 | 4.244 | 表 4 |
| `pricing_flowmap/w10_dw02/` | 可微定价微调（最优 RMSE） | 1 | 0.158 | 3.350 | 表 1/3，反例 |
| `pricing_flowmap/w10_dw1/` | 可微定价微调 w10/λ1.0 | 1 | 0.365 | 3.487 | 表 3 |
| `pricing_flowmap/w30_dw01/` | 可微定价微调 w30/λ0.1 | 1 | 0.747 | 3.349 | 表 3 |

## results/ 索引

| 子目录 | 内容 |
|--------|------|
| `teacher/` | joint-FM teacher NFE120 评测（raw + cal）+ EMA/NFE 选择结果 |
| `distill/` | flow-map / CD / Mean-Flow 原始评测（表 2） |
| `onpolicy/` | on-policy 7 个配置的 raw 评测（表 1/4），最优 h128 另含 cal 评测 |
| `pricing/` | 可微定价微调 3 个配置（表 1/3） |
| `baselines/` | GARCH(1,1)-t、移动块自助、Quant-GAN、DDPM 的 raw/cal 评测 |
| `two_stage_and_finetune/` | 两阶段 FM + 调度采样、SIGMA 路径分布微调（表 5） |
