# FinFlow — 面向区制切换市场的动作条件流匹配世界模型

本仓库是论文 **《面向区制切换市场的动作条件流匹配世界模型：从两阶段到联合转移流匹配，及一步 on-policy flow-map 蒸馏》**（[`paper/main_zh.tex`](paper/main_zh.tex)，编译版 [`paper/main_zh.pdf`](paper/main_zh.pdf)）的完整实现与实验产物。

模型把市场视为一个 **动作条件的世界模型**：外部智能体在每一步选择离散市场区制（正常 / 高波动 / 崩盘）来驱动一个三区制马尔可夫切换 Heston 过程。模型学习单步转移核

```
p_θ(log v_{t+1}, r_t | log v_t, r_{t-1}, a_t)
```

并在自由 rollout（无教师强制）下自回归推演一整个交易年（252 步）。主指标为生成路径上欧式期权价格相对 10 万条 MC oracle 的 **定价 RMSE**。

## 方法主线（与论文一致）

| 阶段 | 方法 | 论文章节 | 代码入口 |
|------|------|----------|----------|
| 基线 | Quant-GAN / GARCH(1,1)-t / 移动块自助 | §相关工作、§实验 | `scripts/train_quant_gan.py`, `scripts/sample_quant_gan.py`, `analysis/remote_scripts/baseline_generate.py` |
| 第一步 | 两阶段转移流匹配 + 调度采样 | §4.1.1 | `scripts/train_vol_trans.py`, `scripts/train_ret_trans.py`, `scripts/train_transition_fm.py` |
| 微调（探索） | SIGMA 路径分布微调（严格正常评分） | §4.1.2 | `scripts/pathwise_teacher_combined.py`, `finflow/pathwise_teacher.py` |
| 微调（反例） | 可微定价微调 | §4.1.3 | `scripts/finetune_flow_map_pricing.py` |
| **主方法** | **联合转移流匹配 teacher（joint-FM）** | §4.1.4 | `scripts/train_joint_trans.py`, `finflow/models/transition_fm.py` |
| 一步蒸馏对比 | CD / Mean-Flow / flow-map | §4.2.1 | `scripts/distill_consistency.py`, `scripts/distill_mean_flow.py`, `scripts/distill_flow_map.py` |
| **主蒸馏** | **on-policy teacher-endpoint 修正** | §4.2.2 | `scripts/finetune_flow_map_onpolicy.py` |

## 核心结果（论文表 1）

| 模型 | RMSE ↓ | MAPE ↓ | 峰度 → 4.60 | 部署 |
|------|--------|--------|-------------|------|
| 真实测试集 vs oracle (10k vs 100k) | 0.165 | 0.0112 | 4.603 | 参照 |
| **Joint-FM teacher** (EMA60, NFE120) | **0.094** | **0.0095** | 4.371 | NFE120 |
| Flow-map（朴素一步） | 0.179 | 0.0145 | 5.974 | NFE1 |
| Pricing-aware flow-map | 0.158 | 0.0174 | 3.350 | NFE1 |
| **On-policy flow-map** | **0.101** | 0.0110 | 4.356 | **NFE1** |

完整的统一对比、三类蒸馏、定价微调、on-policy 消融见 `paper/main_zh.pdf` 表 1–5，对应的原始评测 JSON 见 [`release/results/`](release/results/)。

## 目录结构

```
finflow/                 核心包
  data/                  Heston QE 模拟 + 三区制马尔可夫切换 + 期权定价
  models/                TransitionFM（联合/两阶段速度场）+ MeanFlow + Consistency
  distillation/          Mean-Flow / Consistency 蒸馏器
  pathwise_teacher.py    SIGMA 路径分布微调
  inference/             统一采样器 + 自回归 rollout
  eval/                  风格化事实 + 距离 + 定价 + 报告
  baselines/             Quant-GAN（TCN + Lambert-W + WGAN-GP）
  training.py            联合 / 两阶段训练器
scripts/                 CLI 入口（每条命令一个脚本）
tests/                   pytest 测试套件
data/heston_v3/          数据集（splits + metadata + mc_oracle）
analysis/                make_figures.py + 论文配图 + 可视化数据
paper/                   论文源码（main_zh.tex / pdf / references.bib）
presentation/            课程汇报（pptx）
release/                 终版模型权重 + 关键实验结果
```

## 安装

```bash
pip install -r requirements.txt   # numpy, torch, tqdm, matplotlib, pytest
```

## 端到端流程

```bash
# 1) 数据：三区制切换 Heston + MC oracle
python3 scripts/generate_heston_data.py \
  --output data/heston_v3 --n-train 50000 --n-val 5000 --n-test 10000 \
  --steps 252 --regimes --seed 1234
python3 scripts/generate_mc_oracle.py \
  --data-dir data/heston_v3 --output data/heston_v3/mc_oracle.npz --n-paths 100000

# 2) 主方法：训练联合转移流匹配 teacher
python3 scripts/train_joint_trans.py \
  --data-dir data/heston_v3 --output-dir runs/joint_fm \
  --hidden-dim 512 --num-blocks 6 --batch-size 8192 --epochs 60 --lr 2e-4

# 选择最优 EMA 检查点 / NFE（按定价 RMSE 排名）
python3 scripts/select_joint_checkpoint.py \
  --checkpoints "runs/joint_fm/<run>/checkpoints/ema_epoch_*.pt" \
  --data-dir data/heston_v3 --mc-oracle data/heston_v3/mc_oracle.npz \
  --nfe-steps 120 --rank-by pricing_rmse --regime-actions \
  --output runs/joint_fm/selection.json

# 3) 一步蒸馏：flow-map（最优起点），可对比 CD / Mean-Flow
python3 scripts/distill_flow_map.py --stage joint \
  --teacher-checkpoint runs/joint_fm/<run>/checkpoints/ema_epoch_060.pt \
  --data-dir data/heston_v3 --output-dir runs/joint_distill --epochs 15 --batch-size 4096

# 4) 主蒸馏：on-policy teacher-endpoint 修正（最优配置 h128/s30/e1）
python3 scripts/finetune_flow_map_onpolicy.py \
  --data-dir data/heston_v3 --mc-oracle data/heston_v3/mc_oracle.npz \
  --init-checkpoint runs/joint_distill/<run>/checkpoints/best.pt \
  --teacher-checkpoint runs/joint_fm/<run>/checkpoints/ema_epoch_060.pt \
  --output-dir runs/joint_onpolicy \
  --teacher-n-steps 120 --rollout-horizon 128 --path-batch-size 512 \
  --steps-per-epoch 30 --epochs 1 --lr 5e-6 --flowmap-weight 1

# 5) 评测：自由 rollout + 相对 oracle 定价
python3 scripts/rollout_joint.py \
  --checkpoint runs/joint_onpolicy/<run>/checkpoints/best.pt \
  --output runs/rollout_onpolicy.npz --n-paths 10000 --n-steps 252
python3 scripts/evaluate_rollout.py \
  --real data/heston_v3/test.npz --fake runs/rollout_onpolicy.npz \
  --mc-oracle data/heston_v3/mc_oracle.npz --output runs/eval_onpolicy.json \
  --moneynesses 0.85 0.9 0.95 1.0 1.05 --maturities 0.25 0.5 1.0
```

> **raw / cal 两种口径**：`scripts/rollout_calibration.py` 可在生成器输出上施加一次仿射矩校准（与 Quant-GAN 一致），用于报告 `cal` 口径；joint-FM 与 on-policy flow-map 的 raw 口径无需校准即达 0.094 / 0.101（矩校准后进一步小幅降至 cal 0.085 / 0.092，见论文表 5 统一对比）。

## 交付的权重与结果

见 [`release/README.md`](release/README.md)。`release/checkpoints/` 含 13 个一步学生检查点（flow-map / CD / Mean-Flow / on-policy×7 / pricing×3），`release/results/` 含对应的原始评测 JSON。

### 下载与解压权重

> 📦 **下载链接（北大网盘）**：<https://disk.pku.edu.cn/link/AA5FFF5F5BD0AF446AA0BC210401887C2D>

下载到的文件为 `checkpoints.tar.gz`。在仓库根目录解压，即可还原 `release/checkpoints/` 结构：

```bash
# 1) 把下载到的 checkpoints.tar.gz 放到仓库根目录的 release/ 下
# 2) 解压（会生成 release/checkpoints/...）
tar -xzf release/checkpoints.tar.gz -C release/

# 3) 可选：校验完整性（应与 release/checkpoints.tar.gz.sha256 一致）
sha256sum -c release/checkpoints.tar.gz.sha256
```

解压后目录结构与 [`release/README.md`](release/README.md) 的索引表一致，例如最优交付件为
`release/checkpoints/onpolicy_flowmap/nfe120_h128_s30_e1_BEST/best.pt`。

> **注意**：joint-FM teacher 的权重（`ema_epoch_060.pt`）与 Quant-GAN 检查点在当前本地副本中缺失（仅存 `config.json` / `summary.json`），需用上面的训练命令重新生成；其评测结果 JSON 与配置已完整保留在 `release/`。

## 配图复现

```bash
python3 analysis/make_figures.py     # 读取 analysis/viz_data/，输出到 analysis/figures/
```

## 测试

```bash
python3 -m pytest tests/
```

覆盖：Heston QE 形状/正性、区制模拟、MC-oracle 生成、期权定价、两阶段/联合 FM 训练器、Mean-Flow / Consistency / flow-map 蒸馏、采样器、自回归 rollout、5 项风格化事实、Wasserstein 与签名距离、Quant-GAN、on-policy 与 pricing 微调。
