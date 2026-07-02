# FinFlow — 面向区制切换市场的动作条件流匹配世界模型

本仓库对应论文 **《区制切换市场的动作条件流匹配世界模型》**，包含模型实现、训练脚本、评测结果和中文报告：

- 论文源码：[paper/Report.tex](paper/Report.tex)
- PDF版：[paper/Report.pdf](paper/Report.pdf)
- 终版权重与结果索引：[release/README.md](release/README.md)

FinFlow 将市场建模为动作条件世界模型：外部智能体选择正常、高波动、崩盘三类离散区制，模型学习单步转移核

```text
p_theta(log v_{t+1}, r_t | log v_t, r_{t-1}, a_t)
```

并在无教师强制的自由 rollout 下自回归生成一个交易年。主指标为生成路径上的欧式期权价格相对 10 万条 MC oracle 的定价 RMSE；最终报告还补充亚式期权 RMSE 与 17×3 完整价值度曲面检验。

## 关键结论

- 两阶段转移流匹配能明显优于 GARCH-$t$ 与 Quant-GAN，但方差与收益分阶段建模会在长 rollout 中累积条件误差。
- joint-FM 将下一方差与下一收益作为联合变量建模，把未校准定价 RMSE 降至 `0.094`。
- on-policy flow-map 在学生自身 rollout 状态上匹配冻结 teacher 端点，以 NFE1 一步部署在基础 15 点协议下达到 RMSE `0.101`，同时保持更接近真实值的峰度与杠杆相关。
- 在 17×3 完整价值度曲面与亚式期权检验下，最终 checkpoint 为 `onpolicy_flowmap/nfe120_h64_s30_e4`，欧式 RMSE `0.0917`，Asian RMSE `0.0525`。
- 可微定价微调能降低局部欧式价格误差，但收益未同步迁移到亚式 payoff 与路径统计，因此作为价格敏感性消融。

## 核心结果

| 模型 | RMSE ↓ | MAPE ↓ | 峰度 → 4.60 | 部署 |
|------|--------|--------|-------------|------|
| 真实测试集 vs MC 基准 | 0.1646 | 0.0112 | 4.603 | 有限样本参照 |
| **Joint-FM teacher** | **0.094** | **0.0095** | 4.371 | NFE120 |
| 基础 flow-map 一步模型 | 0.179 | 0.0145 | 5.974 | NFE1 |
| Pricing-aware flow-map | 0.158 | 0.0174 | 3.350 | NFE1 |
| **On-policy flow-map** | **0.101** | 0.0110 | 4.356 | **NFE1** |

完整统一对比、三类蒸馏、定价微调和 on-policy 消融见 [paper/Report.pdf](paper/Report.pdf) 表 1--5；正式评测 JSON 与 full-surface 汇总位于 [release/results/](release/results/)。其中 `0.1646` 是基础 15 点欧式协议下真实测试集相对 MC 基准的有限样本参照，不是模型性能的理论下界。完整 Heston `.npz` 数据按运行约定直接放在 `data/` 根目录；由于多个文件超过 GitHub 100MB 单文件限制，git 只跟踪 `data/*.json` 元数据，完整 `.npz` 数据通过 [release/README.md](release/README.md) 中的网盘压缩包分发，也可按下方命令重新生成。本地运行产生的 `runs/` 是临时实验输出目录，不作为最终交付内容。

## 方法入口

| 目的 | 主要入口 |
|------|----------|
| 数据与 oracle | `scripts/generate_heston_data.py`, `scripts/generate_mc_oracle.py` |
| joint-FM teacher | `scripts/train_joint_trans.py`, `scripts/select_joint_checkpoint.py` |
| 一步蒸馏 | `scripts/distill_flow_map.py`, `scripts/distill_consistency.py`, `scripts/distill_mean_flow.py` |
| on-policy 修正 | `scripts/finetune_flow_map_onpolicy.py` |
| 定价微调 | `scripts/finetune_flow_map_pricing.py` |
| rollout 与评测 | `scripts/rollout_joint.py`, `scripts/evaluate_rollout.py`, `scripts/evaluate_all_checkpoints_full_surface.sh`, `scripts/rollout_calibration.py` |
| 配图 | `analysis/make_figures.py` |

## 项目结构

```text
finflow/                 核心包：数据、模型、采样、评测与基线
  data/                  Heston QE 模拟、区制切换、期权定价
  models/                TransitionFM、MeanFlow、Consistency
  distillation/          一步蒸馏器
  inference/             统一采样与自回归 rollout
  eval/                  定价、风格化事实与距离指标
scripts/                 训练、蒸馏、rollout、评测 CLI
data/                    Heston 数据集、划分与 MC oracle
analysis/                可视化数据与论文配图
paper/                   中文报告源码、PDF 与参考文献
release/                 终版配置、结果 JSON 与权重下载说明
tests/                   pytest 测试套件
```

## 运行命令

```bash
pip install -r requirements.txt

# 1) 生成三区制 Heston 数据与 MC oracle
python3 scripts/generate_heston_data.py \
  --output data --n-train 50000 --n-val 5000 --n-test 10000 \
  --steps 252 --regimes --seed 1234
python3 scripts/generate_mc_oracle.py \
  --data-dir data --output data/mc_oracle.npz --n-paths 100000

# 2) 训练 joint-FM teacher
python3 scripts/train_joint_trans.py \
  --data-dir data --output-dir runs/joint_fm \
  --hidden-dim 512 --num-blocks 6 --batch-size 8192 --epochs 60 --lr 2e-4

# 3) 选择 teacher 检查点
python3 scripts/select_joint_checkpoint.py \
  --checkpoints "runs/joint_fm/<run>/checkpoints/ema_epoch_*.pt" \
  --data-dir data --mc-oracle data/mc_oracle.npz \
  --nfe-steps 120 --rank-by pricing_rmse --regime-actions \
  --output runs/joint_fm/selection.json

# 4) 训练基础 flow-map 一步学生
python3 scripts/distill_flow_map.py --stage joint \
  --teacher-checkpoint runs/joint_fm/<run>/checkpoints/ema_epoch_060.pt \
  --data-dir data --output-dir runs/joint_distill \
  --epochs 15 --batch-size 4096

# 5) on-policy teacher-endpoint 修正
python3 scripts/finetune_flow_map_onpolicy.py \
  --data-dir data --mc-oracle data/mc_oracle.npz \
  --init-checkpoint runs/joint_distill/<run>/checkpoints/best.pt \
  --teacher-checkpoint runs/joint_fm/<run>/checkpoints/ema_epoch_060.pt \
  --output-dir runs/joint_onpolicy \
  --teacher-n-steps 120 --rollout-horizon 128 --path-batch-size 512 \
  --steps-per-epoch 30 --epochs 1 --lr 5e-6 --flowmap-weight 1

# 6) 自由 rollout 与评测
python3 scripts/rollout_joint.py \
  --checkpoint runs/joint_onpolicy/<run>/checkpoints/best.pt \
  --output runs/rollout_onpolicy.npz --n-paths 10000 --n-steps 252
python3 scripts/evaluate_rollout.py \
  --real data/test.npz --fake runs/rollout_onpolicy.npz \
  --mc-oracle data/mc_oracle.npz --output runs/eval_onpolicy.json \
  --moneynesses 0.85 0.9 0.95 1.0 1.05 --maturities 0.25 0.5 1.0

# 可选：测试与配图
python3 -m pytest tests/
python3 analysis/make_figures.py
```

## 权重与结果

GitHub 中保留配置和评测 JSON；超过 GitHub 单文件大小限制的 `best.pt` 权重通过网盘分发。下载链接、解压方式和 checkpoint 索引见 [release/README.md](release/README.md)。
