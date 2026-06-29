# runs/ — 全部实验结果索引

来源:924 GPU 机 `/root/autodl-tmp/Heston-Model/runs/`(2026-06-11 逐字节校验镜像)+ pathwise worktree 的 `smoke_pathwise/`。共约 17G、1644 文件。本目录整体被 .gitignore 排除。

训练日志在 `../logs/`(p3_full_fast / p3_full_fast_seed31 / p3_full_parallel / p3_tuning_20260530)。

## 顶层结构

```
runs/
├── experiments/        全部正式实验(见下)
└── smoke_pathwise/     pathwise teacher 吞吐量冒烟实验(9 组:bs128–768、compile/eager、retonly 变体)
```

## experiments/ — P2 阶段(中等规模验证,2026-05-22 ~ 05-31)

| 目录 | 大小 | 内容 |
|---|---|---|
| `p2_teacher_select_20260522T100554Z` | 286M | teacher checkpoint 选型(配套 `p2_teacher_select_*.nohup.log`) |
| `p2_new_teacher_distill_20260524T144921Z` | 532M | 新 teacher 蒸馏实验(配套 nohup.log;结论摘要见仓库根 `new_teacher_distill_summary.md`) |
| `p2_medium` / `p2_medium_complete` | 632M / 996M | P2 中等规模完整管线(data/evaluation/metadata/oracle/rollouts/training);对比总结在 `summary/medium_model_comparison_summary.md` |
| `p2_ss00` `p2_ss01` `p2_ss02` | 872M ×3 | P2 三个平行 run(ss00–02) |

## experiments/ — P3 阶段(全规模,2026-05-30 ~ 06-03)

| 目录 | 大小 | 内容 |
|---|---|---|
| `p3_full` | 745M | 全规模训练首跑(data/metadata/training) |
| `p3_full_fast` / `p3_full_fast_seed31` | 801M / 745M | fast 配置全跑 + seed31 复跑 |
| **`p3_full_parallel`** | **8.5G** | **主结果根(论文所有数字来源),见下节** |
| `qgan_checkpoint_eval_logs` | 324K | QGAN checkpoint 消融评测日志 |
| `summary` | — | medium 模型对比总结 md |

## p3_full_parallel/ — 主结果根

- **根级文档**:`RESULTS_SUMMARY.md`(总表,+05-03 同步前 .bak)、`RESULTS_403_0603.md`、`RESULTS_FINAL_SPRINT_0603.md`、`selection_ret.json`
- **`data/`(1G)**:三状态 Markov Heston 数据集(seed 20260530):train/val/test(+transitions)、mc_oracle(此数据另有副本在 `../../data/` 与 `project/analysis/p3_full_parallel_data/`)
- **`training/`(4.9G,29 组 checkpoint)**:
  - 第一阶段 FM teacher:`vol_fm`、`ret_fm`
  - LWFM 扫描:`vol_lwfm_d0.03/0.05/0.08/0.12`、`mf_vol_lwfm`
  - 蒸馏:`cd_ret`/`cd_vol`/`mf_ret`/`mf_vol`、`distill_combined_cd`/`distill_combined_mf`
  - 路径损失:`pathwise_teacher`、`combined_0602`
  - 冲刺 A–G:`A_cd_sc`、`A_flowmap_fmlag`/`A_flowmap_sclag`(Lagrangian flow-map)、`B_0602`(sched-sampling teacher 重训)、`b_partner_recipe_pathwise/teacher_0602`、`C_0602`(combined 微调,含 C3_more)、`D_cd_candidates_0603`、`E_flowmap_audit_0603`、`F_teacher_push_0603`、`G_final_sprint_0603`
  - 基线:`quant_gan`、`quant_gan_paper`
  - `ablation_0601`(SIGMA 组件消融)
- **26 个 `eval_*` 目录**:每个含 `evaluation/*.json`(raw 与 cal 报告)。对应关系:`eval_compare_0601`/`eval_ablation_0601`(消融)、`eval_A/B/C/D/E/F/G_*`(对应冲刺)、`eval_baselines_0603`(GARCH-t、bootstrap)、`eval_ddpm_baseline_0603`、`eval_qgan_audit_0601`、`eval_champion`/`eval_calibrated`(最终候选)、`eval_lwfm`/`eval_mf_lwfm`/`eval_pathwise`/`eval_pathwise_guarded`/`eval_b_strongteacher`/`eval_b_partner_recipe_0602`/`eval_distill_compare_0602`/`eval_combined_0602`/`eval_e1_sigmmd`
- 其他:`mf_select`(MeanFlow 选型)、`viz`、`logs`、`metadata`

## 关键结果速查(raw / cal / kurt;评测协议见 REPRODUCE.md,定价下限 0.165)

| 模型 | raw | cal | kurt | 备注 |
|---|---|---|---|---|
| FM teacher | **0.475** | 0.583 | 3.88 | 最佳 raw,自洽 |
| combined(teacher+路径损失) | — | **0.232** | — | 本项目方法,strong teacher 上 |
| combined-CD 蒸馏 | — | **0.170** | 3.11 | 最佳 cal,≈定价下限 |
| FM-CD | — | 0.537 | **4.48** | 最佳尾部形状 |
| B1(FM+sched0.8) | 0.872 | 0.348 | 4.41 | 最佳平衡 |
| C2_rawfix | 2.595 | 0.180 | — | cal 好但 raw 未修复 |
| fmlag(Lagrangian 蒸馏) | k2=1.82 | k1=0.313 | — | NFE 拐点随指标不同 |

系统性发现:raw ↔ cal/kurt 存在折中,无单一全胜模型。
