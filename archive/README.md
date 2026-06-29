# archive/ — 归档（本地保留，已 gitignore）

存放被终版取代或与论文无直接关系的材料，仅供追溯，不属于交付物。

| 目录 | 内容 |
|------|------|
| `runs/` | 旧 `runs/` 轻量索引 README，记录 924 GPU 机镜像来源、P2/P3 run 目录和关键历史结果速查。 |
| `runs_full/` | 完整旧 `runs/`（4GB+）：smoke 冒烟、p2 实验、两阶段时代训练与评测、所有中间 rollout。终版权重已抽取到 `release/checkpoints/`。 |
| `analysis_old/eval_json_backup/` | 两阶段时代（eval_A…G 等）约 200 个旧评测 JSON。 |
| `analysis_old/p3_full_parallel_data/` | 实验数据集副本（`mc_oracle` 已并入 `data/heston_v3/`）。 |
| `server_artifacts/` | 服务器端编排脚本、伙伴节点产物、日志。 |
| `idea/` | 选题、文献、方法设计等过程草稿。 |
| `proj_requirements/` | 课程考核与提交要求文档。 |
| `scripts/` | 服务器编排 `.sh` 脚本与 `.bak` 备份（非论文流程）。 |
| `paper_aux/` | 英文旧版 `main_en.tex`、`main_zh_v3.pdf`、LaTeX 构建中间产物。 |
| `docs/` | 旧版 `README_v3_old.md`、`REPO_MAP.md`、`REPRODUCE.md`、`RESULTS_SUMMARY.md` 等。 |
| `misc/` | `logs/`、`.pytest_cache/`、`github.token`（密钥，勿提交）、杂项。 |
