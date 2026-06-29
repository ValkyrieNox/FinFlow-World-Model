# Paper source — FinFlow Heston world model

Top-conference-style write-up of the project. The Chinese report presents
single-stage joint-FM as teacher and one-step on-policy flow-map as the
deployable student, with pricing-aware and two-stage SIGMA/CD results used as
ablations/baselines.

## Files
- `main_zh.tex` — Chinese report, uses `ctexart`; main method is joint-FM +
  one-step on-policy flow-map.
- `main.tex` — English draft with the earlier two-stage method description.
- `references.bib` — bibliography (method sources + baseline sources), shared.

## Compile — English (`main.tex`)
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
Requires a standard TeX Live (packages: amsmath, amssymb, booktabs, multirow,
natbib, hyperref, authblk, caption, xcolor, geometry).

## Compile — Chinese (`main_zh.tex`), must use XeLaTeX
```bash
source /volume/rhxie/texlive/activate.sh
xelatex main_zh
bibtex  main_zh
xelatex main_zh
xelatex main_zh
```
Requires XeLaTeX, the `ctex` package, and a CJK font. A local TeX Live 2026
installation is available at `/volume/rhxie/texlive/2026`; `pdflatex` will NOT
work for the Chinese version.

## Data provenance
The two-stage baseline result tables use evaluation JSONs under
`runs/experiments/p3_full_parallel/eval_*/evaluation/*.json`
(`pricing_fake_vs_mc_oracle.rmse_overall` for raw/cal pricing RMSE,
`stylized_facts_comparison.kurtosis_fake` for kurtosis). The two econometric
baselines (GARCH(1,1)-t, moving-block bootstrap) were generated and evaluated via
`baseline_generate.py` (CPU only) under `eval_baselines_0603/`.

The joint-FM/on-policy main results in `main_zh.tex` use locally synced
2026-06-15 results:
- `training/joint_distill_0615`, `eval_joint_distill_0615`,
  `logs/0615_joint_distill` for the three joint-FM distillation baselines.
- `training/joint_pricing_flowmap_0615`, `eval_joint_pricing_flowmap_0615`,
  `logs/0615_joint_pricing_flowmap` for pricing-aware flow-map.
- `training/joint_onpolicy_flowmap_0615`, `eval_joint_onpolicy_flowmap_0615`,
  `logs/0615_joint_onpolicy_flowmap` for on-policy teacher-endpoint correction.

Pricing floor (real test vs MC oracle) = 0.165; real kurtosis = 4.60. In the
joint-FM experiments, the teacher best raw RMSE is 0.094, the naive one-step
flow-map is 0.179, pricing-aware flow-map is 0.158, and the best one-step
on-policy flow-map is 0.101.
