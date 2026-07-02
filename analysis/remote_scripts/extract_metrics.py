import json, os
P = os.environ.get("FINFLOW_EXPERIMENT_ROOT", "runs/experiments/p3_full_parallel")
E = P + "/evaluation"  # unused

def j(path):
    return json.load(open(P + "/" + path))

# label -> (raw_json, cal_json_or_None)
M = {
 "real_test":        ("eval_b_strongteacher/evaluation/eval_strongteacher_raw.json", None),  # use real_* fields
 "block_bootstrap":  ("eval_baselines_0603/evaluation/eval_block_bootstrap_raw.json","eval_baselines_0603/evaluation/eval_block_bootstrap_cal.json"),
 "garch_t":          ("eval_baselines_0603/evaluation/eval_garch_t_raw.json","eval_baselines_0603/evaluation/eval_garch_t_cal.json"),
 "ddpm":             ("eval_ddpm_baseline_0603/evaluation/eval_ddpm_raw.json","eval_ddpm_baseline_0603/evaluation/eval_ddpm_cal.json"),
 "qgan_paper_best":  ("eval_compare_0601/evaluation/eval_qgan_paper_best_raw.json","eval_compare_0601/evaluation/eval_qgan_paper_best_cal.json"),
 "qgan_paper_last":  ("eval_compare_0601/evaluation/eval_qgan_paper_last_raw.json","eval_compare_0601/evaluation/eval_qgan_paper_last_cal.json"),
 "FM_teacher":       ("eval_champion/evaluation/eval_fm.json","eval_calibrated/evaluation/eval_fm.json"),
 "FM_SS_B1":         ("eval_B_0602/evaluation/eval_B1_sched08_raw.json","eval_B_0602/evaluation/eval_B1_sched08_cal.json"),
 "LWFM_d05":         ("eval_lwfm/evaluation/eval_d0.05.json", None),
 "bigbatch_teacher": ("eval_b_strongteacher/evaluation/eval_strongteacher_raw.json","eval_b_strongteacher/evaluation/eval_strongteacher_cal.json"),
 "pathloss_base":    ("eval_b_partner_recipe_0602/evaluation/eval_b_partner_recipe_pathwise_combined_bs512_10ep_0602_last_raw.json","eval_b_partner_recipe_0602/evaluation/eval_b_partner_recipe_pathwise_combined_bs512_10ep_0602_last_cal.json"),
 "pathloss_heavy_C2":("eval_C_0602/evaluation/eval_C2_rawfix_raw.json","eval_C_0602/evaluation/eval_C2_rawfix_cal.json"),
 "pathloss_long_C3": ("eval_C_0602/evaluation/eval_C3_more_raw.json","eval_C_0602/evaluation/eval_C3_more_cal.json"),
 "CD_from_B1":       ("eval_D_cd_candidates_0603/evaluation/eval_b1_cd_raw.json","eval_D_cd_candidates_0603/evaluation/eval_b1_cd_cal.json"),
 "CD_from_pathloss": ("eval_distill_compare_0602/evaluation/eval_combined_cd_raw.json","eval_distill_compare_0602/evaluation/eval_combined_cd_cal.json"),
 "CD_pathloss_lowlr":("eval_G_final_sprint_0603/evaluation/eval_c3_cd_lowlr_raw.json","eval_G_final_sprint_0603/evaluation/eval_c3_cd_lowlr_cal.json"),
 # ablation (all on LWFM teacher)
 "abl_moment":       ("eval_ablation_0601/evaluation/eval_ctl_momentonly_raw.json","eval_ablation_0601/evaluation/eval_ctl_momentonly_cal.json"),
 "abl_sigmmd":       ("eval_ablation_0601/evaluation/eval_e1_sigmmd_raw.json","eval_ablation_0601/evaluation/eval_e1_sigmmd_cal.json"),
 "abl_sigw1":        ("eval_ablation_0601/evaluation/eval_e2_sigw1_raw.json","eval_ablation_0601/evaluation/eval_e2_sigw1_cal.json"),
 "abl_energy":       ("eval_ablation_0601/evaluation/eval_e3_energy_raw.json","eval_ablation_0601/evaluation/eval_e3_energy_cal.json"),
 "abl_wgangp":       ("eval_pathwise/evaluation/eval_pathwise_retonly_lwfm_d0.05_bs512_fm4_s2400_opt_0601_raw.json","eval_pathwise/evaluation/eval_pathwise_retonly_lwfm_d0.05_bs512_fm4_s2400_opt_0601_calibrated.json"),
 "abl_combined":     ("eval_combined_0602/evaluation/eval_combined_moment_sigmmd_energy_sigw1_bs512_10ep_0602_raw.json","eval_combined_0602/evaluation/eval_combined_moment_sigmmd_energy_sigw1_bs512_10ep_0602_cal.json"),
 # distill failures
 "FM_MF":            ("eval_champion/evaluation/eval_mf.json","eval_calibrated/evaluation/eval_mf.json"),
 "pathloss_MF":      ("eval_distill_compare_0602/evaluation/eval_combined_mf_raw.json","eval_distill_compare_0602/evaluation/eval_combined_mf_cal.json"),
}

def get(d, *ks, default=None):
    for k in ks:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return default
    return d

rows = {}
for lab, (rawp, calp) in M.items():
    try:
        dr = j(rawp)
    except Exception as e:
        rows[lab] = {"ERR": str(e)}; continue
    sc = dr.get("stylized_facts_comparison", {})
    di = dr.get("distances", {})
    raw_rmse = get(dr, "pricing_fake_vs_mc_oracle", "rmse_overall")
    cal_rmse = None
    if calp:
        try: cal_rmse = get(j(calp), "pricing_fake_vs_mc_oracle", "rmse_overall")
        except Exception: cal_rmse = None
    row = {
        "raw_rmse": raw_rmse,
        "cal_rmse": cal_rmse,
        "kurt": sc.get("kurtosis_fake"),
        "absACF_l1": sc.get("absolute_return_acf_l1"),
        "retACF_l1": sc.get("return_acf_l1"),
        "lev_l1": sc.get("leverage_correlation_l1"),
        "tail_idx": sc.get("tail_index_fake"),
        "sigW_mean": get(di, "signature_wasserstein", "mean"),
        "totretW": di.get("total_return_wasserstein"),
        "margW_mean": di.get("marginal_wasserstein_mean"),
    }
    if lab == "real_test":
        rf = dr.get("real_facts", {})
        row = {"raw_rmse": get(dr,"pricing_real_vs_mc_oracle","rmse_overall"), "cal_rmse": None,
               "kurt": rf.get("kurtosis"), "absACF_l1": 0.0, "retACF_l1": 0.0, "lev_l1": 0.0,
               "tail_idx": rf.get("tail_index"), "sigW_mean": 0.0, "totretW": 0.0, "margW_mean": 0.0}
    rows[lab] = row

hdr = ["raw_rmse","cal_rmse","kurt","absACF_l1","retACF_l1","lev_l1","tail_idx","sigW_mean","totretW","margW_mean"]
print("%-20s | "%"model" + " ".join("%10s"%h for h in hdr))
for lab in M:
    r = rows[lab]
    if "ERR" in r:
        print("%-20s | ERR %s"%(lab, r["ERR"])); continue
    def f(x): return "    -     " if x is None else "%10.5f"%x
    print("%-20s | "%lab + " ".join(f(r[h]) for h in hdr))
print("JSON_DUMP_START")
print(json.dumps(rows))
print("JSON_DUMP_END")
