#!/usr/bin/env python3
"""Generate presentation figures for the FinFlow Heston world-model project.
Run with the local anaconda python. Robust to missing rollout files (skips).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)


def first_existing_dir(*paths):
    for path in paths:
        if os.path.isdir(path):
            return path
    return paths[-1]


DATA = first_existing_dir(os.path.join(ROOT, "p3_full_parallel_data"),
                          os.path.join(PROJECT_ROOT, "data"))
VIZ = os.path.join(ROOT, "viz_data")
JSON = os.path.join(ROOT, "eval_json_backup", "json")
OUT = os.environ.get("FINFLOW_FIGURE_OUT", os.path.join(ROOT, "figures"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 12,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": False})

REGIME_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}
REGIME_NAMES = {0: "Normal", 1: "High-vol", 2: "Crash"}

meta_path = os.path.join(DATA, "metadata.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}


def load_paths(path):
    """Return (s_paths [N,T+1], returns [N,T], actions or None)."""
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
    except Exception as e:
        print(f"[skip] unreadable rollout file {path}: {e}")
        return None
    s = np.asarray(d["s_paths"], float)
    if "log_returns" in d.files:
        r = np.asarray(d["log_returns"], float)
    elif "r_paths" in d.files:
        r = np.asarray(d["r_paths"], float)
    else:
        r = np.diff(np.log(s), axis=1)
    a = np.asarray(d["actions"]) if "actions" in d.files else None
    v = np.asarray(d["v_paths"], float) if "v_paths" in d.files else None
    return dict(s=s, r=r, a=a, v=v)


def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", name)


# ------------------------------------------------------------- metrics (hardcoded, verified)
MET = {
 # name: (raw, cal, kurt, absACF, lev, tail, sigW)
 "FM teacher":        (0.475, 0.583, 3.88, 0.0156, 0.0052, 4.78, 0.01196),
 "FM+SS":             (0.872, 0.348, 4.41, 0.0111, 0.0063, 4.39, 0.01038),
 "Big-batch teacher": (1.057, 0.582, 4.08, 0.0148, 0.0103, 4.61, 0.00993),
 "LWFM":              (0.882, None,  4.33, 0.0059, 0.0033, 4.47, 0.00996),
 "Path-loss (base)":  (2.595, 0.232, 3.66, 0.0307, 0.0143, 5.45, 0.04626),
 "Path-loss (best-cal)":(2.595,0.180, 3.63, 0.0308, 0.0143, 5.49, 0.04483),
 "Path-loss (long)":  (1.483, 0.283, 3.67, 0.0301, 0.0136, 5.40, 0.04884),
 "CD<-FM+SS":         (1.282, 0.371, 4.27, 0.0018, 0.0037, 4.47, 0.00836),
 "CD<-Path-loss":     (1.875, 0.170, 3.11, 0.0285, 0.0078, 6.99, 0.05757),
 "CD<-Path-loss(lowlr)":(2.328,0.118, 2.85, 0.0450, 0.0110, 8.23, 0.05741),
 "GARCH-t":           (1.089, 0.482, 19.17,0.0104, 0.0291, 2.82, 0.00850),
 "DDPM":              (2.855, 0.576, 3.75, 0.0223, 0.0085, 4.85, 0.01659),
 "Quant-GAN (best)":  (5.650, 1.607, 10.21,0.0418, 0.0142, 3.29, 0.02138),
 "Quant-GAN (last)":  (5.429, 0.674, 6.39, 0.0050, 0.0035, 4.03, 0.02033),
 "Block-bootstrap":   (0.269, 0.244, 4.58, 0.0354, 0.0040, 4.30, 0.00588),
}
REAL_KURT, REAL_TAIL, FINITE_SAMPLE_RMSE = 4.60, 4.28, 0.1646
NFE_K = [1, 2, 4, 8]
NFE_RAW = [2.736, 1.819, 3.090, 3.437]
NFE_CAL = [0.313, 2.355, 3.745, 3.857]


def fan(ax, s, color, label, qs=(5, 25, 50, 75, 95)):
    T = s.shape[1]
    x = np.arange(T)
    p = np.percentile(s, qs, axis=0)
    ax.fill_between(x, p[0], p[4], color=color, alpha=0.12, lw=0)
    ax.fill_between(x, p[1], p[3], color=color, alpha=0.25, lw=0)
    ax.plot(x, p[2], color=color, lw=2, label=label)


# ============================================================ FIG 1: controllability
def fig_controllability():
    reg = {k: load_paths(os.path.join(VIZ, f"regime{k}.npz")) for k in (0, 1, 2)}
    if any(v is None for v in reg.values()):
        print("[skip] controllability (missing regime npz)"); return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for k, ax in zip((0, 1, 2), axes):
        s = reg[k]["s"]
        fan(ax, s, REGIME_COLORS[k], "median")
        for i in range(8):
            ax.plot(s[i], color=REGIME_COLORS[k], alpha=0.18, lw=0.7)
        ax.axhline(100, color="k", ls=":", lw=1, alpha=0.5)
        ax.set_title(f"Action = {REGIME_NAMES[k]}", color=REGIME_COLORS[k], fontweight="bold")
        ax.set_xlabel("trading day")
    axes[0].set_ylabel("price (S0=100)")
    fig.suptitle("Controllability: SAME world model driven into 3 forced regimes (1000 paths each)",
                 fontweight="bold", y=1.02)
    savefig(fig, "fig1_controllability.png")


# ============================================================ FIG 2: market movie GIF
def fig_movie():
    src = load_paths(os.path.join(VIZ, "regime2.npz")) or load_paths(os.path.join(VIZ, "fm_markov.npz"))
    if src is None:
        print("[skip] movie (no rollout)"); return
    s, v = src["s"], src.get("v")
    # pick a dramatic path (largest drawdown)
    dd = (np.maximum.accumulate(s, 1) - s).max(1)
    idx = np.argsort(dd)[-1]
    sp = s[idx]; vp = v[idx] if v is not None else None
    T = len(sp)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), gridspec_kw={"height_ratios": [3, 1]})
    ax1.set_xlim(0, T); ax1.set_ylim(sp.min()*0.95, sp.max()*1.05)
    ax1.set_ylabel("price"); ax1.set_title("Day-by-day animation of a generated crash-regime price path", fontweight="bold", fontsize=11)
    ax2.set_xlim(0, T)
    if vp is not None:
        ax2.set_ylim(0, vp.max()*1.1); ax2.set_ylabel("variance")
    ax2.set_xlabel("trading day")
    (line1,) = ax1.plot([], [], color="#1f77b4", lw=2)
    (line2,) = ax2.plot([], [], color="#d62728", lw=1.5)
    pt, = ax1.plot([], [], "o", color="#1f77b4", ms=6)

    def init():
        line1.set_data([], []); line2.set_data([], []); pt.set_data([], [])
        return line1, line2, pt

    step = max(1, T // 80)
    frames = list(range(1, T, step)) + [T-1]

    def update(f):
        x = np.arange(f)
        line1.set_data(x, sp[:f])
        if vp is not None:
            line2.set_data(x, vp[:f])
        pt.set_data([f-1], [sp[f-1]])
        return line1, line2, pt

    try:
        anim = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
        anim.save(os.path.join(OUT, "fig2_crash_animation.gif"),
                  writer=animation.PillowWriter(fps=15))
        plt.close(fig); print("[fig] fig2_crash_animation.gif")
    except Exception as e:
        print("[warn] GIF failed (%s); saving filmstrip" % e)
        plt.close(fig)
        fig, axs = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
        for ax, f in zip(axs, np.linspace(T//5, T-1, 5).astype(int)):
            ax.plot(sp[:f], color="#1f77b4"); ax.set_title(f"day {f}")
        savefig(fig, "fig2_market_filmstrip.png")


# ============================================================ FIG 3: NFE knee
def fig_nfe():
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(NFE_K, NFE_RAW, "o-", color="#1f77b4", lw=2, ms=8, label="raw pricing RMSE")
    ax.plot(NFE_K, NFE_CAL, "s-", color="#d62728", lw=2, ms=8, label="calibrated pricing RMSE")
    ax.axhline(
        FINITE_SAMPLE_RMSE,
        color="gray",
        ls="--",
        lw=1,
        label=f"finite-sample ref {FINITE_SAMPLE_RMSE}",
    )
    ax.annotate("raw knee\n(2 steps)", (2, 1.819), textcoords="offset points", xytext=(10, 18),
                color="#1f77b4", arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    ax.annotate("cal best\n(1 step)", (1, 0.313), textcoords="offset points", xytext=(12, 30),
                color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xticks(NFE_K); ax.set_xlabel("NFE (number of function evals, K)")
    ax.set_ylabel("pricing RMSE"); ax.set_title("Few-step distillation: quality knee differs by metric", fontweight="bold")
    ax.legend()
    savefig(fig, "fig3_nfe_knee.png")


# ============================================================ FIG 4: fan chart real vs FM
def fig_fan():
    real = load_paths(os.path.join(DATA, "test.npz"))
    fm = load_paths(os.path.join(VIZ, "fm_markov.npz"))
    if real is None or fm is None:
        print("[skip] fan (missing data)"); return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fan(ax, real["s"], "#333333", "Real (held-out test)")
    fan(ax, fm["s"], "#1f77b4", "FM world model")
    ax.axhline(100, color="k", ls=":", lw=1, alpha=0.4)
    ax.set_xlabel("trading day"); ax.set_ylabel("price (S0=100)")
    ax.set_title("Price fan chart: real vs generated (median + 25/75 + 5/95 bands)", fontweight="bold")
    ax.legend()
    savefig(fig, "fig4_fanchart.png")


# ============================================================ FIG 5: regime-colored paths
def fig_regime_paths():
    real = load_paths(os.path.join(DATA, "test.npz"))
    if real is None or real["a"] is None:
        print("[skip] regime paths"); return
    s, a = real["s"], real["a"]
    # find paths that visit the crash regime for drama
    has_crash = np.where((a == 2).any(1))[0]
    pick = has_crash[:6] if len(has_crash) >= 6 else np.arange(6)
    fig, ax = plt.subplots(figsize=(9, 5))
    for off, i in enumerate(pick):
        sp = s[i]; ap = a[i]
        x = np.arange(len(sp))
        pts = np.array([x, sp]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        colors = [REGIME_COLORS[int(ap[min(t, len(ap)-1)])] for t in range(len(segs))]
        lc = LineCollection(segs, colors=colors, lw=1.6, alpha=0.9)
        ax.add_collection(lc)
    ax.set_xlim(0, s.shape[1]); ax.autoscale_view()
    ax.set_xlabel("trading day"); ax.set_ylabel("price")
    ax.set_title("Real sample paths colored by latent regime", fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color=REGIME_COLORS[k],lw=3,label=REGIME_NAMES[k]) for k in (0,1,2)])
    savefig(fig, "fig5_regime_paths.png")


# ============================================================ FIG 6: stylized facts report card
def acf(x, nlag):
    x = x - x.mean(); n = len(x); v = (x*x).sum()
    return np.array([1.0] + [ (x[:-k]*x[k:]).sum()/v for k in range(1, nlag+1)])

def fig_stylized():
    real = load_paths(os.path.join(DATA, "test.npz"))
    fm = load_paths(os.path.join(VIZ, "fm_ss.npz")) or load_paths(os.path.join(VIZ, "fm_markov.npz"))
    cd = load_paths(os.path.join(VIZ, "cd_fmss.npz"))
    boot = load_paths(os.path.join(VIZ, "bootstrap.npz"))
    if real is None or fm is None:
        print("[skip] stylized"); return
    series = [("Real", real, "#333333"), ("FM+SS (ours)", fm, "#1f77b4")]
    if cd: series.append(("FM+SS->CD (ours)", cd, "#d62728"))
    if boot: series.append(("Bootstrap (ref)", boot, "#2ca02c"))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    # (a) return distribution log-y (smooth line histogram)
    ax = axs[0, 0]
    edges = np.linspace(-8, 8, 81); ctr = 0.5*(edges[1:]+edges[:-1])
    for name, d, c in series:
        r = d["r"].reshape(-1); r = (r - r.mean())/r.std()
        h, _ = np.histogram(r, bins=edges, density=True)
        ax.semilogy(ctr, h+1e-12, color=c, label=name, lw=1.8)
    ax.semilogy(ctr, np.exp(-ctr**2/2)/np.sqrt(2*np.pi), "k:", label="Normal", lw=1.5)
    ax.set_ylim(1e-5, 1); ax.set_title("(a) Return distribution (log-y: fat tails)")
    ax.set_xlabel("standardized returns"); ax.legend(fontsize=9, loc="lower center", ncol=2)
    # (b) |return| ACF vol clustering
    ax = axs[0, 1]
    for name, d, c in series:
        rr = d["r"][:2000]
        ac = np.mean([acf(np.abs(rr[i]), 40) for i in range(min(800, len(rr)))], axis=0)
        ax.plot(range(1, 41), ac[1:], color=c, lw=1.8, label=name)
    ax.set_title("(b) ACF of |returns| (volatility clustering)"); ax.set_xlabel("lag (days)")
    ax.set_ylim(0, None); ax.legend(fontsize=9)
    # (c) leverage corr(r_t, r_{t+k}^2)
    ax = axs[1, 0]
    for name, d, c in series:
        rr = d["r"][:2000]
        lev = [np.corrcoef(rr[:, :-k].reshape(-1), (rr[:, k:]**2).reshape(-1))[0, 1] for k in range(1, 21)]
        ax.plot(range(1, 21), lev, color=c, lw=1.8, label=name)
    ax.axhline(0, color="k", lw=0.8); ax.set_title("(c) Leverage effect: corr(r_t, sq r_{t+k})")
    ax.set_xlabel("lag k"); ax.legend(fontsize=9)
    # (d) QQ real vs FM
    ax = axs[1, 1]
    rr = (real["r"].reshape(-1)-real["r"].mean())/real["r"].std()
    ff = (fm["r"].reshape(-1)-fm["r"].mean())/fm["r"].std()
    q = np.linspace(0.001, 0.999, 400)
    ax.plot(np.quantile(rr, q), np.quantile(ff, q), color="#1f77b4", lw=2)
    lim = [-8, 8]; ax.plot(lim, lim, "k:")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_title("(d) QQ-plot: FM+SS vs Real")
    ax.set_xlabel("real quantiles"); ax.set_ylabel("FM+SS quantiles")
    fig.suptitle("Stylized-facts report card  (Real vs our FM+SS / FM+SS->CD vs Bootstrap)", fontweight="bold")
    savefig(fig, "fig6_stylized_facts.png")


# ============================================================ FIG 7: raw-cal pareto (CN styled)
def fig_pareto():
    import matplotlib as mpl
    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    NAVY, ORANGE, TEAL, BLUE = "#16263f", "#ef9a2e", "#16a085", "#2f6fd0"
    GRAY, GRAYT, RED, FRONT = "#9aa6b2", "#7a8794", "#e8463a", "#69c5b4"
    fig, ax = plt.subplots(figsize=(11.6, 7.0))
    ax.grid(True, color="#dfe3e8", lw=0.8, zorder=0); ax.set_axisbelow(True)
    # Region below the finite-sample reference can be useful, but should not
    # be read as a theoretical lower bound.
    ax.axhspan(0, FINITE_SAMPLE_RMSE, color="#f4cbc6", alpha=0.55, zorder=0)
    ax.axhline(FINITE_SAMPLE_RMSE, color=RED, ls="--", lw=2.2, zorder=2)
    ax.text(0.12, FINITE_SAMPLE_RMSE + 0.012, "有限样本参照 0.1646",
            color=RED, fontsize=12, va="bottom", ha="left")
    ax.text(0.30, 0.072, "低于有限样本参照的校准区", color=RED, fontsize=14,
            fontweight="bold", va="center", ha="left")
    # trade-off frontier (trend) through the champions
    fx = [0.475, 0.872, 1.875, 2.595]; fy = [0.583, 0.348, 0.170, 0.180]
    ax.plot(fx, fy, ls="--", color=FRONT, lw=2.6, zorder=1)
    ax.text(0.66, 0.47, "raw ↔ 校准  权衡前沿", color=TEAL, fontsize=13,
            rotation=-50, ha="center", va="center")
    # points: (x, y, color, size)
    P = {
        "boot": (0.269, 0.244, GRAY, 130), "fm": (0.475, 0.583, NAVY, 240),
        "ss": (0.872, 0.348, ORANGE, 240), "garch": (1.089, 0.482, GRAY, 130),
        "cd": (1.875, 0.170, TEAL, 240), "pin": (2.595, 0.180, TEAL, 150),
        "art": (2.328, 0.118, BLUE, 200), "ddpm": (2.855, 0.576, GRAY, 130),
    }
    for x, y, c, s in P.values():
        ax.scatter(x, y, s=s, c=c, edgecolor="white", linewidth=1.2, zorder=4)
    def call(key, text, dx, dy, color, ha, bold=False, arrow=True):
        x, y, *_ = P[key]
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=12, ha=ha, va="center", color=color, zorder=5,
                    fontweight=("bold" if bold else "normal"),
                    arrowprops=(dict(arrowstyle="-", color=color, lw=1.3,
                                connectionstyle="arc3,rad=0.25") if arrow else None))
    call("fm",  "FM teacher（基础）\nraw 自洽冠军", -6, 34, NAVY, "left", True)
    call("ss",  "FM + 调度采样\n平衡冠军·峰度 4.41", 18, 40, ORANGE, "left", True)
    call("garch","GARCH-t（峰度爆炸 19）", 12, 14, GRAYT, "left", False, False)
    call("ddpm","DDPM（同架构·扩散）", -12, 16, GRAYT, "right", False, False)
    call("boot","块自助法（数据回放·参照）", 14, 0, GRAYT, "left", False, False)
    call("cd",  "一致性蒸馏 SIGMA-Base→CD\n诚实校准冠军 0.170", 12, 62, TEAL, "left", True)
    call("pin", "SIGMA-Pin（本文·训练）0.180", 14, 26, TEAL, "left", False)
    call("art", "激进蒸馏 cal 0.118\n校准假象（已弃用）", 16, -30, BLUE, "left", True)
    # off-chart Quant-GAN
    ax.text(0.985, 0.97, "↗ Quant-GAN 出界：raw 5.4 · cal 0.67", transform=ax.transAxes,
            ha="right", va="top", color=GRAYT, fontsize=12)
    ax.set_xlim(0, 3.15); ax.set_ylim(0, 0.72)
    ax.set_xlabel("raw  自由演化自洽性（定价 RMSE，越小越好 →）", fontsize=13)
    ax.set_ylabel("cal  矩校准后定价（越低越好 ↓）", fontsize=13)
    savefig(fig, "fig7_pareto_tradeoff.png")


# ============================================================ FIG 8: option price smile
def fig_pricing():
    f = os.path.join(JSON, "eval_champion", "evaluation", "eval_fm.json")
    if not os.path.exists(f):
        print("[skip] pricing"); return
    d = json.load(open(f))
    pf = d["pricing_fake_vs_mc_oracle"]; pr = d.get("pricing_real_vs_mc_oracle")
    mon = pf["moneynesses"]; mats = pf["maturities"]
    gen = pf["mc_prices"]; ora = pf["reference_prices"]
    realp = pr["mc_prices"] if pr else None
    fig, axs = plt.subplots(1, len(mats), figsize=(5*len(mats), 4.2), sharey=False)
    if len(mats) == 1: axs = [axs]
    for j, (ax, T) in enumerate(zip(axs, mats)):
        ax.plot(mon, ora[j], "k-o", lw=2, label="MC oracle (truth)")
        ax.plot(mon, gen[j], "--s", color="#1f77b4", lw=2, label="FM world model")
        if realp: ax.plot(mon, realp[j], ":^", color="#2ca02c", lw=1.5, label="Real test set")
        ax.set_title(f"maturity T={T}y"); ax.set_xlabel("moneyness K/S")
        if j == 0: ax.set_ylabel("call price")
        ax.legend(fontsize=9)
    fig.suptitle("Option pricing vs Monte-Carlo oracle (FM world model)", fontweight="bold", y=1.02)
    savefig(fig, "fig8_option_pricing.png")


# ============================================================ FIG 9: radar
def fig_radar():
    # axes: pricing(raw), kurt-match, vol-clustering, leverage, tail-match  (higher=better)
    sel = ["FM teacher", "FM+SS", "Path-loss (best-cal)", "CD<-FM+SS", "DDPM"]
    DISP9 = {"Path-loss (best-cal)": "SIGMA-Cal", "CD<-FM+SS": "FM+SS->CD"}
    labels = ["pricing\n(raw)", "kurtosis\nmatch", "vol-cluster\n(absACF)", "leverage\nmatch", "tail\nmatch"]
    def score(m):
        raw, cal, kurt, absACF, lev, tail, sigW = MET[m]
        return [
            1/(1+raw),                      # pricing raw (higher better)
            1/(1+abs(kurt-REAL_KURT)),      # kurtosis match
            1/(1+absACF*30),                # vol clustering (lower absACF better)
            1/(1+lev*30),                   # leverage (lower better)
            1/(1+abs(tail-REAL_TAIL)),      # tail match
        ]
    N = len(labels)
    ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cols = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
    for m, c in zip(sel, cols):
        vals = score(m); vals += vals[:1]
        ax.plot(ang, vals, color=c, lw=2, label=DISP9.get(m, m)); ax.fill(ang, vals, color=c, alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([]); ax.set_title("Multi-metric radar (larger = closer to real)", fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    savefig(fig, "fig9_radar.png")


if __name__ == "__main__":
    for fn in [fig_controllability, fig_movie, fig_nfe, fig_fan, fig_regime_paths,
               fig_stylized, fig_pareto, fig_pricing, fig_radar]:
        try:
            fn()
        except Exception as e:
            import traceback; print("[ERR]", fn.__name__, e); traceback.print_exc()
    print("DONE ->", OUT)
