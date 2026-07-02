#!/usr/bin/env python3
"""Supplementary econometric baselines for the FinFlow Heston regime task.

Generates fake paths in the rollout npz format (s_paths, log_returns) for:
  1. GARCH(1,1)-t  : MLE-fit Student-t GARCH(1,1) on pooled training returns,
                     then simulate independent return paths (no regime knowledge).
  2. block-bootstrap: moving-block bootstrap of real training returns.

Each baseline is written in BOTH raw and calibrated (--calibrate-moments mirror)
form so it is directly comparable to the FM / QGAN rows. Calibration mirrors
rollout.py::_calibrate_returns exactly: pool returns, standardize, rescale by the
data's return_std/return_mean, rebuild S from s0.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from scipy import optimize, special

P = Path(os.environ.get("FINFLOW_EXPERIMENT_ROOT", "runs/experiments/p3_full_parallel"))
DATA = P / "data"
meta = json.load(open(DATA / "metadata.json"))
RET_MEAN = meta["normalization"]["return_mean"]
RET_STD = meta["normalization"]["return_std"]
S0 = float(meta["s0"])
N_PATHS, N_STEPS = 5000, 252
SEED = 20260603
rng = np.random.default_rng(SEED)

train = np.load(DATA / "train.npz")
R = np.asarray(train["log_returns"], dtype=np.float64)  # (50000, 252)
print(f"[data] train returns {R.shape} mean={R.mean():.3e} std={R.std():.4f}")


def build_npz(r_paths):
    """r_paths (N,252) -> dict with s_paths (N,253) and log_returns (N,252)."""
    cum = np.cumsum(r_paths, axis=1)
    s_tail = S0 * np.exp(cum)
    s0col = np.full((r_paths.shape[0], 1), S0)
    s_paths = np.concatenate([s0col, s_tail], axis=1)
    return {"s_paths": s_paths.astype(np.float32),
            "log_returns": r_paths.astype(np.float32)}


def calibrate(r_paths):
    """Mirror rollout.py _calibrate_returns."""
    x = r_paths.reshape(-1)
    z = (x - x.mean()) / max(x.std(ddof=0), 1e-6)
    r_cal = (z.reshape(r_paths.shape) * RET_STD + RET_MEAN)
    return r_cal


# ---------------------------------------------------------------- GARCH(1,1)-t
def garch_negloglik(params, eps_paths):
    """eps_paths (M,T) residuals (r-mu) per path; vectorized recursion."""
    omega, alpha, beta, nu = params
    if omega <= 0 or alpha <= 0 or beta <= 0 or nu <= 2.01 or alpha + beta >= 0.9999:
        return 1e10
    M, T = eps_paths.shape
    uncond = omega / (1.0 - alpha - beta)
    sigma2 = np.full(M, uncond)
    e2 = eps_paths ** 2
    c = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
         - 0.5 * np.log((nu - 2) * np.pi))
    ll = 0.0
    for t in range(T):
        if t > 0:
            sigma2 = omega + alpha * e2[:, t - 1] + beta * sigma2
        z2 = e2[:, t] / sigma2
        ll += np.sum(c - 0.5 * np.log(sigma2)
                     - 0.5 * (nu + 1) * np.log1p(z2 / (nu - 2)))
    return -ll


def fit_garch(R, n_fit=4000):
    idx = rng.choice(R.shape[0], size=min(n_fit, R.shape[0]), replace=False)
    sub = R[idx]
    mu = float(sub.mean())
    eps = sub - mu
    var = float(eps.var())
    x0 = [var * 0.05, 0.08, 0.90, 6.0]
    bounds = [(1e-10, var), (1e-4, 0.4), (0.3, 0.998), (2.5, 40.0)]
    res = optimize.minimize(garch_negloglik, x0, args=(eps,), method="L-BFGS-B",
                            bounds=bounds, options={"maxiter": 200})
    omega, alpha, beta, nu = res.x
    print(f"[garch] mu={mu:.3e} omega={omega:.3e} alpha={alpha:.4f} "
          f"beta={beta:.4f} nu={nu:.3f} a+b={alpha+beta:.4f} nll={res.fun:.1f}")
    return mu, omega, alpha, beta, nu


def simulate_garch(mu, omega, alpha, beta, nu, n_paths, n_steps):
    uncond = omega / (1.0 - alpha - beta)
    sigma2 = np.full(n_paths, uncond)
    eps_prev2 = np.full(n_paths, uncond)
    tscale = np.sqrt((nu - 2) / nu)  # standardize Student-t to unit variance
    r = np.empty((n_paths, n_steps))
    for t in range(n_steps):
        sigma2 = omega + alpha * eps_prev2 + beta * sigma2
        z = rng.standard_t(nu, size=n_paths) * tscale
        eps = np.sqrt(sigma2) * z
        r[:, t] = mu + eps
        eps_prev2 = eps ** 2
    return r


# ---------------------------------------------------------------- block bootstrap
def block_bootstrap(R, n_paths, n_steps, block=21):
    M, T = R.shape
    out = np.empty((n_paths, n_steps))
    for i in range(n_paths):
        pos = 0
        while pos < n_steps:
            p = rng.integers(0, M)
            start = rng.integers(0, T - block + 1)
            seg = R[p, start:start + block]
            take = min(block, n_steps - pos)
            out[i, pos:pos + take] = seg[:take]
            pos += take
    return out


def write_and_report(name, r_paths):
    for tag, rp in [("raw", r_paths), ("cal", calibrate(r_paths))]:
        d = build_npz(rp)
        outdir = P / "eval_baselines_0603"
        outdir.mkdir(exist_ok=True)
        f = outdir / f"roll_{name}_{tag}.npz"
        np.savez_compressed(f, **d)
        print(f"[write] {f}")


t0 = time.time()
print("=== GARCH(1,1)-t ===")
params = fit_garch(R)
r_garch = simulate_garch(*params, N_PATHS, N_STEPS)
write_and_report("garch_t", r_garch)

print("=== block-bootstrap (block=21) ===")
r_boot = block_bootstrap(R, N_PATHS, N_STEPS, block=21)
write_and_report("block_bootstrap", r_boot)
print(f"[done] {time.time()-t0:.1f}s")
