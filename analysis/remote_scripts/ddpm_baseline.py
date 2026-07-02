#!/usr/bin/env python3
"""DDPM baseline for the 3-regime Heston world model.

Fair head-to-head vs the Flow-Matching teacher: SAME two-stage transition
factorization, SAME network architecture (finflow TransitionFM, 256/6), SAME
condition layout, SAME free-running rollout (finflow.autoregressive_rollout) and
SAME evaluation (evaluate_rollout.py). The ONLY differences are the training
objective (DDPM epsilon-prediction instead of conditional FM) and the sampler
(ancestral/DDIM diffusion instead of the FM ODE).

Stage alignment (matches finflow.data.heston):
  vol: target = log_v_{t+1}_norm,  cond = [log_v_t_norm, onehot(a_t)]          (dim 1+A)
  ret: target = r_t_norm,          cond = [log_v_{t+1}_norm, log_v_t_norm, r_{t-1}_norm, onehot(a_t)] (dim 3+A)
"""
import argparse, json, os, sys, time, math
from pathlib import Path
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(os.environ.get("FINFLOW_PROJECT_ROOT", ".")).resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from finflow.models import TransitionFM
from finflow.inference.rollout import autoregressive_rollout

P = Path(os.environ.get("FINFLOW_EXPERIMENT_ROOT", "runs/experiments/p3_full_parallel"))
DATA = P / "data"
OUT = P / "eval_ddpm_baseline_0603"
OUT.mkdir(exist_ok=True)
(OUT / "evaluation").mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--batch-size", type=int, default=4096)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--T", type=int, default=1000)
ap.add_argument("--sample-steps", type=int, default=50)
ap.add_argument("--eta", type=float, default=1.0)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--blocks", type=int, default=6)
ap.add_argument("--tdim", type=int, default=64)
ap.add_argument("--action-dropout", type=float, default=0.1)
ap.add_argument("--n-paths", type=int, default=5000)
ap.add_argument("--n-steps", type=int, default=252)
ap.add_argument("--seed", type=int, default=20260603)
args = ap.parse_args()

dev = torch.device("cuda")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

meta = json.load(open(DATA / "metadata.json"))
nz = meta["normalization"]
LVM, LVS = nz["log_v_mean"], nz["log_v_std"]
RM, RS = nz["return_mean"], nz["return_std"]
A = int(meta["num_actions"])
TM = np.asarray(meta["transition_matrix"], dtype=np.float64)
V0, S0 = float(meta["v0"]), float(meta["s0"])

# ---------------- build training pairs from train.npz ----------------
tr = np.load(DATA / "train.npz")
v = np.asarray(tr["v_paths"], np.float64)        # [N,253]
r = np.asarray(tr["log_returns"], np.float64)    # [N,252]
a = np.asarray(tr["actions"], np.int64)          # [N,252]
N, Tn = r.shape
logv = np.log(v)
logv_n = (logv - LVM) / LVS                       # [N,253]
r_n = (r - RM) / RS                               # [N,252]
r_prev_n = np.concatenate([np.full((N, 1), (0.0 - RM) / RS), r_n[:, :-1]], axis=1)  # [N,252]
onehot = np.eye(A, dtype=np.float64)[a]           # [N,252,A]

# flatten over (path, t)
lv_t = logv_n[:, :Tn].reshape(-1, 1)              # log_v_t
lv_next = logv_n[:, 1:Tn + 1].reshape(-1, 1)      # log_v_{t+1}
rt = r_n.reshape(-1, 1)
rp = r_prev_n.reshape(-1, 1)
oh = onehot.reshape(-1, A)

vol_target = torch.tensor(lv_next, dtype=torch.float32, device=dev)
vol_cond = torch.tensor(np.concatenate([lv_t, oh], axis=1), dtype=torch.float32, device=dev)
ret_target = torch.tensor(rt, dtype=torch.float32, device=dev)
ret_cond = torch.tensor(np.concatenate([lv_next, lv_t, rp, oh], axis=1), dtype=torch.float32, device=dev)
print(f"[data] pairs={vol_target.shape[0]} vol_cond={vol_cond.shape[1]} ret_cond={ret_cond.shape[1]} A={A}")

# ---------------- DDPM schedule ----------------
betas = torch.linspace(1e-4, 0.02, args.T, device=dev)
alphas = 1.0 - betas
acp = torch.cumprod(alphas, dim=0)                # [T]
sqrt_acp = acp.sqrt()
sqrt_1macp = (1.0 - acp).sqrt()


def train_stage(name, cond, target, cond_dim, onehot_slice):
    net = TransitionFM(state_dim=1, condition_dim=cond_dim, hidden_dim=args.hidden,
                       time_embedding_dim=args.tdim, num_blocks=args.blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    Np = target.shape[0]
    net.train()
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, Np, (args.batch_size,), device=dev)
        x0 = target[idx]
        c = cond[idx].clone()
        if args.action_dropout > 0:
            drop = torch.rand(c.shape[0], device=dev) < args.action_dropout
            c[drop, onehot_slice[0]:onehot_slice[1]] = 0.0
        t = torch.randint(0, args.T, (args.batch_size,), device=dev)
        eps = torch.randn_like(x0)
        x_t = sqrt_acp[t].unsqueeze(1) * x0 + sqrt_1macp[t].unsqueeze(1) * eps
        tau = (t.float() / (args.T - 1))
        pred = net(x_t, tau, c)
        loss = (pred - eps).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 3000 == 0 or step == args.steps - 1:
            print(f"[{name}] step {step}/{args.steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")
    net.eval()
    return net


class DDPMSampler:
    """Sampler-compatible (state_dim, condition_dim, kind, sample) wrapper."""
    def __init__(self, net, cond_dim, sample_steps, eta):
        self.net = net
        self.state_dim = 1
        self.condition_dim = cond_dim
        self.kind = "ddpm"
        seq = np.linspace(0, args.T - 1, sample_steps).round().astype(int)
        seq = np.unique(seq)
        self.seq = seq

    @torch.no_grad()
    def sample(self, condition, *, noise=None, cfg_w=0.0):
        B = condition.shape[0]
        x = noise if noise is not None else torch.randn(B, 1, device=condition.device)
        x = x.to(condition.device)
        seq = self.seq
        for i in reversed(range(len(seq))):
            t = int(seq[i])
            tau = torch.full((B,), t / (args.T - 1), device=condition.device, dtype=condition.dtype)
            eps = self.net(x, tau, condition)
            a_t = acp[t]
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            x0 = x0.clamp(-6.0, 6.0)
            if i == 0:
                x = x0
            else:
                a_prev = acp[int(seq[i - 1])]
                sigma = args.eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
                z = torch.randn_like(x)
                x = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp_min(0).sqrt() * eps + sigma * z
        return x


t_all = time.time()
print("=== train vol DDPM ===")
vol_net = train_stage("vol", vol_cond, vol_target, vol_cond.shape[1], (1, 1 + A))
print("=== train ret DDPM ===")
ret_net = train_stage("ret", ret_cond, ret_target, ret_cond.shape[1], (3, 3 + A))

vol_s = DDPMSampler(vol_net, vol_cond.shape[1], args.sample_steps, args.eta)
ret_s = DDPMSampler(ret_net, ret_cond.shape[1], args.sample_steps, args.eta)

print("=== free-running rollout ===")
res = autoregressive_rollout(
    vol_s, ret_s, normalization=nz, n_paths=args.n_paths, n_steps=args.n_steps,
    num_actions=A, initial_v=V0, initial_s=S0, transition_matrix=TM,
    initial_regime=0, action_seed=args.seed, noise_seed=args.seed, device=dev,
)


def save_npz(tag, r_paths):
    cum = np.cumsum(r_paths, axis=1)
    s = np.concatenate([np.full((r_paths.shape[0], 1), S0), S0 * np.exp(cum)], axis=1)
    np.savez_compressed(OUT / f"roll_ddpm_{tag}.npz",
                        s_paths=s.astype(np.float32), log_returns=r_paths.astype(np.float32))


r_raw = res.r_paths.astype(np.float64)
save_npz("raw", r_raw)
# calibrated: pool-standardize then rescale to data moments (mirror rollout.py)
x = r_raw.reshape(-1)
z = (x - x.mean()) / max(x.std(ddof=0), 1e-6)
r_cal = (z.reshape(r_raw.shape) * RS + RM)
save_npz("cal", r_cal)
print(f"[done] total {time.time()-t_all:.0f}s; raw kurt={((r_raw-r_raw.mean())**4).mean()/((r_raw-r_raw.mean())**2).mean()**2:.3f}")
print("npz written to", OUT)
