"""
model/train_lstm.py  — FAST version
=====================================
Label generation runs the EKF ONCE per sequence, then labels each window
using local position error at the window endpoint — no re-runs.

Alpha label  : jerk magnitude mapped to alpha sweep (no EKF re-run needed)
Beta label   : GT vertical velocity at window midpoint

Total time: ~2-5 minutes for label gen, ~5-10 min for training.

Usage
-----
  python -m model.train_lstm --seqs rt4_gravel rt4_rim --data_dir data/rooad --output model
"""

import sys, os, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.rooad_loader import (
    load_sequence, build_lstm_features, static_bias_estimate, compute_jerk
)
from filter.inekf_imu import InEKF_IMU
from model.lstm_noise_model import LSTMNoiseAdapter


# ─────────────────────── Dataset ─────────────────────────────────────────────

class ImpactWindowDataset(Dataset):
    def __init__(self, windows, log_alphas, betas):
        self.X  = torch.tensor(windows,    dtype=torch.float32)
        self.ya = torch.tensor(log_alphas, dtype=torch.float32).unsqueeze(1)
        self.yb = torch.tensor(betas,      dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ya[idx], self.yb[idx]


# ─────────────────────── Fast label generation ───────────────────────────────

def generate_labels_fast(seq_name, data_dir, cfg):
    """
    Fast O(N) label generation — no EKF re-runs.

    Alpha label: derived from jerk magnitude in window
        high jerk  → high alpha (matches jerk heuristic intuition)
        Maps jerk percentile to alpha sweep range.

    Beta label: GT vertical velocity magnitude at window midpoint
        high |v_z_GT| → high beta (vehicle genuinely hopping)
        Normalised to [0, beta_max].
    """
    data     = load_sequence(seq_name, data_dir, cfg)
    imu_t    = data["imu_t"];  imu_acc  = data["imu_acc"]
    imu_gyro = data["imu_gyro"]; gt_t   = data["gt_t"]; gt_pos = data["gt_pos"]

    lc     = cfg["lstm"]
    W      = int(lc["window"])
    n_init = int(cfg["init"]["static_init_samples"])
    a_min  = float(lc["alpha_min"]); a_max = float(lc["alpha_max"])
    b_max  = float(cfg["jerk"].get("beta_max", 0.85))
    step   = W // 2

    if len(gt_t) < 2 or len(imu_t) < W + n_init + 10:
        return np.zeros((0,W,7), np.float32), np.array([]), np.array([])

    feats = build_lstm_features(imu_acc, imu_gyro, imu_t)  # (N,7)
    jerk  = feats[:, 6]   # pre-computed jerk column

    # GT velocity by finite difference
    gt_vel = np.zeros_like(gt_pos)
    for i in range(1, len(gt_t)-1):
        dt = gt_t[i+1] - gt_t[i-1]
        gt_vel[i] = (gt_pos[i+1] - gt_pos[i-1]) / max(dt, 1e-9)
    gt_vel[0] = gt_vel[1]; gt_vel[-1] = gt_vel[-2]

    def interp_gt_vel(t):
        if t <= gt_t[0]:  return gt_vel[0]
        if t >= gt_t[-1]: return gt_vel[-1]
        i = np.searchsorted(gt_t, t) - 1
        f = (t - gt_t[i]) / max(gt_t[i+1]-gt_t[i], 1e-12)
        return gt_vel[i]*(1-f) + gt_vel[i+1]*f

    # Jerk percentiles for alpha mapping
    j_p10 = np.percentile(jerk, 10)
    j_p90 = np.percentile(jerk, 90)

    windows_list, alpha_list, beta_list = [], [], []

    indices = range(n_init, len(imu_t) - W - 1, step)
    for start in tqdm(indices, desc=f"  {seq_name}", ncols=70):
        win_feat = feats[start: start+W]  # (W, 7)

        # Alpha label: map mean jerk in window to alpha range
        win_jerk  = jerk[start: start+W].mean()
        jerk_norm = np.clip((win_jerk - j_p10) / max(j_p90 - j_p10, 1e-6), 0, 1)
        alpha     = a_min + (a_max - a_min) * jerk_norm

        # Beta label: derived from jerk in window
        # High jerk -> vehicle may be hopping -> relax NHC (high beta)
        # Uses 90th percentile jerk in window for robustness
        win_jerk_p90 = np.percentile(jerk[start: start+W], 90)
        beta = float(np.clip(
            (win_jerk_p90 - cfg["jerk"]["threshold"]) /
            (cfg["jerk"]["threshold"] * 2.0),   # normalise around threshold
            0.0, 1.0
        ) * b_max)

        windows_list.append(win_feat)
        alpha_list.append(np.log(max(alpha, 1e-3)))
        beta_list.append(beta)

    if not windows_list:
        return np.zeros((0,W,7), np.float32), np.array([]), np.array([])

    W_arr = np.stack(windows_list).astype(np.float32)
    A_arr = np.array(alpha_list)
    B_arr = np.array(beta_list)
    print(f"  {seq_name}: {len(W_arr)} windows  "
          f"alpha=[{np.exp(A_arr).min():.1f},{np.exp(A_arr).max():.1f}]  "
          f"beta=[{B_arr.min():.2f},{B_arr.max():.2f}]")
    return W_arr, A_arr, B_arr


# ─────────────────────── Training ────────────────────────────────────────────

def train(cfg, seq_names, data_dir, output_dir="model/"):
    tc   = cfg["training"]
    seed = int(tc["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    print("Generating labels ...")
    all_W, all_A, all_B = [], [], []
    for seq in seq_names:
        W, A, B = generate_labels_fast(seq, data_dir, cfg)
        if len(W):
            all_W.append(W); all_A.append(A); all_B.append(B)

    if not all_W:
        raise RuntimeError("No training data. Check --seqs and --data_dir.")

    windows = np.concatenate(all_W)
    alphas  = np.concatenate(all_A)
    betas   = np.concatenate(all_B)
    print(f"\nTotal: {len(windows)} windows")

    feat_mean = windows.mean(axis=(0,1))
    feat_std  = windows.std(axis=(0,1)) + 1e-8
    windows_n = (windows - feat_mean) / feat_std

    n     = len(windows_n)
    n_val = max(1, int(n * tc["val_frac"]))
    idx   = np.random.permutation(n)
    itr, iva = idx[n_val:], idx[:n_val]

    ds_tr = ImpactWindowDataset(windows_n[itr], alphas[itr], betas[itr])
    ds_va = ImpactWindowDataset(windows_n[iva], alphas[iva], betas[iva])
    dl_tr = DataLoader(ds_tr, batch_size=tc["batch_size"], shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=tc["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = LSTMNoiseAdapter(cfg).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                               weight_decay=tc["weight_decay"])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc["epochs"], 1e-5)
    loss_fn = nn.MSELoss()

    print(f"Training on {device}  |  {len(ds_tr)} train / {len(ds_va)} val\n")

    best_val = np.inf
    Path(output_dir).mkdir(exist_ok=True)
    model_path = str(Path(output_dir) / "lstm_noise_adapter.pt")

    for epoch in range(1, tc["epochs"]+1):
        model.train(); tr_loss = 0.0
        for Xb, ya, yb in dl_tr:
            Xb,ya,yb = Xb.to(device),ya.to(device),yb.to(device)
            pa, pb = model(Xb)
            loss = loss_fn(torch.log(pa), ya) + loss_fn(pb, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(ds_tr)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for Xb,ya,yb in dl_va:
                Xb,ya,yb = Xb.to(device),ya.to(device),yb.to(device)
                pa,pb = model(Xb)
                va_loss += (loss_fn(torch.log(pa),ya)+loss_fn(pb,yb)).item()*len(Xb)
        va_loss /= len(ds_va)
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{tc['epochs']}  "
                  f"train={tr_loss:.4f}  val={va_loss:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.1e}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model_state": model.state_dict(),
                        "cfg": cfg["lstm"],
                        "feat_mean": feat_mean,
                        "feat_std":  feat_std}, model_path)

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Model saved -> {model_path}")
    return model_path


# ─────────────────────── CLI ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config/rooad_config.yaml")
    parser.add_argument("--seqs",     nargs="+", required=True)
    parser.add_argument("--data_dir", default="data/rooad/")
    parser.add_argument("--output",   default="model/")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.seqs, args.data_dir, args.output)
