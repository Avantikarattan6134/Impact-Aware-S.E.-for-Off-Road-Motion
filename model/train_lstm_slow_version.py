"""
model/train_lstm.py
===================
Train LSTM to predict (alpha, beta) from IMU windows.

Labels
------
  alpha* : Q scale that minimises position error on a short horizon
  beta*  : NHC relaxation = |v_body_vertical_GT| / v_max  (0=no vertical GT motion, 1=full hop)

Usage
-----
  python -m model.train_lstm --config config/rooad_config.yaml ^
      --seqs rt4_gravel rt4_rim --data_dir data/rooad
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
    load_sequence, build_lstm_features, static_bias_estimate,
    compute_jerk
)
from filter.inekf_imu   import InEKF_IMU
from filter.lie_utils   import rot_from_se23
from model.lstm_noise_model import LSTMNoiseAdapter


# ─────────────────────── Dataset ─────────────────────────────────────────────

class ImpactWindowDataset(Dataset):
    def __init__(self, windows, log_alphas, betas):
        self.X  = torch.tensor(windows,    dtype=torch.float32)
        self.ya = torch.tensor(log_alphas, dtype=torch.float32).unsqueeze(1)
        self.yb = torch.tensor(betas,      dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ya[idx], self.yb[idx]


# ─────────────────────── Label generation ────────────────────────────────────

def generate_labels(seq_name, data_dir, cfg):
    data     = load_sequence(seq_name, data_dir, cfg)
    imu_t    = data["imu_t"];  imu_acc  = data["imu_acc"]
    imu_gyro = data["imu_gyro"]; gt_t   = data["gt_t"]; gt_pos = data["gt_pos"]

    tc   = cfg["training"];  lc = cfg["lstm"]
    W    = int(lc["window"]);  horizon = int(tc["horizon"])
    alphas = tc["alpha_sweep"]
    n_init = int(cfg["init"]["static_init_samples"])
    beta_max = float(cfg["jerk"].get("beta_max", 0.85))

    if len(gt_t) < 2 or len(imu_t) < W + horizon + n_init:
        return np.zeros((0, W, 7), np.float32), np.array([]), np.array([])

    feats = build_lstm_features(imu_acc, imu_gyro, imu_t)
    bg0, ba0, R0 = static_bias_estimate(imu_acc, imu_gyro, n_init, cfg["data"]["gravity"])

    def interp_gt(t):
        if t <= gt_t[0]:  return gt_pos[0]
        if t >= gt_t[-1]: return gt_pos[-1]
        i = np.searchsorted(gt_t, t) - 1
        f = (t - gt_t[i]) / max(gt_t[i+1] - gt_t[i], 1e-12)
        return gt_pos[i] * (1-f) + gt_pos[i+1] * f

    def interp_gt_vel(t):
        """Approximate velocity from GT by finite diff."""
        dt = 0.1
        p0 = interp_gt(t - dt/2)
        p1 = interp_gt(t + dt/2)
        return (p1 - p0) / dt

    windows_list, alpha_list, beta_list = [], [], []
    step = W // 2

    for start in range(n_init, len(imu_t) - W - horizon, step):
        win_feat  = feats[start: start + W]
        h_end     = start + W + horizon

        # ── alpha label: sweep Q scale, pick best ATE ───────────────────────
        best_alpha, best_err = 1.0, np.inf
        for alpha in alphas:
            ekf = InEKF_IMU(cfg)
            # disable NHC for alpha sweep (isolate Q effect)
            ekf._nhc_cfg = dict(cfg["nhc"], enabled=False)
            ekf.reset(R0.copy(), np.zeros(3), np.zeros(3),
                      bg0.copy(), ba0.copy(), imu_t[n_init])
            for k in range(n_init, min(h_end, len(imu_t))):
                ekf.propagate(imu_gyro[k], imu_acc[k], imu_t[k], alpha=alpha)
            t_eval = imu_t[min(h_end-1, len(imu_t)-1)]
            err    = np.linalg.norm(ekf.position - interp_gt(t_eval))
            if err < best_err:
                best_err = err; best_alpha = alpha

        # ── beta label: from GT vertical velocity at window midpoint ─────────
        t_mid   = imu_t[start + W // 2]
        v_gt    = interp_gt_vel(t_mid)
        # Project onto body-up direction (approximate as world-up for flat terrain)
        v_vert  = abs(v_gt[2])            # vertical component
        v_max   = 3.0                     # clip at 3 m/s vertical
        beta    = float(np.clip(v_vert / v_max, 0.0, 1.0) * beta_max)

        windows_list.append(win_feat)
        alpha_list.append(np.log(best_alpha))
        beta_list.append(beta)

    if not windows_list:
        return np.zeros((0, W, 7), np.float32), np.array([]), np.array([])

    return (np.stack(windows_list).astype(np.float32),
            np.array(alpha_list),
            np.array(beta_list))


# ─────────────────────── Training loop ───────────────────────────────────────

def train(cfg, seq_names, data_dir, output_dir="model/"):
    tc   = cfg["training"]
    seed = int(tc["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    all_W, all_A, all_B = [], [], []
    for seq in seq_names:
        print(f"  Generating labels for {seq} ...")
        W, A, B = generate_labels(seq, data_dir, cfg)
        if len(W):
            all_W.append(W); all_A.append(A); all_B.append(B)
            print(f"    {len(W)} windows  α=[{np.exp(A).min():.1f},{np.exp(A).max():.1f}]  "
                  f"β=[{B.min():.2f},{B.max():.2f}]")

    if not all_W:
        raise RuntimeError("No training data generated.")

    windows = np.concatenate(all_W)
    alphas  = np.concatenate(all_A)
    betas   = np.concatenate(all_B)

    # Normalise features
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

    print(f"\nTraining on {device}  |  {len(ds_tr)} train / {len(ds_va)} val")

    best_val = np.inf
    Path(output_dir).mkdir(exist_ok=True)
    model_path = str(Path(output_dir) / "lstm_noise_adapter.pt")

    for epoch in range(1, tc["epochs"]+1):
        model.train()
        tr_loss = 0.0
        for Xb, ya, yb in dl_tr:
            Xb, ya, yb = Xb.to(device), ya.to(device), yb.to(device)
            pred_a, pred_b = model(Xb)
            loss = loss_fn(torch.log(pred_a), ya) + loss_fn(pred_b, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(ds_tr)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for Xb, ya, yb in dl_va:
                Xb, ya, yb = Xb.to(device), ya.to(device), yb.to(device)
                pred_a, pred_b = model(Xb)
                va_loss += (loss_fn(torch.log(pred_a), ya) + loss_fn(pred_b, yb)).item() * len(Xb)
        va_loss /= len(ds_va)
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{tc['epochs']}  "
                  f"train={tr_loss:.4f}  val={va_loss:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.1e}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model_state": model.state_dict(), "cfg": cfg["lstm"],
                        "feat_mean": feat_mean, "feat_std": feat_std}, model_path)

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Model saved → {model_path}")
    return model_path


# ─────────────────────── CLI ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config/rooad_config.yaml")
    parser.add_argument("--seqs",     nargs="+", required=True,
                        help="Training sequences e.g. rt4_gravel rt4_rim")
    parser.add_argument("--data_dir", default="data/rooad/")
    parser.add_argument("--output",   default="model/")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.seqs, args.data_dir, args.output)
