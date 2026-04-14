"""
evaluate/metrics.py
===================
Trajectory evaluation metrics.

  ATE  – Absolute Trajectory Error   (after SE(3) alignment)
  RPE  – Relative Pose Error
  Segment analysis – ATE / RPE on high-jerk segments only
"""

import numpy as np
from typing import Optional


# ─────────────────────── Alignment ───────────────────────────────────────────

def umeyama_align(est: np.ndarray, gt: np.ndarray) -> tuple:
    """
    SE(3) alignment of estimated trajectory to ground truth.
    Minimises ‖ (s·R·p_est + t) − p_gt ‖²  (with scale s fixed = 1).

    Parameters
    ----------
    est : (N, 3) estimated positions
    gt  : (N, 3) ground-truth positions  (same timestamps)

    Returns
    -------
    est_aligned : (N, 3) aligned estimates
    R           : (3, 3) optimal rotation
    t           : (3,)   optimal translation
    """
    assert est.shape == gt.shape
    n = est.shape[0]
    mu_e = est.mean(axis=0)
    mu_g = gt.mean(axis=0)
    E = est - mu_e
    G = gt  - mu_g
    W = (G.T @ E) / n
    U, S, Vt = np.linalg.svd(W)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    t = mu_g - R @ mu_e
    est_aligned = (R @ est.T).T + t
    return est_aligned, R, t


# ─────────────────────── Core metrics ────────────────────────────────────────

def ate(est_pos: np.ndarray, gt_pos: np.ndarray,
        align: bool = True) -> dict:
    """
    Absolute Trajectory Error.

    Parameters
    ----------
    est_pos : (N, 3)
    gt_pos  : (N, 3)
    align   : bool  – SE(3)-align before computing error

    Returns
    -------
    dict with keys: rmse, mean, median, max, std, per_step [m]
    """
    if align:
        est_pos, _, _ = umeyama_align(est_pos, gt_pos)
    err = np.linalg.norm(est_pos - gt_pos, axis=1)
    return {
        "rmse":    float(np.sqrt(np.mean(err**2))),
        "mean":    float(np.mean(err)),
        "median":  float(np.median(err)),
        "max":     float(np.max(err)),
        "std":     float(np.std(err)),
        "per_step": err,
    }


def rpe(est_pos: np.ndarray, gt_pos: np.ndarray,
        delta: int = 10) -> dict:
    """
    Relative Pose Error (translation only).

    Parameters
    ----------
    est_pos : (N, 3)
    gt_pos  : (N, 3)
    delta   : int   – frame spacing for relative comparison

    Returns
    -------
    dict with keys: rmse, mean, median, max [m]
    """
    n = len(est_pos) - delta
    if n <= 0:
        return dict(rmse=np.nan, mean=np.nan, median=np.nan, max=np.nan)

    err = np.linalg.norm(
        (est_pos[delta:] - est_pos[:n]) - (gt_pos[delta:] - gt_pos[:n]),
        axis=1
    )
    return {
        "rmse":   float(np.sqrt(np.mean(err**2))),
        "mean":   float(np.mean(err)),
        "median": float(np.median(err)),
        "max":    float(np.max(err)),
    }


# ─────────────────────── Segment analysis ────────────────────────────────────

def segment_ate(est_pos: np.ndarray, gt_pos: np.ndarray,
                segments: list[tuple[int, int]],
                align: bool = True) -> dict:
    """
    Compute ATE restricted to the union of high-dynamic segments.

    Parameters
    ----------
    segments : list of (start_idx, end_idx) in the trajectory index space

    Returns
    -------
    dict with keys: rmse, mean, n_points, per_segment [list of dicts]
    """
    if not segments:
        return {"rmse": np.nan, "mean": np.nan, "n_points": 0,
                "per_segment": []}

    seg_metrics = []
    all_err = []
    for s, e in segments:
        ep = est_pos[s:e]
        gp = gt_pos[s:e]
        if len(ep) < 2:
            continue
        m = ate(ep, gp, align=align)
        seg_metrics.append(m)
        all_err.append(m["per_step"])

    if not all_err:
        return {"rmse": np.nan, "mean": np.nan, "n_points": 0,
                "per_segment": []}

    combined = np.concatenate(all_err)
    return {
        "rmse":        float(np.sqrt(np.mean(combined**2))),
        "mean":        float(np.mean(combined)),
        "n_points":    int(len(combined)),
        "per_segment": seg_metrics,
    }


# ─────────────────────── Interpolation helper ────────────────────────────────

def interpolate_gt(gt_t: np.ndarray, gt_pos: np.ndarray,
                   query_t: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of ground-truth positions to query timestamps.

    Parameters
    ----------
    gt_t    : (M,)    ground-truth timestamps
    gt_pos  : (M, 3)  ground-truth positions
    query_t : (N,)    query timestamps

    Returns
    -------
    pos_interp : (N, 3) interpolated positions
                 NaN for query times outside gt range
    """
    pos_interp = np.full((len(query_t), 3), np.nan)
    for i, t in enumerate(query_t):
        if t < gt_t[0] or t > gt_t[-1]:
            continue
        j = np.searchsorted(gt_t, t) - 1
        j = max(0, min(j, len(gt_t) - 2))
        frac = (t - gt_t[j]) / max(gt_t[j + 1] - gt_t[j], 1e-12)
        pos_interp[i] = gt_pos[j] * (1 - frac) + gt_pos[j + 1] * frac
    return pos_interp


def print_metrics(name: str, ate_d: dict, rpe_d: dict,
                  seg_d: Optional[dict] = None):
    """Pretty-print metric results."""
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  ATE  rmse={ate_d['rmse']:.3f} m  "
          f"mean={ate_d['mean']:.3f} m  "
          f"max={ate_d['max']:.3f} m")
    print(f"  RPE  rmse={rpe_d['rmse']:.3f} m  "
          f"mean={rpe_d['mean']:.3f} m")
    if seg_d and not np.isnan(seg_d.get("rmse", np.nan)):
        print(f"  Seg  rmse={seg_d['rmse']:.3f} m  "
              f"n_pts={seg_d['n_points']}")
    print(f"{'─'*50}")
