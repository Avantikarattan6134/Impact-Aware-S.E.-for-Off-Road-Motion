"""
plot_results.py
===============
Visualise trajectories, ATE over time, jerk + α, and segment analysis.

Usage
-----
  python plot_results.py --results_dir results/ --bag seq1
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FILTER_STYLES = {
    "EKF_baseline": dict(color="#e74c3c", lw=1.5, ls="--",  label="EKF Baseline"),
    "EKF_jerk":     dict(color="#f39c12", lw=1.5, ls="-.",  label="EKF + Jerk"),
    "EKF_LSTM":     dict(color="#2ecc71", lw=2.0, ls="-",   label="EKF + LSTM"),
}
GT_STYLE = dict(color="#2c3e50", lw=2.0, ls="-", label="Ground Truth")


def load_npz(path: str) -> dict:
    return dict(np.load(path, allow_pickle=True))


def plot_trajectories(ax, results: dict, gt: dict, segs: list):
    """Top-down XY trajectory plot."""
    gp = gt["gt_pos"]
    ax.plot(gp[:, 0], gp[:, 1], **GT_STYLE, zorder=5)

    for fname, sty in FILTER_STYLES.items():
        if fname not in results:
            continue
        tp = results[fname]["traj_pos"]
        ax.plot(tp[:, 0], tp[:, 1], **sty, alpha=0.85)

    # shade impact segments (approximate — use fraction of total)
    if segs and len(results):
        first = next(iter(results.values()))
        tp = first["traj_pos"]
        for s, e in segs:
            if s < len(tp) and e < len(tp):
                ax.axvspan(tp[s, 0], tp[e, 0], alpha=0.12,
                           color="#e74c3c", label="_seg")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Trajectory (top-down)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_ate_over_time(ax, results: dict, gt: dict):
    """ATE as a function of time."""
    from evaluate.metrics import interpolate_gt, umeyama_align
    gp = gt["gt_pos"]
    gt_t = gt["gt_t"]

    for fname, sty in FILTER_STYLES.items():
        if fname not in results:
            continue
        r  = results[fname]
        tp = r["traj_pos"]
        tt = r["traj_t"]
        interp = interpolate_gt(gt_t, gp, tt)
        valid  = ~np.isnan(interp[:, 0])
        if valid.sum() < 2:
            continue
        est_v, _, _ = umeyama_align(tp[valid], interp[valid])
        err = np.linalg.norm(est_v - interp[valid], axis=1)
        ax.plot(tt[valid], err, **sty, alpha=0.75)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ATE [m]")
    ax.set_title("Absolute Trajectory Error vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_jerk_and_alpha(ax_j, ax_a, results: dict, cfg: dict):
    """Jerk magnitude and Q-scale α over time."""
    # Pick first filter result for jerk
    first = None
    for fname in ["INEKF_jerk", "INEKF_LSTM", "INEKF_baseline"]:
        if fname in results:
            first = results[fname]
            break
    if first is None:
        return

    jerk = first["jerk"]
    tt   = first["traj_t"]
    # align jerk length to tt
    min_len = min(len(jerk), len(tt))
    jerk = jerk[:min_len]
    tt   = tt[:min_len]

    ax_j.plot(tt, jerk, color="#7f8c8d", lw=1, alpha=0.7, label="Jerk mag")
    ax_j.axhline(cfg["jerk"]["threshold"], color="#e74c3c", ls="--",
                 lw=1.2, label=f"Threshold={cfg['jerk']['threshold']} m/s³")
    ax_j.set_ylabel("Jerk [m/s³]")
    ax_j.legend(fontsize=7)
    ax_j.grid(True, alpha=0.3)
    ax_j.set_title("Jerk magnitude")

    # Alpha logs
    for fname, sty in FILTER_STYLES.items():
        if fname == "EKF_baseline":
            continue
        if fname not in results:
            continue
        r = results[fname]
        if "alpha_log" in r:
            al = np.array(r["alpha_log"])
            ax_a.plot(tt[:len(al)], al[:len(tt)], **sty, alpha=0.85)

    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel("Q-scale α")
    ax_a.set_title("Adaptive Q scale (α)")
    ax_a.legend(fontsize=7)
    ax_a.grid(True, alpha=0.3)


def plot_segment_ate(ax, results: dict, gt: dict, segs: list):
    """Bar chart: ATE per segment for each filter."""
    from evaluate.metrics import interpolate_gt, umeyama_align

    if not segs:
        ax.set_visible(False)
        return

    gp  = gt["gt_pos"]
    gt_t= gt["gt_t"]
    n_segs = len(segs)
    x  = np.arange(n_segs)
    bw = 0.25
    offset = 0

    for fname, sty in FILTER_STYLES.items():
        if fname not in results:
            continue
        r  = results[fname]
        tp = r["traj_pos"]
        tt = r["traj_t"]
        interp = interpolate_gt(gt_t, gp, tt)
        valid  = ~np.isnan(interp[:, 0])

        seg_ates = []
        for s, e in segs:
            sv = valid[s:e]
            if sv.sum() < 2:
                seg_ates.append(np.nan)
                continue
            ep = tp[s:e][sv]
            gv = interp[s:e][sv]
            ep_a, _, _ = umeyama_align(ep, gv)
            seg_ates.append(np.mean(np.linalg.norm(ep_a - gv, axis=1)))

        ax.bar(x + offset * bw, seg_ates, bw, label=sty["label"],
               color=sty["color"], alpha=0.8)
        offset += 1

    ax.set_xticks(x + bw)
    ax.set_xticklabels([f"Seg {i+1}" for i in range(n_segs)])
    ax.set_ylabel("ATE [m]")
    ax.set_title("Per-segment ATE (high-dynamic regions)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def make_figure(results: dict, gt: dict, cfg: dict,
                save_path: str = None):
    """
    Assemble 2×2 figure with all four plots.
    """
    segs = []
    for r in results.values():
        if "segs" in r:
            segs = r["segs"]
            break

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ROOAD Impact-Aware InEKF Results", fontsize=14, y=1.01)

    plot_trajectories(axes[0, 0], results, gt, segs)
    plot_ate_over_time(axes[0, 1], results, gt)
    plot_jerk_and_alpha(axes[1, 0], axes[1, 1], results, cfg)

    # Add segment ATE as inset in top-right if segs exist
    if segs:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        plot_segment_ate(ax2, results, gt, segs)
        fig2.tight_layout()
        if save_path:
            seg_path = save_path.replace(".png", "_segments.png")
            fig2.savefig(seg_path, dpi=150, bbox_inches="tight")
            print(f"Segment plot saved → {seg_path}")
        plt.close(fig2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    else:
        plt.show()


# ─────────────────────── CLI ─────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/rooad_config.yaml")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--bag",         required=True,
                        help="Bag stem name, e.g. 'seq1'")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rdir = Path(args.results_dir)
    gt_data = load_npz(str(rdir / f"{args.bag}_gt.npz"))

    results = {}
    for fname in ["EKF_baseline", "EKF_jerk", "EKF_LSTM"]:
        p = rdir / f"{args.bag}_{fname}.npz"
        if p.exists():
            results[fname] = load_npz(str(p))
        else:
            print(f"[plot] Not found: {p}")

    save_path = str(rdir / f"{args.bag}_results.png")
    make_figure(results, gt_data, cfg, save_path=save_path)
