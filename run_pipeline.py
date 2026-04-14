"""
run_pipeline.py  — GPS + Visual Odometry + IMU pipeline
========================================================
Three sensors fused in the InEKF:
  IMU  @ 400 Hz  — propagation (always)
  GPS  @  10 Hz  — position update (optional)
  VO   @  30 Hz  — rotation update from monocular camera (optional)

Usage
-----
  # Full GPS+VO pipeline
  python run_pipeline.py --seq rt4_updown rt5_updown --data_dir data/rooad

  # GPS only (no camera)
  python run_pipeline.py --seq rt5_updown --data_dir data/rooad --no-vo

  # IMU only (no GPS, no camera)
  python run_pipeline.py --seq rt5_updown --data_dir data/rooad --no-gps --no-vo
"""

import argparse, sys, yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

from data.rooad_loader import (
    load_sequence, load_bag, static_bias_estimate,
    compute_jerk, identify_impact_segments
)
from filter.inekf_imu  import InEKF_IMU
from filter.inekf_jerk import InEKF_Jerk
from filter.inekf_lstm import InEKF_LSTM
from evaluate.metrics  import (
    ate, rpe, segment_ate, interpolate_gt, print_metrics
)


# ── Camera-to-body extrinsic for ROOAD Warthog ────────────────────────────────
# Camera faces forward, IMU z-down.
# Approximate: camera x=forward, y=left, z=up in body frame.
# (Exact calibration from ROOAD kalibr results, approx here)
R_CAM_BODY = np.array([
    [ 0.0, -1.0,  0.0],   # cam_x = -body_y
    [ 0.0,  0.0, -1.0],   # cam_y = -body_z
    [ 1.0,  0.0,  0.0],   # cam_z = +body_x (forward)
], dtype=float)


# ── Load camera data ──────────────────────────────────────────────────────────

def load_camera(seq: str, data_dir: str):
    """Load pre-extracted camera NPZ. Returns (timestamps, frames) or None."""
    p = Path(data_dir) / "camera" / f"{seq}_camera.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["timestamps"], d["frames"]


# ── Core runner ───────────────────────────────────────────────────────────────

def _run_data(data: dict, cfg: dict, filters: dict,
              cam_t=None, cam_frames=None,
              use_gps: bool = True,
              use_vo:  bool = True) -> dict:

    imu_t    = data["imu_t"]
    imu_acc  = data["imu_acc"]
    imu_gyro = data["imu_gyro"]
    gt_t     = data["gt_t"]
    gt_pos   = data["gt_pos"]

    if len(gt_t) < 2:
        print("  [warn] No GT — skipping.")
        return {}

    n_init = int(cfg["init"]["static_init_samples"])
    bg0, ba0, R0 = static_bias_estimate(
        imu_acc, imu_gyro, n_init, cfg["data"]["gravity"]
    )
    p0 = gt_pos[0].copy() if use_gps else np.zeros(3)
    t0 = imu_t[n_init]

    jerk = compute_jerk(imu_acc, imu_t,
                        smooth_window=cfg["segment"]["smooth_window"])
    segs = identify_impact_segments(
        jerk, threshold=cfg["segment"]["jerk_threshold"],
        min_len=cfg["segment"]["min_segment_len"]
    )
    print(f"  Found {len(segs)} high-dynamic segments")

    # Sensor configs
    gps_cfg  = cfg.get("gps", {})
    sh       = float(gps_cfg.get("sigma_h", 0.02))
    sv       = float(gps_cfg.get("sigma_v", 0.05))
    R_gps    = np.diag([sh**2, sh**2, sv**2])

    vo_cfg   = cfg.get("vo", {})
    sigma_vo = float(vo_cfg.get("sigma_rotation", 0.02))

    # VO frontend (one per filter run to avoid shared state)
    vo_available = use_vo and cam_t is not None and cam_frames is not None
    if vo_available:
        from filter.visual_odometry import MonocularVO
        print(f"  VO: {len(cam_t)} camera frames available")
    else:
        if use_vo:
            print("  VO: camera not extracted — run scripts/extract_camera.py")

    results = {}

    for fname, filt in filters.items():
        print(f"  Running {fname} …")
        filt.reset(R0.copy(), np.zeros(3), p0.copy(), bg0.copy(), ba0.copy(), t0)

        # Fresh VO frontend per filter
        vo = MonocularVO() if vo_available else None

        traj_t, traj_pos = [], []
        gps_idx = 0
        cam_idx = 0

        # Fast-forward GPS index to t0
        while gps_idx < len(gt_t) and gt_t[gps_idx] < t0:
            gps_idx += 1
        # Fast-forward camera index to t0
        if vo_available:
            while cam_idx < len(cam_t) and cam_t[cam_idx] < t0:
                cam_idx += 1

        for k in tqdm(range(n_init, len(imu_t)),
                      desc=f"    {fname}", leave=False):
            t_k = imu_t[k]

            # ── GPS update ────────────────────────────────────────────────
            if use_gps:
                while gps_idx < len(gt_t) and gt_t[gps_idx] <= t_k:
                    filt.update_gps(gt_pos[gps_idx], R_gps)
                    gps_idx += 1

            # ── Visual Odometry update ────────────────────────────────────
            if vo is not None:
                while cam_idx < len(cam_t) and cam_t[cam_idx] <= t_k:
                    frame = cam_frames[cam_idx]
                    delta_R = vo.process_frame(frame, cam_t[cam_idx])
                    if delta_R is not None:
                        filt.update_visual_rotation(
                            delta_R, R_CAM_BODY, sigma_vo
                        )
                    cam_idx += 1

            # ── IMU propagation ───────────────────────────────────────────
            filt.propagate(imu_gyro[k], imu_acc[k], t_k)
            traj_t.append(t_k)
            traj_pos.append(filt.position.copy())

        traj_t   = np.array(traj_t)
        traj_pos = np.array(traj_pos)

        gt_interp = interpolate_gt(gt_t, gt_pos, traj_t)
        valid     = ~np.isnan(gt_interp[:, 0])
        if valid.sum() < 2:
            print(f"  [warn] Not enough GT for {fname}")
            continue

        ate_d = ate(traj_pos[valid], gt_interp[valid],
                    align=cfg["eval"]["ate_align"])
        rpe_d = rpe(traj_pos[valid], gt_interp[valid],
                    delta=cfg["eval"]["rpe_delta"])
        seg_d = segment_ate(traj_pos[valid], gt_interp[valid], segs)

        if vo is not None:
            total   = len(vo.success_log)
            success = sum(vo.success_log)
            print(f"    VO success: {success}/{total} frames "
                  f"({100*success/max(total,1):.0f}%)")

        results[fname] = {
            "ate": ate_d, "rpe": rpe_d, "seg": seg_d,
            "traj_t": traj_t, "traj_pos": traj_pos,
            "jerk": jerk, "segs": segs,
            "alpha_log": getattr(filt, "alpha_log", []),
        }
        print_metrics(fname, ate_d, rpe_d, seg_d)

    results["__gt__"] = {"gt_t": gt_t, "gt_pos": gt_pos}
    return results


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_results(results, seq_name, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for fname, res in results.items():
        if fname.startswith("__"):
            np.savez(Path(out_dir) / f"{seq_name}_gt.npz",
                     gt_t=res["gt_t"], gt_pos=res["gt_pos"])
            continue
        np.savez(
            Path(out_dir) / f"{seq_name}_{fname}.npz",
            traj_t=res["traj_t"], traj_pos=res["traj_pos"],
            jerk=res["jerk"],
            alpha_log=np.array(res.get("alpha_log", [])),
        )
    print(f"  Results saved → {out_dir}")


def load_lstm_filter(cfg):
    import torch
    from model.lstm_noise_model import LSTMNoiseAdapter
    model_path = cfg["lstm"]["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"LSTM model not found: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)
    model  = LSTMNoiseAdapter(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    feat_mean = ckpt.get("feat_mean", None)
    feat_std  = ckpt.get("feat_std",  None)
    if feat_mean is not None:
        _orig = model.predict
        def predict_normed(w, device=device):
            return _orig((w - feat_mean) / (feat_std + 1e-8), device=device)
        model.predict = predict_normed
    print(f"  LSTM loaded from {model_path} (device: {device})")
    return InEKF_LSTM(cfg, model, device=device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPS + Visual Odometry + IMU InEKF Pipeline"
    )
    parser.add_argument("--config",    default="config/rooad_config.yaml")
    parser.add_argument("--seq",       nargs="+", default=None)
    parser.add_argument("--data_dir",  default="data/rooad/")
    parser.add_argument("--bag",       default=None)
    parser.add_argument("--gt_csv",    default=None)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--no-gps",    action="store_true")
    parser.add_argument("--no-vo",     action="store_true")
    parser.add_argument("--output",    default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.no_gps:
        cfg.setdefault("gps", {})["enabled"] = False
        print("  [GPS disabled]")

    out_dir  = args.output or cfg["output"]["results_dir"]
    use_gps  = not args.no_gps
    use_vo   = not args.no_vo

    sensors = []
    if use_gps: sensors.append("GPS")
    if use_vo:  sensors.append("VO")
    if not sensors: sensors.append("IMU-only")
    print(f"\nActive sensors: IMU + {' + '.join(sensors)}\n")

    # Filters
    filters = {
        "InEKF_baseline": InEKF_IMU(cfg),
        "InEKF_jerk":     InEKF_Jerk(cfg),
    }
    if not args.skip_lstm:
        try:
            filters["InEKF_LSTM"] = load_lstm_filter(cfg)
        except FileNotFoundError as e:
            print(f"\n[!] {e}\n")

    all_results = {}

    sequences = args.seq
    if not sequences and not args.bag:
        imu_dir = Path(args.data_dir) / "imu"
        if imu_dir.exists():
            sequences = [p.stem.replace("_imu","")
                         for p in sorted(imu_dir.glob("*_imu.csv"))]
        if not sequences:
            print("No sequences found. Run setup_dataset.py first.")
            sys.exit(1)

    for seq in (sequences or []):
        print(f"\n{'═'*60}\n  Sequence: {seq}\n{'═'*60}")
        try:
            data = load_sequence(seq, args.data_dir, cfg)
        except FileNotFoundError as e:
            print(f"  [!] {e}"); continue

        # Load camera if available
        cam_t, cam_frames = (None, None)
        if use_vo:
            cam_t, cam_frames = load_camera(seq, args.data_dir)
            if cam_t is not None:
                print(f"  Camera: {len(cam_t)} frames loaded")
            else:
                print(f"  Camera: not extracted (run scripts/extract_camera.py --seqs {seq})")

        res = _run_data(data, cfg, filters,
                        cam_t=cam_t, cam_frames=cam_frames,
                        use_gps=use_gps, use_vo=use_vo)
        if res:
            all_results[seq] = res
            save_results(res, seq, out_dir)

    if all_results:
        print(f"\n{'═'*60}\n  AGGREGATE SUMMARY\n{'═'*60}")
        for fname in filters.keys():
            ates = [r[fname]["ate"]["rmse"]
                    for r in all_results.values() if fname in r]
            if ates:
                print(f"  {fname:20s}  ATE = "
                      f"{np.mean(ates):.3f} ± {np.std(ates):.3f} m")
    print("\nDone.")


if __name__ == "__main__":
    main()
