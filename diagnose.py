"""
diagnose.py
===========
Inspect raw IMU data and diagnose initialization quality.
"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, '.')
import yaml

with open('config/rooad_config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

seq    = 'rt4_updown'
imu    = np.loadtxt(f'data/rooad/imu/{seq}_imu.csv',    delimiter=',', skiprows=1)
gt     = np.loadtxt(f'data/rooad/gt_enu/{seq}_gt_enu.csv', delimiter=',', skiprows=1)

imu_t    = imu[:, 0]
imu_acc  = imu[:, 1:4]
imu_gyro = imu[:, 4:7]
gt_t     = gt[:, 0]
gt_pos   = gt[:, 1:4]

print("=" * 60)
print("  IMU / GT Timestamp Alignment")
print("=" * 60)
print(f"  IMU  t0={imu_t[0]:.3f}  t1={imu_t[-1]:.3f}  dur={imu_t[-1]-imu_t[0]:.1f}s")
print(f"  GT   t0={gt_t[0]:.3f}  t1={gt_t[-1]:.3f}  dur={gt_t[-1]-gt_t[0]:.1f}s")
overlap = min(imu_t[-1], gt_t[-1]) - max(imu_t[0], gt_t[0])
print(f"  Overlap: {overlap:.1f}s")

print()
print("=" * 60)
print("  IMU Static Analysis (first 1s / 5s / 10s)")
print("=" * 60)
g_true = 9.80665
for n in [400, 2000, 4000]:
    a_mean = imu_acc[:n].mean(axis=0)
    a_std  = imu_acc[:n].std(axis=0)
    g_mean = imu_gyro[:n].mean(axis=0)
    a_mag  = np.linalg.norm(a_mean)
    print(f"\n  First {n} samples ({n/400:.0f}s):")
    print(f"    accel mean: [{a_mean[0]:+.4f}, {a_mean[1]:+.4f}, {a_mean[2]:+.4f}]  |a|={a_mag:.4f} (g={g_true:.4f})")
    print(f"    accel std:  [{a_std[0]:.4f}, {a_std[1]:.4f}, {a_std[2]:.4f}]")
    print(f"    gyro  mean: [{g_mean[0]:+.5f}, {g_mean[1]:+.5f}, {g_mean[2]:+.5f}] rad/s")
    ba_est = a_mean - g_true * (a_mean / a_mag)
    print(f"    accel bias est: [{ba_est[0]:+.4f}, {ba_est[1]:+.4f}, {ba_est[2]:+.4f}]  |ba|={np.linalg.norm(ba_est):.4f}")

print()
print("=" * 60)
print("  Jerk / Motion at start (detecting static window)")
print("=" * 60)
from data.rooad_loader import compute_jerk, identify_impact_segments
jerk = compute_jerk(imu_acc, imu_t, smooth_window=5)
segs = identify_impact_segments(jerk, threshold=cfg['segment']['jerk_threshold'])

# Find first quiet window of 400+ samples
quiet_start = 0
window = 400
for i in range(len(jerk) - window):
    if np.all(jerk[i:i+window] < 2.0):   # very quiet: <2 m/s³
        quiet_start = i
        break

print(f"  First quiet window (jerk < 2 m/s³): starts at sample {quiet_start} ({quiet_start/400:.2f}s)")
print(f"  Jerk at t=0..1s: min={jerk[:400].min():.2f}  max={jerk[:400].max():.2f}  mean={jerk[:400].mean():.2f} m/s^3")

a_quiet = imu_acc[quiet_start:quiet_start+window]
g_quiet = imu_gyro[quiet_start:quiet_start+window]
a_mean  = a_quiet.mean(axis=0)
a_mag   = np.linalg.norm(a_mean)
ba_est  = a_mean - g_true * (a_mean / a_mag)
print(f"\n  Best static window (sample {quiet_start}):")
print(f"    accel mean: [{a_mean[0]:+.4f}, {a_mean[1]:+.4f}, {a_mean[2]:+.4f}]  |a|={a_mag:.4f}")
print(f"    gyro  mean: [{g_quiet.mean(axis=0)[0]:+.5f}, {g_quiet.mean(axis=0)[1]:+.5f}, {g_quiet.mean(axis=0)[2]:+.5f}]")
print(f"    accel bias: [{ba_est[0]:+.4f}, {ba_est[1]:+.4f}, {ba_est[2]:+.4f}]  |ba|={np.linalg.norm(ba_est):.4f}")

print()
print("=" * 60)
print("  Short-window drift test (first 5s only)")
print("=" * 60)
from filter.inekf_imu import InEKF_IMU
from scipy.spatial.transform import Rotation as R_scipy

# Use best static window for init
bg0 = g_quiet.mean(axis=0)
a_norm = a_mean / a_mag
ba0 = ba_est

axis  = np.cross(a_norm, np.array([0., 0., 1.]))
anorm = np.linalg.norm(axis)
if anorm < 1e-9:
    R0 = np.eye(3)
else:
    angle = np.arcsin(np.clip(anorm, -1, 1))
    R0 = R_scipy.from_rotvec(angle * axis / anorm).as_matrix()

print(f"  R0 (initial rotation):\n{R0.round(4)}")
print(f"  bg0: {bg0.round(6)}")
print(f"  ba0: {ba0.round(4)}")

filt = InEKF_IMU(cfg)
t0   = imu_t[quiet_start]
filt.reset(R0, np.zeros(3), np.zeros(3), bg0, ba0, t0)

N5 = min(quiet_start + 2000, len(imu_t))   # 5s from start
for k in range(quiet_start, N5):
    filt.propagate(imu_gyro[k], imu_acc[k], imu_t[k])

p5 = filt.position
v5 = filt.velocity
print(f"\n  After 5s of propagation:")
print(f"    position: [{p5[0]:+.3f}, {p5[1]:+.3f}, {p5[2]:+.3f}] m")
print(f"    velocity: [{v5[0]:+.3f}, {v5[1]:+.3f}, {v5[2]:+.3f}] m/s  |v|={np.linalg.norm(v5):.3f}")

# GT position at 5s
gt_5s = gt_pos[np.searchsorted(gt_t, t0 + 5) - 1]
gt_0s = gt_pos[np.searchsorted(gt_t, t0) - 1]
gt_disp = gt_5s - gt_0s
print(f"    GT displacement (5s): [{gt_disp[0]:+.3f}, {gt_disp[1]:+.3f}, {gt_disp[2]:+.3f}] m")
print(f"    Position error (5s): {np.linalg.norm(p5 - gt_disp):.3f} m")
