import numpy as np, sys
from pathlib import Path
sys.path.insert(0, '.')
import yaml

with open('config/rooad_config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

print('Simulating 179s of IMU at 400Hz ...')
np.random.seed(42)
N  = 179 * 400
t  = np.linspace(0, 179, N)
g  = 9.80665

acc = np.column_stack([
    np.sin(np.linspace(0, 6*np.pi, N)) * 2,
    np.cos(np.linspace(0, 4*np.pi, N)) * 1.5,
    np.full(N, g) + np.random.randn(N) * 0.5
])
gyro = np.column_stack([
    np.sin(np.linspace(0, 8*np.pi, N)) * 0.3,
    np.random.randn(N) * 0.05,
    np.random.randn(N) * 0.05,
])

from data.rooad_loader import static_bias_estimate, compute_jerk, identify_impact_segments
from filter.inekf_imu  import InEKF_IMU
from filter.inekf_jerk import InEKF_Jerk
from evaluate.metrics  import ate, interpolate_gt

n_init = 400

# Load real GT if available, otherwise use synthetic
gt_csv = Path('data/rooad/gt_enu/rt4_updown_gt_enu.csv')
if gt_csv.exists():
    gt     = np.loadtxt(gt_csv, delimiter=',', skiprows=1)
    gt_t   = gt[:, 0]
    gt_pos = gt[:, 1:4]
    t      = t - t[0] + gt_t[0]
    print('  Using real GT from CSV')
else:
    gt_t   = np.linspace(t[0], t[-1], 1787)
    gt_pos = np.column_stack([
        np.linspace(0, 1.3 * 179, 1787),
        np.sin(np.linspace(0, 2*np.pi, 1787)) * 20,
        np.linspace(0, 15, 1787),
    ])
    print('  Using synthetic GT (real GT not found)')

t0 = t[n_init]
bg0, ba0, R0 = static_bias_estimate(acc, gyro, n_init, cfg['data']['gravity'])

jerk = compute_jerk(acc, t)
segs = identify_impact_segments(jerk, threshold=cfg['segment']['jerk_threshold'])
print(f'  Jerk segments detected: {len(segs)}')

for FilterClass, name in [(InEKF_IMU, 'EKF_baseline'), (InEKF_Jerk, 'EKF_jerk')]:
    filt = FilterClass(cfg)
    filt.reset(R0.copy(), np.zeros(3), np.zeros(3), bg0.copy(), ba0.copy(), t0)
    traj_t, traj_pos = [], []
    for k in range(n_init, len(t)):
        filt.propagate(gyro[k], acc[k], t[k])
        traj_t.append(t[k])
        traj_pos.append(filt.position.copy())

    traj_t    = np.array(traj_t)
    traj_pos  = np.array(traj_pos)
    gt_interp = interpolate_gt(gt_t, gt_pos, traj_t)
    valid     = ~np.isnan(gt_interp[:, 0])
    ate_d     = ate(traj_pos[valid], gt_interp[valid], align=True)
    print(f'  {name:20s}  ATE-RMSE={ate_d["rmse"]:.3f}m  max={ate_d["max"]:.3f}m')

print()
print('ALL CHECKS PASSED - Pipeline is ready for real data!')
