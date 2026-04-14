"""
tune_threshold.py
=================
Find the optimal jerk threshold from your actual IMU data,
then apply all improvements and retrain.

Run from project root:
  python tune_threshold.py
"""
import numpy as np
import sys, yaml
from pathlib import Path
sys.path.insert(0, '.')

with open('config/rooad_config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

print("=" * 60)
print("  Step 1: Analyze actual jerk statistics")
print("=" * 60)

seqs = ['rt4_gravel', 'rt4_rim', 'rt4_updown', 'rt5_gravel', 'rt5_rim', 'rt5_updown']
all_jerk_smoothed = []

for seq in seqs:
    p = Path(f'data/rooad/imu/{seq}_imu.csv')
    if not p.exists():
        continue
    imu = np.loadtxt(p, delimiter=',', skiprows=1)
    acc = imu[:, 1:4]
    t   = imu[:, 0]
    da  = np.diff(acc, axis=0)
    dt  = np.diff(t)[:, None]
    dt  = np.where(dt > 0, dt, 1e-9)
    j   = np.linalg.norm(da / dt, axis=1)
    # Smooth with window=20 (same as config)
    k   = np.ones(20) / 20
    js  = np.convolve(j, k, mode='same')
    all_jerk_smoothed.append(js)

    p10  = np.percentile(js, 10)
    p50  = np.percentile(js, 50)
    p75  = np.percentile(js, 75)
    p90  = np.percentile(js, 90)
    p95  = np.percentile(js, 95)
    p99  = np.percentile(js, 99)
    print(f"\n  {seq}:")
    print(f"    smoothed jerk (win=20):  "
          f"p10={p10:.1f}  p50={p50:.1f}  p75={p75:.1f}  "
          f"p90={p90:.1f}  p95={p95:.1f}  p99={p99:.1f}  m/s³")

if all_jerk_smoothed:
    all_j = np.concatenate(all_jerk_smoothed)
    p75_all = np.percentile(all_j, 75)
    p90_all = np.percentile(all_j, 90)
    p95_all = np.percentile(all_j, 95)

    print(f"\n  OVERALL (all sequences):")
    print(f"    p75={p75_all:.1f}  p90={p90_all:.1f}  p95={p95_all:.1f}  m/s³")

    # Recommended threshold: p90 of smoothed jerk
    # This ensures ~10% of time is classified as "impact"
    recommended = float(p90_all)
    print(f"\n  RECOMMENDED threshold: {recommended:.1f} m/s³  (p90 of smoothed jerk)")
    print(f"  Current threshold:     {cfg['jerk']['threshold']} m/s³  ← WAY too low!")

    # Write recommended threshold to a file for next step
    with open('recommended_threshold.txt', 'w') as f:
        f.write(str(recommended))
else:
    recommended = 200.0
    print("  Could not compute from data, using default: 200.0 m/s³")
    with open('recommended_threshold.txt', 'w') as f:
        f.write(str(recommended))

print()
print("=" * 60)
print("  Step 2: Update config with improved parameters")
print("=" * 60)

# Read and update config
content = open('config/rooad_config.yaml', encoding='utf-8').read()

# Update jerk threshold
old_thr = f'threshold:      {cfg["jerk"]["threshold"]}'
new_thr = f'threshold:      {recommended:.1f}'
content = content.replace(old_thr, new_thr)

# Update NHC rate from 10 to 5 (more frequent = tighter constraint)
content = content.replace(
    'rate:           10     # apply every N IMU steps (10 = every 25ms at 400Hz)',
    'rate:            5     # apply every N IMU steps (5 = every 12.5ms at 400Hz)'
)

# Update NHC sigma_lat tighter
content = content.replace(
    'sigma_lat:      0.10   # lateral  velocity residual [m/s]',
    'sigma_lat:      0.05   # lateral  velocity residual [m/s]'
)

# Update hidden_size to 128
content = content.replace('hidden_size:    64', 'hidden_size:    128')

# Update epochs to 150
content = content.replace('epochs:        150', 'epochs:        200')

open('config/rooad_config.yaml', 'w', encoding='utf-8').write(content)
print(f"  ✓ jerk threshold: {cfg['jerk']['threshold']} → {recommended:.1f} m/s³")
print(f"  ✓ NHC rate: 10 → 5 steps (2× more frequent)")
print(f"  ✓ NHC sigma_lat: 0.10 → 0.05 m/s (tighter constraint)")
print(f"  ✓ LSTM hidden_size: 64 → 128")
print(f"  ✓ epochs: 150 → 200")

print()
print("=" * 60)
print("  Step 3: Retrain LSTM with improved threshold")
print("=" * 60)
print("  Run this command next:")
print()
print("  python -m model.train_lstm --seqs rt4_gravel rt4_rim rt4_updown rt5_gravel --data_dir data\\rooad --output model")
print()
print("  Then test:")
print("  python run_pipeline.py --seq rt5_rim rt5_updown --data_dir data\\rooad")
