# Impact-Aware State Estimation for Off-Road Motion

**GPS / IMU / Camera Fusion with Adaptive NHC Relaxation via Jerk Heuristic and LSTM**

> ROB 530 — Mobile Robotics | University of Michigan  
> Pranav Jain & Avantika Rattan | `{psjain, arattan}@umich.edu`

---

## Overview

Standard GPS/IMU filters use a **Non-Holonomic Constraint (NHC)** that assumes zero lateral and vertical vehicle velocity. On off-road terrain this assumption is violated — and critically, it *conflicts* with GPS corrections during hill traversal. When a GPS fix pushes the position estimate vertically, the NHC immediately zeros the resulting velocity correction. The filter oscillates catastrophically.

This project implements a **Right-Invariant EKF (InEKF) on SE₂(3)** with three fused sensors:

| Sensor | Rate | Role |
|---|---|---|
| IMU (VN300) | 400 Hz | Propagation |
| GPS (RTK) | 10 Hz | Position correction |
| Camera (Basler) | 30 Hz | Rotation correction via monocular VO |

And two adaptive mechanisms that learn **when to relax the NHC**:

| Filter | Method |
|---|---|
| **InEKF+Jerk** | Heuristic — relax NHC when IMU jerk > threshold |
| **InEKF+LSTM** | Learned — LSTM predicts Q-scale α and NHC relaxation β |

### Key Results

| Filter | Mean ATE | Improvement |
|---|---|---|
| InEKF Baseline | 6.326 m | — |
| **InEKF+Jerk** | **0.258 m** | **−96%** |
| InEKF+LSTM | 0.277 m | −96% |

InEKF+LSTM achieves the best RPE on 5 of 6 sequences, demonstrating that temporal context enables more precise NHC relaxation than the heuristic.

---

## Project Structure

```
rooad_inekf/
│
├── config/
│   └── rooad_config.yaml          ← All tunable parameters
│
├── data/
│   └── rooad_loader.py            ← Sequence loading, GPS→ENU, jerk, features
│
├── filter/
│   ├── lie_utils.py               ← SO(3) / SE₂(3) math utilities
│   ├── inekf_imu.py               ← Base InEKF + GPS update + NHC + VO update
│   ├── inekf_jerk.py              ← Jerk-adaptive α and β
│   ├── inekf_lstm.py              ← LSTM-adaptive α and β
│   └── visual_odometry.py         ← Monocular VO (LK optical flow + Essential matrix)
│
├── model/
│   ├── lstm_noise_model.py        ← LSTM architecture (dual α + β output heads)
│   └── train_lstm.py              ← O(N) label generation + LSTM training
│
├── evaluate/
│   └── metrics.py                 ← ATE, RPE, segment ATE, Umeyama alignment
│
├── scripts/
│   ├── extract_camera.py          ← Extract camera frames from bags → NPZ
│   └── tune_threshold.py          ← Analyze jerk statistics from IMU data
│
├── run_pipeline.py                ← Main runner — IMU + GPS + VO, all 3 filters
├── generate_plots.py              ← Generate all paper figures
├── eval_velocity.py               ← Velocity-space evaluation (speed, heading)
└── extract_imu.py                 ← Extract IMU + GT from ROS bags → CSV
```

---

## Setup

### 1 — Install Dependencies

```bash
pip install rosbags gdown pyproj pyyaml tqdm scipy matplotlib
pip install torch torchvision          # CUDA recommended for LSTM training
pip install opencv-python              # For Visual Odometry
```

### 2 — Download ROOAD Bags

Save all bags into a `bags/` folder in the project root.

| Sequence | Size | Split | Google Drive |
|---|---|---|---|
| rt4_gravel | 8 GB | Train | [Download](https://drive.google.com/file/d/1dKx6_A1V4wN_0NTKCLrWYgIwozsVrO0F) |
| rt4_rim | 5 GB | Train | [Download](https://drive.google.com/file/d/1m7y33UzYjT-1VgehGPSIzzcWltGRPb-N) |
| rt4_updown | 12 GB | Train | [Download](https://drive.google.com/file/d/1x-nKiURqvLhwyyHBCuPVEdS8MGo1VhOk) |
| rt5_gravel | 7 GB | Train | [Download](https://drive.google.com/file/d/1NBq-YU0YYuI1-D8DxSXdBeoWQ9hCOfj0) |
| rt5_rim | 5 GB | **Test** | [Download](https://drive.google.com/file/d/1sz33CuQ5rxQtYPe5DIpcOMTW9gpvu9Be) |
| rt5_updown | 10 GB | **Test** | [Download](https://drive.google.com/file/d/1Y1CjTEnbPadbg00uw0KLrydDc5-p9Cr0) |

> **Test sequences** (rt5_rim, rt5_updown) are never used during LSTM training — they evaluate generalisation to unseen data.

### 3 — Extract IMU and Ground Truth

```bash
python extract_imu.py
```

This reads all bags and creates:
- `data/rooad/imu/<seq>_imu.csv` — IMU at 400 Hz
- `data/rooad/gt_enu/<seq>_gt_enu.csv` — RTK GPS in ENU coordinates

### 4 — (Optional) Extract Camera for VO

Camera extraction is only needed for Visual Odometry. We extracted for the two updown sequences as a proof-of-concept. Each extraction takes ~10–15 minutes per bag.

```bash
# VO on updown sequences only (recommended)
python scripts/extract_camera.py --seqs rt4_updown rt5_updown

# VO on all 6 sequences
python scripts/extract_camera.py --seqs rt4_gravel rt4_rim rt4_updown rt5_gravel rt5_rim rt5_updown
```

Output: `data/rooad/camera/<seq>_camera.npz` (grayscale frames at 640×400)

---

## Training the LSTM

```bash
python -m model.train_lstm \
    --seqs rt4_gravel rt4_rim rt4_updown rt5_gravel \
    --data_dir data/rooad \
    --output model/
```

- **Input:** 50-sample IMU windows × 9 features `[ax, ay, az, gx, gy, gz, jerk, |a|, |ω|]`
- **Output:** α ∈ [0.1, 20] (Q scale) and β ∈ [0, 0.85] (NHC relaxation)
- **Labels:** Generated analytically in O(N) from jerk percentiles — no EKF re-runs
- **Training:** ~5 minutes on GPU (CUDA auto-detected)
- **Output model:** `model/lstm_noise_adapter.pt`

---

## Running the Pipeline

### GPS + IMU + VO (full pipeline)

```bash
python run_pipeline.py \
    --seq rt4_gravel rt4_rim rt4_updown rt5_gravel rt5_rim rt5_updown \
    --data_dir data/rooad
```

> VO is automatically skipped for sequences without extracted camera files.

### GPS + IMU only (no camera)

```bash
python run_pipeline.py \
    --seq rt4_gravel rt4_rim rt4_updown rt5_gravel rt5_rim rt5_updown \
    --data_dir data/rooad \
    --no-vo
```

### IMU only (no GPS, no camera)

```bash
python run_pipeline.py \
    --seq rt4_gravel rt4_rim rt4_updown rt5_gravel rt5_rim rt5_updown \
    --data_dir data/rooad \
    --no-gps --no-vo
```

### Skip LSTM (faster, baseline + jerk only)

```bash
python run_pipeline.py \
    --seq rt5_rim rt5_updown \
    --data_dir data/rooad \
    --skip-lstm
```

The terminal will always confirm active sensors:

```
Active sensors: IMU + GPS + VO
Active sensors: IMU + GPS
Active sensors: IMU-only
```

---

## Generating Plots

```bash
# All paper figures → plots/ folder
python generate_plots.py

# Velocity-space evaluation (speed RMSE, heading error)
python eval_velocity.py
```

### Output Figures

| File | Description |
|---|---|
| `fig1_trajectory_*.png` | Top-down + elevation trajectory comparison |
| `fig2_ate_over_time.png` | ATE over time — shows filter oscillation |
| `fig3_rpe_ate_bars.png` | Main RPE + ATE bar chart across all sequences |
| `fig4_signals_rt5_updown.png` | Jerk + α signal + ATE — shows LSTM vs Jerk discrimination |
| `fig5_segment_ate.png` | ATE restricted to high-dynamic impact segments |
| `fig6_velocity_metrics.png` | Speed RMSE, velocity RMSE, heading error |
| `fig7_velocity_time_*.png` | Speed, vertical velocity, velocity error over time |

---

## Configuration Reference

All parameters are in `config/rooad_config.yaml`:

| Section | Parameter | Default | Description |
|---|---|---|---|
| `ekf` | `sigma_gyro` | 3e-3 | Gyro noise [rad/s/√Hz] |
| `ekf` | `sigma_accel` | 0.05 | Accel noise [m/s²/√Hz] |
| `jerk` | `threshold` | 8.0 | Impact detection [m/s³] |
| `jerk` | `alpha_max` | 15.0 | Peak Q scale factor |
| `jerk` | `beta_max` | 0.85 | Peak NHC relaxation |
| `jerk` | `alpha_decay` | 0.95 | Q scale decay per step |
| `jerk` | `beta_decay` | 0.90 | NHC decay per step |
| `nhc` | `rate` | 10 | Apply NHC every N IMU steps |
| `nhc` | `sigma_lat` | 0.10 | Lateral velocity noise [m/s] |
| `gps` | `sigma_h` | 0.02 | GPS horizontal noise [m] (RTK) |
| `gps` | `sigma_v` | 0.05 | GPS vertical noise [m] (RTK) |
| `vo` | `sigma_rotation` | 0.15 | VO rotation noise [rad/axis] |
| `lstm` | `window` | 50 | LSTM input window [IMU steps] |
| `lstm` | `hidden_size` | 128 | LSTM hidden units |
| `lstm` | `features` | 9 | Input features per step |
| `training` | `epochs` | 200 | Training epochs |

---

## Why Visual Odometry is Only Active for 2 Sequences

Camera extraction was performed only for `rt4_updown` and `rt5_updown` as a proof-of-concept for the VO extension, because:

1. Each bag is 10–12 GB and extraction takes ~10–15 minutes
2. GPS-only already achieves sub-30 cm ATE on all sequences
3. VO results are mixed due to approximate camera-to-body extrinsics:
   - **rt5_updown:** VO helps — 19% ATE improvement
   - **rt4_updown:** VO hurts slightly — 16% ATE degradation

To enable VO on all sequences, run `scripts/extract_camera.py` on the remaining bags. Exact Kalibr extrinsic calibration from the ROOAD dataset would make VO consistently beneficial.

---

## Technical Background

### Why Right-Invariant EKF?

Standard EKF linearises around the current state estimate. On a tumbling off-road vehicle, large rotations make this linearisation inaccurate, causing inconsistent covariance estimates. The InEKF's error dynamics are **state-independent** — the Jacobian does not depend on the current state — yielding correct uncertainty even after large rotations.

### Why Adaptive NHC?

The NHC (`v_lateral = v_vertical = 0`) is violated during:
- Hill traversal — genuine vertical velocity
- Wheel bounce and rock impacts — transient vertical motion
- Lateral rock hits — large lateral accelerations

With GPS, forcing hard NHC creates a conflict: every 100 ms a GPS fix pushes position up/down the hill, generating a velocity correction, which the NHC zeros 25 ms later. The filter oscillates. Adaptive β > 0 permits the correction to persist, resolving the conflict.

### LSTM Architecture

```
Input  : (batch, 50, 9)   [ax, ay, az, gx, gy, gz, jerk, |a|, |ω|]
LSTM   : 2 layers, hidden=128, dropout=0.2
Head α : Linear → Sigmoid → α ∈ [0.1, 20]   (Q scale)
Head β : Linear → Sigmoid → β ∈ [0, 0.85]   (NHC relaxation)
Loss   : MSE(log α_pred, log α*) + MSE(β_pred, β*)
```

### VO Pipeline

```
Frame pair
  → FAST corner detection
  → Lucas-Kanade optical flow (21×21 window, 3 pyramid levels)
  → Bidirectional consistency check (reject if back-track error > 1px)
  → RANSAC Essential matrix (threshold=1.0px, confidence=99.9%)
  → recoverPose → ΔR_cam
  → Body frame: ΔR_body = R_cb^T · ΔR_cam · R_cb
  → EKF rotation update (σ_vo = 0.15 rad)

Quality gates (reject update if):
  - Inliers < 150
  - Rotation angle > 15° (tracking failure)
  - Jerk > 500 m/s³ (motion blur)
```

---

## Dataset: ROOAD

**RELLIS Off-road Odometry Analysis Dataset**  
George Chustz & Srikanth Saripalli, Texas A&M University  
[[arXiv]](https://arxiv.org/abs/2109.08228) | [[GitHub]](https://github.com/unmannedlab/ROOAD)

| Sensor | Specification |
|---|---|
| IMU | VectorNav VN300 @ 400 Hz — z-axis points DOWN (handled in initialisation) |
| GPS | Ardusimple RTK @ 10 Hz — ~2 cm horizontal, ~5 cm vertical accuracy |
| Camera | Basler Pylon 1920×1200 @ 30 Hz — fx=fy=1636.6 px |
| Platform | Clearpath Warthog UGV |
| Location | RELLIS Campus, Texas A&M University |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'filter.visual_odometry'`**
→ Make sure `visual_odometry.py` is saved in the `filter/` folder, not the project root.

**`EKF_LSTM` appears in results filenames instead of `InEKF_LSTM`**
→ Delete `results/*.npz` and re-run. The old run used an older version of `run_pipeline.py`.

**LSTM model not found**
```bash
python -m model.train_lstm --seqs rt4_gravel rt4_rim rt4_updown rt5_gravel --data_dir data/rooad
```

**VO makes results worse**
→ Expected with approximate extrinsics. Use `--no-vo` for best ATE. See VO section above.

**GT CSV columns not recognised**
→ Check column names in the raw GT file:
```python
import csv
print(list(csv.DictReader(open('data/rooad/raw_gt/rt4_updown_gt.csv')).fieldnames))
```
Expected: `%time`, `E`, `N`, `U`, `heading`

**PowerShell multi-line commands**
→ Use backtick `` ` `` for line continuation, not `\`

---

## Citation

```bibtex
@misc{jain2026impactaware,
  title  = {Impact-Aware State Estimation for Off-Road Motion:
            GPS/IMU Fusion with Adaptive NHC Relaxation},
  author = {Pranav Jain and Avantika Rattan},
  year   = {2026},
  note   = {ROB 530 Final Project, University of Michigan}
}
```

```bibtex
@misc{chustz2021rooad,
  title  = {ROOAD: RELLIS Off-road Odometry Analysis Dataset},
  author = {George Chustz and Srikanth Saripalli},
  year   = {2021},
  eprint = {2109.08228}
}
```

---

*Code released for academic use only.*  
*Dataset © Chustz & Saripalli, used under ROOAD dataset license.*
