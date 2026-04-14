"""
eval_velocity.py
================
Evaluate filters in velocity space instead of position space.

Metrics:
  - Speed RMSE: |v_est| vs |v_gt|
  - Velocity vector RMSE: ||v_est - v_gt||
  - Heading error: angle between estimated and GT heading over time
  - Velocity direction error: angle between v_est and v_gt vectors

GT velocity is derived by finite-differencing RTK positions (10 Hz).

Run from project root:
  python eval_velocity.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import yaml

with open('config/rooad_config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

RESULTS_DIR = Path('results')
PLOTS_DIR   = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

SEQUENCES = ['rt4_gravel', 'rt4_rim', 'rt4_updown',
             'rt5_gravel', 'rt5_rim', 'rt5_updown']

FILTER_STYLES = {
    'InEKF_baseline': dict(color='#e74c3c', lw=2.0, ls='--', label='InEKF Baseline'),
    'InEKF_jerk':     dict(color='#f39c12', lw=2.0, ls='-.', label='InEKF + Jerk'),
    'InEKF_LSTM':     dict(color='#27ae60', lw=2.5, ls='-',  label='InEKF + LSTM'),
}

TERRAIN_LABELS = {
    'rt4_gravel': 'Gravel (rt4)', 'rt4_rim':    'Rim (rt4)',
    'rt4_updown': 'Updown (rt4)', 'rt5_gravel': 'Gravel (rt5)',
    'rt5_rim':    'Rim (rt5)',    'rt5_updown': 'Updown (rt5)',
}


# ─────────────────────── GT velocity from RTK ────────────────────────────────

def load_gt_velocity(seq):
    """
    Load GT position and compute velocity by finite difference.
    Returns gt_t, gt_pos, gt_vel (all Nx3).
    """
    gt_csv = Path(f'data/rooad/gt_enu/{seq}_gt_enu.csv')
    if not gt_csv.exists():
        return None, None, None

    gt = np.loadtxt(gt_csv, delimiter=',', skiprows=1)
    gt_t   = gt[:, 0]
    gt_pos = gt[:, 1:4]

    # Central finite difference velocity (smooth, 10 Hz)
    gt_vel = np.zeros_like(gt_pos)
    for i in range(1, len(gt_t)-1):
        dt = gt_t[i+1] - gt_t[i-1]
        if dt > 0:
            gt_vel[i] = (gt_pos[i+1] - gt_pos[i-1]) / dt
    gt_vel[0]  = gt_vel[1]
    gt_vel[-1] = gt_vel[-2]

    # Smooth velocity (box filter over 3 samples = 0.3s)
    from numpy import convolve, ones
    for d in range(3):
        gt_vel[:, d] = convolve(gt_vel[:, d], ones(3)/3, mode='same')

    return gt_t, gt_pos, gt_vel


def interp_gt_vel(gt_t, gt_vel, query_t):
    """Interpolate GT velocity to filter timestamps."""
    f = interp1d(gt_t, gt_vel, axis=0, bounds_error=False,
                 fill_value=(gt_vel[0], gt_vel[-1]))
    return f(query_t)


# ─────────────────────── Velocity metrics ────────────────────────────────────

def velocity_metrics(v_est, v_gt):
    """
    Compute velocity error metrics.

    Parameters
    ----------
    v_est : (N, 3) estimated velocity
    v_gt  : (N, 3) ground truth velocity

    Returns
    -------
    dict with:
      speed_rmse      : RMSE of speed (scalar)
      vel_rmse        : RMSE of velocity vector magnitude
      heading_err_deg : RMSE of heading angle error [degrees]
      vel_dir_err_deg : RMSE of 3D velocity direction error [degrees]
    """
    # Speed error
    spd_est = np.linalg.norm(v_est, axis=1)
    spd_gt  = np.linalg.norm(v_gt,  axis=1)
    speed_rmse = float(np.sqrt(np.mean((spd_est - spd_gt)**2)))

    # Velocity vector error
    vel_rmse = float(np.sqrt(np.mean(np.sum((v_est - v_gt)**2, axis=1))))

    # Heading error (yaw, from horizontal velocity)
    hdg_est = np.arctan2(v_est[:, 1], v_est[:, 0])   # atan2(N, E)
    hdg_gt  = np.arctan2(v_gt[:,  1], v_gt[:,  0])
    hdg_err = np.abs(np.arctan2(np.sin(hdg_est - hdg_gt),
                                 np.cos(hdg_est - hdg_gt)))
    # Only evaluate when GT speed > 0.3 m/s (heading undefined when static)
    moving  = spd_gt > 0.3
    heading_err_deg = float(np.degrees(np.sqrt(np.mean(hdg_err[moving]**2)))) \
                      if moving.sum() > 0 else float('nan')

    # 3D velocity direction error
    norm_est = spd_est + 1e-9
    norm_gt  = spd_gt  + 1e-9
    cos_ang  = np.clip(np.sum(v_est * v_gt, axis=1) / (norm_est * norm_gt), -1, 1)
    dir_err  = np.degrees(np.arccos(cos_ang))
    vel_dir_err_deg = float(np.sqrt(np.mean(dir_err[moving]**2))) \
                      if moving.sum() > 0 else float('nan')

    return dict(
        speed_rmse=speed_rmse,
        vel_rmse=vel_rmse,
        heading_err_deg=heading_err_deg,
        vel_dir_err_deg=vel_dir_err_deg,
    )


# ─────────────────────── Per-sequence evaluation ─────────────────────────────

def evaluate_sequence(seq):
    print(f'\n  {seq}:')
    gt_t, gt_pos, gt_vel = load_gt_velocity(seq)
    if gt_t is None:
        print('    GT not found'); return None

    results = {}
    for fname in ['InEKF_baseline', 'InEKF_jerk', 'InEKF_LSTM']:
        p = RESULTS_DIR / f'{seq}_{fname}.npz'
        if not p.exists():
            continue
        r       = dict(np.load(p, allow_pickle=True))
        traj_t  = r['traj_t']
        traj_pos= r['traj_pos']

        # Estimate filter velocity by finite difference of position
        v_est = np.zeros_like(traj_pos)
        for i in range(1, len(traj_t)-1):
            dt = traj_t[i+1] - traj_t[i-1]
            if dt > 0:
                v_est[i] = (traj_pos[i+1] - traj_pos[i-1]) / dt
        v_est[0] = v_est[1]; v_est[-1] = v_est[-2]

        # Smooth (1s box filter)
        from numpy import convolve, ones
        k = 400
        for d in range(3):
            v_est[:, d] = convolve(v_est[:, d], ones(k)/k, mode='same')

        # Interpolate GT to filter timestamps
        v_gt_interp = interp_gt_vel(gt_t, gt_vel, traj_t)

        # Only evaluate where GT is valid
        valid = ((traj_t >= gt_t[0]) & (traj_t <= gt_t[-1]))
        m = velocity_metrics(v_est[valid], v_gt_interp[valid])
        results[fname] = dict(metrics=m, v_est=v_est,
                              v_gt=v_gt_interp, traj_t=traj_t, valid=valid)

        print(f'    {fname:20s}  '
              f'speed_rmse={m["speed_rmse"]:.3f}m/s  '
              f'vel_rmse={m["vel_rmse"]:.3f}m/s  '
              f'hdg_err={m["heading_err_deg"]:.2f}°')

    return results


# ─────────────────────── Figure: velocity RMSE bar chart ─────────────────────

def fig_velocity_bars(all_metrics):
    print('\n  Plotting velocity bar chart...')
    seqs    = [s for s in SEQUENCES if s in all_metrics]
    fnames  = ['InEKF_baseline', 'InEKF_jerk', 'InEKF_LSTM']
    metrics = ['speed_rmse', 'vel_rmse', 'heading_err_deg']
    titles  = ['Speed RMSE [m/s]', 'Velocity Vector RMSE [m/s]', 'Heading Error [deg]']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Velocity-Space Evaluation Across All Sequences', fontsize=13)

    x  = np.arange(len(seqs)); bw = 0.25

    for ax, metric, title in zip(axes, metrics, titles):
        for j, fname in enumerate(fnames):
            sty  = FILTER_STYLES[fname]
            vals = []
            for s in seqs:
                if fname in all_metrics[s]:
                    vals.append(all_metrics[s][fname]['metrics'][metric])
                else:
                    vals.append(np.nan)
            bars = ax.bar(x + (j-1)*bw, vals, bw,
                          label=sty['label'], color=sty['color'], alpha=0.85)
            # Value labels
            for k, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(k+(j-1)*bw, v+v*0.02, f'{v:.3f}' if v < 1 else f'{v:.1f}',
                            ha='center', va='bottom', fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([TERRAIN_LABELS[s] for s in seqs],
                           rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(title); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = PLOTS_DIR / 'fig6_velocity_metrics.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {out}')


# ─────────────────────── Figure: speed over time ─────────────────────────────

def fig_speed_over_time(all_metrics, seq='rt5_updown'):
    print(f'  Plotting speed over time ({seq})...')
    if seq not in all_metrics:
        print('    Skipping'); return

    gt_t, _, gt_vel = load_gt_velocity(seq)
    if gt_t is None: return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Velocity Analysis — {TERRAIN_LABELS[seq]}', fontsize=13)

    res = all_metrics[seq]

    # ── Speed over time ───────────────────────────────────────────────────────
    ax = axes[0]
    gt_spd = np.linalg.norm(gt_vel, axis=1)
    ax.plot(gt_t - gt_t[0], gt_spd, color='#2c3e50', lw=2, label='GT Speed', zorder=5)
    for fname, sty in FILTER_STYLES.items():
        if fname not in res: continue
        r   = res[fname]
        tt  = r['traj_t'] - r['traj_t'][0]
        spd = np.linalg.norm(r['v_est'], axis=1)
        ax.plot(tt, spd, color=sty['color'], lw=1.5,
                ls=sty['ls'], label=sty['label'], alpha=0.8)
    ax.set_ylabel('Speed [m/s]')
    ax.set_title('Forward Speed')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Vertical velocity ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(gt_t - gt_t[0], gt_vel[:, 2], color='#2c3e50', lw=2,
            label='GT Vertical Vel', zorder=5)
    for fname, sty in FILTER_STYLES.items():
        if fname not in res: continue
        r  = res[fname]
        tt = r['traj_t'] - r['traj_t'][0]
        ax.plot(tt, r['v_est'][:, 2], color=sty['color'], lw=1.5,
                ls=sty['ls'], label=sty['label'], alpha=0.8)
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_ylabel('Vertical Vel [m/s]')
    ax.set_title('Vertical Velocity (key for updown terrain)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Speed error over time ─────────────────────────────────────────────────
    ax = axes[2]
    for fname, sty in FILTER_STYLES.items():
        if fname not in res: continue
        r    = res[fname]
        tt   = r['traj_t'] - r['traj_t'][0]
        v_gt = r['v_gt']
        v_es = r['v_est']
        err  = np.linalg.norm(v_es - v_gt, axis=1)
        # Smooth
        from numpy import convolve, ones
        err_s = convolve(err, ones(400)/400, mode='same')
        ax.plot(tt, err_s, color=sty['color'], lw=1.5,
                ls=sty['ls'], label=sty['label'], alpha=0.85)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity Error [m/s]')
    ax.set_title('Velocity Vector Error over Time (smoothed)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f'fig7_velocity_time_{seq}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {out}')


# ─────────────────────── Summary table ───────────────────────────────────────

def print_summary_table(all_metrics):
    print('\n' + '='*80)
    print('  VELOCITY METRICS SUMMARY')
    print('='*80)
    print(f'  {"Sequence":15s}  {"Filter":20s}  '
          f'{"Speed RMSE":>12s}  {"Vel RMSE":>10s}  {"Hdg Err":>9s}')
    print('  ' + '-'*70)

    for seq in SEQUENCES:
        if seq not in all_metrics: continue
        for fname in ['InEKF_baseline', 'InEKF_jerk', 'InEKF_LSTM']:
            if fname not in all_metrics[seq]: continue
            m = all_metrics[seq][fname]['metrics']
            print(f'  {seq:15s}  {fname:20s}  '
                  f'{m["speed_rmse"]:>10.4f}m/s  '
                  f'{m["vel_rmse"]:>8.4f}m/s  '
                  f'{m["heading_err_deg"]:>7.2f}°')
        print()

    # Improvement summary
    print('  IMPROVEMENT vs BASELINE (test sequences):')
    print('  ' + '-'*50)
    for seq in ['rt5_rim', 'rt5_updown']:
        if seq not in all_metrics: continue
        if 'InEKF_baseline' not in all_metrics[seq]: continue
        base = all_metrics[seq]['InEKF_baseline']['metrics']
        for fname in ['InEKF_jerk', 'InEKF_LSTM']:
            if fname not in all_metrics[seq]: continue
            m = all_metrics[seq][fname]['metrics']
            spd_imp = (base['speed_rmse'] - m['speed_rmse']) / base['speed_rmse'] * 100
            vel_imp = (base['vel_rmse']   - m['vel_rmse'])   / base['vel_rmse']   * 100
            hdg_imp = (base['heading_err_deg'] - m['heading_err_deg']) / \
                       base['heading_err_deg'] * 100
            print(f'  {seq} {fname}: '
                  f'speed {spd_imp:+.1f}%  vel {vel_imp:+.1f}%  hdg {hdg_imp:+.1f}%')


# ─────────────────────── Main ─────────────────────────────────────────────────

if __name__ == '__main__':
    print('Velocity-space evaluation\n')
    all_metrics = {}
    for seq in SEQUENCES:
        res = evaluate_sequence(seq)
        if res:
            all_metrics[seq] = res

    print_summary_table(all_metrics)
    fig_velocity_bars(all_metrics)
    fig_speed_over_time(all_metrics, 'rt5_updown')
    fig_speed_over_time(all_metrics, 'rt4_updown')

    print(f'\nPlots saved to: {PLOTS_DIR.resolve()}')
