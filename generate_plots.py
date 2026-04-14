"""
generate_plots.py
=================
Generate all plots for the ROOAD InEKF paper.

Produces 4 figures:
  1. Trajectory comparison (top-down) for rt5_updown
  2. ATE over time for all sequences
  3. RPE bar chart across all sequences
  4. Jerk + alpha/beta over time for rt5_updown

Run from project root:
  python generate_plots.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import yaml

# ── Config ───────────────────────────────────────────────────────────────────

with open('config/rooad_config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

RESULTS_DIR = Path('results')
PLOTS_DIR   = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

SEQUENCES = ['rt4_gravel', 'rt4_rim', 'rt4_updown',
             'rt5_gravel', 'rt5_rim', 'rt5_updown']

FILTER_STYLES = {
    'InEKF_baseline': dict(color='#e74c3c', lw=2.0, ls='--',  label='InEKF Baseline',  marker=None),
    'InEKF_jerk':     dict(color='#f39c12', lw=2.0, ls='-.',  label='InEKF + Jerk',    marker=None),
    'InEKF_LSTM':     dict(color='#27ae60', lw=2.5, ls='-',   label='InEKF + LSTM',    marker=None),
}
GT_STYLE = dict(color='#2c3e50', lw=2.0, ls='-', label='Ground Truth')

TERRAIN_LABELS = {
    'rt4_gravel': 'Gravel (rt4)', 'rt4_rim': 'Rim (rt4)',
    'rt4_updown': 'Updown (rt4)', 'rt5_gravel': 'Gravel (rt5)',
    'rt5_rim': 'Rim (rt5)',       'rt5_updown': 'Updown (rt5)',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(seq):
    """Load all filter results and GT for a sequence."""
    out = {}
    for fname in ['InEKF_baseline', 'InEKF_jerk', 'InEKF_LSTM']:
        p = RESULTS_DIR / f'{seq}_{fname}.npz'
        if p.exists():
            out[fname] = dict(np.load(p, allow_pickle=True))
    gt_p = RESULTS_DIR / f'{seq}_gt.npz'
    if gt_p.exists():
        out['__gt__'] = dict(np.load(gt_p, allow_pickle=True))
    return out

def umeyama_align(est, gt):
    n = len(est)
    mu_e, mu_g = est.mean(0), gt.mean(0)
    E, G = est - mu_e, gt - mu_g
    W = (G.T @ E) / n
    U, S, Vt = np.linalg.svd(W)
    d = np.linalg.det(U @ Vt)
    R = U @ np.diag([1,1,d]) @ Vt
    t = mu_g - R @ mu_e
    return (R @ est.T).T + t

def compute_ate_series(traj_pos, gt_t, gt_pos, traj_t):
    """ATE at each timestep after alignment."""
    from scipy.interpolate import interp1d
    f = interp1d(gt_t, gt_pos, axis=0, bounds_error=False,
                 fill_value=(gt_pos[0], gt_pos[-1]))
    gt_interp = f(traj_t)
    valid = ~np.isnan(gt_interp[:,0])
    est_a = umeyama_align(traj_pos[valid], gt_interp[valid])
    err = np.zeros(len(traj_t))
    err[valid] = np.linalg.norm(est_a - gt_interp[valid], axis=1)
    return traj_t, err, valid

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Trajectory comparison (rt5_updown)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_trajectory(seq='rt5_updown'):
    print(f'  Figure 1: Trajectory ({seq})')
    res = load_results(seq)
    if not res or '__gt__' not in res:
        print(f'    Skipping — results not found for {seq}'); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Trajectory Comparison — {TERRAIN_LABELS[seq]}', fontsize=13)

    gt = res['__gt__']
    gt_pos = gt['gt_pos']

    for ax_idx, (ax, dims, labels) in enumerate(zip(
        axes,
        [(0,1), (0,2)],
        [('East [m]', 'North [m]'), ('East [m]', 'Up [m]')]
    )):
        d0, d1 = dims
        ax.plot(gt_pos[:,d0], gt_pos[:,d1], **GT_STYLE, zorder=5)

        for fname, sty in FILTER_STYLES.items():
            if fname not in res: continue
            tp = res[fname]['traj_pos']
            ax.plot(tp[:,d0], tp[:,d1],
                    color=sty['color'], lw=sty['lw'],
                    ls=sty['ls'], label=sty['label'], alpha=0.85)

        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_aspect('equal')

    plt.tight_layout()
    out = PLOTS_DIR / f'fig1_trajectory_{seq}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: ATE over time for key sequences
# ─────────────────────────────────────────────────────────────────────────────

def fig2_ate_over_time():
    print('  Figure 2: ATE over time')
    key_seqs = ['rt4_updown', 'rt5_rim', 'rt5_updown']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Absolute Trajectory Error over Time', fontsize=13)

    for ax, seq in zip(axes, key_seqs):
        res = load_results(seq)
        if not res or '__gt__' not in res:
            ax.set_title(seq); continue

        gt   = res['__gt__']
        gt_t = gt['gt_t']; gt_pos = gt['gt_pos']

        for fname, sty in FILTER_STYLES.items():
            if fname not in res: continue
            r  = res[fname]
            tt, err, valid = compute_ate_series(
                r['traj_pos'], gt_t, gt_pos, r['traj_t'])
            # Smooth for readability
            from numpy import convolve, ones
            k = 400   # 1s smoothing
            err_s = convolve(err, ones(k)/k, mode='same')
            t_rel = tt - tt[0]
            ax.plot(t_rel, err_s,
                    color=sty['color'], lw=sty['lw'],
                    ls=sty['ls'], label=sty['label'], alpha=0.85)

        ax.set_xlabel('Time [s]'); ax.set_ylabel('ATE [m]')
        ax.set_title(TERRAIN_LABELS[seq])
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / 'fig2_ate_over_time.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: RPE bar chart across all sequences
# ─────────────────────────────────────────────────────────────────────────────

def fig3_rpe_bars():
    print('  Figure 3: RPE bar chart')

    # Hardcoded from your results
    data = {
        'rt4_gravel': {'InEKF_baseline':0.070, 'InEKF_jerk':0.074, 'InEKF_LSTM':0.074},
        'rt4_rim':    {'InEKF_baseline':0.046, 'InEKF_jerk':0.042, 'InEKF_LSTM':0.042},
        'rt4_updown': {'InEKF_baseline':0.150, 'InEKF_jerk':0.134, 'InEKF_LSTM':0.134},
        'rt5_gravel': {'InEKF_baseline':0.058, 'InEKF_jerk':0.059, 'InEKF_LSTM':0.059},
        'rt5_rim':    {'InEKF_baseline':0.066, 'InEKF_jerk':0.060, 'InEKF_LSTM':0.060},
        'rt5_updown': {'InEKF_baseline':0.127, 'InEKF_jerk':0.103, 'InEKF_LSTM':0.103},
    }
    ate_data = {
        'rt4_gravel': {'InEKF_baseline':27.78, 'InEKF_jerk':32.97, 'InEKF_LSTM':32.90},
        'rt4_rim':    {'InEKF_baseline':19.48, 'InEKF_jerk':18.14, 'InEKF_LSTM':18.14},
        'rt4_updown': {'InEKF_baseline':132.36,'InEKF_jerk':131.37,'InEKF_LSTM':131.32},
        'rt5_gravel': {'InEKF_baseline':10.23, 'InEKF_jerk':11.50, 'InEKF_LSTM':11.47},
        'rt5_rim':    {'InEKF_baseline':26.09, 'InEKF_jerk':24.18, 'InEKF_LSTM':24.19},
        'rt5_updown': {'InEKF_baseline':86.72, 'InEKF_jerk':83.41, 'InEKF_LSTM':83.37},
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle('Filter Comparison Across All Sequences', fontsize=13)

    x     = np.arange(len(SEQUENCES))
    bw    = 0.25
    seqs  = SEQUENCES
    fnames = ['InEKF_baseline', 'InEKF_jerk', 'InEKF_LSTM']

    for i, (ax, d, ylabel, title) in enumerate(zip(
        [ax1, ax2],
        [rpe_d := data, ate_d := ate_data],
        ['RPE-RMSE [m]', 'ATE-RMSE [m]'],
        ['Relative Pose Error (RPE) — lower is better',
         'Absolute Trajectory Error (ATE) — lower is better']
    )):
        for j, fname in enumerate(fnames):
            sty = FILTER_STYLES[fname]
            vals = [d[s][fname] for s in seqs]
            bars = ax.bar(x + (j-1)*bw, vals, bw,
                          label=sty['label'], color=sty['color'], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([TERRAIN_LABELS[s] for s in seqs], rotation=15, ha='right')
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for j, fname in enumerate(fnames):
            vals = [d[s][fname] for s in seqs]
            for k, v in enumerate(vals):
                ax.text(k + (j-1)*bw, v + v*0.02, f'{v:.2f}' if i==0 else f'{v:.0f}',
                        ha='center', va='bottom', fontsize=6, rotation=90)

    plt.tight_layout()
    out = PLOTS_DIR / 'fig3_rpe_ate_bars.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Jerk + alpha/beta signals for rt5_updown
# ─────────────────────────────────────────────────────────────────────────────

def fig4_jerk_signals(seq='rt5_updown'):
    print(f'  Figure 4: Jerk + signals ({seq})')
    res = load_results(seq)
    if not res:
        print('    Skipping — results not found'); return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'Impact Detection Signals — {TERRAIN_LABELS[seq]}', fontsize=13)

    # Jerk
    ax = axes[0]
    if 'InEKF_baseline' in res:
        r  = res['InEKF_baseline']
        tt = r['traj_t'] - r['traj_t'][0]
        jk = r.get('jerk', np.zeros(len(tt)))
        if len(jk) > len(tt): jk = jk[:len(tt)]
        elif len(jk) < len(tt): jk = np.pad(jk, (0, len(tt)-len(jk)))
        ax.plot(tt, jk, color='#7f8c8d', lw=0.8, alpha=0.7)
        ax.axhline(cfg['jerk']['threshold'], color='#e74c3c',
                   ls='--', lw=1.5, label=f"Threshold ({cfg['jerk']['threshold']} m/s³)")
        ax.fill_between(tt, jk, cfg['jerk']['threshold'],
                        where=jk > cfg['jerk']['threshold'],
                        alpha=0.25, color='#e74c3c', label='Impact detected')
    ax.set_ylabel('Jerk [m/s³]'); ax.set_title('IMU Jerk Magnitude')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Alpha
    ax = axes[1]
    for fname, sty in [('InEKF_jerk', FILTER_STYLES['InEKF_jerk']),
                        ('InEKF_LSTM', FILTER_STYLES['InEKF_LSTM'])]:
        if fname not in res: continue
        r  = res[fname]
        al = r.get('alpha_log', np.array([]))
        if len(al) == 0: continue
        tt = r['traj_t'][:len(al)] - r['traj_t'][0]
        ax.plot(tt, al, color=sty['color'], lw=1.2,
                ls=sty['ls'], label=sty['label'], alpha=0.85)
    ax.axhline(1.0, color='gray', ls=':', lw=1, label='InEKF Baseline (α=1)')
    ax.set_ylabel('Q-scale α'); ax.set_title('Process Noise Scaling (α)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ATE over time
    ax = axes[2]
    if '__gt__' in res:
        gt = res['__gt__']
        gt_t = gt['gt_t']; gt_pos = gt['gt_pos']
        for fname, sty in FILTER_STYLES.items():
            if fname not in res: continue
            r  = res[fname]
            tt, err, _ = compute_ate_series(
                r['traj_pos'], gt_t, gt_pos, r['traj_t'])
            from numpy import convolve, ones
            err_s = convolve(err, ones(400)/400, mode='same')
            ax.plot(tt - tt[0], err_s,
                    color=sty['color'], lw=sty['lw'],
                    ls=sty['ls'], label=sty['label'], alpha=0.85)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('ATE [m]')
    ax.set_title('Trajectory Error over Time')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f'fig4_signals_{seq}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Segment-level ATE comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig5_segment_ate():
    print('  Figure 5: Segment ATE')

    seg_data = {
        'rt4_gravel': {'InEKF_baseline':18.46, 'InEKF_jerk':22.62, 'InEKF_LSTM':22.56},
        'rt4_rim':    {'InEKF_baseline':17.22, 'InEKF_jerk':16.09, 'InEKF_LSTM':16.10},
        'rt4_updown': {'InEKF_baseline':127.77,'InEKF_jerk':126.52,'InEKF_LSTM':126.48},
        'rt5_gravel': {'InEKF_baseline':8.39,  'InEKF_jerk':9.75,  'InEKF_LSTM':9.73},
        'rt5_rim':    {'InEKF_baseline':22.86, 'InEKF_jerk':21.33, 'InEKF_LSTM':21.34},
        'rt5_updown': {'InEKF_baseline':83.55, 'InEKF_jerk':80.24, 'InEKF_LSTM':80.21},
    }
    segs_count = {
        'rt4_gravel':63,'rt4_rim':42,'rt4_updown':42,
        'rt5_gravel':58,'rt5_rim':62,'rt5_updown':85
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('Segment-Level ATE on High-Dynamic Regions', fontsize=13)

    x  = np.arange(len(SEQUENCES)); bw = 0.25
    for j, fname in enumerate(['InEKF_baseline','InEKF_jerk','InEKF_LSTM']):
        sty  = FILTER_STYLES[fname]
        vals = [seg_data[s][fname] for s in SEQUENCES]
        ax.bar(x + (j-1)*bw, vals, bw,
               label=sty['label'], color=sty['color'], alpha=0.85)

    # Annotate with segment counts
    for i, seq in enumerate(SEQUENCES):
        ax.text(i, 1.0, f'n={segs_count[seq]}',
                ha='center', va='bottom', fontsize=7, color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels([TERRAIN_LABELS[s] for s in SEQUENCES], rotation=15, ha='right')
    ax.set_ylabel('Segment ATE-RMSE [m]')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = PLOTS_DIR / 'fig5_segment_ate.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating plots...\n')
    fig1_trajectory('rt5_updown')
    fig1_trajectory('rt4_updown')
    fig2_ate_over_time()
    fig3_rpe_bars()
    fig4_jerk_signals('rt5_updown')
    fig5_segment_ate()
    print(f'\nAll plots saved to: {PLOTS_DIR.resolve()}')
    print('\nFiles generated:')
    for f in sorted(PLOTS_DIR.glob('*.png')):
        size = f.stat().st_size // 1024
        print(f'  {f.name}  ({size} KB)')
