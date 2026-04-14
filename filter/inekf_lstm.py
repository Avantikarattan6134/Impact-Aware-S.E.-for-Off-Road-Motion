"""
filter/inekf_lstm.py
====================
InEKF with LSTM-predicted adaptive process-noise scaling.

At each step:
  1. Append [ax, ay, az, gx, gy, gz, jerk_mag] to a rolling buffer.
  2. Once buffer is full, query the LSTM for α.
  3. Pass α to parent propagate().
"""

import numpy as np
from collections import deque
from .inekf_imu import InEKF_IMU


class InEKF_LSTM(InEKF_IMU):
    """
    LSTM-adaptive InEKF.

    Parameters
    ----------
    cfg   : full config dict
    model : LSTMNoiseAdapter instance (already loaded / trained)
    device: torch device string ("cpu" or "cuda")
    """

    def __init__(self, cfg: dict, model, device: str = "cpu"):
        super().__init__(cfg)
        lc = cfg["lstm"]
        self.win        = int(lc["window"])
        self.alpha_min  = float(lc["alpha_min"])
        self.alpha_max  = float(lc["alpha_max"])
        self.model      = model
        self.device     = device
        self.n_feat     = int(lc["features"])   # 7

        # rolling window of features
        self._feat_buf = deque(maxlen=self.win)
        self._prev_a   = None
        self._prev_t   = None
        self._alpha    = 1.0    # default until buffer fills

        # logs
        self.alpha_log: list[float] = []

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._feat_buf.clear()
        self._prev_a = None
        self._prev_t = None
        self._alpha  = 1.0
        self._beta   = 0.0
        self.alpha_log.clear()

    def propagate(self, omega_m: np.ndarray, accel_m: np.ndarray,
                  t_new: float, alpha: float = 1.0):
        """Override: compute LSTM α, then propagate."""
        # jerk
        if self._prev_a is not None and self._prev_t is not None:
            dt = t_new - self._prev_t
            jerk = np.linalg.norm((accel_m - self._prev_a) / dt) if dt > 0 else 0.0
        else:
            jerk = 0.0
        self._prev_a = accel_m.copy()
        self._prev_t = t_new

        acc_mag  = np.linalg.norm(accel_m)
        gyro_mag = np.linalg.norm(omega_m)
        feat = np.array([
            accel_m[0], accel_m[1], accel_m[2],
            omega_m[0], omega_m[1], omega_m[2],
            jerk, acc_mag, gyro_mag
        ], dtype=np.float32)
        self._feat_buf.append(feat)

        if len(self._feat_buf) == self.win:
            window_np = np.stack(list(self._feat_buf), axis=0)  # (W,7)
            result = self.model.predict(window_np, device=self.device)
            if isinstance(result, tuple):
                self._alpha, self._beta = result
            else:
                self._alpha = result   # backward compat

        self.alpha_log.append(self._alpha)
        super().propagate(omega_m, accel_m, t_new, alpha=self._alpha)

    def update_nhc(self, beta: float = 0.0):
        """LSTM-predicted beta controls NHC relaxation."""
        super().update_nhc(beta=self._beta)
