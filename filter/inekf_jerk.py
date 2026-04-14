"""
filter/inekf_jerk.py
====================
InEKF with jerk-triggered Q inflation AND NHC relaxation.

The key insight: during impacts, not only is process noise higher (Q scaled),
but the vehicle may genuinely have vertical velocity (hopping, bouncing).
We relax the NHC proportionally to jerk magnitude.

beta schedule
-------------
  jerk < threshold  → beta = 0.0  (full NHC, normal terrain)
  jerk >= threshold → beta ramps up to beta_max  (NHC relaxed)
  After impact: beta decays back to 0 at rate beta_decay
"""

import numpy as np
from collections import deque
from .inekf_imu import InEKF_IMU


class InEKF_Jerk(InEKF_IMU):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        jk = cfg["jerk"]
        self.win        = int(jk["window"])
        self.threshold  = float(jk["threshold"])
        self.alpha_min  = float(jk["alpha_min"])
        self.alpha_max  = float(jk["alpha_max"])
        self.decay      = float(jk["alpha_decay"])

        # beta: NHC relaxation (0=full constraint, 1=no constraint)
        self.beta_max   = float(jk.get("beta_max",  0.85))
        self.beta_decay = float(jk.get("beta_decay", 0.90))

        self._alpha = self.alpha_min
        self._beta  = 0.0
        self._accel_buf = deque(maxlen=self.win + 1)
        self._t_buf     = deque(maxlen=self.win + 1)

        self.alpha_log: list = []
        self.beta_log:  list = []
        self.jerk_log:  list = []

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._alpha = self.alpha_min
        self._beta  = 0.0
        self._accel_buf.clear()
        self._t_buf.clear()
        self.alpha_log.clear()
        self.beta_log.clear()
        self.jerk_log.clear()

    def propagate(self, omega_m: np.ndarray, accel_m: np.ndarray,
                  t_new: float, alpha: float = 1.0):
        self._accel_buf.append(accel_m.copy())
        self._t_buf.append(t_new)

        jerk_mag    = self._compute_jerk()
        self._alpha = self._update_alpha(jerk_mag)
        self._beta  = self._update_beta(jerk_mag)

        self.alpha_log.append(self._alpha)
        self.beta_log.append(self._beta)
        self.jerk_log.append(jerk_mag)

        super().propagate(omega_m, accel_m, t_new, alpha=self._alpha)

    def update_nhc(self, beta: float = 0.0):
        """Use jerk-derived beta instead of caller's beta."""
        super().update_nhc(beta=self._beta)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _compute_jerk(self) -> float:
        buf_a = list(self._accel_buf)
        buf_t = list(self._t_buf)
        if len(buf_a) < 2:
            return 0.0
        jerks = []
        for i in range(1, len(buf_a)):
            dt = buf_t[i] - buf_t[i-1]
            if dt > 0:
                jerks.append(np.linalg.norm((buf_a[i] - buf_a[i-1]) / dt))
        return float(np.mean(jerks)) if jerks else 0.0

    def _update_alpha(self, jerk_mag: float) -> float:
        if jerk_mag > self.threshold:
            return self.alpha_max
        return max(self.alpha_min, self._alpha * self.decay)

    def _update_beta(self, jerk_mag: float) -> float:
        if jerk_mag > self.threshold:
            return self.beta_max
        return max(0.0, self._beta * self.beta_decay)
