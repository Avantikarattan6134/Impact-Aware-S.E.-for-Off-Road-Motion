"""
filter/inekf_imu.py
===================
Baseline IMU-only Right-Invariant Extended Kalman Filter on SE_2(3).

State
-----
  X  ∈ SE_2(3)  – 5×5 matrix : [R | v | p]
  b  ∈ R^6      – bias vector : [b_gyro(3); b_accel(3)]
  P  ∈ R^{15×15} – error covariance (right-invariant error + bias error)

Error state
-----------
  ε = [δφ(3), δv(3), δp(3), δb_g(3), δb_a(3)] ∈ R^15

  Right-invariant error: X̃ = X̂ X⁻¹ ≈ exp(δξ^) ,  δξ = [δφ; δv; δp]

Propagation (discrete, Euler + Rodrigues)
-----------------------------------------
  ω_c = ω_m − b_g
  a_c = a_m − b_a
  R'  = R  Exp(ω_c dt)
  v'  = v  + (R a_c + g) dt
  p'  = p  + v dt + ½(R a_c + g) dt²
  b'  = b

Linearised error Jacobian (15×15)
----------------------------------
  F = [[−[ω_c]×,       0,     0,     −I,       0   ],
       [−R[a_c]×,   −[ω_c]×,  0,      0,     −R    ],
       [   0,          I,   −[ω_c]×,  0,       0   ],
       [   0,          0,     0,      0,       0   ],
       [   0,          0,     0,      0,       0   ]]  × dt  +  I

  Discretised: Φ = I + F_c·dt  (first-order, sufficient for 400 Hz IMU)

Process-noise covariance (discrete)
-------------------------------------
  Q_d = diag([σ_g²·dt, σ_a²·dt, 0, σ_bg²·dt, σ_ba²·dt]) mapped through G

  G (15×12):
    rows 0:3  ← σ_g  (gyro noise → rotation)
    rows 3:6  ← σ_a  (accel noise → velocity)
    rows 6:9  ← 0    (no direct noise on position)
    rows 9:12 ← σ_bg (gyro bias)
    rows 12:15← σ_ba (accel bias)
"""

import numpy as np
from .lie_utils import (
    skew, so3_exp, so3_log, build_se23, se23_adjoint,
    rot_from_se23, vel_from_se23, pos_from_se23
)


class InEKF_IMU:
    """
    Right-Invariant EKF (IMU-only, fixed process noise).

    Parameters
    ----------
    cfg : dict  – from config/rooad_config.yaml  (full config)
    """

    DIM_STATE = 15   # δφ(3) + δv(3) + δp(3) + δb_g(3) + δb_a(3)

    def __init__(self, cfg: dict):
        ekf_cfg  = cfg["ekf"]
        init_cfg = cfg["init"]

        # --- gravity (ENU) ------------------------------------------------
        self.g = np.array(cfg["data"]["gravity"], dtype=float)   # [0,0,−9.81]

        # --- process noise spectral densities (continuous) ----------------
        self.σ_g  = ekf_cfg["sigma_gyro"]
        self.σ_a  = ekf_cfg["sigma_accel"]
        self.σ_bg = ekf_cfg["sigma_bg"]
        self.σ_ba = ekf_cfg["sigma_ba"]

        # --- initial P diagonal -------------------------------------------
        self.P0_rot = init_cfg["P0_rot"]
        self.P0_vel = init_cfg["P0_vel"]
        self.P0_pos = init_cfg["P0_pos"]
        self.P0_bg  = init_cfg["P0_bg"]
        self.P0_ba  = init_cfg["P0_ba"]

        # --- state variables (set by reset()) -----------------------------
        self.X  = None          # 5×5 SE_2(3)
        self.b  = None          # (6,)  [b_g; b_a]
        self.P  = None          # (15,15)
        self.t  = None          # current timestamp (seconds)

        self._initialized = False
        self._nhc_cfg = cfg.get("nhc", {
            "enabled": True, "rate": 10,
            "sigma_lat": 0.10, "sigma_vert": 0.50
        })
        self._nhc_step = 0

    # ── Public interface ────────────────────────────────────────────────────

    def reset(self, R0: np.ndarray, v0: np.ndarray, p0: np.ndarray,
              bg0: np.ndarray, ba0: np.ndarray, t0: float):
        """
        Initialise the filter.

        Parameters
        ----------
        R0  : (3,3)  initial orientation (world←body)
        v0  : (3,)   initial velocity in world frame
        p0  : (3,)   initial position in world frame
        bg0 : (3,)   initial gyro bias
        ba0 : (3,)   initial accel bias
        t0  : float  initial timestamp [s]
        """
        self.X = build_se23(R0, v0, p0)
        self.b = np.concatenate([bg0, ba0])
        self.P = np.diag([
            self.P0_rot,  self.P0_rot,  self.P0_rot,
            self.P0_vel,  self.P0_vel,  self.P0_vel,
            self.P0_pos,  self.P0_pos,  self.P0_pos,
            self.P0_bg,   self.P0_bg,   self.P0_bg,
            self.P0_ba,   self.P0_ba,   self.P0_ba,
        ])
        self.t = t0
        self._initialized = True

    def propagate(self, omega_m: np.ndarray, accel_m: np.ndarray,
                  t_new: float, alpha: float = 1.0):
        """
        IMU propagation step.

        Parameters
        ----------
        omega_m : (3,)  raw gyro  measurement  [rad/s]
        accel_m : (3,)  raw accel measurement  [m/s²]
        t_new   : float new timestamp [s]
        alpha   : float Q scaling factor (1.0 for baseline, >1 under impacts)
        """
        assert self._initialized, "Call reset() first."
        dt = t_new - self.t
        if dt <= 0:
            return
        self.t = t_new

        R = rot_from_se23(self.X)
        v = vel_from_se23(self.X)
        p = pos_from_se23(self.X)
        b_g = self.b[:3]
        b_a = self.b[3:]

        # --- corrected IMU ------------------------------------------------
        ω_c = omega_m - b_g
        a_c = accel_m - b_a

        # --- state propagation --------------------------------------------
        R_new = R @ so3_exp(ω_c * dt)
        acc_world = R @ a_c + self.g
        v_new = v + acc_world * dt
        p_new = p + v * dt + 0.5 * acc_world * dt**2

        self.X = build_se23(R_new, v_new, p_new)
        # biases unchanged

        # --- linearised Jacobian Φ (15×15) --------------------------------
        Φ = self._state_jacobian(R, ω_c, a_c, dt)

        # --- discrete process noise Q_d -----------------------------------
        Q_d = self._process_noise(R, dt, alpha)

        # --- covariance propagation ---------------------------------------
        self.P = Φ @ self.P @ Φ.T + Q_d

        # --- NHC pseudo-measurement update -----------------------------------
        nhc = self._nhc_cfg
        if nhc.get("enabled", True):
            self._nhc_step += 1
            if self._nhc_step >= nhc.get("rate", 10):
                self._nhc_step = 0
                self.update_nhc(beta=0.0)   # baseline: full constraint

    # ── State accessors ─────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return pos_from_se23(self.X)

    @property
    def velocity(self) -> np.ndarray:
        return vel_from_se23(self.X)

    @property
    def rotation(self) -> np.ndarray:
        return rot_from_se23(self.X)

    @property
    def bias_gyro(self) -> np.ndarray:
        return self.b[:3].copy()

    @property
    def bias_accel(self) -> np.ndarray:
        return self.b[3:].copy()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _state_jacobian(self, R: np.ndarray, ω_c: np.ndarray,
                        a_c: np.ndarray, dt: float) -> np.ndarray:
        """
        First-order discrete Jacobian Φ ≈ I + F_c·dt  (15×15).

        Block layout (rows/cols): [δφ | δv | δp | δb_g | δb_a]
                                   0:3  3:6  6:9  9:12  12:15
        """
        Kω = skew(ω_c)
        Ka = skew(a_c)
        Z  = np.zeros((3, 3))
        I3 = np.eye(3)

        # Continuous-time F_c
        F_c = np.zeros((15, 15))

        # δφ̇ = −[ω_c]× δφ − δb_g
        F_c[0:3, 0:3]   = -Kω
        F_c[0:3, 9:12]  = -I3

        # δv̇ = −R[a_c]× δφ − [ω_c]× δv − R δb_a
        F_c[3:6, 0:3]   = -R @ Ka
        F_c[3:6, 3:6]   = -Kω
        F_c[3:6, 12:15] = -R

        # δṗ = δv − [ω_c]× δp
        F_c[6:9, 3:6]   = I3
        F_c[6:9, 6:9]   = -Kω

        # biases: no dynamics
        # F_c[9:15, :] = 0

        Φ = np.eye(15) + F_c * dt
        return Φ

    def _process_noise(self, R: np.ndarray, dt: float,
                       alpha: float = 1.0) -> np.ndarray:
        """
        Discrete process noise Q_d (15×15).

        Noise sources (body frame → world frame via R):
          n_g  ~ N(0, σ_g²)   gyro  white noise → affects δφ
          n_a  ~ N(0, σ_a²)   accel white noise → affects δv
          n_bg ~ N(0, σ_bg²)  gyro  bias rw     → affects δb_g
          n_ba ~ N(0, σ_ba²)  accel bias rw     → affects δb_a
        """
        # Continuous-time noise input matrix G (15×12)
        G = np.zeros((15, 12))
        G[0:3,  0:3]  = -R          # gyro noise → rot error
        G[3:6,  3:6]  = -R          # accel noise → vel error
        # position error has no direct noise injection
        G[9:12,  6:9]  = np.eye(3)  # gyro bias rw
        G[12:15, 9:12] = np.eye(3)  # accel bias rw

        # Continuous-time noise covariance Qc
        Qc = np.diag([
            self.σ_g**2,  self.σ_g**2,  self.σ_g**2,
            self.σ_a**2,  self.σ_a**2,  self.σ_a**2,
            self.σ_bg**2, self.σ_bg**2, self.σ_bg**2,
            self.σ_ba**2, self.σ_ba**2, self.σ_ba**2,
        ]) * alpha

        # Discrete: Q_d = G Qc G^T dt
        return G @ Qc @ G.T * dt


    # ── Visual Odometry rotation update ─────────────────────────────────────

    def update_visual_rotation(self,
                                delta_R_cam: np.ndarray,
                                R_cam_body:  np.ndarray,
                                sigma_vo:    float = 0.02):
        """
        EKF update from monocular VO relative rotation.

        The camera measures the relative rotation between consecutive frames:
            delta_R_cam = R_cam(t-1)^T @ R_cam(t)

        Converting to body frame and comparing with filter prediction:
            delta_R_body = R_cam_body^T @ delta_R_cam @ R_cam_body

        Innovation (rotation vector):
            nu = Log(delta_R_pred^T @ delta_R_body)

        Measurement Jacobian H (3x15):
            d(Log(...))/d(delta_phi) = I_3  (linearised around zero)
            d(Log(...))/d(other)     = 0

        Parameters
        ----------
        delta_R_cam  : (3,3) relative camera rotation from VO
        R_cam_body   : (3,3) fixed camera-to-body extrinsic rotation
        sigma_vo     : float  VO rotation noise [rad] (per axis)
        """
        from .lie_utils import so3_log, so3_exp

        R_body = rot_from_se23(self.X)

        # Predicted body-frame relative rotation from filter
        # (we store the previous body rotation implicitly through integration)
        # Approximate: delta_R_pred = I  (short interval, update is a correction)
        # Better: compare against accumulated rotation since last VO update
        # For 30 Hz camera, delta_t ~ 33ms, so small rotation expected

        # Convert VO rotation from camera frame to body frame
        delta_R_body = R_cam_body.T @ delta_R_cam @ R_cam_body

        # Innovation: rotation vector of residual
        innov = so3_log(delta_R_body)           # (3,) in body frame

        # Transform to world frame for filter correction
        innov_world = R_body @ innov             # (3,)

        # Jacobian H (3x15): rotation error directly affects rotation state
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)

        # Measurement noise
        R_vo = np.eye(3) * sigma_vo**2

        # Kalman gain
        S = H @ self.P @ H.T + R_vo
        K = self.P @ H.T @ np.linalg.inv(S)    # (15,3)

        # Correction
        dx = K @ innov_world                    # (15,)

        # Apply state correction
        R_new = so3_exp(dx[0:3]) @ R_body
        v_new = vel_from_se23(self.X) + dx[3:6]
        p_new = pos_from_se23(self.X) + dx[6:9]
        self.X  = build_se23(R_new, v_new, p_new)
        self.b += dx[9:15]

        # Covariance — Joseph form
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_vo @ K.T

    # ── GPS position update ──────────────────────────────────────────────────

    def update_gps(self, gps_pos: np.ndarray, R_gps: np.ndarray = None):
        """
        GPS position measurement update (EKF correction step).

        Measurement model:  z = p  (position in world/ENU frame)
        Jacobian H (3×15):  H = [0|0|I|0|0]  — position directly observable

        Parameters
        ----------
        gps_pos : (3,)  GPS position in ENU frame [m]
        R_gps   : (3,3) GPS measurement noise covariance
                        Default: diag([0.02², 0.02², 0.05²]) m²
                        (RTK: ~2cm horizontal, ~5cm vertical)
        """
        if R_gps is None:
            R_gps = np.diag([0.02**2, 0.02**2, 0.05**2])

        p_est = pos_from_se23(self.X)
        R_mat = rot_from_se23(self.X)
        v_est = vel_from_se23(self.X)

        # Innovation
        innov = gps_pos - p_est          # (3,)

        # Jacobian H (3×15): position error directly maps to position state
        H = np.zeros((3, 15))
        H[0:3, 6:9] = np.eye(3)         # d(pos)/d(δp)

        # Kalman gain
        S = H @ self.P @ H.T + R_gps    # (3,3)
        K = self.P @ H.T @ np.linalg.inv(S)  # (15,3)

        # Correction vector
        dx = K @ innov                   # (15,)

        # Apply to state
        from .lie_utils import so3_exp
        R_new = so3_exp(dx[0:3]) @ R_mat
        v_new = v_est + dx[3:6]
        p_new = p_est + dx[6:9]
        self.X  = build_se23(R_new, v_new, p_new)
        self.b += dx[9:15]

        # Covariance — Joseph form for numerical stability
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_gps @ K.T

    # ── NHC velocity constraint ──────────────────────────────────────────────

    def update_nhc(self, beta: float = 0.0):
        """
        Non-Holonomic Constraint with learnable relaxation factor beta.

        beta = 0.0  →  full hard constraint  (body y,z velocity → 0)
        beta = 1.0  →  no constraint         (vehicle airborne / hopping)
        beta in (0,1) → partial constraint   (during impact recovery)

        Parameters
        ----------
        beta : float in [0, 1]
            NHC relaxation. Baseline always uses 0.
            Jerk/LSTM filters learn when to relax.
        """
        R = rot_from_se23(self.X)
        v = vel_from_se23(self.X)

        v_body    = R.T @ v
        # Partial zero: v_body[i] → v_body[i] * beta
        v_body[1] *= beta     # lateral  (0 = zeroed, 1 = unchanged)
        v_body[2] *= beta     # vertical (0 = zeroed, 1 = unchanged)
        v_corrected = R @ v_body

        self.X = build_se23(R, v_corrected, pos_from_se23(self.X))

        # Shrink covariance in constrained directions
        sigma = self._nhc_cfg.get("sigma_lat", 0.1)
        shrink = (1.0 - beta) * 0.5   # shrink more when beta is small
        if shrink > 0:
            for idx in [4, 5]:
                self.P[idx, :] *= (1.0 - shrink)
                self.P[:, idx] *= (1.0 - shrink)
                self.P[idx, idx] = max(self.P[idx, idx], sigma**2 * (beta + 0.01))

    def nees(self, pos_error: np.ndarray) -> float:
        """NEES for position — filter consistency metric."""
        P_pos = self.P[6:9, 6:9]
        try:
            return float(pos_error @ np.linalg.inv(P_pos) @ pos_error)
        except np.linalg.LinAlgError:
            return float('nan')
