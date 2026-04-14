"""
filter/visual_odometry.py
=========================
Monocular Visual Odometry frontend for ROOAD InEKF.

Pipeline (per frame pair)
--------------------------
1. Detect FAST corners in current frame
2. Track with Lucas-Kanade optical flow to next frame
3. Filter outliers with RANSAC Essential matrix
4. Decompose Essential matrix -> relative rotation deltaR
5. Return deltaR as EKF rotation measurement

Camera (ROOAD Basler Pylon 1920x1200)
--------------------------------------
  fx = fy = 1636.6 px
  cx = 960.8,  cy = 596.7
  Distortion: k1=-0.170, k2=0.023, p1=0.0002, p2=0.0001

Scale ambiguity: monocular VO cannot recover translation scale.
We only use the ROTATION component as a measurement -> constrains
heading drift without needing scale.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


# ── ROOAD Basler Pylon calibration ────────────────────────────────────────────
# Source: unmannedlab/ROOAD yamls/
ROOAD_K = np.array([
    [1636.6,    0.0,   960.8],
    [   0.0, 1636.6,   596.7],
    [   0.0,    0.0,     1.0],
], dtype=np.float64)

ROOAD_D = np.array([-0.170, 0.023, 0.0002, 0.0001, 0.0], dtype=np.float64)

# Working resolution (resize to speed up)
WORK_W, WORK_H = 640, 400   # 1/3 of original
SCALE_X = WORK_W / 1920.0
SCALE_Y = WORK_H / 1200.0

# Scale K for working resolution
WORK_K = ROOAD_K.copy()
WORK_K[0, 0] *= SCALE_X;  WORK_K[1, 1] *= SCALE_Y
WORK_K[0, 2] *= SCALE_X;  WORK_K[1, 2] *= SCALE_Y


class MonocularVO:
    """
    Lightweight monocular VO frontend.
    Outputs relative rotation deltaR between consecutive frames.

    Parameters
    ----------
    max_features : int   max FAST corners to detect
    min_features : int   re-detect when tracked count falls below this
    ransac_th    : float RANSAC inlier threshold [px]
    min_inliers  : int   minimum Essential matrix inliers to accept
    """

    def __init__(self,
                 max_features: int = 300,
                 min_features: int = 100,
                 ransac_th:    float = 1.0,
                 min_inliers:  int   = 30):
        self.max_features = max_features
        self.min_features = min_features
        self.ransac_th    = ransac_th
        self.min_inliers  = min_inliers

        # LK optical flow parameters
        self.lk_params = dict(
            winSize  = (21, 21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30, 0.01),
        )

        # FAST detector
        self.detector = cv2.FastFeatureDetector_create(
            threshold=15, nonmaxSuppression=True
        )

        # State
        self._prev_gray  : Optional[np.ndarray] = None
        self._prev_pts   : Optional[np.ndarray] = None
        self._t_prev     : float                 = -1.0

        # Diagnostics
        self.n_tracked   : int = 0
        self.n_inliers   : int = 0
        self.success_log : list = []

    def reset(self):
        self._prev_gray = None
        self._prev_pts  = None
        self._t_prev    = -1.0

    def process_frame(self, img_bgr: np.ndarray,
                      t: float) -> Optional[np.ndarray]:
        """
        Process a new camera frame.

        Parameters
        ----------
        img_bgr : (H, W, 3) or (H, W)  BGR or grayscale image
        t       : float  frame timestamp [s]

        Returns
        -------
        delta_R : (3,3) relative rotation  OR  None if not enough inliers
        """
        # Preprocess
        gray = self._preprocess(img_bgr)

        if self._prev_gray is None:
            # First frame — just store
            self._prev_gray = gray
            self._prev_pts  = self._detect(gray)
            self._t_prev    = t
            return None

        # Track features LK
        curr_pts, mask_track = self._track(self._prev_gray, gray,
                                           self._prev_pts)
        self.n_tracked = int(mask_track.sum())

        if self.n_tracked < self.min_features:
            # Re-detect on current frame
            self._prev_gray = gray
            self._prev_pts  = self._detect(gray)
            self._t_prev    = t
            self.success_log.append(False)
            return None

        prev_good = self._prev_pts[mask_track]
        curr_good = curr_pts[mask_track]

        # Essential matrix -> R
        delta_R, n_in = self._recover_rotation(prev_good, curr_good)
        self.n_inliers = n_in

        # Update state
        self._prev_gray = gray
        self._prev_pts  = curr_good if n_in >= self.min_inliers \
                          else self._detect(gray)
        self._t_prev    = t

        if n_in < self.min_inliers or delta_R is None:
            self.success_log.append(False)
            return None

        # Quality gate: reject large rotations (likely tracking failure)
        # Off-road: inter-frame rotation should be < 15 deg at 30 Hz
        import math
        cos_angle = (np.trace(delta_R) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg > 15.0:
            self.success_log.append(False)
            return None

        # Re-detect periodically to avoid feature loss
        if len(self._prev_pts) < self.min_features:
            self._prev_pts = self._detect(gray)

        self.success_log.append(True)
        return delta_R

    # ── Private helpers ──────────────────────────────────────────────────────

    def _preprocess(self, img) -> np.ndarray:
        """Resize + convert to grayscale."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        return cv2.resize(gray, (WORK_W, WORK_H))

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        """Detect FAST corners, return (N,1,2) float32."""
        kps = self.detector.detect(gray)
        if len(kps) > self.max_features:
            kps = sorted(kps, key=lambda k: -k.response)[:self.max_features]
        pts = np.array([[k.pt] for k in kps], dtype=np.float32)
        return pts

    def _track(self, prev_gray, curr_gray, prev_pts):
        """
        Lucas-Kanade tracking.
        Returns (curr_pts, mask) where mask[i]=True means tracked.
        """
        if prev_pts is None or len(prev_pts) == 0:
            return np.zeros((0, 1, 2), np.float32), np.zeros(0, bool)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )
        # Back-track for consistency check
        prev_back, status_b, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, curr_pts, None, **self.lk_params
        )
        dist_back = np.linalg.norm(prev_pts - prev_back, axis=2).squeeze()
        mask = (status.squeeze() == 1) & \
               (status_b.squeeze() == 1) & \
               (dist_back < 1.0)
        return curr_pts, mask

    def _recover_rotation(self, pts1: np.ndarray,
                          pts2: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Estimate relative rotation via Essential matrix.

        Returns (R, n_inliers) or (None, 0) on failure.
        R maps points from frame1 coords to frame2 coords:
          x2 ~ R @ x1 + t  (monocular: t direction only, no scale)
        """
        if len(pts1) < 8:
            return None, 0

        p1 = pts1.reshape(-1, 2).astype(np.float64)
        p2 = pts2.reshape(-1, 2).astype(np.float64)

        E, mask_E = cv2.findEssentialMat(
            p1, p2, WORK_K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.ransac_th,
        )
        if E is None or mask_E is None:
            return None, 0

        n_in = int(mask_E.sum())
        if n_in < self.min_inliers:
            return None, n_in

        _, R, _t, mask_pose = cv2.recoverPose(
            E, p1, p2, WORK_K, mask=mask_E
        )
        # R: rotation from cam1 to cam2
        return R, int(mask_pose.sum())
