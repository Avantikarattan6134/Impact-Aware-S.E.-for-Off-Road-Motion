"""
filter/lie_utils.py
===================
Lie-group math for the Right-Invariant EKF on SE_2(3).

Groups used
-----------
  SO(3)    – 3-D rotation matrices
  SE_2(3)  – 5×5 matrix group embedding (R, v, p):
               X = | R  v  p |
                   | 0  1  0 |
                   | 0  0  1 |

Conventions
-----------
  All angles in radians, distances in metres.
  "hat"  (⋅^)   : R^n → algebra  (vector → skew / Lie-algebra element)
  "vee"  (⋅^∨)  : algebra → R^n
  Exp / Log      : (un)map algebra ↔ group
"""

import numpy as np


# ─────────────────────────────── SO(3) ───────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """Return 3×3 skew-symmetric matrix for v ∈ R³."""
    v = v.ravel()
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ]
    ])


def vee(S: np.ndarray) -> np.ndarray:
    """Return the 3-vector from a 3×3 skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues' rotation formula: R^3 → SO(3)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + skew(phi)
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) → R^3 (principal log, angle in [0, π])."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-10:
        return vee(R - R.T) / 2.0
    return (angle / (2.0 * np.sin(angle))) * vee(R - R.T)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3) J_l(φ)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3) + 0.5 * skew(phi)
    axis = phi / angle
    K = skew(axis)
    return (np.sin(angle) / angle) * np.eye(3) + \
           (1 - np.sin(angle) / angle) * np.outer(axis, axis) + \
           ((1 - np.cos(angle)) / angle) * K


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3) J_r(φ) = J_l(-φ)."""
    return so3_left_jacobian(-phi)


def so3_right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of right Jacobian J_r^{-1}(φ)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3) - 0.5 * skew(phi)
    axis = phi / angle
    K = skew(axis)
    half = angle / 2.0
    return (half / np.tan(half)) * np.eye(3) + \
           (1 - half / np.tan(half)) * np.outer(axis, axis) - \
           half * K


# ─────────────────────── SE_2(3) helpers ─────────────────────────────────────

def se23_hat(xi: np.ndarray) -> np.ndarray:
    """
    Map xi = [phi(3), nu(3), rho(3)] ∈ R^9 → 5×5 Lie algebra element.
    """
    phi, nu, rho = xi[:3], xi[3:6], xi[6:9]
    X = np.zeros((5, 5))
    X[:3, :3] = skew(phi)
    X[:3, 3]  = nu
    X[:3, 4]  = rho
    return X


def se23_vee(X: np.ndarray) -> np.ndarray:
    """Map 5×5 Lie algebra element → xi = [phi, nu, rho] ∈ R^9."""
    phi = vee(X[:3, :3])
    nu  = X[:3, 3]
    rho = X[:3, 4]
    return np.concatenate([phi, nu, rho])


def se23_exp(xi: np.ndarray) -> np.ndarray:
    """
    SE_2(3) exponential map: R^9 → 5×5 group element.
    Result: X = | R  Jl*nu  Jl*rho |
                | 0    1       0    |
                | 0    0       1    |
    """
    phi, nu, rho = xi[:3], xi[3:6], xi[6:9]
    R  = so3_exp(phi)
    Jl = so3_left_jacobian(phi)
    v  = Jl @ nu
    p  = Jl @ rho
    X  = np.eye(5)
    X[:3, :3] = R
    X[:3, 3]  = v
    X[:3, 4]  = p
    return X


def se23_log(X: np.ndarray) -> np.ndarray:
    """SE_2(3) logarithm: 5×5 → R^9."""
    R  = X[:3, :3]
    v  = X[:3, 3]
    p  = X[:3, 4]
    phi = so3_log(R)
    Jl_inv = np.linalg.inv(so3_left_jacobian(phi))
    nu  = Jl_inv @ v
    rho = Jl_inv @ p
    return np.concatenate([phi, nu, rho])


def se23_inv(X: np.ndarray) -> np.ndarray:
    """Invert a 5×5 SE_2(3) element."""
    Xinv = np.eye(5)
    RT = X[:3, :3].T
    Xinv[:3, :3] = RT
    Xinv[:3, 3]  = -RT @ X[:3, 3]
    Xinv[:3, 4]  = -RT @ X[:3, 4]
    return Xinv


def se23_adjoint(X: np.ndarray) -> np.ndarray:
    """
    9×9 Adjoint of X ∈ SE_2(3).
    Used to map right-invariant error across the group.
    Ad_X = | R        0    0  |
           | [v]× R   R    0  |
           | [p]× R   0    R  |
    """
    R = X[:3, :3]
    v = X[:3, 3]
    p = X[:3, 4]
    Ad = np.zeros((9, 9))
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = skew(v) @ R
    Ad[3:6, 3:6] = R
    Ad[6:9, 0:3] = skew(p) @ R
    Ad[6:9, 6:9] = R
    return Ad


# ─────────────────── Convenience extractors ──────────────────────────────────

def rot_from_se23(X: np.ndarray) -> np.ndarray:
    return X[:3, :3].copy()

def vel_from_se23(X: np.ndarray) -> np.ndarray:
    return X[:3, 3].copy()

def pos_from_se23(X: np.ndarray) -> np.ndarray:
    return X[:3, 4].copy()


def build_se23(R: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Assemble 5×5 SE_2(3) from R, v, p."""
    X = np.eye(5)
    X[:3, :3] = R
    X[:3, 3]  = v
    X[:3, 4]  = p
    return X
