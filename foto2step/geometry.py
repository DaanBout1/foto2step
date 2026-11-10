"""Basic 3D geometry helpers: rigid/similarity fitting, triangulation, screw decomposition."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def to_h(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    p = np.asarray(p, float)
    return p @ T[:3, :3].T + T[:3, 3]


def make_T(R: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False, weights=None):
    """Least-squares transform dst ~ s R src + t. Returns (s, R, t)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if weights is None:
        w = np.ones(len(src))
    else:
        w = np.asarray(weights, float)
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (w[:, None] * xd).T @ xs
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (w * (xs ** 2).sum(1)).sum()
        s = np.trace(np.diag(D) @ S) / var_s
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rigid_T(src, dst, weights=None) -> np.ndarray:
    s, R, t = umeyama(src, dst, False, weights)
    return make_T(R, t)


def sim_T(src, dst, weights=None) -> np.ndarray:
    s, R, t = umeyama(src, dst, True, weights)
    return make_T(R, t, s)


def ransac_rigid(src, dst, thresh, iters=2000, min_pts=3, rng=None):
    """RANSAC rigid fit; returns (T, inlier_mask)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(src)
    best = None
    best_cnt = -1
    if n < min_pts:
        return None, np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, min_pts, replace=False)
        try:
            T = rigid_T(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        r = np.linalg.norm(to_h(T, src) - dst, axis=1)
        inl = r < thresh
        c = inl.sum()
        if c > best_cnt:
            best_cnt = c
            best = inl
    if best is None or best.sum() < min_pts:
        return None, np.zeros(n, bool)
    T = rigid_T(src[best], dst[best])
    for _ in range(3):
        r = np.linalg.norm(to_h(T, src) - dst, axis=1)
        best = r < thresh
        if best.sum() < min_pts:
            break
        T = rigid_T(src[best], dst[best])
    return T, best


def triangulate_dlt(P_list, x_list) -> np.ndarray:
    """Linear triangulation. P_list: list of 3x4 (normalized camera, i.e. [R|t]); x_list: normalized 2D."""
    A = []
    for P, x in zip(P_list, x_list):
        A.append(x[0] * P[2] - P[0])
        A.append(x[1] * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_refine(P_list, x_list, X0, sigmas=None):
    """Gauss-Newton refinement minimising normalized reprojection error (robust)."""
    P = np.asarray(P_list)
    x = np.asarray(x_list)
    w = np.ones(len(x)) if sigmas is None else 1.0 / np.asarray(sigmas)

    def res(X):
        Xh = np.append(X, 1.0)
        pc = P @ Xh  # n x 3
        proj = pc[:, :2] / pc[:, 2:3]
        return ((proj - x) * w[:, None]).ravel()

    sol = least_squares(res, X0, loss="huber", f_scale=1.0 if sigmas is not None else 1e-3)
    return sol.x


def rotation_angle(R) -> float:
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def screw_from_T(T: np.ndarray, rot_thresh_deg: float = 0.5) -> dict:
    """Decompose rigid motion into a joint description.

    Returns dict with type 'prismatic' (axis, travel) or 'revolute' (axis, point, angle, pitch).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    ang = rotation_angle(R)
    if ang < rot_thresh_deg:
        d = np.linalg.norm(t)
        axis = t / d if d > 1e-9 else np.array([1.0, 0, 0])
        return {"type": "prismatic", "axis": axis.tolist(), "travel": float(d), "residual_rotation_deg": ang}
    w, V = np.linalg.eig(R)
    k = np.argmin(np.abs(w - 1))
    u = np.real(V[:, k])
    u /= np.linalg.norm(u)
    # sign so that rotation angle is positive about u
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    s = np.trace(K.T @ R) / 2  # = sin(theta)
    if s < 0:
        u = -u
    along = float(t @ u)
    tp = t - along * u
    # (I-R)c = tp, least squares with c orthogonal to u
    A = np.vstack([np.eye(3) - R, u[None, :]])
    b = np.append(tp, 0.0)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return {"type": "revolute", "axis": u.tolist(), "point": c.tolist(), "angle_deg": ang,
            "pitch_mm": along}


def fit_plane(pts, weights=None):
    """Return (normal, d, rms) for plane n.x + d = 0."""
    pts = np.asarray(pts, float)
    c = pts.mean(0) if weights is None else (weights[:, None] * pts).sum(0) / weights.sum()
    X = pts - c
    if weights is not None:
        X = X * np.sqrt(weights)[:, None]
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1]
    d = -n @ c
    r = pts @ n + d
    return n, d, float(np.sqrt(np.mean(r ** 2)))


def fit_cylinder(pts, normals=None, axis0=None):
    """Least-squares cylinder fit. Returns dict(axis, point, radius, rms) or None."""
    pts = np.asarray(pts, float)
    if len(pts) < 20:
        return None
    if axis0 is None:
        if normals is None:
            return None
        # axis is direction most orthogonal to all normals
        _, _, Vt = np.linalg.svd(normals - 0.0, full_matrices=False)
        axis0 = Vt[-1]
    a0 = axis0 / np.linalg.norm(axis0)
    c0 = pts.mean(0)
    # initial radius/centre from 2D circle fit in plane orthogonal to axis
    e1 = np.cross(a0, [1, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(a0, [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a0, e1)
    q = np.c_[(pts - c0) @ e1, (pts - c0) @ e2]
    A = np.c_[2 * q, np.ones(len(q))]
    b = (q ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[:2]
    r0 = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9))
    p0 = c0 + cx * e1 + cy * e2

    def unpack(x):
        a = x[:3] / np.linalg.norm(x[:3])
        return a, x[3:6], x[6]

    def res(x):
        a, p, r = unpack(x)
        v = pts - p
        dperp = np.linalg.norm(v - np.outer(v @ a, a), axis=1)
        return dperp - r

    x0 = np.r_[a0, p0, r0]
    try:
        sol = least_squares(res, x0, loss="soft_l1", f_scale=0.5)
    except Exception:
        return None
    a, p, r = unpack(sol.x)
    # move point to projection of centroid on axis
    p = p + ((c0 - p) @ a) * a
    rr = res(np.r_[a, p, r])
    return {"axis": a, "point": p, "radius": float(abs(r)), "rms": float(np.sqrt(np.mean(rr ** 2)))}
