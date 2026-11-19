"""CAD reconstruction: points -> fitted surfaces -> real B-rep solid (STEP with analytic/NURBS faces).

Instead of turning the noisy scan into millions of triangles, we do what a modeller does in a
reverse-engineering package:

1. segment the points into surface regions and fit a primitive to each (plane, cylinder) with least
   squares, so the noise averages out (thousands of points per face);
2. make every fitted surface a bit larger than its region and split them all against each other with
   the OpenCASCADE general-fuse algorithm -> a set of candidate face fragments, plus the fitted floor
   as a candidate for the unseen underside;
3. choose the fragments with an integer program (PolyFit-style): maximise how well the chosen faces
   are supported by scan points, penalise unsupported area, and require that every edge is used by
   exactly 0 or 2 chosen faces -> a closed, manifold solid;
4. sew, fix and export as STEP.

All lengths in mm. Freeform regions (NURBS) plug into the same arrangement/selection machinery.
"""
from __future__ import annotations

import os
import numpy as np
import time

from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .log import log

# ----------------------------------------------------------------------------------------------
# 1. segmentation + primitive fitting
# ----------------------------------------------------------------------------------------------


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def fit_plane(P, robust_sigma=None, init=None):
    """Least-squares plane; with robust_sigma an IRLS fit with Tukey weights (c = 2.5 sigma), started
    from `init` when given, so a second nearby level (a 2 mm step) cannot tilt the plane."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    if robust_sigma and len(P) > 10:
        if init is not None:
            n = init["n"] / np.linalg.norm(init["n"])
            c = P[np.argsort(np.abs(P @ n + init["d"]))[: max(10, len(P) // 2)]].mean(0)
        cc = 2.5 * robust_sigma
        for _ in range(15):
            r = (P - c) @ n
            r = r - np.median(r)
            u = r / cc
            w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
            if w.sum() < 10:
                break
            c = (w[:, None] * P).sum(0) / w.sum()
            X = (P - c) * np.sqrt(w)[:, None]
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            n_new = vt[2] * np.sign(vt[2] @ n)
            if np.linalg.norm(n_new - n) < 1e-7:
                n = n_new
                break
            n = n_new
    return {"type": "plane", "n": n, "d": -float(n @ c), "c": c}


def plane_dist(prim, P):
    return P @ prim["n"] + prim["d"]


def _perp_basis(a):
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(a, t))
    return u, np.cross(a, u)


def cyl_dist(prim, P):
    a, p0, r = prim["a"], prim["p"], prim["r"]
    q = P - p0
    radial = q - np.outer(q @ a, a)
    return np.linalg.norm(radial, axis=1) - r


def cyl_normals(prim, P):
    a, p0 = prim["a"], prim["p"]
    q = P - p0
    return _unit(q - np.outer(q @ a, a))


def fit_cylinder(P, a0, p0, r0):
    """Nonlinear LS: axis (2 angles), axis point (2 coords in the perpendicular plane), radius."""
    u0, v0 = _perp_basis(a0)

    def unpack(x):
        a = _unit(a0 + x[0] * u0 + x[1] * v0)
        p = p0 + x[2] * u0 + x[3] * v0
        return a, p, x[4]

    def res(x):
        a, p, r = unpack(x)
        q = P - p
        return np.linalg.norm(q - np.outer(q @ a, a), axis=1) - r

    sol = least_squares(res, np.array([0, 0, 0, 0, r0]), loss="soft_l1", f_scale=0.5, max_nfev=200)
    a, p, r = unpack(sol.x)
    # put the axis point at the centre of the region along the axis
    t = (P - p) @ a
    p = p + a * float(np.median(t))
    return {"type": "cylinder", "a": a, "p": p, "r": float(abs(r))}


def _extr_nearest(prim, P):
    """Extrusion of a 2D polyline: (2D coords of P, index of the nearest polyline sample)."""
    from scipy.spatial import cKDTree
    t = prim.get("_tree")
    if t is None:
        t = cKDTree(prim["poly"])
        prim["_tree"] = t
    X = np.c_[np.asarray(P) @ prim["u"], np.asarray(P) @ prim["v"]]
    _, j = t.query(X)
    return X, j


def extr_dist(prim, P):
    X, j = _extr_nearest(prim, P)
    return np.sum((X - prim["poly"][j]) * prim["pnrm"][j], axis=1)


def extr_normals(prim, P):
    _, j = _extr_nearest(prim, P)
    n2 = prim["pnrm"][j]
    return n2[:, :1] * prim["u"][None] + n2[:, 1:2] * prim["v"][None]


def prim_dist(prim, P):
    if prim["type"] == "extrusion":
        return extr_dist(prim, P)
    return plane_dist(prim, P) if prim["type"] == "plane" else cyl_dist(prim, P)


def prim_normals(prim, P):
    """Surface normals; with an orientation sign 's' (set once the data normals are oriented) they
    point to the side the scan saw, so the two faces of a thin sheet are told apart."""
    sg = prim.get("s", 1.0)
    if prim["type"] == "plane":
        return np.broadcast_to(sg * prim["n"], P.shape)
    if prim["type"] == "extrusion":
        return sg * extr_normals(prim, P)
    return sg * cyl_normals(prim, P)


def signed_cos(prim, Q, NQ):
    """cos(angle) between surface and data normals after choosing the primitive's orientation sign
    from the majority of the data (stored in prim['s'])."""
    raw = dict(prim)
    raw["s"] = 1.0
    d = np.sum(prim_normals(raw, Q) * NQ, axis=1)
    sg = 1.0 if (len(d) == 0 or np.median(d) >= 0) else -1.0
    prim["s"] = sg
    return sg * d


def _largest_component(P, idx, link, min_size=None):
    """Largest spatially connected subset of P[idx] (points closer than `link` are connected).
    With min_size: all components of at least that size (a face split into strips by grooves)."""
    if len(idx) < 3:
        return idx
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = cKDTree(P[idx]).query_pairs(link, output_type="ndarray")
    n = len(idx)
    if len(pairs) == 0:
        return idx[:1]
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)),
                                  directed=False)
    cnt = np.bincount(lab)
    if min_size is not None:
        big = np.where(cnt >= min_size)[0]
        if len(big):
            return idx[np.isin(lab, big)]
    return idx[lab == cnt.argmax()]


def segment_primitives(P, N, noise_mm, min_pts=None, max_prims=60, seed=0, cylinders=True, oriented=False):
    """Efficient-RANSAC style multi-primitive detection (planes + cylinders) with connectivity.

    Returns (prims, labels): labels[i] = index into prims or -1."""
    rng = np.random.default_rng(seed)
    n_all = len(P)
    eps = max(2.5 * noise_mm, 0.8)
    cos_t = np.cos(np.radians(min(30.0, 12.0 + 15.0 * noise_mm)))  # PCA normals get noisier with noise
    link = 3.0 * max(noise_mm, 0.5) + 1.5
    min_pts = min_pts or max(120, int(0.0015 * n_all))
    labels = -np.ones(n_all, int)
    prims = []
    tree = cKDTree(P)
    fails = 0
    while len(prims) < max_prims:
        free = np.where(labels < 0)[0]
        if len(free) < min_pts:
            break
        probe = free if len(free) <= 6000 else rng.choice(free, 6000, replace=False)
        cands = []
        # plane candidates from point + local PCA normal
        for s in rng.choice(free, min(250, len(free)), replace=False):
            nb = tree.query_ball_point(P[s], 4.0)
            if len(nb) < 8:
                continue
            pr = fit_plane(P[nb])
            cands.append(pr)
        if cylinders:
            free_mask = labels < 0
            for i in rng.choice(free, min(400, len(free)), replace=False):
                nb = [j for j in tree.query_ball_point(P[i], 12.0) if free_mask[j]]
                if len(nb) < 10:
                    continue
                j = nb[rng.integers(len(nb))]
                if np.linalg.norm(P[j] - P[i]) < 3.0:
                    continue
                a = np.cross(N[i], N[j])
                if np.linalg.norm(a) < 0.15:
                    continue
                a = _unit(a)
                u, v = _perp_basis(a)
                # 2D: intersect the two normal lines in the plane perpendicular to the axis
                pi = np.array([P[i] @ u, P[i] @ v]); pj = np.array([P[j] @ u, P[j] @ v])
                ni = np.array([N[i] @ u, N[i] @ v]); nj = np.array([N[j] @ u, N[j] @ v])
                A = np.c_[ni, -nj]
                if abs(np.linalg.det(A)) < 1e-3:
                    continue
                ti, tj = np.linalg.solve(A, pj - pi)
                c2 = pi + ti * ni
                r = np.linalg.norm(pi - c2)
                if not (1.5 < r < 150.0) or abs(np.linalg.norm(pj - c2) - r) > eps:
                    continue
                p0 = c2[0] * u + c2[1] * v + (P[i] @ a) * a
                cands.append({"type": "cylinder", "a": a, "p": p0, "r": float(r)})
        if not cands:
            break
        # quick score on the probe subset
        Q, NQ = P[probe], N[probe]
        scores = []
        for c in cands:
            d = np.abs(prim_dist(c, Q))
            ok = d < eps
            if ok.sum() < 3:
                scores.append(0)
                continue
            cs = signed_cos(c, Q[ok], NQ[ok]) if oriented else np.abs(np.sum(prim_normals(c, Q[ok]) * NQ[ok], axis=1))
            scores.append(int((cs > cos_t).sum()))
        order = np.argsort(scores)[::-1][:6]
        best, best_idx = None, None
        for k in order:
            c = cands[k]
            for _ in range(3):  # inliers -> refit -> inliers
                d = np.abs(prim_dist(c, P[free]))
                ok = d < eps
                cs = (signed_cos(c, P[free][ok], N[free][ok]) if oriented
                      else np.abs(np.sum(prim_normals(c, P[free][ok]) * N[free][ok], axis=1)))
                idx = free[ok][cs > cos_t]
                # refit on the connected region only: letting distant strips join during the
                # iterations makes a plane tilt to bridge e.g. a pad top and a pocket floor
                idx = _largest_component(P, idx, link)
                if len(idx) < min_pts:
                    break
                if c["type"] == "plane":
                    c = fit_plane(P[idx])
                else:
                    c = fit_cylinder(P[idx], c["a"], c["p"], c["r"])
                    if not (1.0 < c["r"] < 300.0):
                        idx = idx[:0]
                        break
            if c["type"] == "cylinder" and len(idx) >= min_pts:
                # a cylinder must really curve over its region, otherwise it is a plane in disguise
                u, v = _perp_basis(c["a"])
                q = P[idx] - c["p"]
                ang = np.arctan2(q @ v, q @ u)
                span = _angular_span(ang)
                if span < np.radians(25) or c["r"] > 4 * np.ptp(q @ c["a"]) + 50:
                    continue
                # model selection: a cylinder must fit clearly better than a plane, close to the noise,
                # and its arc must be evenly covered (not two flat walls of a channel)
                # the curvature must be clearly measurable: sagitta over the covered arc >= 3 x noise,
                # otherwise a flat face (plus a bit of a neighbour) passes as a large-radius arc
                if c["r"] * (1 - np.cos(min(span, np.pi) / 2)) < 3.0 * max(noise_mm, 0.2):
                    continue
                rc = np.sqrt(np.mean(prim_dist(c, P[idx]) ** 2))
                rp = np.sqrt(np.mean(plane_dist(fit_plane(P[idx]), P[idx]) ** 2))
                if rc > 1.8 * max(noise_mm, 0.2) or rp < 1.3 * rc:
                    continue
                a_s = np.mod(ang - ang.min(), 2 * np.pi)
                hist, _ = np.histogram(a_s, bins=8, range=(0, max(span, 1e-6)))
                if (hist > 0.25 * len(idx) / 8).mean() < 0.75:
                    continue
                c["ang"] = span
            if best is None or len(idx) > len(best_idx):
                best, best_idx = c, idx
        if best is None or len(best_idx) < min_pts:
            fails += 1
            if fails >= 4:
                break
            continue
        fails = 0
        if best["type"] == "plane":
            # converged: now also take the other strips of the same plane (split by grooves etc.)
            d = np.abs(prim_dist(best, P[free]))
            ok = d < eps
            cs = (signed_cos(dict(best), P[free][ok], N[free][ok]) if oriented
                  else np.abs(np.sum(prim_normals(best, P[free][ok]) * N[free][ok], axis=1)))
            more = _largest_component(P, free[ok][cs > cos_t], link, min_size=min_pts)
            if len(more) > len(best_idx):
                best_idx = np.union1d(best_idx, more)
        if oriented:
            signed_cos(best, P[best_idx], N[best_idx])
        r = np.abs(prim_dist(best, P[best_idx]))
        best["rms"] = float(np.sqrt(np.mean(r ** 2)))
        best["n_pts"] = int(len(best_idx))
        labels[best_idx] = len(prims)
        prims.append(best)
    # grow: unlabeled points within eps of a primitive whose region is nearby
    free = np.where(labels < 0)[0]
    if len(free) and prims:
        lab_tree = cKDTree(P[labels >= 0])
        lab_of = labels[labels >= 0]
        dnb, inb = lab_tree.query(P[free], k=8, distance_upper_bound=link * 2)
        for row, i in enumerate(free):
            cand = {lab_of[j] for j, dd in zip(inb[row], dnb[row]) if np.isfinite(dd)}
            best_k, best_d = -1, eps
            for k in cand:
                dd = abs(prim_dist(prims[k], P[i:i + 1])[0])
                cs = prim_normals(prims[k], P[i:i + 1])[0] @ N[i]
                cs = cs if oriented else abs(cs)
                if dd < best_d and cs > np.cos(np.radians(35)):
                    best_k, best_d = k, dd
            labels[i] = best_k
    return prims, labels


def _angular_span(ang):
    a = np.sort(np.mod(ang, 2 * np.pi))
    gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
    return 2 * np.pi - gaps.max()


def merge_similar(prims, labels, P, noise_mm):
    """Merge primitives that describe the same surface (split by RANSAC because of noise/occlusion)."""
    eps = max(2.5 * noise_mm, 0.8)
    changed = True
    while changed:
        changed = False
        for i in range(len(prims)):
            for j in range(i + 1, len(prims)):
                a, b = prims[i], prims[j]
                if a is None or b is None or a["type"] != b["type"] or a["type"] == "extrusion":
                    continue
                Pi, Pj = P[labels == i], P[labels == j]
                if len(Pi) == 0 or len(Pj) == 0:
                    continue
                if a.get("s", 1.0) * b.get("s", 1.0) * (a["n"] @ b["n"] if a["type"] == "plane" else 1.0) < 0:
                    continue  # opposite faces (e.g. both sides of a sheet) are never the same surface
                if a["type"] == "plane" and abs(a["n"] @ b["n"]) < np.cos(np.radians(3)):
                    continue
                if a["type"] == "cylinder" and (abs(a["a"] @ b["a"]) < np.cos(np.radians(3)) or abs(a["r"] - b["r"]) > 2 * eps):
                    continue
                tol = max(0.3, 6.0 * noise_mm / np.sqrt(min(len(Pi), len(Pj))))
                if a["type"] == "cylinder":
                    tol = max(tol, 0.5 * eps)
                if abs(np.median(prim_dist(a, Pj))) < tol and abs(np.median(prim_dist(b, Pi))) < tol:
                    both = np.r_[Pi, Pj]
                    new = fit_plane(both) if a["type"] == "plane" else fit_cylinder(both, a["a"], a["p"], a["r"])
                    if "s" in a:
                        if a["type"] == "plane":
                            new["s"] = a["s"] * float(np.sign(new["n"] @ a["n"]) or 1.0)
                        else:
                            new["s"] = a["s"]
                    new["rms"] = float(np.sqrt(np.mean(prim_dist(new, both) ** 2)))
                    new["n_pts"] = len(both)
                    prims[i] = new
                    prims[j] = None
                    labels[labels == j] = i
                    changed = True
    keep = [k for k, p in enumerate(prims) if p is not None]
    remap = -np.ones(len(prims) + 1, int)
    remap[keep] = np.arange(len(keep))
    labels = np.where(labels >= 0, remap[labels], -1)
    return [prims[k] for k in keep], labels


# ----------------------------------------------------------------------------------------------
# 2. candidate surfaces (oversized faces) and arrangement
# ----------------------------------------------------------------------------------------------

def _occ():
    """Import the OpenCASCADE pieces we need (cadquery-ocp)."""
    import OCP  # noqa: F401
    return True


def make_face(prim, P, margin, box=None):
    """Bounded OCC face for a primitive. With `box` (lo, hi) the surface is extended over the whole
    (expanded) bounding box of the object, so neighbouring surfaces always meet (PolyFit-style);
    otherwise it covers its own region plus `margin` mm."""
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Ax3, gp_Cylinder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    if prim["type"] == "plane":
        n = prim["n"]
        c = P.mean(0) if len(P) else prim.get("c", -prim["d"] * n)
        c = c - (c @ n + prim["d"]) * n
        u, v = _perp_basis(n)
        q = P - c
        if box is not None:
            lo, hi = box
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            q = np.r_[q, corners - c]
        uu, vv = q @ u, q @ v
        ax = gp_Ax3(gp_Pnt(*c), gp_Dir(*n), gp_Dir(*u))
        f = BRepBuilderAPI_MakeFace(gp_Pln(ax), float(uu.min() - margin), float(uu.max() + margin),
                                    float(vv.min() - margin), float(vv.max() + margin)).Face()
        return f
    a, p0, r = prim["a"], prim["p"], prim["r"]
    u, v = _perp_basis(a)
    q = P - p0
    t = q @ a
    ang = np.arctan2(q @ v, q @ u)
    if box is not None:
        lo, hi = box
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        t = np.r_[t, (corners - p0) @ a]
    # angular window: centre of the occupied arc, widened by the margin (as arc length)
    a_sorted = np.sort(np.mod(ang, 2 * np.pi))
    gaps = np.diff(np.r_[a_sorted, a_sorted[0] + 2 * np.pi])
    k = gaps.argmax()
    start = a_sorted[(k + 1) % len(a_sorted)]
    span = 2 * np.pi - gaps[k]
    dm = margin / max(r, 1e-3)
    a0, a1 = start - dm, start + span + dm
    if a1 - a0 >= 2 * np.pi - 1e-6:
        a0, a1 = 0.0, 2 * np.pi
    ax = gp_Ax3(gp_Pnt(*p0), gp_Dir(*a), gp_Dir(*u))
    return BRepBuilderAPI_MakeFace(gp_Cylinder(ax, r), float(a0), float(a1),
                                   float(t.min() - margin), float(t.max() + margin)).Face()


def _faces_of(shape):
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    out = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        out.append(TopoDS.Face(ex.Current()))
        ex.Next()
    return out


def _sample_face(face, step):
    """Points (and their face) sampled on a face via its triangulation, ~one per step^2 mm^2."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    face = TopoDS.Face(face)
    BRepMesh_IncrementalMesh(face, step * 0.5, False, 0.3, True)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    _sample_face.node = None
    if tri is None:
        return np.zeros((0, 3)), 0.0
    tr = loc.Transformation()
    V = np.array([[tri.Node(i).Transformed(tr).X(), tri.Node(i).Transformed(tr).Y(), tri.Node(i).Transformed(tr).Z()]
                  for i in range(1, tri.NbNodes() + 1)])

    T = np.array([[tri.Triangle(i).Value(j) - 1 for j in (1, 2, 3)] for i in range(1, tri.NbTriangles() + 1)])
    if len(T) == 0:
        return np.zeros((0, 3)), 0.0
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    n = np.maximum(1, np.round(area / (step * step)).astype(int))
    rng = np.random.default_rng(0)
    idx = np.repeat(np.arange(len(T)), n)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    s1 = np.sqrt(r1)
    pts = (1 - s1)[:, None] * A[idx] + (s1 * (1 - r2))[:, None] * B[idx] + (s1 * r2)[:, None] * C[idx]
    # random samples on a curved face lie on the chords of its mesh (up to the deflection away):
    # one sample projected onto the surface itself tells which primitive the face belongs to
    _sample_face.node = pts[len(pts) // 2][None]
    try:
        from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
        from OCP.gp import gp_Pnt
        pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, pts[len(pts) // 2])), BRep_Tool.Surface_s(face))
        if pr.NbPoints():
            q = pr.NearestPoint()
            _sample_face.node = np.array([[q.X(), q.Y(), q.Z()]])
    except Exception:
        pass
    return pts, float(area.sum())


def build_solid(prims, labels, P, noise_mm, floor_z=None, margin=6.0, cover_r=None, step=1.5,
                w_cover=0.6, w_complex=0.3, time_limit=300.0, ref_surface=None, keep_frac=0.6,
                ref_labels=None):
    """Arrangement of oversized primitive faces + integer selection -> closed solid.

    Returns (shape, report) where shape is a TopoDS_Shape (solid if closing succeeded, else shell)."""
    from OCP.BOPAlgo import BOPAlgo_Builder
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix

    eps = max(2.5 * noise_mm, 0.8)
    cover_r = cover_r or max(2.0, 4 * noise_mm)
    src_faces, src_kind = [], []
    lo, hi = P.min(0) - margin, P.max(0) + margin
    if floor_z is not None:
        lo[2] = min(lo[2], floor_z - margin)
    for k, pr in enumerate(prims):
        Pk = P[labels == k]
        if len(Pk) < 3:
            continue
        src_faces.append(make_face(pr, Pk, margin, box=(lo, hi)))
        src_kind.append(k)
    if floor_z is not None:  # unseen underside: the floor the part stands on closes the solid
        f = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, float(floor_z)), gp_Dir(0, 0, 1)),
                                    float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])).Face()
        src_faces.append(f)
        src_kind.append(-2)

    bop = BOPAlgo_Builder()
    for f in src_faces:
        bop.AddArgument(f)
    bop.SetRunParallel(True)
    bop.SetFuzzyValue(1e-4)
    bop.Perform()
    if bop.HasErrors():
        raise RuntimeError("general fuse mislukt")
    res = bop.Shape()

    # fragments with their source surface
    frags, frag_src = [], []
    for f, k in zip(src_faces, src_kind):
        lst = list(_iter_list(bop.Modified(f))) or [f]
        for g in lst:
            frags.append(g)
            frag_src.append(k)
    nF = len(frags)

    # candidate pruning: a fragment can only be part of the model if it lies on the scanned surface
    # (scan points + watertight reference mesh samples). This removes the thousands of fragments far
    # inside or outside the object that full-bbox surfaces create, and keeps the ILP small.
    if ref_labels is not None:
        ref, ref_lab = ref_surface, ref_labels
    else:
        ref = P if ref_surface is None else np.r_[P, ref_surface]
        ref_lab = None
    ref_tree = cKDTree(ref)
    d_keep = max(3.0, 4 * noise_mm)
    keep_ids, agree_all = [], []
    foot = _footprint(P, floor_z) if floor_z is not None else None
    samples_all = []
    for i, g in enumerate(frags):
        S, _ = _sample_face(g, step)
        samples_all.append(S)
        if len(S) == 0:
            continue
        dd, jj = ref_tree.query(S, distance_upper_bound=d_keep)
        ok = np.isfinite(dd)
        if ref_lab is not None and frag_src[i] == -2:
            # floor under the part: neutral (unseen), allowed inside the footprint of the part
            if _in_footprint(S, foot):
                keep_ids.append(i)
                agree_all.append(0.5)
            continue
        if ref_lab is not None:
            agree = float((ref_lab[jj[ok]] == frag_src[i]).sum() / len(S))
            # unknown (-1) reference samples count half: the surface may well continue there
            agree += 0.5 * float((ref_lab[jj[ok]] == -1).sum() / len(S))
            if ok.mean() >= 0.5:
                keep_ids.append(i)
                agree_all.append(agree)
            continue
        near = ok.mean()
        if frag_src[i] == -2:
            if ref_surface is None or near >= 0.3:
                keep_ids.append(i)
                agree_all.append(near)
        elif near >= keep_frac:
            keep_ids.append(i)
            agree_all.append(near)
    frags = [frags[i] for i in keep_ids]
    frag_src = [frag_src[i] for i in keep_ids]
    pre_samples = [samples_all[i] for i in keep_ids]
    nF = len(frags)

    # edge -> incident fragments (OCC shapes hash by TShape+location, orientation-independent)
    edge_faces = {}
    for i, g in enumerate(frags):
        ex = TopExp_Explorer(g, TopAbs_EDGE)
        seen = set()
        while ex.More():
            e = ex.Current()
            h = hash(e)
            if h not in seen:
                seen.add(h)
                edge_faces.setdefault(h, set()).add(i)
            ex.Next()

    # support / coverage per fragment
    supp = np.zeros(nF)
    cover = np.zeros(nF)
    area = np.zeros(nF)
    samples = pre_samples
    for i, S in enumerate(samples):
        area[i] = len(S) * step * step
    for k in sorted(set(frag_src)):
        ids = [i for i in range(nF) if frag_src[i] == k]
        if k < 0:
            continue
        Pk = P[labels == k]
        allS = np.vstack([samples[i] for i in ids if len(samples[i])]) if ids else np.zeros((0, 3))
        owner = np.concatenate([[i] * len(samples[i]) for i in ids if len(samples[i])]).astype(int) if ids else np.zeros(0, int)
        if len(allS) == 0 or len(Pk) == 0:
            continue
        d, j = cKDTree(allS).query(Pk, distance_upper_bound=cover_r)
        hit = np.isfinite(d)
        np.add.at(supp, owner[j[hit]], 1.0)
        tk = cKDTree(Pk)
        for i in ids:
            if len(samples[i]):
                dd, _ = tk.query(samples[i], distance_upper_bound=cover_r)
                cover[i] = np.isfinite(dd).mean()
    n_tot = max(1, (labels >= 0).sum())
    A_tot = max(1.0, area[[i for i in range(nF) if frag_src[i] >= 0]].sum())

    # objective (minimise): -data fit + uncovered area. Floor fragments: no penalty (expected unseen).
    c = np.zeros(nF)
    agree_kept = np.array(agree_all)
    if ref_lab is not None:
        # fragment wanted where the labelled reference mesh says "this surface is here"
        c = -(agree_kept - 0.5) * area / A_tot
    for i in (range(nF) if ref_lab is None else []):
        if frag_src[i] == -2:
            c[i] = 1e-4 * area[i] / A_tot
        else:
            c[i] = -supp[i] / n_tot + w_cover * (1 - cover[i]) * area[i] / A_tot

    # manifold constraints: for each edge, sum of selected incident faces = 2 * y_e
    rows = [sorted(v) for v in edge_faces.values()]
    nE = len(rows)
    # complexity (PolyFit): an edge where faces of two different surfaces are both chosen is a sharp
    # edge of the model; z_p >= x_i + x_j - 1 for every such pair, penalised per unit edge length.
    pairs, plen = [], []
    for inc in rows:
        for a_ in range(len(inc)):
            for b_ in range(a_ + 1, len(inc)):
                i, j = inc[a_], inc[b_]
                if frag_src[i] != frag_src[j]:
                    pairs.append((i, j))
    nP = len(pairs)
    nV = nF + nE + nP
    A = lil_matrix((nE + nP, nV))
    lo_b = np.zeros(nE + nP)
    hi_b = np.zeros(nE + nP)
    for r, inc in enumerate(rows):
        for i in inc:
            A[r, i] = 1
        A[r, nF + r] = -2
    for q, (i, j) in enumerate(pairs):
        r = nE + q
        A[r, i] = 1
        A[r, j] = 1
        A[r, nF + nE + q] = -1
        lo_b[r] = -np.inf
        hi_b[r] = 1
    cc = np.r_[c, np.zeros(nE), np.full(nP, w_complex / max(1, nE))]
    sol = milp(cc, constraints=[LinearConstraint(A.tocsr(), lo_b, hi_b)], integrality=np.ones(nV),
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "disp": False})
    if sol.x is None:
        raise RuntimeError(f"selectie onoplosbaar ({sol.message})")
    chosen = [i for i in range(nF) if sol.x[i] > 0.5]
    shape = _sew_solid([frags[i] for i in chosen])
    build_solid.debug = dict(frags=frags, frag_src=frag_src, supp=supp, cover=cover, area=area, chosen=chosen, rows=rows)
    rep = {"surfaces": len(src_faces), "fragments": nF, "edges": nE, "chosen": len(chosen),
           "floor_faces": sum(1 for i in chosen if frag_src[i] == -2),
           "status": sol.message}
    return shape, rep


def _footprint(P, floor_z, cell=2.0, band=20.0, dilate=3):
    from scipy.ndimage import binary_dilation, binary_fill_holes
    Q = P[P[:, 2] < floor_z + band]
    if len(Q) == 0:
        Q = P
    lo = Q[:, :2].min(0) - (dilate + 2) * cell
    ij = np.floor((Q[:, :2] - lo) / cell).astype(int)
    shp = ij.max(0) + dilate + 3
    g = np.zeros(shp, bool)
    g[ij[:, 0], ij[:, 1]] = True
    g = binary_fill_holes(binary_dilation(g, iterations=dilate))
    return lo, cell, g


def _in_footprint(S, foot, frac=0.5):
    if foot is None or len(S) == 0:
        return True
    lo, cell, g = foot
    ij = np.floor((S[:, :2] - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < g.shape[0]) & (ij[:, 1] < g.shape[1])
    inside = np.zeros(len(S), bool)
    inside[ok] = g[ij[ok, 0], ij[ok, 1]]
    return inside.mean() >= frac


def _iter_list(lst):
    for x in lst:
        yield x


def _sew_solid(faces, tol=1e-3):
    """Sew faces; turn every closed shell into a solid and return the largest solid (by volume),
    or a compound of shells when nothing closes."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopoDS import TopoDS
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    sew = BRepBuilderAPI_Sewing(tol)
    for f in faces:
        sew.Add(f)
    sew.Perform()
    sh = sew.SewedShape()
    best, best_v = None, 0.0
    ex = TopExp_Explorer(sh, TopAbs_SHELL)
    while ex.More():
        shell = TopoDS.Shell(ex.Current())
        ex.Next()
        ms = BRepBuilderAPI_MakeSolid(shell)
        if not ms.IsDone():
            continue
        fix = ShapeFix_Solid(ms.Solid())
        fix.Perform()
        so = fix.Solid()
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(so, g)
        v = abs(g.Mass())
        if v > best_v:
            best, best_v = so, v
    if best is None:
        return sh
    try:
        u = ShapeUpgrade_UnifySameDomain(best, True, True, False)
        u.Build()
        best = u.Shape()
    except Exception:
        pass
    return best


def shape_report(shape, P=None, step=0.3):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.TopAbs import TopAbs_SOLID
    rep = {"valid": bool(BRepCheck_Analyzer(shape).IsValid()), "solid": shape.ShapeType() == TopAbs_SOLID,
           "faces": len(_faces_of(shape))}
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    rep["volume_mm3"] = float(g.Mass())
    if P is not None:
        S = np.vstack([_sample_face(f, step)[0] for f in _faces_of(shape)])
        d, _ = cKDTree(S).query(P)
        rep["dev_median"] = float(np.median(d))
        rep["dev_p95"] = float(np.percentile(d, 95))
    return rep


def write_step(shape, path, name="part"):
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.Interface import Interface_Static
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214")
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    return w.Write(str(path)) == 1


def label_reference_mesh(V, F, prims, labels, P, noise_mm, floor_z=None, n_samples=200_000, seed=0):
    """Sample the watertight reference mesh and label every sample with the primitive it belongs to
    (majority of nearby labelled scan points, confirmed by distance to that primitive), the floor (-2)
    for the closed underside, or -1 (unknown / freeform)."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    S = np.asarray(m.sample_points_uniformly(n_samples).points)
    eps = max(2.5 * noise_mm, 0.8)
    lab = -np.ones(len(S), int)
    good = labels >= 0
    tree = cKDTree(P[good])
    pl = labels[good]
    d, j = tree.query(S, k=7, distance_upper_bound=3 * eps)
    for i in range(len(S)):
        ls = [pl[x] for x, dd in zip(j[i], d[i]) if np.isfinite(dd)]
        if not ls:
            continue
        vals, cnt = np.unique(ls, return_counts=True)
        k = vals[cnt.argmax()]
        if cnt.max() >= 0.5 * len(ls) and abs(prim_dist(prims[k], S[i:i + 1])[0]) < 2 * eps:
            lab[i] = k
    if floor_z is not None:
        lab[(lab < 0) & (S[:, 2] < floor_z + eps)] = -2
    return S, lab


# ----------------------------------------------------------------------------------------------
# 3. mesh-guided B-rep: topology from the watertight mesh, geometry from the fitted surfaces
# ----------------------------------------------------------------------------------------------
#
# The arrangement + selection above is elegant but fragile on noisy, partially segmented scans
# (many small surfaces slice everything). The mesh already has the right topology (it is closed and
# follows the scan), so we let it decide which surface meets which, and only replace its geometry:
# every vertex is projected onto its surface, onto the intersection curve of two surfaces, or onto the
# corner point of three; every region becomes one B-rep face on the exact analytic surface.


def _project_multi(x, prs, iters=30):
    """Point on (approximately) the common intersection of several surfaces: alternating projection."""
    y = x.copy()
    for _ in range(iters):
        y_old = y.copy()
        for pr in prs:
            d = prim_dist(pr, y[None])[0]
            y = y - d * prim_normals(pr, y[None])[0]
        if np.linalg.norm(y - y_old) < 1e-7:
            break
    return y


def label_mesh(V, F, prims, labels, P, N, noise_mm, floor_z=None, min_tris=25):
    """Primitive label per triangle (-1 = freeform/unknown), smoothed over the mesh."""
    eps = max(2.5 * noise_mm, 0.8)
    C = V[F].mean(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = 0.5 * np.linalg.norm(fn, axis=1)
    # mesh triangle normals are far too noisy (tens of degrees); use the local PCA normal of the
    # scan points around each triangle instead
    _, nn = cKDTree(P).query(C, k=24)
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    cov = np.einsum("tki,tkj->tij", Q, Q)
    _, vecs = np.linalg.eigh(cov)
    fn = vecs[:, :, 0]
    good = labels >= 0
    tree = cKDTree(P[good])
    pl = labels[good]
    d, j = tree.query(C, k=40, distance_upper_bound=8 * eps)
    lab = -np.ones(len(F), int)
    near_labels = [set(pl[j[t][np.isfinite(d[t])]].tolist()) for t in range(len(F))]
    if floor_z is not None:
        k_floor = len(prims)
        fl = (lab < 0) & (np.abs(C[:, 2] - floor_z) < eps) & (np.abs(fn[:, 2]) > 0.8)
        lab[fl] = k_floor
    # neighbours by shared edges
    from collections import defaultdict
    e2t = defaultdict(list)
    for t, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            e2t[(min(u, v), max(u, v))].append(t)
    nb = [[] for _ in range(len(F))]
    for ts in e2t.values():
        if len(ts) == 2:
            nb[ts[0]].append(ts[1])
            nb[ts[1]].append(ts[0])
    all_prims = prims + ([{"type": "plane", "n": np.array([0, 0, 1.0]), "d": -float(floor_z)}] if floor_z is not None else [])

    def ok_for(t, k):
        return abs(prim_dist(all_prims[k], C[t:t + 1])[0]) < 3 * eps

    log(f"    labels vooraf: onbekend {np.mean(lab < 0):.3f}")
    # Regularise with ICM on a Potts energy: data = squared distance to the surface (in eps) plus
    # normal disagreement; unknown (freeform) has a constant cost; neighbours with different labels
    # pay `smooth`. Removes the speckle that per-triangle decisions leave behind.
    K = len(all_prims)
    unk_cost = 3.0
    smooth = 1.2

    def data_cost(t, k):
        if k < 0:
            return unk_cost
        pr = all_prims[k]
        dd = prim_dist(pr, C[t:t + 1])[0] / eps
        cs = abs(prim_normals(pr, C[t:t + 1])[0] @ fn[t])
        return min(dd * dd, 4.0) + 1.5 * (1.0 - cs)

    # initial labels: cheapest surface among those whose scan points are nearby
    for t in range(len(F)):
        if lab[t] >= 0:
            continue
        best_k, best_e = -1, unk_cost
        for k in near_labels[t]:
            e = data_cost(t, k)
            if e < best_e:
                best_k, best_e = k, e
        lab[t] = best_k
    log(f"    labels start: onbekend {np.mean(lab < 0):.3f}, zonder kandidaat {np.mean([len(x) == 0 for x in near_labels]):.3f}")
    for _ in range(30):
        changed = 0
        for t in range(len(F)):
            cands = {lab[t], -1} | {lab[s_] for s_ in nb[t]} | near_labels[t]
            best_k, best_e = lab[t], None
            for k in cands:
                e = data_cost(t, k) + smooth * sum(1 for s_ in nb[t] if lab[s_] != k)
                if best_e is None or e < best_e - 1e-9:
                    best_k, best_e = k, e
            if best_k != lab[t]:
                lab[t] = best_k
                changed += 1
        log(f"    ICM: {changed} gewijzigd, onbekend {np.mean(lab < 0):.3f}")
        if changed == 0:
            break
    # tiny components -> unknown
    comp = _components(lab, nb)
    for cid in np.unique(comp):
        m = comp == cid
        if lab[m][0] >= 0 and m.sum() < min_tris:
            lab[m] = -1
    return lab, nb, all_prims, area


def _components(lab, nb):
    comp = -np.ones(len(lab), int)
    c = 0
    for s in range(len(lab)):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            t = stack.pop()
            for u in nb[t]:
                if comp[u] < 0 and lab[u] == lab[s]:
                    comp[u] = c
                    stack.append(u)
        c += 1
    return comp


def snap_vertices(V, F, lab, all_prims, noise_mm, max_move=None):
    """Move every vertex onto its surface / intersection curve / corner. Freeform vertices stay."""
    eps = max(2.5 * noise_mm, 0.8)
    max_move = max_move or 4 * eps
    vlabs = [set() for _ in range(len(V))]
    vfree = np.zeros(len(V), bool)
    for t, tri in enumerate(F):
        for v in tri:
            if lab[t] >= 0:
                vlabs[v].add(int(lab[t]))
            else:
                vfree[v] = True
    W = V.copy()
    moved = np.zeros(len(V))
    kinds = np.zeros(len(V), int)
    for v in range(len(V)):
        ks = sorted(vlabs[v])
        if not ks:
            continue
        prs = [all_prims[k] for k in ks]
        # drop near-parallel duplicates (no well-defined intersection)
        sel = []
        for pr in prs:
            nrm = prim_normals(pr, V[v][None])[0]
            if all(abs(nrm @ prim_normals(q, V[v][None])[0]) < np.cos(np.radians(12)) for q in sel):
                sel.append(pr)
        sel = sel[:3]
        y = _project_multi(V[v], sel)
        mv = np.linalg.norm(y - V[v])
        if mv > max_move and len(sel) > 1:
            y = _project_multi(V[v], sel[:1])
            mv = np.linalg.norm(y - V[v])
        if mv <= max_move:
            W[v] = y
            moved[v] = mv
            kinds[v] = len(sel)
    return W, kinds, moved


def mesh_brep(W, F, lab, all_prims, tol=1e-3):
    """Build a sewn B-rep from a labelled, snapped mesh. Returns (shape, info)."""
    from collections import defaultdict
    from OCP.gp import gp_Pnt, gp_Pln, gp_Ax3, gp_Dir, gp_Cylinder
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
                                    BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid)
    from OCP.Geom import Geom_Plane, Geom_CylindricalSurface
    from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Solid
    from OCP.TopoDS import TopoDS

    e2t = defaultdict(list)
    for t, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            e2t[(min(u, v), max(u, v))].append(t)
    nb = [[] for _ in range(len(F))]
    for ts in e2t.values():
        if len(ts) == 2:
            nb[ts[0]].append(ts[1])
            nb[ts[1]].append(ts[0])
    comp = _components(lab, nb)

    verts = {}

    def vx(i):
        if i not in verts:
            verts[i] = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, W[i]))).Vertex()
        return verts[i]

    seg_edges = {}

    def seg(i, j):
        key = (min(i, j), max(i, j))
        if key not in seg_edges:
            mk = BRepBuilderAPI_MakeEdge(vx(key[0]), vx(key[1]))
            seg_edges[key] = mk.Edge() if mk.IsDone() else None
        return seg_edges[key]

    chain_edges = {}

    def other_comp(t, u, v):
        for s in e2t[(min(u, v), max(u, v))]:
            if s != t:
                return comp[s]
        return -1

    faces = []
    n_analytic = n_free = 0
    mesh_brep.debug = []
    for cid in np.unique(comp):
        tris = np.where(comp == cid)[0]
        k = lab[tris[0]]
        if k < 0:
            # freeform fallback: one planar face per triangle (to be replaced by NURBS patches)
            for t in tris:
                a, b, c = F[t]
                es = [seg(a, b), seg(b, c), seg(c, a)]
                if any(e is None for e in es):
                    continue
                w = BRepBuilderAPI_MakeWire(*es)
                if w.IsDone():
                    mf = BRepBuilderAPI_MakeFace(w.Wire(), True)
                    if mf.IsDone():
                        faces.append(mf.Face())
                        n_free += 1
            continue
        # directed boundary edges of this region with the neighbour component across
        nxt = {}
        across = {}
        for t in tris:
            a, b, c = F[t]
            for u, v in ((a, b), (b, c), (c, a)):
                oc = other_comp(t, u, v)
                if oc != cid:
                    nxt.setdefault(u, []).append(v)
                    across[(u, v)] = oc
        loops = []
        used = set()
        for u0 in list(nxt):
            for v0 in nxt[u0]:
                if (u0, v0) in used:
                    continue
                loop = [u0]
                u, v = u0, v0
                while (u, v) not in used:
                    used.add((u, v))
                    loop.append(v)
                    cand = [w for w in nxt.get(v, []) if (v, w) not in used]
                    if not cand:
                        break
                    u, v = v, cand[0]
                if loop[-1] == loop[0] and len(loop) > 3:
                    loops.append(loop[:-1])
        if not loops:
            continue
        wires = []
        for loop in loops:
            n = len(loop)
            oc = [across[(loop[i], loop[(i + 1) % n])] for i in range(n)]
            # rotate so the loop starts at a change of neighbour (chain start)
            starts = [i for i in range(n) if oc[i] != oc[i - 1]]
            r = starts[0] if starts else 0
            loop = loop[r:] + loop[:r]
            oc = oc[r:] + oc[:r]
            edges = []
            i = 0
            while i < n:
                j = i
                while j + 1 < n and oc[j + 1] == oc[i]:
                    j += 1
                chain = loop[i:j + 2] if j + 1 < n else loop[i:] + [loop[0]]
                nb_lab = lab[np.where(comp == oc[i])[0][0]] if oc[i] >= 0 else -1
                pts = W[chain]
                straight = False
                if nb_lab >= 0 and len(chain) > 2:
                    d0 = pts[-1] - pts[0]
                    L = np.linalg.norm(d0)
                    if L > 1e-6:
                        q = pts - pts[0]
                        off = np.linalg.norm(q - np.outer(q @ (d0 / L), d0 / L), axis=1)
                        straight = off.max() < 0.02
                if straight:
                    key = tuple(sorted((chain[0], chain[-1]))) + ("L", tuple(sorted(chain)))
                    if key not in chain_edges:
                        chain_edges[key] = BRepBuilderAPI_MakeEdge(vx(chain[0]), vx(chain[-1])).Edge()
                    edges.append(chain_edges[key])
                else:
                    for a_, b_ in zip(chain[:-1], chain[1:]):
                        e_ = seg(a_, b_)
                        if e_ is not None:
                            edges.append(e_)
                i = j + 1
            mw = BRepBuilderAPI_MakeWire()
            for e in edges:
                mw.Add(e)
            if mw.IsDone():
                wires.append(mw.Wire())
        if not wires:
            continue
        pr = all_prims[k]
        if pr["type"] == "plane":
            n_ = pr["n"]
            c_ = -pr["d"] * n_
            u_, _ = _perp_basis(n_)
            surf = Geom_Plane(gp_Pln(gp_Ax3(gp_Pnt(*c_), gp_Dir(*n_), gp_Dir(*u_))))
        else:
            u_, _ = _perp_basis(pr["a"])
            surf = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(*pr["p"]), gp_Dir(*pr["a"]), gp_Dir(*u_)), pr["r"])
        # outer wire = the one enclosing the largest area (by its points' spread)
        sizes = []
        for loop in loops:
            q = W[loop]
            sizes.append(np.ptp(q, axis=0).prod() + np.ptp(q, axis=0).sum())
        order = np.argsort(sizes)[::-1]
        mf = BRepBuilderAPI_MakeFace(surf, wires[order[0]], True)
        for o in order[1:]:
            if o < len(wires):
                mf.Add(wires[o])
        if not mf.IsDone():
            continue
        fx = ShapeFix_Face(mf.Face())
        fx.SetPrecision(tol)
        fx.Perform()
        fc = fx.Face()
        # sanity: the face must have (about) the area of its mesh region; a reversed outer wire
        # makes OCC build the complement, which we then flip
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(fc, g)
        a_face = abs(g.Mass())
        tri = W[F[tris]]
        a_mesh = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        mesh_brep.debug.append((int(k), int(cid), a_face, a_mesh, len(loops)))
        faces.append(fc)
        n_analytic += 1
    sew = BRepBuilderAPI_Sewing(max(tol, 1e-3))
    for f in faces:
        sew.Add(f)
    sew.Perform()
    sh = sew.SewedShape()
    info = {"analytic_faces": n_analytic, "freeform_triangles": n_free, "free_edges": sew.NbFreeEdges()}
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SHELL
    ex = TopExp_Explorer(sh, TopAbs_SHELL)
    best, best_n = None, -1
    while ex.More():
        s_ = TopoDS.Shell(ex.Current())
        ex.Next()
        nf = len(_faces_of(s_))
        if nf > best_n:
            best, best_n = s_, nf
    if best is not None and best.Closed():
        ms = BRepBuilderAPI_MakeSolid(best)
        if ms.IsDone():
            fix = ShapeFix_Solid(ms.Solid())
            fix.Perform()
            so = fix.Solid()
            try:
                from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
                u = ShapeUpgrade_UnifySameDomain(so, True, True, False)
                u.Build()
                so = u.Shape()
            except Exception:
                pass
            info["closed"] = True
            return so, info
    info["closed"] = False
    return sh, info


def weld(W, F, lab, tol=1e-3):
    """Merge vertices that snapping put on the same spot; drop collapsed triangles."""
    q = np.round(W / tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    W2 = W[first]
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return W2, F2[ok], lab[ok]


# ----------------------------------------------------------------------------------------------
# 4. volumetric cell complex (robust): split the bounding box by all fitted surfaces into cells,
#    classify every cell inside/outside with a watertight reference volume, keep the boundary.
#    The result is closed by construction and every face lies exactly on a fitted surface.
# ----------------------------------------------------------------------------------------------

def _surface_face(prim, lo, hi):
    """Unbounded-enough face of a primitive covering the whole box (lo, hi)."""
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Ax3, gp_Cylinder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    diag = float(np.linalg.norm(hi - lo))
    ctr = 0.5 * (lo + hi)
    if prim["type"] == "plane":
        n = prim["n"]
        c = ctr - (ctr @ n + prim["d"]) * n
        u, _ = _perp_basis(n)
        ax = gp_Ax3(gp_Pnt(*c), gp_Dir(*n), gp_Dir(*u))
        return BRepBuilderAPI_MakeFace(gp_Pln(ax), -diag, diag, -diag, diag).Face()
    if prim["type"] == "extrusion":
        # the profile curve (B-spline through the fitted polyline, ends extended along their
        # tangents) swept along the extrusion direction over the whole box
        from OCP.Geom import Geom_SurfaceOfLinearExtrusion
        a = prim["a"]
        crv = extr_bspline(prim, float(ctr @ a) - diag)
        surf = Geom_SurfaceOfLinearExtrusion(crv, gp_Dir(*map(float, a)))
        return BRepBuilderAPI_MakeFace(surf, crv.FirstParameter(), crv.LastParameter(), 0.0, 2 * diag, 1e-6).Face()
    a, p0, r = prim["a"], prim["p"], prim["r"]
    t0 = (ctr - p0) @ a
    u, _ = _perp_basis(a)
    ax = gp_Ax3(gp_Pnt(*(p0 + t0 * a)), gp_Dir(*a), gp_Dir(*u))
    return BRepBuilderAPI_MakeFace(gp_Cylinder(ax, r), 0.0, 2 * np.pi, -diag, diag).Face()


def extr_bspline(prim, t0=0.0):
    """The B-spline profile curve of an extrusion primitive (3D, in the plane at height t0 along a)."""
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.OCP.collections import Array1_gp_Pnt
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.gp import gp_Pnt
    a, u, v = prim["a"], prim["u"], prim["v"]
    pts = prim["poly_ext"]
    arr = Array1_gp_Pnt(1, len(pts))
    for i, (x, y) in enumerate(pts):
        arr.SetValue(i + 1, gp_Pnt(*map(float, x * u + y * v + t0 * a)))
    return GeomAPI_PointsToBSpline(arr, 3, 3, GeomAbs_C2, 0.01).Curve()


def extr_resample(prim, step=0.2):
    """Make the distance model of an extrusion the B-spline itself: dense samples of the curve the
    face is built from, normals turned to the side of the given ones (prim['pnrm'])."""
    from OCP.GeomAdaptor import GeomAdaptor_Curve
    from OCP.GCPnts import GCPnts_UniformAbscissa
    crv = extr_bspline(prim, 0.0)
    ad = GeomAdaptor_Curve(crv)
    ua = GCPnts_UniformAbscissa(ad, step)
    u, v = prim["u"], prim["v"]
    X, T = [], []
    from OCP.gp import gp_Pnt, gp_Vec
    for i in range(1, ua.NbPoints() + 1):
        p, d1 = gp_Pnt(), gp_Vec()
        crv.D1(ua.Parameter(i), p, d1)
        q = np.array([p.X(), p.Y(), p.Z()])
        t = np.array([d1.X(), d1.Y(), d1.Z()])
        X.append([q @ u, q @ v])
        T.append([t @ u, t @ v])
    X, T = np.array(X), np.array(T)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    Nn = np.c_[-T[:, 1], T[:, 0]]
    t_old = cKDTree(prim["poly"])
    _, j = t_old.query(X)
    if np.sum(Nn * prim["pnrm"][j]) < 0:
        Nn = -Nn
    prim["poly"], prim["pnrm"] = X, Nn
    prim.pop("_tree", None)
    return prim


def _mesh_of_shape(shape, defl=0.2):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopAbs import TopAbs_REVERSED
    BRepMesh_IncrementalMesh(shape, defl, False, 0.3, True)
    Vs, Fs, off = [], [], 0
    for f in _faces_of(shape):
        loc = TopLoc_Location()
        t = BRep_Tool.Triangulation_s(f, loc)
        if t is None:
            continue
        tr = loc.Transformation()
        V = np.array([[t.Node(i).Transformed(tr).X(), t.Node(i).Transformed(tr).Y(), t.Node(i).Transformed(tr).Z()]
                      for i in range(1, t.NbNodes() + 1)])
        T = np.array([t.Triangle(i).Get() for i in range(1, t.NbTriangles() + 1)]) - 1
        if f.Orientation() == TopAbs_REVERSED:
            T = T[:, ::-1]
        Vs.append(V)
        Fs.append(T + off)
        off += len(V)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(Vs), np.vstack(Fs)


def reference_volume(P, Nrm, floor_z=None, depth=9):
    """Watertight occupancy from the oriented scan points (untrimmed screened Poisson).
    Returns occ(points)->bool. Below the floor is always outside."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(Nrm)
    m, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, scale=1.3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    def occ(Q):
        q = o3d.core.Tensor(np.asarray(Q, np.float32))
        inside = scene.compute_occupancy(q, nsamples=3).numpy() > 0.5
        if floor_z is not None:
            inside &= Q[:, 2] > floor_z
        return inside
    return occ, m


def _edge_position(t, noise_mm, bin_w=0.2):
    """Where a point cloud ends along a direction, for a side that no fitted surface bounds (the
    2 mm end face of a sheet: too narrow for normals and segmentation). t: coordinates, the edge is
    at the high end. Model of the density near the edge: a blurred step (faces running up to the
    edge) + a Gaussian bump of width noise (the end face itself, seen edge-on) + a constant
    (outliers). A step's midpoint is unbiased whatever its blur, unlike the extreme percentile,
    which lies ~3 noise outside. Returns None when the fit is not convincing."""
    from scipy.special import ndtr
    q = np.percentile(t, 99.9)
    lo_, hi_ = q - 12.0, q + 3.0
    tt = t[(t > lo_) & (t < hi_)]
    if len(tt) < 200:
        return None
    edges = np.arange(lo_, hi_ + bin_w, bin_w)
    h, _ = np.histogram(tt, edges)
    x = 0.5 * (edges[:-1] + edges[1:])
    sig = max(noise_mm, 0.1)
    best = None
    for st in (0.3 * sig, 0.6 * sig, sig):
        for e in np.arange(q - 4 * sig, q + 1.0, 0.02):
            M = np.c_[ndtr((e - x) / st), np.exp(-0.5 * ((x - e) / sig) ** 2), np.ones_like(x)]
            c, *_ = np.linalg.lstsq(M, h, rcond=None)
            if c[0] <= 0 or c[1] < 0 or c[2] < 0:
                continue
            r = float(np.sum((M @ c - h) ** 2))
            if best is None or r < best[0]:
                best = (r, e, c)
    if best is None:
        return None
    r, e, c = best
    plateau = c[0]
    if plateau < 3 * max(c[2], 0.5):  # no clear step above the background
        return None
    return float(e)


def _tight_box(P, prims, noise_mm, margin):
    """Bounding box of the cell complex. Sides bounded by a fitted surface facing that way stay at
    the robust extent + margin (the surface cuts the part off inside it). A side where the extreme
    points lie only on surfaces running along it (sheet edges) is put at the estimated edge: the box
    face is then the part's face."""
    lo = np.percentile(P, 0.02, axis=0) - margin
    hi = np.percentile(P, 99.98, axis=0) + margin
    eps = max(2.5 * noise_mm, 0.8)
    for ax in range(3):
        for sgn in (1.0, -1.0):
            t = sgn * P[:, ax]
            q = np.percentile(t, 99.9)
            S = P[t > q - 2.0]
            d = np.zeros(3)
            d[ax] = sgn
            facing = np.zeros(len(S), bool)
            for pr in prims:
                if pr.get("hyp"):
                    continue
                near = np.abs(prim_dist(pr, S)) < eps
                if near.any():
                    nn = prim_normals(pr, S[near])
                    facing[np.where(near)[0][np.abs(nn @ d) > np.cos(np.radians(30))]] = True
            if facing.mean() > 0.2:
                # the outermost fitted plane facing this way bounds the part: put the box just
                # outside it, so noise outliers leave no room for slivers (thin ridge tips)
                outer = [-(sgn * pr["d"] * np.sign(pr["n"][ax])) for pr in prims
                         if pr["type"] == "plane" and not pr.get("hyp") and abs(pr["n"][ax]) > 0.999]
                if outer:
                    pos = max(outer)
                    if np.mean(t > pos + 3 * noise_mm) < 1e-3 and q - 3 * noise_mm < pos:
                        if sgn > 0:
                            hi[ax] = min(hi[ax], pos + margin)
                        else:
                            lo[ax] = max(lo[ax], -(pos + margin))
                continue
            e = _edge_position(t, noise_mm)
            if e is None:
                continue
            if sgn > 0:
                hi[ax] = min(hi[ax], e)
            else:
                lo[ax] = max(lo[ax], -e)
    return lo, hi


def cell_solid(prims, P, occ, floor_z=None, margin=0.3, grid=0.6, min_frac=0.5, max_cells=60000, smooth=0.3,
               noise_mm=0.5, floor_plane=None, N=None, cams=None):
    """Cell complex of the bounding box split by every fitted surface; keep inside cells; return the
    boundary as one solid. Returns (shape, report)."""
    import open3d as o3d
    from OCP.BOPAlgo import BOPAlgo_MakerVolume
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    # tight box at the robust data extent: where no fitted surface bounds the part (narrow sheet
    # edges that got too few points) the box face is the best estimate, so it must not sit mm outside
    lo, hi = _tight_box(P, [p for p in prims], noise_mm, margin)
    if floor_z is not None:
        lo[2] = floor_z
        if floor_plane is not None:  # a tilted bottom may dip a little under the floor level
            lo[2] = min(lo[2], float(np.percentile(P[:, 2], 0.02)) - margin)
    box = BRepPrimAPI_MakeBox(gp_Pnt(*lo), gp_Pnt(*hi)).Shape()
    mv = BOPAlgo_MakerVolume()
    mv.AddArgument(box)
    for pr in prims:
        # untrimmed (trimming the faces to the box first makes OCC lose cells); cells that form
        # outside the box are discarded below
        mv.AddArgument(_surface_face(pr, lo, hi))
    mv.SetRunParallel(True)
    mv.SetIntersect(True)
    mv.Perform()
    if mv.HasErrors():
        raise RuntimeError("cellencomplex mislukt")
    cplx = mv.Shape()
    if any(pr["type"] == "extrusion" for pr in prims):
        # faces cut by a free-form curve can lack their 2D edge curves (they then cannot be meshed
        # or classified): let the standard healing add them
        from OCP.ShapeFix import ShapeFix_Shape
        fx = ShapeFix_Shape(cplx)
        fx.SetPrecision(1e-5)
        fx.SetMaxTolerance(1e-3)
        fx.Perform()
        cplx = fx.Shape()
    cells = []
    ex = TopExp_Explorer(cplx, TopAbs_SOLID)
    while ex.More():
        c = TopoDS.Solid(ex.Current())
        ex.Next()
        Vc, _ = _mesh_of_shape(c, defl=0.5)
        if len(Vc) and np.all(Vc.min(0) >= lo - 0.01) and np.all(Vc.max(0) <= hi + 0.01):
            cells.append(c)
        elif os.environ.get("FOTO2STEP_DEBUG_CELLS") and len(Vc):
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            g_ = GProp_GProps()
            BRepGProp.VolumeProperties_s(c, g_)
            log(f"    cel buiten doos: vol {g_.Mass():.0f} lo {np.round(Vc.min(0) - lo, 2)} hi {np.round(Vc.max(0) - hi, 2)}")
    if len(cells) > max_cells:
        raise RuntimeError(f"te veel cellen ({len(cells)})")
    # grid samples once, occupancy of the reference once
    axes = [np.arange(lo[i] + grid / 2, hi[i], grid) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    ref_in = occ(G)
    keep, fracs, vols = [], [], []
    for c in cells:
        V, T = _mesh_of_shape(c, defl=0.1)
        if len(T) == 0 and os.environ.get("FOTO2STEP_DEBUG_CELLS"):
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            g_ = GProp_GProps()
            BRepGProp.VolumeProperties_s(c, g_)
            log(f"    cel zonder mesh: vol {g_.Mass():.0f}")
        if len(T) == 0:  # keep fracs/keep aligned with cells (review finding)
            fracs.append(0.0)
            vols.append(0.0)
            keep.append(False)
            continue
        bl, bh = V.min(0), V.max(0)
        sel = np.all((G >= bl) & (G <= bh), axis=1)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            # tiny cell: classify its centroid
            ctr = V.mean(0)[None]
            f = float(occ(ctr)[0])
            fracs.append(f)
            vols.append(0.0)
            keep.append(f >= min_frac)
            continue
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(T.astype(np.uint32)))
        incell = sc.compute_occupancy(o3d.core.Tensor(G[idx].astype(np.float32))).numpy() > 0.5
        n_in = incell.sum()
        if n_in == 0:
            ctr = V.mean(0)[None]
            f = float(occ(ctr)[0])
        else:
            f = float(ref_in[idx][incell].mean())
        fracs.append(f)
        vols.append(n_in * grid ** 3)
        keep.append(f >= min_frac)
    fr = np.array(fracs)
    vol = np.array(vols)
    if smooth > 0 and len(cells) > 1:
        keep = _graphcut_cells(cells, fr, vol, smooth, P=P, eps=max(1.0, 2.5 * noise_mm),
                               floor_z=floor_z, prims=prims, floor_plane=floor_plane,
                               vis=_visibility_test(P, N, cams) if cams is not None and N is not None else None)
    chosen = [c for c, k in zip(cells, keep) if k]
    cell_solid.last_chosen = chosen
    cell_solid.debug = (cells, fr, vol, keep)
    from OCP.GProp import GProp_GProps as _GP
    from OCP.BRepGProp import BRepGProp as _BG
    vsum = 0.0
    for c in chosen:
        g = _GP()
        _BG.VolumeProperties_s(c, g)
        vsum += abs(g.Mass())
    shape = _boundary_solid(chosen)
    rep = {"surfaces": len(prims), "cells": len(cells), "inside": len(chosen), "cells_volume": vsum,
           "ambiguous_cells": int(((fr > 0.2) & (fr < 0.8)).sum())}
    return shape, rep


def _visibility_test(P, N, cams, min_cams=2, max_cams=64, splat_factor=2.0, ignore_mm=4.0):
    """vis(S) -> bool per sample: is the spot in the line of sight of >= min_cams cameras, with the
    scan points (as small discs) as occluders? Used to tell 'no points because it does not exist'
    from 'no points because no camera could see it'. Occluders closer than ignore_mm do not count:
    they belong to the surface the cell selection is deciding about (a 1 mm sliver behind a wall
    is hidden by that wall only as long as the sliver is kept)."""
    import open3d as o3d
    P = np.asarray(P, float)
    N = np.asarray(N, float)
    C = np.asarray(cams, float)
    if len(C) > max_cams:
        C = C[np.linspace(0, len(C) - 1, max_cams).astype(int)]
    d, _ = cKDTree(P).query(P[:: max(1, len(P) // 20000)], k=2)
    sp = float(np.clip(splat_factor * np.median(d[:, 1]), 0.4, 3.0))
    t1 = np.cross(N, np.where(np.abs(N[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(N, t1)
    V = np.concatenate([P + sp * (np.cos(a) * t1 + np.sin(a) * t2) for a in np.radians([90.0, 210.0, 330.0])])
    n = len(P)
    F = np.c_[np.arange(n), np.arange(n) + n, np.arange(n) + 2 * n]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    off = float(ignore_mm)

    def vis(S, nrm=None, chunk=20000, max_angle=75.0):
        S = np.asarray(S, float)
        out = np.zeros(len(S), bool)
        for i in range(0, len(S), chunk):
            Q = S[i:i + chunk]
            D = C[None] - Q[:, None]
            L = np.linalg.norm(D, axis=2)
            U = D / L[..., None]
            facing = (np.abs(np.einsum("ijk,ik->ij", U, np.asarray(nrm[i:i + chunk], float))) >
                      np.cos(np.radians(max_angle))) if nrm is not None else np.ones(U.shape[:2], bool)
            O = Q[:, None] + off * U
            rays = np.concatenate([O.reshape(-1, 3), U.reshape(-1, 3)], 1).astype(np.float32)
            hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(len(Q), len(C))
            out[i:i + chunk] = ((hit >= L - off - 0.5) & facing).sum(1) >= min_cams
        return out
    return vis


def _graphcut_cells(cells, frac, vol, lam, P=None, eps=1.0, step=1.0, w_unsup=4.0, floor_z=None,
                    prims=None, w_orient=5.0, floor_plane=None, vis=None):
    """Binary labelling of cells by exact min-cut.
    unary: volume that disagrees with the reference occupancy;
    pairwise: every face that would become part of the model surface (between an inside and an outside
    cell, or between an inside cell and the bounding box) costs lam x area, plus w_unsup x area for the
    part of it that no scan point supports. A real surface was seen by the cameras; a protrusion made of
    unsupported faces is a classification error, not geometry."""
    import networkx as nx
    owner = {}
    farea = {}
    fsup = {}   # 1 - fraction that is unsupported although a camera could have seen it (penalty)
    fdata = {}  # fraction actually supported by points (evidence for orientation)
    face_pts = {}
    face_node = {}
    tree = cKDTree(P) if P is not None else None
    for i, c in enumerate(cells):
        for f in _faces_of(c):
            h = hash(f)
            owner.setdefault(h, []).append(i)
            if h not in farea:
                S, a = _sample_face(f, step)
                farea[h] = a
                face_pts[h] = S
                face_node[h] = _sample_face.node
                if floor_plane is not None:
                    on_floor = len(S) and np.all(np.abs(S @ floor_plane[0] - floor_plane[1]) < 0.05)
                else:
                    on_floor = floor_z is not None and len(S) and np.all(np.abs(S[:, 2] - floor_z) < 0.05)
                if on_floor:
                    fsup[h] = 1.0  # the underside on the floor is never seen: no penalty
                elif tree is not None and len(S):
                    d, _ = tree.query(S, distance_upper_bound=eps)
                    uns = ~np.isfinite(d)
                    fdata[h] = float(1.0 - uns.mean())  # real support by measured points
                    if vis is not None and uns.any():
                        # only a face the cameras could have seen is suspicious when it has no
                        # points; the deep part of a hole is hidden by its own rim, and a camera
                        # looking along a bore sees its wall only at a grazing angle (no points)
                        nrm_s = None
                        if prims is not None:
                            xm = face_node[h]
                            dk = [abs(prim_dist(pr, xm)[0]) for pr in prims]
                            kk = int(np.argmin(dk))
                            if dk[kk] < 0.05:
                                raw = dict(prims[kk])
                                raw["s"] = 1.0
                                nrm_s = prim_normals(raw, S[uns])
                        uns[np.where(uns)[0][~vis(S[uns], nrm_s)]] = False
                    fsup[h] = float(1.0 - uns.mean())
                else:
                    fsup[h] = 1.0
    G = nx.DiGraph()
    vol = np.maximum(vol, 1e-3)
    unary_in = (1 - frac) * vol
    unary_out = frac * vol
    # oriented evidence: a supported face on a fitted surface whose normal points to the side the
    # cameras saw -> the cell in front of it is outside, the cell behind it inside. This decides thin
    # walls/sheets where the point-based occupancy mixes both sides.
    if prims is not None:
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_IN
        for h, own in owner.items():
            if len(own) != 2 or fdata.get(h, fsup.get(h, 0)) < 0.3:
                continue
            S = face_pts.get(h)
            if S is None or len(S) == 0:
                continue
            x = face_node[h][0]
            dk = [abs(prim_dist(pr, x[None])[0]) for pr in prims]
            k = int(np.argmin(dk))
            if dk[k] > 0.05 or "s" not in prims[k]:
                continue
            nrm = prim_normals(prims[k], x[None])[0]
            q = x + 0.05 * nrm
            cls = BRepClass3d_SolidClassifier(cells[own[0]], gp_Pnt(*map(float, q)), 1e-6)
            front, back = (own[0], own[1]) if cls.State() == TopAbs_IN else (own[1], own[0])
            w = w_orient * fdata.get(h, fsup[h]) * farea[h]  # only measured surface is evidence
            unary_in[front] += w
            unary_out[back] += w
    for h, own in owner.items():
        w = farea[h] * (lam + w_unsup * (1.0 - fsup[h]))
        if len(own) == 1:          # face on the bounding box: boundary if the cell is inside
            unary_in[own[0]] += w
        elif len(own) == 2:
            a, b = own
            for u, v in ((a, b), (b, a)):
                if G.has_edge(u, v):
                    G[u][v]["capacity"] += w
                else:
                    G.add_edge(u, v, capacity=w)
    for i in range(len(cells)):
        G.add_edge("s", i, capacity=float(unary_out[i]))  # cut when i is outside
        G.add_edge(i, "t", capacity=float(unary_in[i]))   # cut when i is inside
    _, (S, _) = nx.minimum_cut(G, "s", "t")
    _graphcut_cells.debug = dict(owner=owner, farea=farea, fsup=fsup, unary_in=unary_in, unary_out=unary_out)
    return [i in S for i in range(len(cells))]


def _boundary_solid(cells):
    """Union of the chosen cells by a glued boolean fuse (cells share faces exactly, so the 'full glue'
    option is exact and fast). Proper topology: sewing the boundary faces instead produced solids that
    BRepCheck accepted but that broke on STEP export (real scan)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.OCP.collections import List_TopoDS_Shape
    from OCP.BOPAlgo import BOPAlgo_GlueFull
    if not cells:
        raise RuntimeError("geen gesloten solid: geen enkele cel als binnen geclassificeerd")
    if len(cells) == 1:
        return cells[0]
    a, t = List_TopoDS_Shape(), List_TopoDS_Shape()
    a.Append(cells[0])
    for c in cells[1:]:
        t.Append(c)
    f = BRepAlgoAPI_Fuse()
    f.SetArguments(a)
    f.SetTools(t)
    f.SetGlue(BOPAlgo_GlueFull)
    f.SetRunParallel(True)
    f.Build()
    if not f.IsDone():
        return _boundary_solid_sewn(cells)
    return f.Shape()


def _boundary_solid_sewn(cells):
    """Faces used by exactly one chosen cell form the outer boundary; sew -> solid -> merge coplanar."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS
    cnt, rep = {}, {}
    for c in cells:
        for f in _faces_of(c):
            h = hash(f)
            cnt[h] = cnt.get(h, 0) + 1
            rep.setdefault(h, f)
    sew = BRepBuilderAPI_Sewing(1e-4)
    for h, n in cnt.items():
        if n == 1:
            sew.Add(rep[h])
    sew.Perform()
    sh = sew.SewedShape()
    solids = []
    ex = TopExp_Explorer(sh, TopAbs_SHELL)
    while ex.More():
        shell = TopoDS.Shell(ex.Current())
        ex.Next()
        ms = BRepBuilderAPI_MakeSolid(shell)
        if ms.IsDone():
            fx = ShapeFix_Solid(ms.Solid())
            fx.Perform()
            solids.append(fx.Solid())
    if not solids:
        raise RuntimeError("geen gesloten solid: geen enkele cel als binnen geclassificeerd")
    if len(solids) == 1:
        res = solids[0]
    else:
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound
        b = BRep_Builder()
        res = TopoDS_Compound()
        b.MakeCompound(res)
        for s_ in solids:
            b.Add(res, s_)
    u = ShapeUpgrade_UnifySameDomain(res, True, True, False)
    u.Build()
    return u.Shape()


def refine_primitives(prims, labels, P, N, noise_mm, iters=4, oriented=False):
    """Re-collect inliers for every primitive from ALL points near its region (distance + normal) and
    refit with fixed orientation. Segmentation inliers are gathered around the first, biased fit; this
    removes that self-consistent bias (e.g. radius of a groove that is only a short arc)."""
    eps = max(2.5 * noise_mm, 0.8)
    tree = cKDTree(P)
    for k, p in enumerate(prims):
        own = np.where(labels == k)[0]
        if len(own) < 10:
            continue
        near = np.unique(np.concatenate(tree.query_ball_point(P[own], 2 * eps)))
        for _ in range(iters):
            Q, NQ = P[near], N[near]
            d = np.abs(prim_dist(p, Q))
            cs = np.sum(prim_normals(p, Q) * NQ, axis=1)
            if not oriented:
                cs = np.abs(cs)
            inl = (d < eps) & (cs > np.cos(np.radians(20)))
            if inl.sum() < 10:
                break
            Qi = Q[inl]
            if p["type"] == "extrusion":
                pass  # fitted on the whole profile already
            elif p["type"] == "plane":
                if p.get("snapped"):
                    p["d"] = -float(np.mean(Qi @ p["n"]))
                else:
                    sg = p.get("s")
                    fp = fit_plane(Qi, robust_sigma=noise_mm, init=p)
                    if fp["n"] @ p["n"] < 0:
                        fp["n"], fp["d"] = -fp["n"], -fp["d"]
                    p["n"], p["d"] = fp["n"], fp["d"]
                    if sg is not None:
                        p["s"] = sg
            else:
                a = p["a"]
                u, v = _perp_basis(a)
                q2 = np.c_[Qi @ u, Qi @ v]
                c0 = np.array([p["p"] @ u, p["p"] @ v])
                sol = least_squares(lambda x: np.linalg.norm(q2 - x[:2], axis=1) - x[2], np.r_[c0, p["r"]],
                                    loss="soft_l1", f_scale=noise_mm, max_nfev=200)
                p["p"] = sol.x[0] * u + sol.x[1] * v + float(np.median(Qi @ a)) * a
                p["r"] = float(abs(sol.x[2]))
        prims[k] = p
    return prims


def dedupe_surfaces(prims, tol=None, noise_mm=0.5):
    """Parallel planes (or coaxial equal cylinders) closer than tol create sliver cells that can split
    the solid into pieces. Keep the best-supported one of each such group (measured surfaces before
    hypotheses)."""
    tol = tol if tol is not None else max(0.3, 1.0 * noise_mm)
    order = sorted(range(len(prims)), key=lambda k: (prims[k].get("hyp", False), -prims[k].get("n_pts", 0)))
    kept = []
    for k in order:
        p = prims[k]
        dup = False
        for j in kept:
            q = prims[j]
            if p["type"] != q["type"] or p["type"] == "extrusion":
                continue
            if p["type"] == "plane":
                c = p["n"] @ q["n"]
                if abs(c) > np.cos(np.radians(1.0)) and abs(-p["d"] - np.sign(c) * -q["d"]) < tol:
                    dup = True
            else:
                if (abs(p["a"] @ q["a"]) > np.cos(np.radians(1.0)) and abs(p["r"] - q["r"]) < tol and
                        np.linalg.norm(np.cross(p["p"] - q["p"], q["a"])) < tol):
                    dup = True
            if dup:
                break
        if not dup:
            kept.append(k)
    return [prims[k] for k in sorted(kept)]


def drop_ghost_layers(prims, labels, P, max_gap=5.0, cell=3.0, overlap=0.6, noise_mm=0.25):
    """Reflections on shiny metal give a second, parallel copy of a face a few mm off. An opaque
    surface cannot be seen at two levels in the same spot (a real pad or pocket hides the level
    under it), so a same-facing parallel plane whose footprint lies on a larger plane's footprint is
    dropped; its points become unlabelled."""
    planes = [k for k, p in enumerate(prims) if p["type"] == "plane"]
    size = {k: int((labels == k).sum()) for k in planes}
    dead = set()
    drop_ghost_layers.debug = []
    def foot(k, u, v, dense=False):
        Q = P[labels == k]
        ij = np.floor(np.c_[Q @ u, Q @ v] / cell).astype(int)
        cells, cnt = np.unique(ij, axis=0, return_counts=True)
        if dense:  # only where the surface was really measured, not a few stray noisy points
            cells = cells[cnt >= max(3, 0.5 * np.median(cnt))]
        return set(map(tuple, cells))
    for a in sorted(planes, key=lambda k: -size[k]):
        if a in dead:
            continue
        pa = prims[a]
        u, v = _perp_basis(pa["n"])
        Fa = None
        for b in planes:
            if b == a or b in dead or size[b] > size[a]:
                continue
            pb = prims[b]
            if abs(pa["n"] @ pb["n"]) < np.cos(np.radians(5)):
                continue
            if pa.get("s", 1) * pb.get("s", 1) * (pa["n"] @ pb["n"]) < 0:
                continue  # facing the other way: other side of a wall, not a ghost
            gap = abs(np.median(prim_dist(pa, P[labels == b])))
            if not (max(0.5, 2.0 * noise_mm) < gap < max_gap):  # closer: same surface, split by noise
                continue
            Fa = Fa if Fa is not None else foot(a, u, v, dense=True)
            Fb = foot(b, u, v)
            if Fb and len(Fb & Fa) / len(Fb) >= overlap:
                drop_ghost_layers.debug.append((a, b, round(gap, 2), size[a], size[b], round(len(Fb & Fa) / len(Fb), 2)))
                dead.add(b)
    drop_ghost_layers.ghost_mask = np.zeros(len(P), bool)
    if not dead:
        return prims, labels
    # all points of the ghost layer go, also unlabelled ones near it: they would otherwise make the
    # occupancy test see a surface that is not there
    for b in dead:
        pb = prims[b]
        a = [a_ for a_, b_, *_ in drop_ghost_layers.debug if b_ == b][0]
        u, v = _perp_basis(prims[a]["n"])
        # the reflection can lie over the whole real face, while RANSAC may have given only part of
        # it to the ghost plane (the rest is unlabelled or sits on walls at the rim): use the real
        # face's footprint. The gap is > 0.5 mm * 2, so a band of half the gap keeps the real points.
        Fa = foot(a, u, v)
        gap_b = abs(np.median(prim_dist(prims[a], P[labels == b])))
        band = np.abs(prim_dist(pb, P)) < 0.5 * gap_b
        ij = np.floor(np.c_[P @ u, P @ v] / cell).astype(int)
        Fd = {(i + di, j + dj) for i, j in Fa for di in (-1, 0, 1) for dj in (-1, 0, 1)}
        inside = np.array([tuple(x) in Fd for x in ij])
        drop_ghost_layers.ghost_mask |= band & inside
    keep = [k for k in range(len(prims)) if k not in dead]
    remap = -np.ones(len(prims) + 1, int)
    remap[keep] = np.arange(len(keep))
    labels = np.where(labels >= 0, remap[labels], -1)
    log(f"  CAD: {len(dead)} spiegel-/dubbellaag(en) verwijderd")
    return [prims[k] for k in keep], labels


def prune_redundant(prims, labels, P, noise_mm, frac=0.75, small=0.5, N=None):
    """Drop small primitives whose points are already explained (by distance) by other primitives:
    strips along an edge where the noise made neither neighbour claim the points. They would add
    false cuts and faces. Points are handed to the nearest remaining primitive."""
    eps = max(2.5 * noise_mm, 0.8)
    n_max = max((labels == k).sum() for k in range(len(prims))) if prims else 0
    alive = list(range(len(prims)))
    order = sorted(alive, key=lambda k: (labels == k).sum())
    for k in order:
        Q = P[labels == k]
        if len(Q) == 0 or len(Q) > small * n_max:
            continue
        tq = cKDTree(Q)
        others = [j for j in alive if j != k and np.any(np.isfinite(
            tq.query(P[labels == j][:: max(1, (labels == j).sum() // 3000)], distance_upper_bound=3 * eps)[0]))]
        if not others:
            continue
        D = np.stack([np.abs(prim_dist(prims[j], Q)) for j in others], 1)
        if N is not None:
            # a point only counts as explained by a surface facing the same way
            NQ = N[labels == k]
            for c_, j in enumerate(others):
                cs = np.sum(prim_normals(prims[j], Q) * NQ, axis=1)
                if "s" not in prims[j]:
                    cs = np.abs(cs)
                # 45 deg: at concave edges the PCA normal is a blend of both faces (seen ~37 deg);
                # perpendicular (90) and opposite (180) faces stay well apart
                D[cs < np.cos(np.radians(45)), c_] = np.inf
        best_j = D.argmin(1)
        expl = D.min(1) < 1.5 * eps
        # the explaining surface must pass through the points, not run parallel 2 mm away
        sd = np.array([prim_dist(prims[others[j]], Q[i:i + 1])[0] for i, j in enumerate(best_j)])
        centred = abs(np.median(sd[expl])) < 0.5 * eps if expl.any() else False
        if expl.mean() >= frac and centred:
            alive.remove(k)
            idx = np.where(labels == k)[0]
            labels[idx] = np.where(expl, np.array(others)[best_j], -1)
    remap = -np.ones(len(prims) + 1, int)
    remap[alive] = np.arange(len(alive))
    labels = np.where(labels >= 0, remap[labels], -1)
    return [prims[k] for k in alive], labels


def _frame_from_planes(planes, w, tol_deg=10.0):
    """Orthogonal frame best fitting all plane normals that lie near one of its axes (weighted
    orthogonal Procrustes, iterated). Start: largest plane + the largest one perpendicular to it."""
    if len(planes) < 2:
        return None
    order = np.argsort(-w)
    e1 = planes[order[0]][1]["n"].copy()
    e2 = None
    for o in order[1:]:
        n = planes[o][1]["n"]
        if abs(n @ e1) < np.sin(np.radians(10)):
            e2 = _unit(n - (n @ e1) * e1)
            break
    if e2 is None:
        return None
    R = np.array([e1, e2, np.cross(e1, e2)])
    for _ in range(5):
        A = np.zeros((3, 3))
        for (k, p), wk in zip(planes, w):
            c = R @ p["n"]
            i = int(np.argmax(np.abs(c)))
            if abs(c[i]) >= np.cos(np.radians(tol_deg)):
                A += wk * np.outer(np.sign(c[i]) * p["n"], np.eye(3)[i])
        U, _, Vt = np.linalg.svd(A)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = (U @ D @ Vt).T
    return R


def regularize_orientations(prims, labels, P, tol_deg=4.0, noise_mm=None, up=None):
    """Machined parts are mostly built on one orthogonal frame: snap plane normals and cylinder axes
    that are within tol_deg of a frame axis exactly onto it, then refit offsets / radii. Removes the
    0.5-2 deg wobble that noisy fits give and makes parallel faces exactly parallel."""
    planes = [(k, p) for k, p in enumerate(prims) if p["type"] == "plane"]
    if not planes:
        return prims
    w = np.array([p["n_pts"] if "n_pts" in p else (labels == k).sum() for k, p in planes], float)
    order = np.argsort(-w)
    R_data = _frame_from_planes(planes, w)
    if up is not None and R_data is not None:
        # use the floor only when the part really stands square on it (a part lying on a bent sheet
        # of paper is a few degrees off; forcing the floor axis would tilt every side face)
        upn = _unit(np.asarray(up, float))
        if np.max(np.abs(R_data @ upn)) < np.cos(np.radians(1.5)):
            up = None
    regularize_orientations.frame_from_floor = up is not None
    if up is None and R_data is not None:
        e1, e2, e3 = R_data
    elif up is not None:
        # the floor fixes one axis exactly; the other two come from the dominant vertical planes,
        # averaged (modulo 90 deg) over all of them, weighted by size
        e3 = _unit(np.asarray(up, float))
        u0, v0 = _perp_basis(e3)
        acc = 0j
        for (k, p), wk in zip(planes, w):
            n = p["n"] - (p["n"] @ e3) * e3
            if np.linalg.norm(n) < np.cos(np.radians(10)):
                continue
            ang = np.arctan2(n @ v0, n @ u0)
            acc += wk * np.exp(4j * ang)
        if acc == 0:
            return prims
        th = np.angle(acc) / 4
        e1 = np.cos(th) * u0 + np.sin(th) * v0
        e2 = np.cross(e3, e1)
    else:
        return prims
    R = np.array([e1, e2, e3])
    out = []
    for k, p in enumerate(prims):
        p = dict(p)
        Q = P[labels == k]
        # orientation uncertainty grows when a region is small compared to the noise: allow a larger
        # snap angle for narrow faces (e.g. a 12 mm high end face scanned with 1 mm noise)
        tol = tol_deg
        if noise_mm and len(Q) > 3:
            ext = np.sort(np.ptp(Q - Q.mean(0), axis=0))[-2] if p["type"] == "plane" else np.ptp(Q @ p["a"])
            tol = min(15.0, max(tol_deg, np.degrees(np.arctan(3.0 * noise_mm / max(ext, 1e-6)))))
        cos_t = np.cos(np.radians(tol))
        if p["type"] == "plane":
            c = R @ p["n"]
            i = int(np.argmax(np.abs(c)))
            if abs(c[i]) < cos_t and abs(c[i]) >= np.cos(np.radians(12)) and len(Q) > 10:
                # model selection instead of a hard angle: on a thick noisy layer (reflections) a free
                # fit tilts by several degrees; snap when the frame-aligned plane explains the points
                # just as well
                n_s = np.sign(c[i]) * R[i]
                d_s = -float(np.median(Q @ n_s))
                r_t = np.sqrt(np.mean(plane_dist(p, Q) ** 2))
                r_s = np.sqrt(np.mean((Q @ n_s + d_s) ** 2))
                if r_s <= 1.10 * r_t + 0.1 * (noise_mm or 0.5):
                    cos_t = -1.0
            if abs(c[i]) >= cos_t:
                n = np.sign(c[i]) * R[i]
                p["n"] = n
                p["d"] = -float(np.median(Q @ n)) if len(Q) else p["d"]
                p["snapped"] = True
        else:
            c = R @ p["a"]
            i = int(np.argmax(np.abs(c)))
            if abs(c[i]) >= cos_t:
                a = R[i]
                u, v = _perp_basis(a)
                q2 = np.c_[Q @ u, Q @ v]
                # geometric circle fit with the axis fixed (algebraic fits are biased on short arcs)
                c0 = np.array([p["p"] @ u, p["p"] @ v])

                def res(x):
                    return np.linalg.norm(q2 - x[:2], axis=1) - x[2]
                sol = least_squares(res, np.r_[c0, p["r"]], loss="soft_l1", f_scale=0.5, max_nfev=200)
                cx, cy, r = sol.x[0], sol.x[1], float(abs(sol.x[2]))
                if 0.5 * p["r"] < r < 2 * p["r"]:
                    p["a"] = a
                    p["p"] = cx * u + cy * v + float(np.median(Q @ a)) * a
                    p["r"] = r
                    p["snapped"] = True
        out.append(p)
    return out


def hoppe_occupancy(P, Nrm, floor_z=None, k=24, floor_plane=None):
    """Inside test from oriented scan points (normals pointing outward, towards the cameras):
    a query is inside when it lies behind the local tangent planes of its nearest scan points."""
    tree = cKDTree(P)

    def occ(Q, chunk=200_000):
        Q = np.asarray(Q)
        inside = np.zeros(len(Q), bool)
        for i in range(0, len(Q), chunk):  # chunked: a full grid x k neighbours does not fit in RAM
            q = Q[i:i + chunk]
            _, j = tree.query(q, k=k)
            s = np.einsum("qkd,qkd->qk", q[:, None, :] - P[j], Nrm[j])
            inside[i:i + chunk] = np.median(s, axis=1) < 0
        if floor_plane is not None:  # (n, h): the part lies on the side x.n > h
            inside &= Q @ floor_plane[0] > floor_plane[1]
        elif floor_z is not None:
            inside &= Q[:, 2] > floor_z
        return inside
    return occ


# ----------------------------------------------------------------------------------------------
# 5. driver
# ----------------------------------------------------------------------------------------------

def reconstruct(P, cams, noise_mm, floor_z=0.0, normals=None, voxel=None, log_fn=log, smooth=0.3):
    """Scan points of one part (+ camera centres) -> (solid, info). Cell-complex method."""
    import open3d as o3d
    from .meshing import orient_by_visibility
    o3d.utility.random.seed(0)  # RANSAC inside open3d: reproducible results
    t0 = time.time()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float)))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    est = estimate_noise(np.asarray(pc.points))
    if est is not None:
        log_fn(f"  CAD: ruis uit vlakfit {est:.2f} mm (opgegeven {noise_mm:.2f} mm)")
        noise_mm = max(noise_mm, est)
    voxel = voxel or max(0.5, 1.2 * noise_mm)
    P_full = np.asarray(pc.points).copy()  # all points: the profile fit averages over all of them
    pc = pc.voxel_down_sample(voxel)
    r_n = max(3.0, 5 * voxel)
    dens = len(pc.points) / max(1e-9, _approx_area(np.asarray(pc.points), voxel))
    def _normals(Pq):
        if normals is not None:
            return np.asarray(normals)[cKDTree(np.asarray(P)).query(Pq)[1]], 1.0
        pq = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pq))
        pq.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=r_n, max_nn=int(np.clip(4 * np.pi * r_n ** 2 * dens, 40, 400))))
        return orient_by_visibility(Pq, np.asarray(pq.normals), cams)

    Pd = np.asarray(pc.points)
    Nd, dec = _normals(Pd)
    # all points take part (thin flanges and steps just above the floor too); floor points were cropped
    Ps, Ns = Pd, Nd
    min_pts = max(40, int(25.0 * dens))  # ~25 mm^2 of surface, independent of point density
    up_axis = (0, 0, 1.0) if floor_z is not None else None

    def _segment(Ps, Ns):
        prims, lab = segment_primitives(Ps, Ns, noise_mm, min_pts=min_pts, oriented=True)
        prims, lab = merge_similar(prims, lab, Ps, noise_mm)
        prims = regularize_orientations(prims, lab, Ps, noise_mm=noise_mm, up=up_axis)
        return prims, lab

    prims, lab = _segment(Ps, Ns)
    prims, lab = drop_ghost_layers(prims, lab, Ps, noise_mm=noise_mm)
    gm = getattr(drop_ghost_layers, "ghost_mask", None)
    if gm is not None and gm.any():
        # reflection layer found: remove its points and segment again, so the strips it caused at
        # the edges disappear as well
        # (normals again too: near the layer they were estimated from a mix of both levels)
        Ps = Ps[~gm]
        Ns, dec = _normals(Ps)
        Pd, Nd = Ps, Ns  # Pd/Ps are the same set here
        prims, lab = _segment(Ps, Ns)
    # first finding about the part: is it (partly) a constant profile along one of its axes?
    from .profiles import profile_primitives
    dfull, ifull = cKDTree(Ps).query(P_full, distance_upper_bound=1.5 * voxel)
    okf = np.isfinite(dfull)  # (ghost layers and outliers removed from Ps are dropped here too)
    prims, lab, findings = profile_primitives(Ps, Ns, prims, lab, _frame_axes(prims), noise_mm, min_pts,
                                              P_full=P_full[okf], N_full=Ns[ifull[okf]])
    reconstruct.findings = findings
    auth = [np.asarray(f["axis"]) for f in findings if f.get("authority")]  # empty unless AUTHORITY
    prims, lab = frame_cylinders(prims, lab, Ps, Ns, noise_mm, min_pts, skip_axes=auth)
    prims, lab = prune_redundant(prims, lab, Ps, noise_mm, N=Ns)
    prims = refine_primitives(prims, lab, Ps, Ns, noise_mm, oriented=True)
    prims, lab = merge_similar(prims, lab, Ps, noise_mm)  # snapping can make planes coincide
    prims, lab = split_false_chamfers(prims, lab, Ps, Ns, noise_mm)
    prims, lab = detail_primitives(prims, lab, Ps, Ns, noise_mm, min_pts, skip_axes=auth)
    log_fn(f"  CAD: {len(prims)} oppervlakken ({sum(p['type'] == 'plane' for p in prims)} vlak, "
           f"{sum(p['type'] == 'cylinder' for p in prims)} cilinder), {np.mean(lab >= 0):.0%} van de punten verklaard")
    Ns, n_flip = harmonize_normals(prims, lab, Ps, Ns, noise_mm)
    Pd, Nd = Ps, Ns
    hyp = thickness_hypotheses(prims, lab, Ps, floor_z=floor_z)
    floor_plane = None  # set only for a tilted part; otherwise the level floor floor_z is used
    if floor_z is not None and not getattr(regularize_orientations, "frame_from_floor", True):
        # part stands a little tilted on the floor (bent marker sheet, chip under one edge): its
        # own underside is square to the part, through its lowest point. It replaces the floor as the
        # unseen bottom; the floor would leave a wedge of air under the part in the solid.
        axes = [p_["n"] for p_ in prims if p_["type"] == "plane" and p_.get("snapped")]
        if axes:
            a = max(axes, key=lambda n: abs(n[2]))
            a = a * np.sign(a[2])
            tilt = np.degrees(np.arccos(min(1.0, abs(a[2]))))
            if tilt < 6.0:  # more is not "standing a bit crooked" but a part placed on purpose
                zmin = float(np.percentile(Ps @ a, 0.5))
                hyp.append({"type": "plane", "n": a.copy(), "d": -zmin, "hyp": True, "snapped": True,
                            "rms": 0.0, "n_pts": 0})
                floor_plane = (a.copy(), zmin)
                log_fn(f"  CAD: deel staat {tilt:.1f}° scheef op de vloer; onderkant haaks op het deel")
    if hyp:
        log_fn(f"  CAD: {len(hyp)} verborgen achtervlak(ken) voorgesteld uit gemeten wanddikte(s)")
    bnd = boundary_hypotheses(prims, lab, Ps, Ns, noise_mm)
    if bnd:
        log_fn(f"  CAD: {len(bnd)} randwand(en) voorgesteld bij kleine treden")
    hyp = hyp + bnd
    reconstruct.last = dict(prims=prims, lab=lab, P=Ps, N=Ns, hyp=hyp)
    prims = dedupe_surfaces(prims + hyp, noise_mm=noise_mm)
    occ = hoppe_occupancy(Pd, Nd, floor_z=floor_z, floor_plane=floor_plane)
    blo, bhi = Pd.min(0), Pd.max(0)
    if floor_z is not None:
        blo[2] = floor_z
    caps, occ = blind_hole_caps(prims, Pd, Nd, occ, noise_mm, blo, bhi, log_fn=log_fn)
    prims = prims + caps
    shape, rep = cell_solid(prims, Pd, occ, floor_z=floor_z, noise_mm=noise_mm, smooth=smooth,
                            floor_plane=floor_plane, N=Nd, cams=cams)
    shape = clean_solid(shape)
    # simplification: a surface that ends up bounding (almost) nothing only adds cells, slivers and
    # tiny faces. Drop those and solve again on the remaining surfaces.
    a_min = max(30.0, 10 * (2.5 * noise_mm) ** 2)
    used = _surface_areas(shape, prims)
    drop = [k for k in range(len(prims)) if used[k] < a_min and not prims[k].get("cap")]  # a cap is small by nature
    if drop and len(drop) < len(prims):
        keep_p = [p for k, p in enumerate(prims) if k not in drop]
        log_fn(f"  CAD: {len(drop)} oppervlak(ken) zonder wezenlijke bijdrage weggelaten, opnieuw opgelost")
        try:
            shape2, rep2 = cell_solid(keep_p, Pd, occ, floor_z=floor_z, noise_mm=noise_mm, smooth=smooth,
                                      floor_plane=floor_plane, N=Nd, cams=cams)
            shape2 = clean_solid(shape2)
            q1 = quality(shape, Pd, noise_mm, floor_z)
            q2 = quality(shape2, Pd, noise_mm, floor_z)
            reconstruct.simpl_q = (q1, q2, [prims[k]["type"] for k in drop], used.round(1).tolist())
            # keep the simpler model unless it fits clearly worse
            if q2["dev_p95"] <= q1["dev_p95"] + 0.25 * noise_mm and q2["frac_far"] <= q1["frac_far"] + 0.01:
                shape, rep, prims = shape2, rep2, keep_p
        except Exception as e:  # the first solution stands
            log_fn(f"  CAD: vereenvoudiging mislukt ({e}); eerste oplossing blijft")
    if any(p["type"] == "extrusion" for p in prims):
        from OCP.BRepCheck import BRepCheck_Analyzer
        if not BRepCheck_Analyzer(shape).IsValid():
            # a free-form surface can cut a flat face into a boundary OCC rejects; a valid solid
            # without the spline beats an invalid one with it
            log_fn("  CAD: solid met vrije-kromme-vlak ongeldig; opnieuw opgelost zonder vrije krommen")
            keep_p = [p for p in prims if p["type"] != "extrusion"]
            try:
                shape2, rep2 = cell_solid(keep_p, Pd, occ, floor_z=floor_z, noise_mm=noise_mm, smooth=smooth,
                                          floor_plane=floor_plane, N=Nd, cams=cams)
                shape2 = clean_solid(shape2)
                if BRepCheck_Analyzer(shape2).IsValid():
                    shape, rep, prims = shape2, rep2, keep_p
            except Exception as e:
                log_fn(f"  CAD: oplossen zonder vrije krommen mislukt ({e})")
    info = {"findings": getattr(reconstruct, "findings", []),
            "surfaces": len(prims), "cells": rep["cells"], "inside": rep["inside"],
            "ambiguous": rep["ambiguous_cells"], "oriented": dec, "explained": float(np.mean(lab >= 0)),
            "noise_mm": float(noise_mm), "cells_volume": rep.get("cells_volume"),
            "t": time.time() - t0}
    return shape, info


def _surface_areas(shape, prims, tol=0.05):
    """Area of the solid's boundary lying on each primitive."""
    out = np.zeros(len(prims))
    for f in _faces_of(shape):
        S, a = _sample_face(f, 2.0)
        if len(S) == 0:
            continue
        x = _sample_face.node
        d = [abs(prim_dist(p, x)[0]) for p in prims]
        k = int(np.argmin(d)) if d else -1
        if k >= 0 and d[k] < tol:
            out[k] += a
    return out


def _same_surface(p, q, tol):
    """Two primitives describing the same geometric surface (orientation ignored)."""
    if p["type"] != q["type"]:
        return False
    if p["type"] == "extrusion":
        return p is q
    if p["type"] == "plane":
        c = p["n"] @ q["n"]
        return abs(c) > 0.999 and abs(p["d"] - np.sign(c) * q["d"]) < tol
    if abs(p["a"] @ q["a"]) < 0.999 or abs(p["r"] - q["r"]) > tol:
        return False
    w = q["p"] - p["p"]
    return np.linalg.norm(w - (w @ p["a"]) * p["a"]) < tol


def _frame_axes(prims):
    axes = []
    for p in prims:
        if p["type"] == "plane" and p.get("snapped"):
            if not any(abs(p["n"] @ a) > 0.999 for a in axes):
                axes.append(_unit(p["n"]))
    return axes


def _verify_frame_cylinder(P, N, a, c, r, idx, r_cur, noise_mm, eps, link, min_pts, cos45):
    """Refine a frame-axis cylinder hypothesis and test it. Returns (prim, won point indices, reason)."""
    u, v = _perp_basis(a)
    X = np.c_[P @ u, P @ v]
    NX = np.c_[N @ u, N @ v]
    across = np.abs(N @ a) < np.sin(np.radians(45))
    for _ in range(3):
        sol = least_squares(lambda x: np.linalg.norm(X[idx] - x[:2], axis=1) - x[2], np.r_[c, r],
                            loss="soft_l1", f_scale=noise_mm)
        c, r = sol.x[:2], float(abs(sol.x[2]))
        if not (1.5 < r < 60.0):
            return None, None, "radius"
        rad = X - c
        rn = np.linalg.norm(rad, axis=1)
        d = np.abs(rn - r)
        radial = np.abs(np.sum(NX * rad, 1) / np.maximum(rn, 1e-9))
        idx = np.where((d < eps) & (radial > cos45) & across)[0]
        if len(idx) < min_pts:
            return None, None, "few"
    idx = _largest_component(P, idx, link)
    pr = {"type": "cylinder", "a": a.copy(), "p": c[0] * u + c[1] * v + a * float(np.median(P[idx] @ a)),
          "r": r, "snapped": True, "frame": True}
    d = np.abs(prim_dist(pr, P[idx]))
    r_c = r_cur[idx]
    win = idx[~np.isfinite(r_c) | (d < r_c)]
    if len(win) < min_pts:
        return None, None, "wins few"
    # most of what it wins must have been badly explained: otherwise it is a rounded version of
    # an edge or ridge that the existing faces already describe
    rw = r_cur[win]
    if np.mean(np.isfinite(rw) & (rw < eps)) > 0.5:
        return None, None, "explained"
    ang = np.arctan2(X[win, 1] - c[1], X[win, 0] - c[0])
    span = _angular_span(ang)
    if span < np.radians(120):  # groove/bore/boss; a 90 deg arc is two faces meeting at an edge
        return None, None, "span"
    a_s = np.mod(ang - ang.min(), 2 * np.pi)
    hist, _ = np.histogram(a_s, bins=8, range=(0, max(span, 1e-6)))
    if (hist > 0.25 * len(win) / 8).mean() < 0.75:
        return None, None, "uneven"
    # a real cylinder holds nearly all points of a wide band around its radius over its own arc and
    # length (a staircase of planes, a square pocket or a big circle through a flat face does not)
    hh = P[win] @ a
    rn = np.linalg.norm(X - c, axis=1)
    angs = np.arctan2(X[:, 1] - c[1], X[:, 0] - c[0])
    in_arc = np.mod(angs - ang.min(), 2 * np.pi) <= span
    band = (np.abs(rn - r) < 3 * eps) & (P @ a >= hh.min()) & (P @ a <= hh.max()) & across & in_arc
    frac = (np.abs(rn[band] - r) < eps).mean() if band.any() else 0.0
    if frac < 0.7:
        return None, None, f"band {frac:.2f}"
    rms = float(np.sqrt(np.mean(prim_dist(pr, P[win]) ** 2)))
    if rms > 1.5 * noise_mm:
        return None, None, f"rms {rms:.2f}"
    signed_cos(pr, P[win], N[win])
    pr.update(rms=rms, n_pts=int(len(win)))
    return pr, win, "ok"


def frame_cylinders(prims, labels, P, N, noise_mm, min_pts, max_new=12, n_hyp=500, seed=0, skip_axes=()):
    """Grooves, bores and bosses along the part's own axes, searched over ALL points.

    At ~0.7 mm noise the PCA normals over a 5 mm neighbourhood smear a groove of r = 8 mm, so the
    normal-guided RANSAC misses it and the groove ends up as a staircase of parallel planes or one
    slanted plane. Here the axis is fixed to a frame axis (machined parts: grooves and bores run along
    the part's edges), so only centre and radius are fitted, from 3 positions; normals only need to
    agree loosely. A candidate is scored by how much better it explains points than their current
    surface; the points it wins move to the cylinder, and planes left without points are pruned."""
    # along an axis whose profile was recognised, the profile alone decides the surfaces
    axes = [a for a in _frame_axes(prims) if not any(abs(a @ b) > 0.999 for b in skip_axes)]
    if not axes:
        return prims, labels
    rng = np.random.default_rng(seed)
    eps = max(2.0 * noise_mm, 0.8)
    link = 3.0 * max(noise_mm, 0.5) + 1.5
    cos45 = np.cos(np.radians(45))
    labels = labels.copy()
    prims = list(prims)
    tree = cKDTree(P)

    def current_residual():
        r = np.full(len(P), np.inf)
        for k, p in enumerate(prims):
            m = labels == k
            if m.any():
                r[m] = np.abs(prim_dist(p, P[m]))
        return r

    n_new = 0
    frame_cylinders.debug = []
    rejected = []  # (axis, centre, r) verified and rejected: not tried again
    for _round in range(max_new):
        r_cur = current_residual()
        hyps = []
        for a in axes:
            u, v = _perp_basis(a)
            X = np.c_[P @ u, P @ v]
            Na = N @ a
            cand = np.where(np.abs(Na) < np.sin(np.radians(45)))[0]
            if len(cand) < min_pts:
                continue
            # seeds preferably where the current model explains the points badly
            w = 1.0 + np.minimum(np.nan_to_num(r_cur[cand], posinf=5 * noise_mm) / noise_mm, 5.0)
            w = w / w.sum()
            NX = np.c_[N[cand] @ u, N[cand] @ v]
            for _ in range(n_hyp):
                i = cand[rng.choice(len(cand), p=w)]
                nb = tree.query_ball_point(P[i], 12.0)
                nb = np.array([j for j in nb if abs(Na[j]) < np.sin(np.radians(45)) and j != i])
                if len(nb) < 2:
                    continue
                j, k = rng.choice(nb, 2, replace=False)
                A_ = 2 * np.array([X[j] - X[i], X[k] - X[i]])
                if abs(np.linalg.det(A_)) < 1e-3:
                    continue
                c = np.linalg.solve(A_, [X[j] @ X[j] - X[i] @ X[i], X[k] @ X[k] - X[i] @ X[i]])
                r = float(np.linalg.norm(X[i] - c))
                if not (1.5 < r < 60.0):
                    continue
                if any(abs(a @ ra) > 0.999 and np.linalg.norm(c - rc_) < eps and abs(r - rr_) < eps
                       for ra, rc_, rr_ in rejected):
                    continue
                rad = X[cand] - c
                rn = np.linalg.norm(rad, axis=1)
                d = np.abs(rn - r)
                ok = d < eps
                if ok.sum() < min_pts:
                    continue
                radial = np.abs(np.sum(NX[ok] * rad[ok], 1) / np.maximum(rn[ok], 1e-9))
                okk = np.where(ok)[0][radial > cos45]
                rc = r_cur[cand[okk]]
                # only a clear improvement counts: on a flat face the residuals are noise anyway,
                # and a big circle grazing it would collect "gain" from that noise alone
                gain = np.where(np.isfinite(rc), np.clip((rc - d[okk] - noise_mm) / noise_mm, 0, 1), 1.0).sum()
                if gain >= 0.5 * min_pts:
                    hyps.append((gain, a, c, r, cand[okk]))
        # verify the best few distinct hypotheses; accept the first that is a real cylinder
        hyps.sort(key=lambda h: -h[0])
        tried, accepted = [], False
        for gain, a, c, r, idx in hyps:
            if len(tried) >= 120:
                break
            if any(abs(a @ ta) > 0.999 and np.linalg.norm(c - tc) < eps and abs(r - tr) < eps for ta, tc, tr in tried):
                continue
            tried.append((a, c, r))
            pr, win, why = _verify_frame_cylinder(P, N, a, c, r, idx, r_cur, noise_mm, eps, link, min_pts, cos45)
            frame_cylinders.debug.append((round(float(gain), 1), np.round(a, 2), np.round(c, 1), round(r, 2), why))
            if pr is None or any(_same_surface(pr, q, eps) for q in prims):
                rejected.append((a, c, r))
                continue
            labels[win] = len(prims)
            prims.append(pr)
            n_new += 1
            accepted = True
            break
        if not accepted:
            break
    if n_new:
        log(f"  CAD: {n_new} groef/boring/naaf langs de assen van het deel gevonden")
    return prims, labels


def split_false_chamfers(prims, labels, P, N, noise_mm):
    """A small step (a 3 x 3 mm notch along an edge) at ~0.5 mm noise is also fitted by one slanted
    plane across it (a false chamfer). For a plane that is not on the part's frame but lies along a
    frame axis, compare with the frame-aligned step (two planes, each point on the nearer one); the
    step wins when it fits clearly better and both arms are well populated."""
    axes = []
    for p in prims:
        if p["type"] == "plane" and p.get("snapped"):
            if not any(abs(p["n"] @ a) > 0.999 for a in axes):
                axes.append(_unit(p["n"]))
    if len(axes) < 3:
        return prims, labels
    out = list(prims)
    labels = labels.copy()
    n_split = 0
    for k, p in enumerate(prims):
        idx = np.where(labels == k)[0]
        if len(idx) < 60 or p["type"] == "extrusion":
            continue
        if p["type"] == "plane":
            if p.get("snapped"):
                continue
            along = [a for a in axes if abs(a @ p["n"]) < np.sin(np.radians(2))]
        else:
            # a small arc along a frame axis can be a false fillet on a sharp edge: the same test
            along = [a for a in axes if abs(a @ p["a"]) > np.cos(np.radians(2))]
            if len(along) == 1:
                uu, vv = _perp_basis(p["a"])
                w_ = P[idx] - p["p"]
                if _angular_span(np.arctan2(w_ @ vv, w_ @ uu)) >= np.radians(120):
                    continue
        if len(along) != 1:
            continue
        a1, a2 = [a for a in axes if abs(a @ along[0]) < 0.5][:2]
        Q = P[idx]
        u, v = Q @ a1, Q @ a2
        rms_plane = float(np.sqrt(np.mean(prim_dist(p, Q) ** 2)))
        is_cyl = p["type"] == "cylinder"
        # the neighbouring frame faces (top, back wall) also hold some of these points
        d0 = np.full(len(Q), np.inf)
        for q in prims:
            if q["type"] == "plane" and q.get("snapped") and any(abs(q["n"] @ a) > 0.999 for a in (a1, a2)):
                d0 = np.minimum(d0, np.abs(prim_dist(q, Q)))
        best = None
        for t1 in np.percentile(u, np.linspace(2, 98, 33)):
            for t2 in np.percentile(v, np.linspace(2, 98, 33)):
                d1, d2 = np.abs(u - t1), np.abs(v - t2)
                r = np.minimum(np.minimum(d1, d2), d0)
                rms = float(np.sqrt(np.mean(r ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, t1, t2)
        rms, t1, t2 = best
        d1, d2 = np.abs(u - t1), np.abs(v - t2)
        own = np.minimum(d1, d2) <= d0  # points that need the new arms
        on1, on2 = own & (d1 < d2), own & (d1 >= d2)
        # a real chamfer fits its plane at the noise level; a false one leaves the step's variance
        if is_cyl:
            # the edge model (2 planes) has no curvature to spare: as good a fit is enough
            accept = rms <= 1.0 * rms_plane and min(on1.sum(), on2.sum()) > 0.15 * own.sum()
        else:
            accept = (rms_plane > 1.15 * noise_mm and rms < 0.92 * rms_plane
                      and min(on1.sum(), on2.sum()) > 0.15 * own.sum())
        if not accept:
            continue
        # refine: reassign to the nearer arm (or neighbour face) and re-estimate, a few times; the
        # grid optimum is biased by the ambiguous points in the inner corner
        for _ in range(8):
            t1, t2 = float(np.median(u[on1])), float(np.median(v[on2]))
            d1, d2 = np.abs(u - t1), np.abs(v - t2)
            own = np.minimum(d1, d2) <= d0
            on1, on2 = own & (d1 < d2), own & (d1 >= d2)
            if min(on1.sum(), on2.sum()) < 10:
                break
        if min(on1.sum(), on2.sum()) < 10:
            continue
        new = []
        for a, t, m in ((a1, t1, on1), (a2, t2, on2)):
            q = {"type": "plane", "n": a.copy(), "d": -t, "snapped": True}
            signed_cos(q, P[idx[m]], N[idx[m]])
            q.update(rms=float(np.sqrt(np.mean(prim_dist(q, P[idx[m]]) ** 2))), n_pts=int(m.sum()))
            new.append((q, m))
        out[k] = new[0][0]
        labels[idx[~own]] = -1  # the rest is handed to the neighbours by the later steps
        labels[idx[on2]] = len(out)
        out.append(new[1][0])
        n_split += 1
    if n_split:
        log(f"  CAD: {n_split} schuin vlak(ken) vervangen door een kleine trede")
    return out, labels


def blind_hole_caps(prims, P, N, occ, noise_mm, box_lo, box_hi, max_depth_ratio=6.0, log_fn=log):
    """Holes whose bottom no camera saw.

    A hole is measured only as deep as its wall was seen; below that nothing is known. If the hole
    would have to run on for more than max_depth_ratio x its diameter to come out on the other side
    of the part, it is taken as a blind hole that ends where its wall was last seen (a 5 mm hole is
    not drilled through 90 mm of steel); a shorter one is left open (through hole).
    Returns (cap planes, occupancy function with the unseen part of blind holes set to material)."""
    eps = max(2.0 * noise_mm, 0.8)
    caps, zones, opens = [], [], []
    for p in prims:
        if p["type"] != "cylinder" or p.get("s", 1.0) > 0 or p["r"] > 15.0:
            continue
        a, p0, r = _unit(p["a"]), p["p"], p["r"]
        q = P - p0
        t = q @ a
        rad = q - np.outer(t, a)
        rn = np.linalg.norm(rad, axis=1)
        mem = (np.abs(rn - r) < eps) & (np.sum(N * rad, 1) / np.maximum(rn, 1e-9) < -np.cos(np.radians(45)))
        if mem.sum() < 30:
            continue
        t0, t1 = np.percentile(t[mem], [0.5, 99.5])
        u, v = _perp_basis(a)
        # wall seen at both ends with an unseen stretch between (a long bore seen into from both
        # sides): it runs through, so the stretch between is open
        ts = np.sort(t[mem])
        gaps = np.diff(ts)
        if len(gaps) and gaps.max() > max(5.0, 2 * r):
            opens.append((a, p0, r, t0, t1))
            continue
        for end, sgn in ((t1, 1.0), (t0, -1.0)):
            # deep side: just past this end, beside the hole, there is material
            side = p0 + (end + sgn * 2.0) * a + (r + 2.5) * np.array([u, -u, v, -v])
            if occ(side).mean() < 0.75:
                continue  # this end opens to the outside
            # how far would the hole have to run to come out of the part?
            L = 0.0
            for step in np.arange(1.0, 400.0, 1.0):
                x = p0 + (end + sgn * step) * a
                if np.any(x < box_lo) or np.any(x > box_hi):
                    break
                ring = x + (r + 2.5) * np.array([u, -u, v, -v])
                if occ(ring).mean() < 0.5:
                    break
                L = step
            if L > max_depth_ratio * 2 * r:
                tc = end + sgn * 0.3
                caps.append({"type": "plane", "n": a * sgn, "d": -float((p0 + tc * a) @ (a * sgn)), "hyp": True,
                             "snapped": True, "rms": 0.0, "n_pts": 0, "cap": True})
                zones.append((a, p0, r, tc, sgn))
    if caps:
        log_fn(f"  CAD: {len(caps)} blind(e) gat(en): bodem niet gezien, gat eindigt waar de wand nog gezien is")

    if opens:
        log_fn(f"  CAD: {len(opens)} doorgaande boring(en): van beide kanten gezien, midden open")

    def occ2(Q):
        o = occ(Q)
        for a, p0, r, tc, sgn in zones:
            q = Q - p0
            t = q @ a
            rn = np.linalg.norm(q - np.outer(t, a), axis=1)
            o = o | ((rn < r + 0.5) & ((t - tc) * sgn > 0))
        for a, p0, r, t0, t1 in opens:
            q = Q - p0
            t = q @ a
            rn = np.linalg.norm(q - np.outer(t, a), axis=1)
            o = o & ~((rn < r - 0.3) & (t > t0) & (t < t1))
        return o
    return caps, (occ2 if (zones or opens) else occ)


def harmonize_normals(prims, labels, P, N, noise_mm):
    """Visibility orientation fails where few cameras see a point (the deep part of a hole: its own
    rim hides it); one flipped patch then turns the occupancy test around there. Points that lie on a
    fitted surface take that surface's orientation (decided by the majority of its points)."""
    eps = max(2.5 * noise_mm, 0.8)
    N = N.copy()
    n_flip = 0
    for k, pr in enumerate(prims):
        if "s" not in pr:
            continue
        cand = np.where((labels == k) | (labels < 0))[0]
        if len(cand) == 0:
            continue
        cand = cand[np.abs(prim_dist(pr, P[cand])) < eps]
        if len(cand) == 0:
            continue
        pn = prim_normals(pr, P[cand])
        c = np.sum(pn * N[cand], axis=1)
        flip = cand[c < -np.cos(np.radians(30))]
        N[flip] *= -1
        n_flip += len(flip)
    return N, n_flip


def detail_primitives(prims, labels, P, N, noise_mm, min_pts, max_new=12, seed=0, skip_axes=()):
    """Small features (a counterbore 8 mm wide and 5 mm deep, a short boss) are lost by the main
    segmentation: their PCA normals are blended with the neighbouring faces, so the normal-guided
    RANSAC never collects enough agreeing points. Here the unexplained clusters are fitted with
    models whose orientation comes from the part's frame (planes square to a frame axis, cylinders
    along one): only positions are fitted, normals only need to agree loosely (45 deg)."""
    rng = np.random.default_rng(seed)
    eps = max(2.0 * noise_mm, 0.8)
    link = 3.0 * max(noise_mm, 0.5) + 1.5
    axes = []
    for p in prims:
        a = p["n"] if p["type"] == "plane" and p.get("snapped") else (p["a"] if p["type"] == "cylinder" else None)
        if a is not None and not any(abs(a @ b) > 0.999 for b in axes):
            axes.append(_unit(a))
    if not axes:
        return prims, labels
    cos45 = np.cos(np.radians(45))
    labels = labels.copy()
    out = []
    free = np.where(labels < 0)[0]
    if len(free) < min_pts:
        return prims, labels
    pairs = cKDTree(P[free]).query_pairs(link, output_type="ndarray")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    _, comp = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                                              shape=(len(free), len(free))), directed=False)
    cnt = np.bincount(comp)
    for cl in np.argsort(-cnt):
        if cnt[cl] < min_pts or len(out) >= max_new:
            break
        idx = free[comp == cl]
        for _ in range(4):  # a few surfaces per cluster (counterbore: wall + floor)
            if len(idx) < min_pts:
                break
            Q, NQ = P[idx], N[idx]
            best = None
            for a in axes:
                in_profile = any(abs(a @ b) < 0.01 for b in skip_axes)  # plane running along a profile
                t = Q @ a
                along = np.abs(NQ @ a) > cos45
                if in_profile:
                    along = np.zeros(len(Q), bool)
                for t0 in rng.choice(t[along], min(40, int(along.sum())), replace=False) if along.any() else []:
                    ok = along & (np.abs(t - t0) < eps)
                    if best is None or ok.sum() > best[0]:
                        t1 = float(np.median(t[ok]))
                        best = (int(ok.sum()), {"type": "plane", "n": a.copy(), "d": -t1})
                if any(abs(a @ b) > 0.999 for b in skip_axes):
                    continue  # cylinders along a recognised profile come from the profile only
                u, v = _perp_basis(a)
                X = np.c_[Q @ u, Q @ v]
                across = np.abs(NQ @ a) < np.sin(np.radians(45))
                if across.sum() < 3:
                    continue
                ia = np.where(across)[0]
                for _ in range(60):
                    i, j, k = rng.choice(ia, 3, replace=False)
                    A_ = np.c_[2 * (X[[j, k]] - X[i])]
                    if abs(np.linalg.det(A_)) < 1e-6:
                        continue
                    b_ = np.sum(X[[j, k]] ** 2, 1) - np.sum(X[i] ** 2)
                    c = np.linalg.solve(A_, b_)
                    r = float(np.linalg.norm(X[i] - c))
                    if not (1.5 < r < 100.0):
                        continue
                    rad = X - c
                    rn = np.linalg.norm(rad, axis=1)
                    ok = across & (np.abs(rn - r) < eps)
                    if best is not None and ok.sum() <= best[0]:
                        continue
                    nr = (NQ[ok] @ u) * rad[ok, 0] / np.maximum(rn[ok], 1e-9) + (NQ[ok] @ v) * rad[ok, 1] / np.maximum(rn[ok], 1e-9)
                    ok[np.where(ok)[0][np.abs(nr) < cos45]] = False
                    if ok.sum() >= 3 and (best is None or ok.sum() > best[0]):
                        ang = np.arctan2(rad[ok, 1], rad[ok, 0])
                        span = _angular_span(ang)
                        if span < np.radians(120):  # a hole or boss; <= 90 deg is an edge strip / fillet
                            continue
                        # evenly covered arc, not two opposite walls of a pocket
                        a_s = np.mod(ang - ang.min(), 2 * np.pi)
                        hist, _ = np.histogram(a_s, bins=8, range=(0, max(span, 1e-6)))
                        if (hist > 0.25 * ok.sum() / 8).mean() < 0.75:
                            continue
                        def res(x, Xo=X[ok]):
                            return np.linalg.norm(Xo - x[:2], axis=1) - x[2]
                        sol = least_squares(res, np.r_[c, r], loss="soft_l1", f_scale=0.5 * eps)
                        cx, cy, rr = sol.x
                        p0 = cx * u + cy * v + a * float(np.median(Q[ok] @ a))
                        best = (int(ok.sum()), {"type": "cylinder", "a": a.copy(), "p": p0, "r": float(abs(rr))})
            if best is None or best[0] < min_pts:
                break
            pr = best[1]
            if any(_same_surface(pr, q, eps) for q in prims + out):
                # already a fitted surface (e.g. the deep, badly oriented part of a hole): not a new
                # one; harmonize_normals gives these points the surface's orientation
                idx = np.setdiff1d(idx, idx[np.abs(prim_dist(pr, Q)) < eps])
                continue
            d = np.abs(prim_dist(pr, Q))
            cs = signed_cos(pr, Q, NQ)
            sel = (d < eps) & (cs > cos45)
            sel_idx = _largest_component(P, idx[sel], link)
            if len(sel_idx) < min_pts:
                break
            if pr["type"] == "cylinder":
                # a square pocket's four walls also pass for a ring: of the points in a wide band
                # around the radius, a real cylinder holds nearly all within the inlier band
                hq = (Q - pr["p"]) @ pr["a"]
                hs = (P[sel_idx] - pr["p"]) @ pr["a"]
                band = (np.abs(d) < 3 * eps) & (hq >= hs.min()) & (hq <= hs.max())
                if (np.abs(d[band]) < eps).mean() < 0.8:
                    idx = np.setdiff1d(idx, sel_idx)
                    continue
            # an edge strip (points of two faces near their sharp edge) fits a small cylinder too,
            # a false fillet: the points must not be explained already by the existing surfaces
            Qs, Ns_ = P[sel_idx], N[sel_idx]
            expl = np.zeros(len(sel_idx), bool)
            for q in prims:
                near = np.abs(prim_dist(q, Qs)) < eps
                if near.any():
                    cq = np.sum(prim_normals(q, Qs[near]) * Ns_[near], axis=1)
                    cq = cq if "s" in q else np.abs(cq)
                    expl[np.where(near)[0][cq > cos45]] = True
            if expl.mean() > 0.5:
                idx = np.setdiff1d(idx, sel_idx)
                continue
            rms = float(np.sqrt(np.mean(prim_dist(pr, P[sel_idx]) ** 2)))
            if rms > 1.5 * max(noise_mm, 0.2):
                break
            signed_cos(pr, P[sel_idx], N[sel_idx])
            pr.update(rms=rms, n_pts=int(len(sel_idx)), snapped=True, detail=True)
            labels[sel_idx] = len(prims) + len(out)
            out.append(pr)
            idx = np.setdiff1d(idx, sel_idx)
    if out:
        log(f"  CAD: {len(out)} klein(e) detail(s) gevonden "
            f"({sum(p['type'] == 'cylinder' for p in out)} cilinder)")
    return prims + out, labels


def boundary_hypotheses(prims, labels, P, N, noise_mm, max_new=40):
    """Small steps (a 2 mm pad, a 3 mm pocket, a notch) have walls too low to be segmented on their own.
    Where a frame-aligned plane region ends, propose a perpendicular, frame-aligned wall at that edge;
    its position is refined on the few wall points there. The cell selection decides whether it exists."""
    eps = max(2.5 * noise_mm, 0.8)
    axes = []
    for p in prims:
        if p["type"] == "plane" and p.get("snapped"):
            if not any(abs(p["n"] @ a) > 0.999 for a in axes):
                axes.append(p["n"] / np.linalg.norm(p["n"]))
    if len(axes) < 2:
        return []
    existing = [(p["n"], -p["d"]) for p in prims if p["type"] == "plane"]
    cyls = [p for p in prims if p["type"] in ("cylinder", "extrusion")]
    tree = cKDTree(P)
    out = []
    for k, p in enumerate(prims):
        if p["type"] != "plane" or not p.get("snapped"):
            continue
        Q = P[labels == k]
        if len(Q) < 30:
            continue
        for e in axes:
            if abs(e @ p["n"]) > 0.1:
                continue
            t = Q @ e
            for pos in (np.percentile(t, 0.5), np.percentile(t, 99.5)):
                if any(abs(n @ e) > 0.999 and abs(off * np.sign(n @ e) - pos) < 1.5 for n, off in existing):
                    continue
                # the region may end on a known curved surface (counterbore floor -> its bore wall, a flat
                # face running tangentially into a profile curve)
                E = Q[np.abs(t - pos) < 2 * eps]
                if len(E) and cyls and np.mean(np.min(np.stack([np.abs(prim_dist(c_, E)) for c_ in cyls], 1), 1)
                                               < 1.5 * eps) > 0.5:
                    continue
                # refine on points near the candidate wall with a compatible normal
                near = np.asarray(tree.query_ball_point(Q[np.argmin(np.abs(t - pos))], 40.0))
                if len(near) == 0:
                    continue
                Rn = P[near]
                dd = Rn @ e - pos
                sel = (np.abs(dd) < eps) & (np.abs(N[near] @ e) > 0.7)
                if sel.sum() >= 8:
                    pos = float(np.median(Rn[sel] @ e))
                q = {"type": "plane", "n": e.copy(), "d": -pos, "hyp": True, "snapped": True,
                     "rms": 0.0, "n_pts": int(sel.sum())}
                existing.append((e, pos))
                out.append(q)
                if len(out) >= max_new:
                    return out
    return out


def thickness_hypotheses(prims, labels, P, max_t=6.0, max_new=12, floor_z=None):
    """Sheet metal / thin walls: when a plane is only seen from one side, its back face is unseen.
    Wall thicknesses measured elsewhere in the part (pairs of opposite, parallel, overlapping planes)
    are used to propose the hidden back faces; the cell selection only uses them where the occupancy
    needs a closure. Returns a list of hypothesis planes (marked 'hyp')."""
    planes = [(k, p) for k, p in enumerate(prims) if p["type"] == "plane" and "s" in p]
    def on(p):
        return p["s"] * p["n"]
    def overlap(i, j, n):
        u, v = _perp_basis(n)
        A, B = P[labels == i], P[labels == j]
        if len(A) == 0 or len(B) == 0:
            return False
        a2, b2 = np.c_[A @ u, A @ v], np.c_[B @ u, B @ v]
        lo = np.maximum(a2.min(0), b2.min(0))
        hi = np.minimum(a2.max(0), b2.max(0))
        return bool(np.all(hi - lo > 2.0))
    ts, paired = [], set()
    thickness_hypotheses.pairs = []
    for a in range(len(planes)):
        for b in range(a + 1, len(planes)):
            (i, p), (j, q) = planes[a], planes[b]
            if on(p) @ on(q) > -np.cos(np.radians(3)):
                continue
            t = abs((-p["d"]) - (-q["d"] * float(np.sign(p["n"] @ q["n"]))))
            ci, cj = P[labels == i].mean(0), P[labels == j].mean(0)
            is_wall = (cj - ci) @ on(p) < 0  # q lies behind p: material in between (not a slot/pocket)
            if is_wall and 0.3 < t < max_t and overlap(i, j, p["n"]):
                # the back of each must lie behind the front (wall material between them)
                thickness_hypotheses.pairs.append((i, j, round(t, 2)))
                ts.append(t)
                paired.update((i, j))
    if not ts:
        return []
    ts = sorted(ts)
    uniq = []
    for t in ts:
        if not uniq or t - uniq[-1] > 0.5:
            uniq.append(t)
    uniq = uniq[:3]
    out = []
    for k, p in planes:
        if k in paired:
            continue
        for t in uniq:
            n = on(p)
            q = {"type": "plane", "n": p["n"].copy(), "d": p["d"] + p["s"] * t, "s": -p["s"], "hyp": True,
                 "rms": 0.0, "n_pts": 0}
            if floor_z is not None and abs(n[2]) > 0.99 and abs(-q["d"] * q["n"][2] - floor_z) < 1.5:
                continue  # the floor itself closes it
            out.append(q)
            if len(out) >= max_new:
                return out
    return out


def estimate_noise(P, n_planes=4, band=1.5):
    """Surface noise (sigma, mm) from robust residuals of the largest planes; None if no plane.
    A narrow band keeps the opposite side of thin sheets (>= 2 mm away) out of the statistics."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4.0, max_nn=60))
    sig = []
    for _ in range(n_planes):
        if len(pc.points) < 500:
            break
        m, idx = pc.segment_plane(1.0, 3, 1000)
        Q = np.asarray(pc.points)
        r = Q @ np.asarray(m[:3]) + m[3]
        # only points whose local surface is parallel to the plane (not walls touching it)
        par = np.abs(np.asarray(pc.normals) @ np.asarray(m[:3])) > 0.97
        r = np.where(par, r, np.inf)
        b = band
        for _ in range(3):
            near = np.abs(r) < b
            if near.sum() < 300:
                break
            rr = r[near] - np.median(r[near])
            sg = 1.4826 * np.median(np.abs(rr))
            sg /= _trunc_mad_factor(b / max(sg, 1e-6))
            break  # single pass: growing the band pulls in the other side of thin sheets
        if near.sum() < 300:
            break
        sig.append((near.sum(), sg))
        pc = pc.select_by_index(np.where(np.abs(r) < max(b, 2.0))[0], invert=True)
    if not sig:
        return None
    # a plane that straddles two nearby parallel faces (thin sheet) only inflates its estimate:
    # take the smallest estimate among well-supported planes
    good = [b_ for a, b_ in sig if a >= 1000] or [b_ for _, b_ in sig]
    return float(min(good))


def _trunc_mad_factor(c):
    """MAD of a normal distribution truncated at +-c sigma, relative to the untruncated MAD."""
    from scipy.stats import norm
    c = max(float(c), 0.7)
    mass = 2 * norm.cdf(c) - 1
    q = norm.ppf(0.5 + 0.25 * mass)
    return float(q / 0.6744897501960817)


def _approx_area(P, voxel):
    """Surface area estimate: number of occupied surface voxels x voxel^2."""
    q = np.unique(np.floor(P / (2 * voxel)).astype(np.int64), axis=0)
    return len(q) * (2 * voxel) ** 2


def clean_solid(shape):
    """Merge coplanar faces, keep the largest solid without internal cavities, fix it, and return it
    as a bare TopoDS_Solid (not wrapped in a compound)."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.BRepClass3d import BRepClass3d
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid

    def largest(sh):
        best, bv = None, -1.0
        ex = TopExp_Explorer(sh, TopAbs_SOLID)
        while ex.More():
            so = TopoDS.Solid(ex.Current())
            ex.Next()
            g = GProp_GProps()
            BRepGProp.VolumeProperties_s(so, g)
            if abs(g.Mass()) > bv:
                best, bv = so, abs(g.Mass())
        return best

    try:
        u = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        u.Build()
        shape = u.Shape()
    except Exception:
        pass
    best = largest(shape)
    if best is None:
        return shape
    # closed internal cavities can never be seen by a camera: they are noise -> keep the outer shell
    outer = BRepClass3d.OuterShell_s(best)
    if not outer.IsNull():
        ms = BRepBuilderAPI_MakeSolid(outer)
        if ms.IsDone():
            best = ms.Solid()
    fx = ShapeFix_Shape(best)
    fx.Perform()
    return largest(fx.Shape()) or fx.Shape()


def quality(shape, P, noise_mm, floor_z=None):
    """Deviation of the scan points from the CAD solid, and the acceptance decision."""
    rep = shape_report(shape, P, step=0.5)
    S = np.vstack([_sample_face(f, 0.5)[0] for f in _faces_of(shape)])
    d, _ = cKDTree(S).query(P)
    rep["frac_far"] = float(np.mean(d > 3.0 * noise_mm))
    rep["dev_p98"] = float(np.percentile(d, 98))
    ok = bool(rep["valid"] and rep["solid"] and rep["faces"] >= 4
              and rep["dev_median"] <= max(1.0, 2.0 * noise_mm)
              and rep["frac_far"] <= 0.05)
    rep["accepted"] = ok
    return rep


def roundtrip_ok(path, shape, vol_tol=0.005, ref_volume=None):
    """Read a written STEP back and compare: must be a valid solid with the same volume and no face
    larger than the part (a face whose boundary got lost on export becomes a huge sheet)."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    roundtrip_ok.healed = False
    r = STEPControl_Reader()
    if r.ReadFile(str(path)) != 1:
        return False, "niet leesbaar"
    r.TransferRoots()
    rs = r.OneShape()
    if not BRepCheck_Analyzer(rs).IsValid():
        # a few faces (of thousands) can come back with a wire the checker rejects; CAD importers heal
        # that on import. Accept only if the standard healing makes it valid and every other check
        # (one solid, closed, same faces, same volume) still passes.
        from OCP.ShapeFix import ShapeFix_Shape
        fx = ShapeFix_Shape(rs)
        fx.SetPrecision(1e-4)
        fx.SetMaxTolerance(1e-2)
        fx.Perform()
        rs = fx.Shape()
        if not BRepCheck_Analyzer(rs).IsValid():
            return False, "teruggelezen STEP ongeldig"
        roundtrip_ok.healed = True
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_WIRE
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    def count(sh, t):
        ex, n = TopExp_Explorer(sh, t), 0
        while ex.More():
            n += 1
            ex.Next()
        return n
    if count(rs, TopAbs_SOLID) != 1 or count(rs, TopAbs_SHELL) != 1:
        return False, f"{count(rs, TopAbs_SOLID)} solids / {count(rs, TopAbs_SHELL)} schillen i.p.v. 1"
    fb = ShapeAnalysis_FreeBounds(rs, 1e-4)
    if count(fb.GetClosedWires(), TopAbs_WIRE) + count(fb.GetOpenWires(), TopAbs_WIRE) > 0:
        return False, "open randen"
    if len(_faces_of(rs)) != len(_faces_of(shape)):
        return False, f"{len(_faces_of(rs))} vlakken i.p.v. {len(_faces_of(shape))}"
    def vol(x):
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(x, g)
        return g.Mass()
    v0, v1 = vol(shape), vol(rs)
    if v1 <= 0:
        return False, "negatief volume (omgekeerde oriëntatie)"
    if ref_volume:  # independent reference: the summed volume of the chosen cells
        v0 = ref_volume
        vol_tol = max(vol_tol, 0.02)  # removed internal cavities may add a little
    if abs(v1 - v0) > vol_tol * abs(v0):
        return False, f"volume na teruglezen {v1:.0f} i.p.v. {v0:.0f} mm3"
    b0 = Bnd_Box()
    BRepBndLib.Add_s(shape, b0, False)
    ext0 = b0.CornerMax().Distance(b0.CornerMin())
    for f in _faces_of(rs):
        b = Bnd_Box()
        BRepBndLib.Add_s(f, b, False)
        if b.CornerMax().Distance(b.CornerMin()) > 1.5 * ext0:
            return False, "vlak verloor zijn begrenzing bij export"
    return True, "ok"


def export_solid(path, name, shape, rgb, ref_volume=None):
    """Write one part and make sure it reads back correctly. Repair cascade when it does not:
    1. small-edge / wire fixes, 2. the whole solid as B-spline (NURBS) geometry: same shape,
    universally readable. Returns (ok, method, shape_written)."""
    from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Wireframe
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
    write_named_step(path, [(name, shape, rgb)])
    ok, why = roundtrip_ok(path, shape, ref_volume=ref_volume)
    if ok and roundtrip_ok.healed:
        return True, "analytisch (bij inlezen standaard hersteld)", shape
    if ok:
        return True, "analytisch", shape
    sf = ShapeFix_Shape(shape)
    sf.Perform()
    wf = ShapeFix_Wireframe(sf.Shape())
    wf.SetPrecision(0.01)
    wf.FixSmallEdges()
    wf.FixWireGaps()
    s1 = wf.Shape()
    write_named_step(path, [(name, s1, rgb)])
    if roundtrip_ok(path, s1, ref_volume=ref_volume)[0]:
        return True, "analytisch (gerepareerd)", s1
    s2 = BRepBuilderAPI_NurbsConvert(shape, True).Shape()
    write_named_step(path, [(name, s2, rgb)])
    ok, why2 = roundtrip_ok(path, s2, ref_volume=ref_volume)
    if ok:
        return True, "NURBS", s2
    return False, f"{why}; NURBS: {why2}", shape


def write_named_step(path, parts, assembly_name=None):
    """STEP (AP214) with names and colours. parts: list of (name, shape, rgb 0..1). One part -> a
    single named solid; several -> an assembly with one component per part (moving parts stay separate)."""
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.Interface import Interface_Static
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    if len(parts) == 1:
        name, shape, rgb = parts[0]
        lab = st.AddShape(shape, False)
        TDataStd_Name.Set_s(lab, TCollection_ExtendedString(name))
        ct.SetColor(lab, Quantity_Color(*map(float, rgb), Quantity_TOC_RGB), XCAFDoc_ColorSurf)
    else:
        comp = TopoDS_Compound()
        BRep_Builder().MakeCompound(comp)
        asm = st.NewShape()
        TDataStd_Name.Set_s(asm, TCollection_ExtendedString(assembly_name or "samenstelling"))
        from OCP.TopLoc import TopLoc_Location
        for name, shape, rgb in parts:
            plab = st.AddShape(shape, False)
            TDataStd_Name.Set_s(plab, TCollection_ExtendedString(name))
            ct.SetColor(plab, Quantity_Color(*map(float, rgb), Quantity_TOC_RGB), XCAFDoc_ColorSurf)
            cl = st.AddComponent(asm, plab, TopLoc_Location())
            TDataStd_Name.Set_s(cl, TCollection_ExtendedString(name))
        st.UpdateAssemblies()
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214")
    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    w.SetColorMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    return w.Write(str(path)) == 1
