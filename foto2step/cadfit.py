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


def fit_plane(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
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


def prim_dist(prim, P):
    return plane_dist(prim, P) if prim["type"] == "plane" else cyl_dist(prim, P)


def prim_normals(prim, P):
    """Surface normals; with an orientation sign 's' (set once the data normals are oriented) they
    point to the side the scan saw, so the two faces of a thin sheet are told apart."""
    sg = prim.get("s", 1.0)
    if prim["type"] == "plane":
        return np.broadcast_to(sg * prim["n"], P.shape)
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
                idx = _largest_component(P, idx, link, min_size=min_pts if c["type"] == "plane" else None)
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
                if a is None or b is None or a["type"] != b["type"]:
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
    a, p0, r = prim["a"], prim["p"], prim["r"]
    t0 = (ctr - p0) @ a
    u, _ = _perp_basis(a)
    ax = gp_Ax3(gp_Pnt(*(p0 + t0 * a)), gp_Dir(*a), gp_Dir(*u))
    return BRepBuilderAPI_MakeFace(gp_Cylinder(ax, r), 0.0, 2 * np.pi, -diag, diag).Face()


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


def cell_solid(prims, P, occ, floor_z=None, margin=0.3, grid=0.6, min_frac=0.5, max_cells=20000, smooth=0.3,
               noise_mm=0.5):
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
    lo = np.percentile(P, 0.02, axis=0) - margin
    hi = np.percentile(P, 99.98, axis=0) + margin
    if floor_z is not None:
        lo[2] = floor_z
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
    cells = []
    ex = TopExp_Explorer(mv.Shape(), TopAbs_SOLID)
    while ex.More():
        c = TopoDS.Solid(ex.Current())
        ex.Next()
        Vc, _ = _mesh_of_shape(c, defl=0.5)
        if len(Vc) and np.all(Vc.min(0) >= lo - 0.01) and np.all(Vc.max(0) <= hi + 0.01):
            cells.append(c)
    if len(cells) > max_cells:
        raise RuntimeError(f"te veel cellen ({len(cells)})")
    # grid samples once, occupancy of the reference once
    axes = [np.arange(lo[i] + grid / 2, hi[i], grid) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    ref_in = occ(G)
    keep, fracs, vols = [], [], []
    for c in cells:
        V, T = _mesh_of_shape(c, defl=0.1)
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
                               floor_z=floor_z, prims=prims)
    chosen = [c for c, k in zip(cells, keep) if k]
    shape = _boundary_solid(chosen)
    rep = {"surfaces": len(prims), "cells": len(cells), "inside": len(chosen),
           "ambiguous_cells": int(((fr > 0.2) & (fr < 0.8)).sum())}
    return shape, rep


def _graphcut_cells(cells, frac, vol, lam, P=None, eps=1.0, step=1.0, w_unsup=4.0, floor_z=None,
                    prims=None, w_orient=5.0):
    """Binary labelling of cells by exact min-cut.
    unary: volume that disagrees with the reference occupancy;
    pairwise: every face that would become part of the model surface (between an inside and an outside
    cell, or between an inside cell and the bounding box) costs lam x area, plus w_unsup x area for the
    part of it that no scan point supports. A real surface was seen by the cameras; a protrusion made of
    unsupported faces is a classification error, not geometry."""
    import networkx as nx
    owner = {}
    farea = {}
    fsup = {}
    face_pts = {}
    tree = cKDTree(P) if P is not None else None
    for i, c in enumerate(cells):
        for f in _faces_of(c):
            h = hash(f)
            owner.setdefault(h, []).append(i)
            if h not in farea:
                S, a = _sample_face(f, step)
                farea[h] = a
                face_pts[h] = S
                if floor_z is not None and len(S) and np.all(np.abs(S[:, 2] - floor_z) < 0.05):
                    fsup[h] = 1.0  # the underside on the floor is never seen: no penalty
                elif tree is not None and len(S):
                    d, _ = tree.query(S, distance_upper_bound=eps)
                    fsup[h] = float(np.isfinite(d).mean())
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
            if len(own) != 2 or fsup.get(h, 0) < 0.3:
                continue
            S = face_pts.get(h)
            if S is None or len(S) == 0:
                continue
            x = S[len(S) // 2]
            dk = [abs(prim_dist(pr, x[None])[0]) for pr in prims]
            k = int(np.argmin(dk))
            if dk[k] > 0.05 or "s" not in prims[k]:
                continue
            nrm = prim_normals(prims[k], x[None])[0]
            q = x + 0.05 * nrm
            cls = BRepClass3d_SolidClassifier(cells[own[0]], gp_Pnt(*map(float, q)), 1e-6)
            front, back = (own[0], own[1]) if cls.State() == TopAbs_IN else (own[1], own[0])
            w = w_orient * fsup[h] * farea[h]
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
    return [i in S for i in range(len(cells))]


def _boundary_solid(cells):
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
            if p["type"] == "plane":
                p["d"] = -float(np.mean(Qi @ p["n"]))
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


def prune_redundant(prims, labels, P, noise_mm, frac=0.75, small=0.08):
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
        if (D.min(1) < 1.5 * eps).mean() >= frac:
            alive.remove(k)
            idx = np.where(labels == k)[0]
            labels[idx] = np.array(others)[D.argmin(1)]
    remap = -np.ones(len(prims) + 1, int)
    remap[alive] = np.arange(len(alive))
    labels = np.where(labels >= 0, remap[labels], -1)
    return [prims[k] for k in alive], labels


def regularize_orientations(prims, labels, P, tol_deg=4.0, noise_mm=None, up=None):
    """Machined parts are mostly built on one orthogonal frame: snap plane normals and cylinder axes
    that are within tol_deg of a frame axis exactly onto it, then refit offsets / radii. Removes the
    0.5-2 deg wobble that noisy fits give and makes parallel faces exactly parallel."""
    planes = [(k, p) for k, p in enumerate(prims) if p["type"] == "plane"]
    if not planes:
        return prims
    w = np.array([p["n_pts"] if "n_pts" in p else (labels == k).sum() for k, p in planes], float)
    order = np.argsort(-w)
    if up is not None:
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
        e1 = planes[order[0]][1]["n"].copy()
        e2 = None
        for o in order[1:]:
            n = planes[o][1]["n"]
            if abs(n @ e1) < np.sin(np.radians(10)):
                e2 = _unit(n - (n @ e1) * e1)
                break
        if e2 is None:
            return prims
        e3 = np.cross(e1, e2)
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


def hoppe_occupancy(P, Nrm, floor_z=None, k=24):
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
        if floor_z is not None:
            inside &= Q[:, 2] > floor_z
        return inside
    return occ


# ----------------------------------------------------------------------------------------------
# 5. driver
# ----------------------------------------------------------------------------------------------

def reconstruct(P, cams, noise_mm, floor_z=0.0, normals=None, voxel=None, log_fn=log):
    """Scan points of one part (+ camera centres) -> (solid, info). Cell-complex method."""
    import open3d as o3d
    from .meshing import orient_by_visibility
    t0 = time.time()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float)))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    est = estimate_noise(np.asarray(pc.points))
    if est is not None:
        log_fn(f"  CAD: ruis uit vlakfit {est:.2f} mm (opgegeven {noise_mm:.2f} mm)")
        noise_mm = max(noise_mm, est)
    voxel = voxel or max(0.5, 1.2 * noise_mm)
    pc = pc.voxel_down_sample(voxel)
    r_n = max(3.0, 5 * voxel)
    dens = len(pc.points) / max(1e-9, _approx_area(np.asarray(pc.points), voxel))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r_n, max_nn=int(np.clip(4 * np.pi * r_n ** 2 * dens, 40, 400))))
    Pd = np.asarray(pc.points)
    if normals is not None:
        Nd = np.asarray(normals)
        Nd = Nd[cKDTree(np.asarray(P)).query(Pd)[1]]
        dec = 1.0
    else:
        Nd, dec = orient_by_visibility(Pd, np.asarray(pc.normals), cams)
    # all points take part (thin flanges and steps just above the floor too); floor points were cropped
    Ps, Ns = Pd, Nd
    min_pts = max(40, int(25.0 * dens))  # ~25 mm^2 of surface, independent of point density
    prims, lab = segment_primitives(Ps, Ns, noise_mm, min_pts=min_pts, oriented=True)
    prims, lab = merge_similar(prims, lab, Ps, noise_mm)
    prims = regularize_orientations(prims, lab, Ps, noise_mm=noise_mm,
                                    up=(0, 0, 1.0) if floor_z is not None else None)
    prims, lab = prune_redundant(prims, lab, Ps, noise_mm)
    prims = refine_primitives(prims, lab, Ps, Ns, noise_mm, oriented=True)
    prims, lab = merge_similar(prims, lab, Ps, noise_mm)  # snapping can make planes coincide
    log_fn(f"  CAD: {len(prims)} oppervlakken ({sum(p['type'] == 'plane' for p in prims)} vlak, "
           f"{sum(p['type'] == 'cylinder' for p in prims)} cilinder), {np.mean(lab >= 0):.0%} van de punten verklaard")
    hyp = thickness_hypotheses(prims, lab, Ps, floor_z=floor_z)
    if hyp:
        log_fn(f"  CAD: {len(hyp)} verborgen achtervlak(ken) voorgesteld uit gemeten wanddikte(s)")
    reconstruct.last = dict(prims=prims, lab=lab, P=Ps, N=Ns, hyp=hyp)
    prims = prims + hyp
    occ = hoppe_occupancy(Pd, Nd, floor_z=floor_z)
    shape, rep = cell_solid(prims, Pd, occ, floor_z=floor_z, noise_mm=noise_mm)
    shape = clean_solid(shape)
    info = {"surfaces": len(prims), "cells": rep["cells"], "inside": rep["inside"],
            "ambiguous": rep["ambiguous_cells"], "oriented": dec, "explained": float(np.mean(lab >= 0)),
            "t": time.time() - t0}
    return shape, info


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
            # check: plane moved by t against its outward normal
            x = -p["d"] * p["n"]
            assert abs((x - t * n) @ q["n"] + q["d"]) < 1e-6
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
    """Keep the largest solid, fix it and merge coplanar faces."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    best, bv = None, -1.0
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        so = TopoDS.Solid(ex.Current())
        ex.Next()
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(so, g)
        if abs(g.Mass()) > bv:
            best, bv = so, abs(g.Mass())
    if best is None:
        return shape
    fx = ShapeFix_Shape(best)
    fx.Perform()
    s = fx.Shape()
    u = ShapeUpgrade_UnifySameDomain(s, True, True, False)
    u.Build()
    return u.Shape()


# ----------------------------------------------------------------------------------------------
# 6. pipeline helpers: quality gate, named single-part STEP, assembly of OCC shapes
# ----------------------------------------------------------------------------------------------

def quality(shape, P, noise_mm, floor_z=None):
    """Deviation of the scan points from the CAD solid, and the acceptance decision."""
    rep = shape_report(shape, P, step=0.5)
    ok = bool(rep["valid"] and rep["solid"] and rep["faces"] >= 4
              and rep["dev_median"] <= max(1.0, 2.0 * noise_mm))
    rep["accepted"] = ok
    return rep


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
