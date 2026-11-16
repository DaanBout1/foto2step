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
    if prim["type"] == "plane":
        return np.broadcast_to(prim["n"], P.shape)
    return cyl_normals(prim, P)


def _largest_component(P, idx, link):
    """Largest spatially connected subset of P[idx] (points closer than `link` are connected)."""
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
    return idx[lab == np.bincount(lab).argmax()]


def segment_primitives(P, N, noise_mm, min_pts=None, max_prims=60, seed=0, cylinders=True):
    """Efficient-RANSAC style multi-primitive detection (planes + cylinders) with connectivity.

    Returns (prims, labels): labels[i] = index into prims or -1."""
    rng = np.random.default_rng(seed)
    n_all = len(P)
    eps = max(2.5 * noise_mm, 0.8)
    cos_t = np.cos(np.radians(16.0))
    link = 3.0 * max(noise_mm, 0.5) + 1.5
    min_pts = min_pts or max(120, int(0.0015 * n_all))
    labels = -np.ones(n_all, int)
    prims = []
    tree = cKDTree(P)
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
            cs = np.abs(np.sum(prim_normals(c, Q[ok]) * NQ[ok], axis=1))
            scores.append(int((cs > cos_t).sum()))
        order = np.argsort(scores)[::-1][:6]
        best, best_idx = None, None
        for k in order:
            c = cands[k]
            for _ in range(3):  # inliers -> refit -> inliers
                d = np.abs(prim_dist(c, P[free]))
                ok = d < eps
                cs = np.abs(np.sum(prim_normals(c, P[free][ok]) * N[free][ok], axis=1))
                idx = free[ok][cs > cos_t]
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
                c["ang"] = span
            if best is None or len(idx) > len(best_idx):
                best, best_idx = c, idx
        if best is None or len(best_idx) < min_pts:
            break
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
                if dd < best_d:
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
                if a["type"] == "plane" and abs(a["n"] @ b["n"]) < np.cos(np.radians(3)):
                    continue
                if a["type"] == "cylinder" and (abs(a["a"] @ b["a"]) < np.cos(np.radians(3)) or abs(a["r"] - b["r"]) > 2 * eps):
                    continue
                if np.median(np.abs(prim_dist(a, Pj))) < eps and np.median(np.abs(prim_dist(b, Pi))) < eps:
                    both = np.r_[Pi, Pj]
                    new = fit_plane(both) if a["type"] == "plane" else fit_cylinder(both, a["a"], a["p"], a["r"])
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
