"""Split the reference-state mesh into rigid bodies using free-space reasoning across machine poses.

For every surface patch and every rigid-body hypothesis H (from markers), the patch is moved by H
into another pose and checked against what the cameras of that pose actually saw:
  - camera sees a surface exactly there        -> support
  - camera sees *through* that location         -> contradiction (that space is empty)
  - camera's view is blocked before it          -> no information
A patch belongs to the body whose motion is not contradicted. Patches without evidence (e.g. faces
parallel to a linear motion) inherit the label of their neighbours, preferring convex edges.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .features import detect_regions, face_adjacency
from .geometry import to_h
from .log import log


def _scene(mesh):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return sc


def make_patches(V, F, tol, angle_deg=12.0, crease_deg=35.0, chunk_mm=25.0):
    labels, regions = detect_regions(V, F, tol, angle_deg, min_area=200.0, cylinders=True)
    patch = labels.copy()
    nxt = len(regions)
    free = patch < 0
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    adj = face_adjacency(F)
    dots = (fn[adj[:, 0]] * fn[adj[:, 1]]).sum(1)
    sel = free[adj[:, 0]] & free[adj[:, 1]] & (dots > np.cos(np.radians(crease_deg)))
    A = coo_matrix((np.ones(sel.sum()), (adj[sel, 0], adj[sel, 1])), shape=(len(F), len(F)))
    _, comp = connected_components(A, directed=False)
    # leftover (free-form / rounded) faces: split components into spatial chunks so that a component
    # spanning two parts (e.g. a rounded contact edge) can be divided between them
    cen = tri.mean(1)
    cell = np.floor(cen / chunk_mm).astype(np.int64)
    key = comp.astype(np.int64) * 1_000_003 + ((cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791))
    fk = key[free]
    _, inv = np.unique(fk, return_inverse=True)
    patch[free] = nxt + inv
    # relabel compactly
    u, patch = np.unique(patch, return_inverse=True)
    return patch, fn, adj


def patch_graph(V, F, patch, fn, adj):
    """Edges between patches with weight = boundary length * (1 convex | 0.3 concave)."""
    pa = patch[adj[:, 0]]
    pb = patch[adj[:, 1]]
    cross = pa != pb
    if not cross.any():
        return {}
    a = adj[cross]
    # shared edge length
    Fa = F[a[:, 0]]
    Fb = F[a[:, 1]]
    shared = np.array([list(set(x) & set(y))[:2] for x, y in zip(Fa, Fb)])
    L = np.linalg.norm(V[shared[:, 0]] - V[shared[:, 1]], axis=1)
    # convexity: centroid of b relative to plane of a
    ca = V[Fa].mean(1)
    cb = V[Fb].mean(1)
    convex = ((cb - ca) * fn[a[:, 0]]).sum(1) < 0
    w = L * np.where(convex, 1.0, 0.3)
    G = {}
    for i, j, ww in zip(pa[cross], pb[cross], w):
        key = (min(i, j), max(i, j))
        G[key] = G.get(key, 0.0) + ww
    return G


def _sample_faces(V, F, faces, n, rng):
    tri = V[F[faces]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    p = area / area.sum() if area.sum() > 0 else None
    k = rng.choice(len(faces), size=min(n, max(len(faces) * 3, 1)), p=p)
    r1 = np.sqrt(rng.random(len(k)))
    r2 = rng.random(len(k))
    t = tri[k]
    pts = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
    return pts, faces[k]


_FOV = {}


def _fov_radius(cam):
    key = id(cam)
    r = _FOV.get(key)
    if r is None:
        w, h = cam.width, cam.height
        pts = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, 0], [0, h / 2]], float)
        try:
            r = float(np.linalg.norm(np.asarray(cam.cam_from_img(pts)), axis=1).max()) * 1.02
        except Exception:
            r = 10.0
        _FOV[key] = r
    return r


def evidence(q, nq, scene, cams, tau, max_cams=60):
    """Return (n_support, n_contra) per point for points q (state-S coords)."""
    n = len(q)
    sup = np.zeros(n, int)
    con = np.zeros(n, int)
    if not cams:
        return sup, con
    rays = []
    meta = []
    cam_list = cams if len(cams) <= max_cams else [cams[i] for i in np.linspace(0, len(cams) - 1, max_cams).astype(int)]
    for cam, M, C in cam_list:
        pc = q @ M[:3, :3].T + M[:3, 3]
        z = pc[:, 2]
        # guard against lens-distortion fold-back: only points inside the true field of view
        rmax = _fov_radius(cam)
        xn = pc[:, :2] / np.maximum(z[:, None], 1e-9)
        ok = (z > 50.0) & (np.linalg.norm(xn, axis=1) < rmax)
        uv = np.full((n, 2), -1e9)
        if ok.any():
            uv[ok] = cam.img_from_cam(pc[ok], check_cheirality=False)
        ok &= (uv[:, 0] > 5) & (uv[:, 1] > 5) & (uv[:, 0] < cam.width - 5) & (uv[:, 1] < cam.height - 5)
        idx = np.where(ok)[0]
        if not len(idx):
            continue
        d = q[idx] - C
        dist = np.linalg.norm(d, axis=1)
        d /= dist[:, None]
        facing = (nq[idx] * (-d)).sum(1) > 0.15
        rays.append(np.c_[np.broadcast_to(C, d.shape), d])
        meta.append((idx, dist, facing))
    if not rays:
        return sup, con
    R = np.vstack(rays).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(R))["t_hit"].numpy()
    o = 0
    for idx, dist, facing in meta:
        th = t[o:o + len(idx)]
        o += len(idx)
        s = (np.abs(th - dist) < tau) & facing
        c = th > dist + tau
        np.add.at(sup, idx[s], 1)
        np.add.at(con, idx[c], 1)
    return sup, con


def _patch_area(V, F, faces):
    t = V[F[faces]]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def label_reference(meshA, hyps: list, others: list, tol, tau, rng=None, manual=None, smooth_mm=8.0):
    """Label faces of the reference mesh.

    hyps: list of dicts {name, T: {state: 4x4 A->state}} (index 0 = static/basis).
    others: list of dicts {state, mesh, cams}.
    Returns face labels (index into hyps) and per-patch diagnostics.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    V = np.asarray(meshA.vertices)
    F = np.asarray(meshA.triangles)
    patch, fn, adj = make_patches(V, F, tol)
    nP = patch.max() + 1
    G = patch_graph(V, F, patch, fn, adj)
    scenes = {o["state"]: _scene(o["mesh"]) for o in others}
    nH = len(hyps)
    fc = np.zeros((nP, nH))
    info = np.zeros((nP, nH))
    face_lab = -np.ones(len(F), int)
    face_ids_by_patch = [np.where(patch == p)[0] for p in range(nP)]
    sample_cache = {}
    for p in range(nP):
        faces = face_ids_by_patch[p]
        pts, fidx = _sample_faces(V, F, faces, 300, rng)
        nr = fn[fidx]
        sample_cache[p] = (pts, nr, fidx)
        for h, H in enumerate(hyps):
            n_con = 0
            n_inf = 0
            for o in others:
                T = H["T"].get(o["state"])
                if T is None:
                    continue
                q = to_h(T, pts)
                nq = nr @ T[:3, :3].T
                s, c = evidence(q, nq, scenes[o["state"]], o["cams"], tau)
                contra = (c >= 2) & (c > 2 * s)
                informed = contra | (s >= 1)
                n_con += contra.sum()
                n_inf += informed.sum()
            fc[p, h] = n_con / max(n_inf, 1)
            info[p, h] = n_inf
    # ---- global labelling: unary (evidence) + boundary smoothness, minimised with ICM
    active = [h for h, H in enumerate(hyps) if h == 0 or any(H["T"].get(o["state"]) is not None for o in others)]
    p_area = np.array([_patch_area(V, F, face_ids_by_patch[p]) for p in range(nP)])
    conf = np.minimum(info, 60.0) / 60.0
    U = p_area[:, None] * fc * conf * 4.0          # contradicted surface costs (mm^2)
    U[:, [h for h in range(nH) if h not in active]] = np.inf
    U[:, 1:] += 1e-3 * p_area[:, None]              # tiny prior: unexplained surface belongs to the base
    nbrs = {}
    for (i, j), w in G.items():
        nbrs.setdefault(i, []).append((j, w * smooth_mm))
        nbrs.setdefault(j, []).append((i, w * smooth_mm))
    lab = np.argmin(U, axis=1)
    order = np.argsort(-p_area)
    for it in range(60):
        changed = 0
        for p in order:
            cost = U[p].copy()
            for q_, w in nbrs.get(p, []):
                cost += w
                cost[lab[q_]] -= w
            nl = int(np.argmin(cost))
            if nl != lab[p]:
                lab[p] = nl
                changed += 1
        if changed == 0:
            break
    split = [p for p in range(nP) if fc[p, lab[p]] > 0.3 and info[p, lab[p]] >= 20 and len(face_ids_by_patch[p]) > 50]
    cand = {}
    face_lab = lab[patch]
    # split patches where no hypothesis fits: per face, best hypothesis from point evidence
    for p in split:
        pts, nr, fidx = sample_cache[p]
        cons = np.zeros((len(pts), nH))
        for h, H in enumerate(hyps):
            for o in others:
                T = H["T"].get(o["state"])
                if T is None:
                    continue
                s, c = evidence(to_h(T, pts), nr @ T[:3, :3].T, scenes[o["state"]], o["cams"], tau)
                cons[:, h] += (c >= 2) & (c > 2 * s)
        best = np.argmin(cons + 1e-3 * np.arange(nH), axis=1)
        faces = face_ids_by_patch[p]
        cen = V[F[faces]].mean(1)
        _, nn = cKDTree(pts).query(cen)
        face_lab[faces] = best[nn]
    names = [h["name"] for h in hyps]
    if manual:
        face_lab = apply_manual(V, F, face_lab, manual, names)
    diag = {"n_patches": int(nP), "n_split": len(split), "icm_iterations": it + 1, "names": names}
    for h, H in enumerate(hyps):
        diag[f"faces_{H['name']}"] = int((face_lab == h).sum())
    log(f"  segmentatie: {nP} vlakdelen, " + ", ".join(f"{n}: {(face_lab == h).sum():,} driehoeken"
                                                        for h, n in enumerate(names)))
    return face_lab, diag


def apply_manual(V, F, face_lab, manual, names):
    cen = V[F].mean(1)
    for mb in manual:
        lo = np.asarray(mb["box_min"], float)
        hi = np.asarray(mb["box_max"], float)
        inside = np.all((cen >= lo) & (cen <= hi), axis=1)
        if mb["name"] in names:
            idx = names.index(mb["name"])
        else:
            names.append(mb["name"])
            idx = len(names) - 1
        face_lab[inside] = idx
    return face_lab


def submesh(mesh, face_mask):
    m = o3d.geometry.TriangleMesh(mesh)
    m.remove_triangles_by_mask(~face_mask)
    m.remove_unreferenced_vertices()
    return m


def label_points(points, meshA, face_lab):
    sc = _scene(meshA)
    r = sc.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
    pid = r["primitive_ids"].numpy()
    d = np.linalg.norm(r["points"].numpy() - points, axis=1)
    return face_lab[pid], d


def fuse_other_state(points_S, body_meshes_A: list, hyps: list, state: str, tau: float, grow_r: float,
                     normals_S=None):
    """Label points of state S by inverse-transforming each hypothesis onto the A-pose body meshes.

    Returns list (per body) of points expressed in A pose.
    """
    nH = len(hyps)
    D = np.full((len(points_S), nH), np.inf)
    for h, H in enumerate(hyps):
        T = H["T"].get(state)
        if T is None or body_meshes_A[h] is None or len(body_meshes_A[h].triangles) == 0:
            continue
        p = to_h(np.linalg.inv(T), points_S)
        D[:, h] = _scene(body_meshes_A[h]).compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()
    best = np.argmin(D, axis=1)
    dmin = D[np.arange(len(D)), best]
    srt = np.sort(D, axis=1)
    margin = srt[:, 1] - srt[:, 0] if nH > 1 else np.full(len(D), np.inf)
    lab = np.where((dmin < tau) & (margin > tau * 0.5), best, -1)
    # region growing into newly visible surfaces: a point may only adopt a neighbour's body if, moved back
    # with that body's motion, it lies close (< grow_max) to that body's reference surface
    grow_max = max(10 * tau, 25.0)
    allowed = D < grow_max
    tree = cKDTree(points_S)
    for _ in range(40):
        un = np.where((lab < 0) & allowed.any(1))[0]
        if not len(un):
            break
        nn = tree.query_ball_point(points_S[un], grow_r)
        new = lab.copy()
        cnt = 0
        for i, lst in zip(un, nn):
            ls = lab[lst]
            ls = ls[(ls >= 0)]
            ls = ls[allowed[i, ls]] if len(ls) else ls
            if len(ls):
                vals, c = np.unique(ls, return_counts=True)
                new[i] = vals[np.argmax(c)]
                cnt += 1
        lab = new
        if cnt == 0:
            break
    out = []
    out_n = []
    for h, H in enumerate(hyps):
        sel = lab == h
        T = H["T"].get(state)
        if T is None or not sel.any():
            out.append(np.zeros((0, 3)))
            out_n.append(np.zeros((0, 3)))
        else:
            Ti = np.linalg.inv(T)
            out.append(to_h(Ti, points_S[sel]))
            out_n.append(normals_S[sel] @ Ti[:3, :3].T if normals_S is not None else np.zeros((sel.sum(), 3)))
    return out, out_n, lab
