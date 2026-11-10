"""Detect planes and cylinders on a reconstructed mesh, snap vertices onto them and sharpen plane-plane edges.

Machines are mostly made of planes and cylinders. Projecting noisy photogrammetry vertices onto
robustly fitted surfaces removes most of the random noise and restores sharp edges that
Poisson meshing rounds off.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geometry import fit_cylinder, fit_plane
from .log import log


def face_adjacency(F: np.ndarray):
    """Return array of (f1, f2) pairs sharing an edge."""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    e.sort(1)
    key = e[:, 0].astype(np.int64) * (F.max() + 1) + e[:, 1]
    order = np.argsort(key)
    k = key[order]
    same = np.where(k[1:] == k[:-1])[0]
    return np.c_[fid[order][same], fid[order][same + 1]]


def _neighbors(nF, adj):
    rows = np.r_[adj[:, 0], adj[:, 1]]
    cols = np.r_[adj[:, 1], adj[:, 0]]
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    cols = cols[order]
    start = np.searchsorted(rows, np.arange(nF + 1))
    return cols, start


def detect_regions(V, F, tol_mm, angle_deg=12.0, min_area=400.0, cylinders=True):
    """Return (labels per face [-1 none], list of region dicts)."""
    nF = len(F)
    tri = V[F]
    cen = tri.mean(1)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    fn = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-12)
    adj = face_adjacency(F)
    nbr, start = _neighbors(nF, adj)
    cos_t = np.cos(np.radians(angle_deg))
    # local flatness score for seed ordering: mean angle to neighbours
    dots = (fn[adj[:, 0]] * fn[adj[:, 1]]).sum(1)
    flat = np.zeros(nF)
    cnt = np.zeros(nF)
    np.add.at(flat, adj[:, 0], 1 - dots)
    np.add.at(flat, adj[:, 1], 1 - dots)
    np.add.at(cnt, adj[:, 0], 1)
    np.add.at(cnt, adj[:, 1], 1)
    flat = flat / np.maximum(cnt, 1)
    labels = -np.ones(nF, int)
    regions = []
    order = np.argsort(flat)
    # ---------------- planes
    for seed in order:
        if labels[seed] != -1 or flat[seed] > 0.02:
            continue
        rid = len(regions)
        members = [seed]
        labels[seed] = rid
        n = fn[seed].copy()
        d = -n @ cen[seed]
        q = deque([seed])
        S_a = area[seed]
        S_c = cen[seed] * area[seed]
        S_cc = np.outer(cen[seed], cen[seed]) * area[seed]
        next_refit = 8
        while q:
            f = q.popleft()
            for g in nbr[start[f]:start[f + 1]]:
                if labels[g] != -1:
                    continue
                if abs(fn[g] @ n) < cos_t:
                    continue
                if np.max(np.abs(tri[g] @ n + d)) > tol_mm:
                    continue
                labels[g] = rid
                members.append(g)
                q.append(g)
                S_a += area[g]
                S_c += cen[g] * area[g]
                S_cc += np.outer(cen[g], cen[g]) * area[g]
                if len(members) >= next_refit:
                    c = S_c / S_a
                    C = S_cc / S_a - np.outer(c, c)
                    w, U = np.linalg.eigh(C)
                    nn = U[:, 0]
                    if nn @ n < 0:
                        nn = -nn
                    n = nn
                    d = -n @ c
                    next_refit = int(len(members) * 1.5) + 1
        m = np.array(members)
        if area[m].sum() < min_area or len(m) < 6:
            labels[m] = -2  # too small: mark temporarily
            regions.append(None)
            continue
        pts = tri[m].reshape(-1, 3)
        n, d, rms = fit_plane(pts, np.repeat(area[m], 3))
        regions.append({"type": "plane", "normal": n, "d": d, "rms": rms, "area": float(area[m].sum()), "faces": m})
    labels[labels == -2] = -1
    _merge_coplanar(labels, regions, tri, area, tol_mm)
    # ---------------- cylinders on remaining smooth patches
    if cylinders:
        free = labels == -1
        sel = free[adj[:, 0]] & free[adj[:, 1]] & (dots > np.cos(np.radians(angle_deg * 1.5)))
        A = coo_matrix((np.ones(sel.sum()), (adj[sel, 0], adj[sel, 1])), shape=(nF, nF))
        ncomp, comp = connected_components(A, directed=False)
        comp[~free] = -1
        for c in np.unique(comp[comp >= 0]):
            m = np.where(comp == c)[0]
            if area[m].sum() < min_area or len(m) < 30:
                continue
            pts = tri[m].reshape(-1, 3)
            nr = np.repeat(fn[m], 3, axis=0)
            cyl = fit_cylinder(pts, nr)
            if cyl is None or cyl["radius"] > 2000 or cyl["radius"] < 2:
                continue
            if cyl["rms"] > tol_mm / 2:
                continue
            # angular coverage check (avoid fitting cylinders to near-planes)
            a = cyl["axis"]
            v = cen[m] - cyl["point"]
            v -= np.outer(v @ a, a)
            e1 = v[0] / np.linalg.norm(v[0])
            e2 = np.cross(a, e1)
            ang = np.degrees(np.arctan2(v @ e2, v @ e1))
            cover = np.ptp(np.unwrap(np.radians(np.sort(ang)))) * 180 / np.pi
            if cover < 45:
                continue
            rid = len(regions)
            labels[m] = rid
            t = (cen[m] - cyl["point"]) @ a
            regions.append({"type": "cylinder", "axis": a, "point": cyl["point"], "radius": cyl["radius"],
                            "rms": cyl["rms"], "area": float(area[m].sum()), "faces": m,
                            "length": float(np.ptp(t)), "coverage_deg": float(min(cover, 360))})
    return labels, regions


def _merge_coplanar(labels, regions, tri, area, tol_mm, ang_deg=2.0):
    """Merge plane regions that describe the same plane (over-segmentation by noise)."""
    ids = [i for i, r in enumerate(regions) if r is not None and r["type"] == "plane"]
    if len(ids) < 2:
        return
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cos_a = np.cos(np.radians(ang_deg))
    for a_i, i in enumerate(ids):
        ri = regions[i]
        for j in ids[a_i + 1:]:
            rj = regions[j]
            c = ri["normal"] @ rj["normal"]
            if abs(c) < cos_a:
                continue
            sg = 1.0 if c > 0 else -1.0
            if abs(ri["d"] - sg * rj["d"]) > tol_mm * 0.5:
                continue
            # check: points of j lie on plane i within tolerance
            pj = tri[rj["faces"]].reshape(-1, 3)
            if np.quantile(np.abs(pj @ ri["normal"] + ri["d"]), 0.95) > tol_mm:
                continue
            parent[find(j)] = find(i)
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    for root, mem in groups.items():
        if len(mem) == 1:
            continue
        faces = np.concatenate([regions[k]["faces"] for k in mem])
        pts = tri[faces].reshape(-1, 3)
        n, d, rms = fit_plane(pts, np.repeat(area[faces], 3))
        regions[root] = {"type": "plane", "normal": n, "d": d, "rms": rms, "area": float(area[faces].sum()), "faces": faces}
        labels[faces] = root
        for k in mem:
            if k != root:
                regions[k] = None


def snap(V, F, labels, regions, max_move, sharpen=True):
    """Project vertices onto detected surfaces; sharpen plane-plane edges. Returns new V and stats.

    Vectorised: the common case (vertex inside one region) is done per region with numpy; only
    vertices on region boundaries (a small fraction) are handled one by one.
    """
    from scipy.spatial import cKDTree
    V = V.copy()
    nV = len(V)
    planes = {i: r for i, r in enumerate(regions) if r is not None and r["type"] == "plane"}
    cyls = {i: r for i, r in enumerate(regions) if r is not None and r["type"] == "cylinder"}
    vf_v = F.ravel()
    vf_l = np.repeat(labels, 3)
    lmin = np.full(nV, np.iinfo(np.int64).max)
    lmax = np.full(nV, -2)
    np.minimum.at(lmin, vf_v, vf_l)
    np.maximum.at(lmax, vf_v, vf_l)
    used = lmax > -2
    moved = np.zeros(nV)
    single = used & (lmin == lmax) & (lmin >= 0)
    for rid in np.unique(lmin[single]):
        r = regions[rid]
        idx = np.where(single & (lmin == rid))[0]
        p = V[idx]
        if r["type"] == "plane":
            q = p - (p @ r["normal"] + r["d"])[:, None] * r["normal"]
        else:
            a = r["axis"]
            v = p - r["point"]
            ax = (v @ a)[:, None] * a
            rad = v - ax
            nr = np.linalg.norm(rad, axis=1, keepdims=True)
            q = r["point"] + ax + rad / np.maximum(nr, 1e-9) * r["radius"]
        mv = np.linalg.norm(q - p, axis=1)
        ok = mv <= max_move
        V[idx[ok]] = q[ok]
        moved[idx[ok]] = mv[ok]
    multi = np.where(used & (lmin != lmax))[0]
    n_line = n_corner = 0
    if len(multi):
        sel = np.isin(vf_v, multi)
        pairs = np.unique(np.c_[vf_v[sel], vf_l[sel]], axis=0)
        by_v = {}
        for v, l in pairs:
            if l >= 0:
                by_v.setdefault(int(v), []).append(int(l))
        for v, ls in by_v.items():
            p = V[v]
            pl = [l for l in ls if l in planes]
            cy = [l for l in ls if l in cyls]
            newp = None
            if len(pl) >= 2:
                newp = _project_planes(p, [planes[l] for l in pl])
                if len(pl) == 2:
                    n_line += 1
                else:
                    n_corner += 1
            elif len(pl) == 1 and not cy:
                r = planes[pl[0]]
                newp = p - (p @ r["normal"] + r["d"]) * r["normal"]
            elif len(cy) == 1 and not pl:
                newp = _project_cyl(p, cyls[cy[0]])
            if newp is not None and np.linalg.norm(newp - p) <= max_move:
                moved[v] = np.linalg.norm(newp - p)
                V[v] = newp
    n_sharp = 0
    if sharpen and planes:
        un = np.zeros(nV, bool)
        un[vf_v[vf_l < 0]] = True
        cand_v = np.where(un)[0]
        lab_f = np.where(np.isin(labels, list(planes)))[0]
        if len(cand_v) and len(lab_f):
            tri_c = V[F[lab_f]].mean(1)
            tree = cKDTree(tri_c)
            for v, nb in zip(cand_v, tree.query_ball_point(V[cand_v], max_move * 2.5)):
                if len(nb) < 2:
                    continue
                cl = np.unique(labels[lab_f[nb]])
                if len(cl) < 2:
                    continue
                p = V[v]
                close = [int(l) for l in cl if abs(p @ planes[l]["normal"] + planes[l]["d"]) <= max_move]
                if len(close) < 2:
                    continue
                newp = _project_planes(p, [planes[l] for l in close])
                if newp is not None and np.linalg.norm(newp - p) <= max_move:
                    moved[v] = np.linalg.norm(newp - p)
                    V[v] = newp
                    n_sharp += 1
    stats = {"snapped": int((moved > 0).sum()), "edge_vertices": n_line, "corner_vertices": n_corner,
             "sharpened": n_sharp, "mean_move_mm": float(moved[moved > 0].mean()) if (moved > 0).any() else 0.0}
    return V, stats


def _project_planes(p, planes):
    """Closest point to p on intersection of planes (2 -> line, 3+ -> point), only if well-conditioned."""
    N = np.array([r["normal"] for r in planes])
    D = np.array([r["d"] for r in planes])
    # drop near-parallel duplicates
    keep = [0]
    for i in range(1, len(N)):
        if all(abs(N[i] @ N[j]) < np.cos(np.radians(25)) for j in keep):
            keep.append(i)
    N = N[keep]
    D = D[keep]
    if len(N) < 2:
        r = planes[0]
        return p - (p @ r["normal"] + r["d"]) * r["normal"]
    if len(N) >= 3:
        N = N[:3]
        D = D[:3]
        if abs(np.linalg.det(N)) > 0.3:
            return np.linalg.solve(N, -D)
        N = N[:2]
        D = D[:2]
    # min |x-p|^2 s.t. N x + D = 0
    A = N @ N.T
    lam = np.linalg.solve(A, N @ p + D)
    return p - N.T @ lam


def _project_cyl(p, r):
    a = r["axis"]
    v = p - r["point"]
    ax = (v @ a) * a
    rad = v - ax
    nr = np.linalg.norm(rad)
    if nr < 1e-9:
        return None
    return r["point"] + ax + rad / nr * r["radius"]


def regularize_mesh(mesh, noise_mm, cfg_f):
    """Full feature pass on an Open3D mesh. Returns (mesh, report dict)."""
    import open3d as o3d
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles).copy()
    tol = float(cfg_f.get("plane_tol_mm") or 0) or max(3.0 * noise_mm, 0.3)
    labels, regions = detect_regions(V, F, tol, cfg_f.get("angle_deg", 12.0), cfg_f.get("min_region_area_mm2", 400.0),
                                     cfg_f.get("cylinders", True))
    nV, stats = snap(V, F, labels, regions, max_move=max(4 * tol, 1.5), sharpen=cfg_f.get("sharpen_edges", True))
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(nV), o3d.utility.Vector3iVector(F))
    m.merge_close_vertices(1e-3)
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    rep = []
    tot_area = 0.0
    for r in regions:
        if r is None:
            continue
        tot_area += r["area"]
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k != "faces"}
        rep.append(d)
    a_all = float(mesh.get_surface_area())
    n_pl = sum(1 for r in rep if r["type"] == "plane")
    n_cy = sum(1 for r in rep if r["type"] == "cylinder")
    log(f"  features: {n_pl} vlakken, {n_cy} cilinders ({tot_area / max(a_all, 1):.0%} van oppervlak), "
        f"tol {tol:.2f} mm, {stats['snapped']:,} vertices gesnapt, {stats['sharpened']:,} randen aangescherpt")
    return m, {"tolerance_mm": tol, "regions": rep, "coverage": tot_area / max(a_all, 1), **stats}
