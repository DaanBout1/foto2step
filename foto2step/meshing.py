"""Point cloud cleanup and watertight-where-observed surface meshing."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .log import log


def crop_cloud(pc: o3d.geometry.PointCloud, markers_world: dict, cfg_mesh: dict, extra_box=None):
    pts = np.asarray(pc.points)
    keep = np.ones(len(pts), bool)
    if markers_world:
        allc = np.array([c for m in markers_world.values() for c in m["corners"]])
        lo = allc.min(0) - cfg_mesh["crop_margin_mm"]
        hi = allc.max(0) + cfg_mesh["crop_margin_mm"]
        lo[2] = -50.0
        hi[2] = max(hi[2], 0) + 4000.0  # machine can be taller than markers
        keep &= np.all((pts >= lo) & (pts <= hi), axis=1)
    if extra_box is not None:
        keep &= np.all((pts >= extra_box[0]) & (pts <= extra_box[1]), axis=1)
    if cfg_mesh.get("remove_floor", True):
        keep &= pts[:, 2] > float(cfg_mesh.get("floor_clearance_mm", 3.0))
    return pc.select_by_index(np.where(keep)[0])


def clean_cloud(pc, voxel):
    n0 = len(pc.points)
    pc = pc.voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.5)
    pc, _ = pc.remove_radius_outlier(nb_points=6, radius=voxel * 10)
    log(f"  puntenwolk opgeschoond: {n0:,} -> {len(pc.points):,}")
    return pc


def noise_estimate(pc, k=16, sample=20000):
    """Robust local surface noise (mm): RMS distance to local plane over k-NN, median over sample."""
    pts = np.asarray(pc.points)
    if len(pts) < 100:
        return 0.5
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), min(sample, len(pts)), replace=False)
    tree = cKDTree(pts)
    _, nn = tree.query(pts[idx], k=k)
    Q = pts[nn] - pts[nn].mean(1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", Q, Q) / k)
    return float(np.median(np.sqrt(np.maximum(ev[:, 0], 0))))


def poisson_mesh(pc, voxel, depth=0, trim_dist=None, density_q=0.0, max_depth=11):
    """Screened Poisson. depth 0 = automatic from voxel size, capped at max_depth (memory: depth 11 ~4 GB,
    depth 12 ~12-16 GB)."""
    pts = np.asarray(pc.points)
    ext = float(np.max(pts.max(0) - pts.min(0)))
    if not depth:
        depth = int(np.clip(np.ceil(np.log2(ext / voxel * 1.1)), 6, max_depth))
    log(f"  Poisson-meshing depth {depth} ({len(pts):,} punten, {ext:.0f} mm)")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, scale=1.1, linear_fit=True)
    dens = np.asarray(dens)
    v = np.asarray(mesh.vertices)
    tree = cKDTree(pts)
    d, _ = tree.query(v)
    td = trim_dist if trim_dist else max(3.0 * voxel, 2.0)
    F = np.asarray(mesh.triangles)
    fd = d[F].max(1)
    badf = fd > td
    # keep small unobserved gaps (Poisson interpolates them well); remove large/far extrapolations
    keep_gap_area = (15.0 * voxel) ** 2
    if badf.any():
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from .features import face_adjacency
        adj = face_adjacency(F)
        sel = badf[adj[:, 0]] & badf[adj[:, 1]]
        A = coo_matrix((np.ones(sel.sum()), (adj[sel, 0], adj[sel, 1])), shape=(len(F), len(F)))
        _, comp = connected_components(A, directed=False)
        tri = v[F]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        cb = comp[badf]
        comp_area = np.bincount(cb, weights=area[badf], minlength=comp.max() + 1)
        comp_maxd = np.zeros(comp.max() + 1)
        np.maximum.at(comp_maxd, cb, fd[badf])
        remove_comp = (comp_area > keep_gap_area) | (comp_maxd > 4 * td)
        remove = badf & remove_comp[comp]
        if density_q > 0:
            dens_bad = dens < np.quantile(dens, density_q)
            remove |= dens_bad[F].any(1) & badf
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
    mesh = clean_mesh(mesh)
    return mesh


def cap_triangles(mesh, n_max):
    """Light QEM decimation to keep memory bounded (Poisson output is heavily oversampled)."""
    if n_max and len(mesh.triangles) > n_max:
        n0 = len(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(int(n_max))
        mesh = clean_mesh(mesh, min_component_frac=0, min_component_tris=0)
        log(f"  werk-mesh: {n0:,} -> {len(mesh.triangles):,} driehoeken")
    return mesh


def clean_mesh(mesh, min_component_frac=0.002, min_component_tris=50):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles) == 0:
        return mesh
    cl, n_tri, _ = mesh.cluster_connected_triangles()
    cl = np.asarray(cl)
    n_tri = np.asarray(n_tri)
    thr = max(min_component_tris, min_component_frac * len(mesh.triangles))
    mesh.remove_triangles_by_mask(n_tri[cl] < thr)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_arrays(mesh):
    return np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()


def arrays_to_mesh(V, F):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, float)),
                                  o3d.utility.Vector3iVector(np.asarray(F, np.int32)))
    m.compute_vertex_normals()
    return m


def decimate_bounded(mesh, max_err_mm=0.15, max_tris=400000, ref_pts=None):
    """Quadric decimation: reduce as far as possible while the deviation stays <= max_err_mm.

    Deviation = max(99.9% of original-surface samples -> new mesh, every new vertex -> original mesh),
    so both lost detail and misplaced vertices are caught.
    """
    if ref_pts is None:
        ref = mesh.sample_points_uniformly(min(300000, max(20000, len(mesh.triangles) * 2)))
        ref_pts = np.asarray(ref.points)
    orig = o3d.t.geometry.RaycastingScene()
    orig.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def err_of(cand):
        e1 = _max_dev(cand, ref_pts)
        dv = orig.compute_distance(o3d.core.Tensor(np.asarray(cand.vertices, np.float32))).numpy()
        return max(e1, float(dv.max()) if len(dv) else 0.0)

    n = len(mesh.triangles)
    best = mesh
    best_err = 0.0
    target = n
    while target > 1000:
        target = int(target * 0.5)
        cand = mesh.simplify_quadric_decimation(target)
        err = err_of(cand)
        if err > max_err_mm:
            break
        best, best_err = cand, err
    if len(best.triangles) > max_tris:
        cand = mesh.simplify_quadric_decimation(max_tris)
        e = err_of(cand)
        if e < 20 * max_err_mm:
            best, best_err = cand, e
            log(f"  WAARSCHUWING: driehoekslimiet {max_tris:,} bereikt, afwijking {best_err:.3f} mm")
        else:
            log(f"  WAARSCHUWING: decimatie onbetrouwbaar ({e:.1f} mm), mesh blijft {len(best.triangles):,} driehoeken")
    best = clean_mesh(best, min_component_frac=0, min_component_tris=0)
    log(f"  decimatie: {n:,} -> {len(best.triangles):,} driehoeken (afwijking max {best_err:.3f} mm)")
    return best


def _max_dev(mesh, ref_pts):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d = sc.compute_distance(o3d.core.Tensor(ref_pts.astype(np.float32))).numpy()
    return float(np.quantile(d, 0.999))


def boundary_loops(F):
    """Boundary loops (vertex index lists, following triangle winding) of a triangle mesh."""
    F = np.asarray(F)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    s = np.sort(e, axis=1)
    _, inv, cnt = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    b = e[cnt[inv.ravel()] == 1]
    nxt = {}
    for a, c in b:
        nxt.setdefault(int(a), []).append(int(c))
    used = set()
    loops = []
    for a0, c0 in b:
        a0, c0 = int(a0), int(c0)
        if (a0, c0) in used:
            continue
        loop = [a0]
        a, c = a0, c0
        ok = True
        for _ in range(len(b) + 2):
            used.add((a, c))
            if c == a0:
                break
            loop.append(c)
            cands = [x for x in nxt.get(c, []) if (c, x) not in used]
            if not cands:
                ok = False
                break
            a, c = c, cands[0]
        if ok and len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_holes(mesh, max_radius=3000.0, planar_tol=None):
    """Close boundary loops: planar loops with constrained ear-clipping, small curved loops with a fan.

    Returns (mesh, n_filled, n_left).
    """
    import mapbox_earcut as earcut
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles).copy()
    loops = boundary_loops(F)
    newF = []
    newV = []
    nV = len(V)
    filled = left = 0
    for lp in loops:
        P = V[lp]
        c = P.mean(0)
        rad = float(np.max(np.linalg.norm(P - c, axis=1)))
        if rad > max_radius:
            left += 1
            continue
        X = P - c
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        n = Vt[2]
        rms = float(np.sqrt(np.mean((X @ n) ** 2)))
        tol = planar_tol if planar_tol else max(0.02 * rad, 0.5)
        # boundary follows triangle winding: the cap must have the opposite orientation
        # (non-planar loops, e.g. unobservable contact areas between parts, are still capped when their
        # projection is a simple polygon: checked by area consistency of the triangulation)
        if rms <= max(tol, 0.15 * rad):
            e1, e2 = Vt[0], Vt[1]
            uv = np.c_[X @ e1, X @ e2].astype(np.float64)
            try:
                tri = earcut.triangulate_float64(uv, np.array([len(uv)], np.uint32)).reshape(-1, 3)
            except Exception:
                tri = np.zeros((0, 3), int)
            simple = False
            if len(tri) == len(lp) - 2:
                e1 = uv[tri[:, 1]] - uv[tri[:, 0]]
                e2 = uv[tri[:, 2]] - uv[tri[:, 0]]
                ta = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
                pa = 0.5 * abs(np.sum(uv[:, 0] * np.roll(uv[:, 1], -1) - np.roll(uv[:, 0], -1) * uv[:, 1]))
                simple = pa > 0 and abs(ta - pa) <= 0.01 * pa
            if simple:
                # orientation: compare with loop direction
                lpa = np.array(lp)
                T = lpa[tri]
                nn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]).sum(0)
                # loop direction normal (right-hand rule along boundary)
                ln = np.cross(P - c, np.roll(P, -1, 0) - c).sum(0)
                if nn @ ln > 0:
                    T = T[:, [0, 2, 1]]
                newF.append(T)
                filled += 1
                continue
        if rad <= max(10.0, 0.05 * max_radius):
            ci = nV + len(newV)
            newV.append(c)
            T = np.c_[np.roll(lp, -1), lp, np.full(len(lp), ci)]
            newF.append(T)
            filled += 1
        else:
            left += 1
    if newF:
        V2 = np.vstack([V] + ([np.array(newV)] if newV else []))
        F2 = np.vstack([F] + newF)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V2), o3d.utility.Vector3iVector(F2.astype(np.int32)))
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
    return mesh, filled, left


def make_manifold(mesh):
    """Remove non-manifold edges/vertices and slivers so the B-rep is valid (edge shared by exactly <=2 faces)."""
    for _ in range(3):
        n0 = len(mesh.triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        nm = np.asarray(mesh.get_non_manifold_vertices())
        if len(nm):
            F = np.asarray(mesh.triangles)
            bad = np.isin(F, nm).any(1)
            mesh.remove_triangles_by_mask(bad)
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) == n0:
            break
    mesh.compute_vertex_normals()
    return mesh
