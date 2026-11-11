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
        if not len(dv):
            return e1
        # robust: 99.9% of vertices within tolerance; a single exploded vertex (QEM failure) is caught by the cap
        if dv.max() > 20 * max_err_mm:
            return float(dv.max())
        return max(e1, float(np.quantile(dv, 0.999)))

    n = len(mesh.triangles)
    # 1) lossless pass: removes vertices that carry no shape (inside snapped planes) without measurable error
    try:
        import pyfqmr
        s_ = pyfqmr.Simplify()
        s_.setMesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        s_.simplify_mesh_lossless(verbose=False, epsilon=min(1e-3, (0.1 * max_err_mm) ** 2), preserve_border=True)
        v_, f_, _ = s_.getMesh()
        cand = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v_), o3d.utility.Vector3iVector(f_.astype(np.int32)))
        if len(f_) and err_of(cand) <= max_err_mm:
            log(f"  verliesvrije vereenvoudiging: {n:,} -> {len(f_):,} driehoeken")
            mesh = cand
    except Exception as e:
        log(f"  (verliesvrije vereenvoudiging niet beschikbaar: {e})")
    best = mesh
    best_err = 0.0
    target = len(mesh.triangles)
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
        try:  # second, independent decimator (more robust on exactly planar, snapped regions)
            import fast_simplification
            v2, f2 = fast_simplification.simplify(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                                  1.0 - max_tris / max(len(mesh.triangles), 1))
            cand2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v2), o3d.utility.Vector3iVector(f2.astype(np.int32)))
            e2 = err_of(cand2)
            if e2 < e:
                cand, e = cand2, e2
        except Exception:
            pass
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
    """Boundary loops (vertex index lists, oriented along the triangle winding) of a triangle mesh.

    Walks the boundary as an undirected graph (robust to pinch vertices and to small orientation
    defects from decimation), splits off every closed cycle as soon as a vertex repeats, then orients
    each loop by majority vote of its directed boundary edges."""
    F = np.asarray(F)
    if len(F) == 0:
        return []
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    s = np.sort(e, axis=1)
    _, inv, cnt = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    b = e[cnt[inv.ravel()] == 1]
    directed = set(map(tuple, b.tolist()))
    adj = {}
    for a_, c_ in b:
        a_, c_ = int(a_), int(c_)
        adj.setdefault(a_, []).append(c_)
        adj.setdefault(c_, []).append(a_)
    used = set()

    def take(u):
        for w in adj.get(u, []):
            k = (u, w) if u < w else (w, u)
            if k not in used:
                used.add(k)
                return w
        return None

    loops = []
    for start in list(adj):
        while True:
            nx = take(start)
            if nx is None:
                break
            path = [start]
            pos = {start: 0}
            cur = nx
            while True:
                if cur in pos:
                    k = pos[cur]
                    cyc = path[k:]
                    if len(cyc) >= 3:
                        loops.append(cyc)
                    for v in path[k + 1:]:
                        pos.pop(v, None)
                    path = path[:k + 1]
                    if cur == start and len(path) == 1:
                        break
                else:
                    pos[cur] = len(path)
                    path.append(cur)
                n2 = take(cur)
                if n2 is None:
                    break  # dangling chain (should not happen on a clean mesh)
                cur = n2
    out = []
    for lp in loops:
        fwd = sum((lp[i], lp[(i + 1) % len(lp)]) in directed for i in range(len(lp)))
        out.append(lp if fwd * 2 >= len(lp) else lp[::-1])
    return out


def _zipper(V, lp, forbidden=frozenset()):
    """Greedy ear clipping by shortest diagonal; never creates an edge that already exists in the mesh."""
    lp = list(lp)
    if len(lp) < 3:
        return None
    out = []
    while len(lp) > 3:
        a = np.array(lp)
        prv = np.roll(a, 1)
        nxt = np.roll(a, -1)
        d = np.linalg.norm(V[prv] - V[nxt], axis=1)
        for i in np.argsort(d):
            e = (min(prv[i], nxt[i]), max(prv[i], nxt[i]))
            if e not in forbidden:
                break
        else:
            return None
        out.append((nxt[i], a[i], prv[i]))
        forbidden = forbidden | {e}
        lp.pop(int(i))
    e = (min(lp[0], lp[2]), max(lp[0], lp[2]))
    out.append((lp[2], lp[1], lp[0]))
    return np.array(out, np.int64)


def _existing_edges_among(F, verts):
    """Set of existing mesh edges whose both endpoints are in `verts`."""
    vs = np.zeros(F.max() + 1, bool)
    vs[np.asarray(list(verts))] = True
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = e[vs[e[:, 0]] & vs[e[:, 1]]]
    e.sort(1)
    return frozenset(map(tuple, e.tolist()))


def fill_holes(mesh, max_radius=3000.0, planar_tol=None, zipper_max_mm=60.0):
    """Close boundary loops: planar loops with constrained ear-clipping, small curved loops with a fan.

    Returns (mesh, n_filled, n_left).
    """
    import mapbox_earcut as earcut
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles).copy()
    loops = boundary_loops(F)
    bverts = set(v for lp in loops for v in lp)
    forbidden = _existing_edges_among(F, bverts) if bverts else frozenset()
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
                lpa = np.array(lp)
                T = lpa[tri]
                ne = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
                # boundary edges of the loop itself are allowed; any other existing edge is not
                loop_e = set(tuple(sorted((lp[i], lp[(i + 1) % len(lp)]))) for i in range(len(lp)))
                simple = not any((tuple(x) in forbidden) and (tuple(x) not in loop_e) for x in ne.tolist())
            if simple:
                # orientation: compare with loop direction
                nn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]).sum(0)
                # loop direction normal (right-hand rule along boundary)
                ln = np.cross(P - c, np.roll(P, -1, 0) - c).sum(0)
                if nn @ ln > 0:
                    T = T[:, [0, 2, 1]]
                newF.append(T)
                filled += 1
                continue
        # general fallback: greedy "zipper" ear clipping (shortest diagonal first). Works for thin slits
        # and wiggly or non-planar loops; such caps only close surface that was never observed.
        # zipper only for small or nearly planar openings; large irregular openings are left open rather
        # than closed with an inaccurate "tent" (the STEP then becomes a shell)
        if not (rad <= zipper_max_mm or rms <= max(tol, 0.05 * rad)):
            left += 1
            continue
        loop_e = frozenset(tuple(sorted((lp[i], lp[(i + 1) % len(lp)]))) for i in range(len(lp)))
        T = _zipper(V, lp, forbidden - loop_e)
        if T is not None:
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


def split_nonmanifold_vertices(mesh, nm=None):
    """Give every edge-connected fan of triangles around a pinch vertex its own copy of that vertex."""
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles).copy()
    if nm is None:
        nm = np.asarray(mesh.get_non_manifold_vertices())
    newV = []
    for v in nm:
        tris = np.where((F == v).any(1))[0]
        if len(tris) < 2:
            continue
        # union-find over triangles sharing an edge that contains v
        parent = {t: t for t in tris}

        def find(t):
            while parent[t] != t:
                parent[t] = parent[parent[t]]
                t = parent[t]
            return t
        other = {t: set(F[t][F[t] != v]) for t in tris}
        tl = list(tris)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                if other[tl[i]] & other[tl[j]]:
                    parent[find(tl[i])] = find(tl[j])
        fans = {}
        for t in tris:
            fans.setdefault(find(t), []).append(t)
        for k, ts in enumerate(fans.values()):
            if k == 0:
                continue
            ni = len(V) + len(newV)
            newV.append(V[v])
            for t in ts:
                F[t][F[t] == v] = ni
    if newV:
        V = np.vstack([V, np.array(newV)])
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    return mesh


def make_manifold(mesh):
    """Remove non-manifold edges/vertices and slivers so the B-rep is valid (edge shared by exactly <=2 faces)."""
    for _ in range(3):
        n0 = len(mesh.triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        nm = np.asarray(mesh.get_non_manifold_vertices())
        if len(nm):
            mesh = split_nonmanifold_vertices(mesh, nm)
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) == n0:
            break
    mesh.compute_vertex_normals()
    return mesh


def _fix_t_junctions(mesh):
    """Zero-area 3-vertex holes (vertex m lying on edge a-b): split the triangle across a-b at m."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles).copy()
    changed = False
    for lp in boundary_loops(F):
        if len(lp) != 3:
            continue
        P = V[lp]
        area = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
        L = max(np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[2] - P[1]), np.linalg.norm(P[0] - P[2]))
        if area > 1e-3 * L * L:
            continue
        # the middle vertex m is the one opposite the longest side
        d = [np.linalg.norm(P[(i + 1) % 3] - P[(i + 2) % 3]) for i in range(3)]
        i = int(np.argmax(d))
        m, a, b = lp[i], lp[(i + 1) % 3], lp[(i + 2) % 3]
        hit = np.where(((F == a).any(1)) & ((F == b).any(1)))[0]
        if len(hit) != 1:
            continue
        t = F[hit[0]].copy()
        k = [j for j in range(3) if t[j] not in (a, b)][0]
        x = t[k]
        # keep winding: rotate so t = (x, p, q)
        t = np.roll(t, -k)
        p_, q_ = t[1], t[2]
        F[hit[0]] = (x, p_, m)
        F = np.vstack([F, [x, m, q_]])
        changed = True
    if changed:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    return mesh


def close_mesh(mesh, max_radius=3000.0, rounds=6, force=False):
    """Alternate hole filling and manifold repair until the mesh is closed (or no further progress)."""
    total = 0

    def edges_closed(m):
        F = np.asarray(m.triangles)
        if len(F) == 0:
            return False
        e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _, c = np.unique(e, axis=0, return_counts=True)
        return bool(np.all(c == 2))

    for _ in range(rounds):
        mesh = make_manifold(mesh)
        if not boundary_loops(np.asarray(mesh.triangles)):
            break
        mesh = _fix_t_junctions(mesh)
        mesh, n, left = fill_holes(mesh, max_radius, zipper_max_mm=(1e9 if force else 60.0))
        total += n
        if edges_closed(mesh) or n == 0:
            break
    mesh.compute_vertex_normals()
    return mesh, total, len(boundary_loops(np.asarray(mesh.triangles)))
