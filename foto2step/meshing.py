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
    pc, _ = pc.remove_radius_outlier(nb_points=8, radius=voxel * 4)
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


def poisson_mesh(pc, voxel, depth=0, trim_dist=None, density_q=0.02):
    pts = np.asarray(pc.points)
    ext = float(np.max(pts.max(0) - pts.min(0)))
    if not depth:
        depth = int(np.clip(np.ceil(np.log2(ext / voxel * 1.1)), 6, 12))
    log(f"  Poisson-meshing depth {depth} ({len(pts):,} punten, {ext:.0f} mm)")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth, scale=1.1, linear_fit=True)
    dens = np.asarray(dens)
    v = np.asarray(mesh.vertices)
    tree = cKDTree(pts)
    d, _ = tree.query(v)
    td = trim_dist if trim_dist else max(3.0 * voxel, 2.0)
    bad = (d > td) | (dens < np.quantile(dens, density_q))
    mesh.remove_vertices_by_mask(bad)
    mesh = clean_mesh(mesh)
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
    """Quadric decimation, halving until the max deviation (sampled) exceeds max_err_mm."""
    if ref_pts is None:
        ref = mesh.sample_points_uniformly(min(200000, max(20000, len(mesh.triangles) * 2)))
        ref_pts = np.asarray(ref.points)
    best = mesh
    n = len(mesh.triangles)
    target = n
    while target > 2000:
        target = int(target * 0.6)
        cand = mesh.simplify_quadric_decimation(target)
        err = _max_dev(cand, ref_pts)
        if err > max_err_mm and len(best.triangles) <= max_tris:
            break
        best = cand
    if len(best.triangles) > max_tris:
        best = mesh.simplify_quadric_decimation(max_tris)
        log(f"  WAARSCHUWING: driehoekslimiet {max_tris} bereikt, afwijking {_max_dev(best, ref_pts):.3f} mm")
    best = clean_mesh(best, min_component_frac=0, min_component_tris=0)
    log(f"  decimatie: {n:,} -> {len(best.triangles):,} driehoeken (max afwijking {_max_dev(best, ref_pts):.3f} mm)")
    return best


def _max_dev(mesh, ref_pts):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d = sc.compute_distance(o3d.core.Tensor(ref_pts.astype(np.float32))).numpy()
    return float(np.quantile(d, 0.999))
