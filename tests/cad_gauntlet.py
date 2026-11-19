"""Gauntlet for the CAD reconstruction (cadfit): synthetic machined parts with exact ground truth.

Each case builds a ground-truth solid with OpenCASCADE, simulates a photogrammetry scan of it
(surface samples, unseen underside, Gaussian noise, unsigned normals, ring of cameras), runs the
reconstruction and grades the resulting STEP solid against the truth.

    python tests/cad_gauntlet.py [noise_mm] [case ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.spatial import cKDTree  # noqa: E402

from foto2step import cadfit  # noqa: E402


# ---------------------------------------------------------------- ground-truth parts (mm, z up)

def _box(x0, y0, z0, x1, y1, z1):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x1, y1, z1)).Shape()


def _cyl(p, a, r, h):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p), gp_Dir(*a)), r, h).Shape()


def _cut(a, b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    return BRepAlgoAPI_Cut(a, b).Shape()


def _fuse(a, b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    return BRepAlgoAPI_Fuse(a, b).Shape()


def case_box():
    return _box(0, 0, 0, 100, 60, 80)


def case_tool_block():
    """Like the user's steel block: 104 x 80 x 90 with three round grooves along x on the front,
    a stepped back (T-profile), a vertical slot and a vertical hole."""
    s = _box(0, 0, 0, 104, 80, 90)
    for z in (20.0, 45.0, 70.0):  # round grooves r 7.8 along x, axis 3 mm outside the front face
        s = _cut(s, _cyl((-1, -3.0, z), (1, 0, 0), 7.8, 106))
    s = _cut(s, _box(-1, 62, -1, 30, 81, 91))   # step on the back left
    s = _cut(s, _box(74, 62, -1, 105, 81, 91))  # step on the back right -> T-profile
    s = _cut(s, _box(45, 20, 60, 59, 50, 91))   # pocket in the top
    s = _cut(s, _cyl((80, 40, 30), (0, 0, 1), 4.3, 61))  # hole from the top
    return s


def case_real_block():
    """Measured from the user's real scan (test_01): 105 x 63 x 90 block, two lugs 17 wide sticking
    out 14 mm at the back, and three deep round grooves along x on the front (r 13, 11 deep, 2 mm
    ridges between them)."""
    s = _box(0, 0, 0, 105, 63, 90)
    for x0 in (33.0, 82.0):
        s = _fuse(s, _box(x0, -14, 0, x0 + 17, 0.5, 90))
    for z in (15.0, 42.0, 70.0):
        s = _cut(s, _cyl((-1, 65.0, z), (1, 0, 0), 13.0, 107))
    return s


def case_cove_block():
    """The test_01 block as the photos show it: front (y = 63) with three wide concave coves (r 11.3),
    a lip above each and a 1.5 mm undercut slot under each lip (not visible from above), two lugs
    at the back and a 5 mm hole in the top. The whole front is one profile along x."""
    s = _box(0, 0, 0, 105, 63, 90)
    for x0 in (33.0, 82.0):
        s = _fuse(s, _box(x0, -14, 0, x0 + 17, 0.5, 90))
    for k in range(3):
        zk = 2.0 + 29.0 * k
        s = _cut(s, _cyl((-1, 64.5, zk + 15.0), (1, 0, 0), 11.3, 107))
        s = _cut(s, _box(-1, 57.0, zk + 26.0, 106, 64, zk + 27.5))
    s = _cut(s, _cyl((52, 45, 75), (0, 0, 1), 2.5, 16))
    return s


def case_ogee_block():
    """Block 100 x 60 x 50 with a freeform S-shaped (ogee) moulding along x on the top front edge:
    the profile is a smooth spline, neither a line nor one arc."""
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    ys = np.linspace(60.0, 40.0, 41)
    zs = 40.0 - 10.0 * np.cos(np.pi * (60.0 - ys) / 20.0)
    arr = TColgp_Array1OfPnt(1, len(ys))
    for i, (y, z) in enumerate(zip(ys, zs)):
        arr.SetValue(i + 1, gp_Pnt(0.0, float(y), float(z)))
    from OCP.GeomAbs import GeomAbs_C2
    spl = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-4).Curve()
    w = BRepBuilderAPI_MakeWire()
    pts = [(40, 50), (0, 50), (0, 0), (60, 0), (60, 30)]
    for (y0, z0), (y1, z1) in zip(pts[:-1], pts[1:]):
        w.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(0, y0, z0), gp_Pnt(0, y1, z1)).Edge())
    w.Add(BRepBuilderAPI_MakeEdge(spl).Edge())
    f = BRepBuilderAPI_MakeFace(w.Wire()).Face()
    return BRepPrimAPI_MakePrism(f, gp_Vec(100, 0, 0)).Shape()


def case_tslot_rail():
    """Aluminium 40x40 T-slot profile, 120 long, lying on the floor: slot opening 8 wide / 3 deep,
    cavity 16 wide / 4 deep behind it (an undercut), on the top and both sides; centre bore r 4.2."""
    s = _box(0, 0, 0, 120, 40, 40)
    for side in ("top", "y0", "y1"):
        if side == "top":
            op, cav = _box(-1, 16, 37, 121, 24, 41), _box(-1, 12, 33, 121, 28, 37)
        elif side == "y0":
            op, cav = _box(-1, -1, 16, 121, 3, 24), _box(-1, 3, 12, 121, 7, 28)
        else:
            op, cav = _box(-1, 37, 16, 121, 41, 24), _box(-1, 33, 12, 121, 37, 28)
        s = _cut(_cut(s, op), cav)
    s = _cut(s, _cyl((-1, 20, 20), (1, 0, 0), 4.2, 122))
    return s


def case_bracket():
    """L-bracket with a boss and a through hole: plane + cylinder interplay."""
    s = _fuse(_box(0, 0, 0, 120, 50, 12), _box(0, 0, 0, 14, 50, 70))
    s = _fuse(s, _cyl((70, 25, 12), (0, 0, 1), 12.0, 18))
    s = _cut(s, _cyl((70, 25, -1), (0, 0, 1), 6.0, 40))
    return s


def case_shallow():
    """Shallow features (review): 30x30x2 pad, 20x20x3 pocket, 3x3 notch along an edge."""
    s = _box(0, 0, 0, 100, 80, 20)
    s = _fuse(s, _box(10, 10, 20, 40, 40, 22))
    s = _cut(s, _box(60, 40, 17, 80, 60, 21))
    s = _cut(s, _box(-1, 77, 17, 101, 81, 21))
    return s


def case_zprofile():
    """2 mm sheet-metal Z-profile (review): flange on the floor, 50 mm web, top flange."""
    s = _box(0, 0, 0, 80, 60, 2)
    s = _fuse(s, _box(78, 0, 0, 80, 60, 50))
    s = _fuse(s, _box(78, 0, 48, 108, 60, 50))
    return s


def case_offframe():
    """Sloped face 30 deg, tilted hole, counterbore (review)."""
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    s = _box(0, 0, 0, 120, 80, 40)
    n = np.array([np.sin(np.radians(30)), 0, np.cos(np.radians(30))])
    f = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(90, 0, 40), gp_Dir(*n))).Face()
    hs = BRepPrimAPI_MakeHalfSpace(f, gp_Pnt(*(np.array([90, 40, 40]) + 50 * n))).Solid()
    s = _cut(s, hs)
    a = np.array([np.sin(np.radians(30)), 0, -np.cos(np.radians(30))])
    s = _cut(s, _cyl(tuple(np.array([30, 25, 41]) - 0 * a), tuple(a), 5.0, 60))
    s = _cut(s, _cyl((40, 55, 35), (0, 0, 1), 8.0, 6))
    s = _cut(s, _cyl((40, 55, -1), (0, 0, 1), 4.0, 42))
    return s


def _tilted(shape, deg=2.0):
    """Part standing tilted on the floor (e.g. on a bent marker sheet): rotate about x, lift to z>=0."""
    from OCP.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir, gp_Vec
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), np.radians(deg))
    s = BRepBuilderAPI_Transform(shape, t, True).Shape()
    V, _ = cadfit._mesh_of_shape(s, 0.05)
    t2 = gp_Trsf()
    t2.SetTranslation(gp_Vec(0, 0, -float(V[:, 2].min())))
    return BRepBuilderAPI_Transform(s, t2, True).Shape()


def case_tilted_tool_block():
    return _tilted(case_tool_block(), 2.0)


CASES = {"box": case_box, "tool_block": case_tool_block, "real_block": case_real_block,
         "cove_block": case_cove_block, "ogee_block": case_ogee_block, "tslot_rail": case_tslot_rail, "bracket": case_bracket,
         "shallow": case_shallow, "zprofile": case_zprofile, "offframe": case_offframe,
         "tilted_tool_block": case_tilted_tool_block, "ghost_box": case_box, "ghost_tool_block": case_tool_block}


# ---------------------------------------------------------------- scan simulation

def simulate_scan(shape, noise, density=1.6, seed=0, ghost=False):
    """Area-uniform samples on all faces except the underside (z = 0), Gaussian noise along the
    surface normal plus a little tangential jitter; also returns camera centres on two rings."""
    rng = np.random.default_rng(seed)
    V, T = cadfit._mesh_of_shape(shape, defl=0.05)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2 * area[:, None], 1e-12)
    up = ~((np.abs(n[:, 2] + 1) < 1e-3) & (np.maximum(np.maximum(A[:, 2], B[:, 2]), C[:, 2]) < 1e-6))
    area = area * up
    k = rng.poisson(area * density)
    idx = np.repeat(np.arange(len(T)), k)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    s1 = np.sqrt(r1)
    P = (1 - s1)[:, None] * A[idx] + (s1 * (1 - r2))[:, None] * B[idx] + (s1 * r2)[:, None] * C[idx]
    N = n[idx]
    P0 = P
    # white noise + correlated (wavy, 15 mm wavelength) noise along the normal + 1% outliers
    k3 = rng.normal(size=(6, 3)) * (2 * np.pi / 15.0)
    ph = rng.uniform(0, 2 * np.pi, 6)
    wav = (np.sin(P0 @ k3.T + ph).sum(1) / np.sqrt(3.0)) * (0.3 * noise)
    P = P + N * (rng.normal(0, noise, (len(P), 1)) + wav[:, None]) + rng.normal(0, 0.3 * noise, P.shape)
    ctr = 0.5 * (V.min(0) + V.max(0))
    cams = []
    for h, R, m in ((60.0, 400.0, 24), (250.0, 350.0, 16), (500.0, 150.0, 8)):
        for a in np.linspace(0, 2 * np.pi, m, endpoint=False):
            cams.append([ctr[0] + R * np.cos(a), ctr[1] + R * np.sin(a), h])
    cams = np.array(cams)
    # photogrammetry only reconstructs what at least 2 cameras see (not the inside of deep holes)
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(T.astype(np.uint32)))
    O = (P0 + N * 0.05)  # visibility of the true surface point (noise must not bias what is kept)
    seen = np.zeros(len(P), int)
    for c in cams:
        D = c - O
        L = np.linalg.norm(D, axis=1)
        front = np.einsum("nd,nd->n", D, N) > 0.17 * L  # not more grazing than ~80 deg
        rays = np.c_[O, D / L[:, None]].astype(np.float32)
        hit = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        seen += front & (hit >= L - 1e-3)
    keep = seen >= 2
    P, N = P[keep], N[keep]
    if ghost:  # reflections on shiny tops: a second layer 2.5 mm off with 40% of the density
        top = np.where((N[:, 2] > 0.9) & (rng.random(len(P)) < 0.4))[0]
        P = np.vstack([P, P[top] + 2.5 * N[top]])
        N = np.vstack([N, N[top]])
    n_out = int(0.01 * len(P))
    out_i = rng.choice(len(P), n_out, replace=False)
    P[out_i] += rng.normal(0, 5 * noise, (n_out, 3))
    simulate_scan.visible_truth = (P0[keep], N)
    return P, N, cams


# ---------------------------------------------------------------- grading

def _seen(S, V, T, cams, Nrm=None):
    """Surface the scan can contain: >= 2 cameras with a free line of sight, and (as in the scan
    simulation) not more grazing than ~80 deg. A face seen only edge-on (the inside of a 1.5 mm slot,
    a long bore from its ends) gives no points and cannot be judged."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(T.astype(np.uint32)))
    cnt = np.zeros(len(S), int)
    for c in cams:
        D = c - S
        L = np.linalg.norm(D, axis=1)
        O = S + 0.05 * D / L[:, None]
        rays = np.c_[O, D / L[:, None]].astype(np.float32)
        hit = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        ok = hit >= L - 0.1
        if Nrm is not None:
            ok &= np.abs(np.einsum("nd,nd->n", D, Nrm)) > 0.17 * L
        cnt += ok
    return cnt >= 2


def grade(result, truth, scan_P, cams):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    rt = cadfit.shape_report(truth)
    rr = cadfit.shape_report(result)
    out = {"valid": rr["valid"], "solid": rr["solid"], "faces": rr["faces"], "faces_truth": rt["faces"],
           "vol_err_pct": 100 * (rr["volume_mm3"] - rt["volume_mm3"]) / rt["volume_mm3"]}
    try:
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(BRepAlgoAPI_Common(result, truth).Shape(), g)
        inter = g.Mass()
        out["iou"] = inter / (rr["volume_mm3"] + rt["volume_mm3"] - inter)
    except Exception:
        out["iou"] = float("nan")
    # surface distances. Truth samples are split into 'seen' (>= 2 cameras) and 'unseen'; only seen
    # surface can be judged. Max-type metrics (99.9%) so a missing feature cannot hide in the 5% tail.
    Vt, Tt = cadfit._mesh_of_shape(truth, defl=0.05)
    Vr, Tr = cadfit._mesh_of_shape(result, defl=0.05)
    St = _area_samples(Vt, Tt, 0.5)
    Nt = _area_samples.normals
    keep_t = St[:, 2] > 0.5
    St, Nt = St[keep_t], Nt[keep_t]
    Sr = _area_samples(Vr, Tr, 0.5)
    Sr = Sr[Sr[:, 2] > 0.5]
    seen = _seen(St, Vt, Tt, cams, Nt)
    d_tr = _dist_to_mesh(St, Vr, Tr)
    d_rt = _dist_to_mesh(Sr, Vt, Tt)
    # result samples whose closest truth point is unseen are not judged either
    _, jt = cKDTree(St).query(Sr)
    r_judged = seen[jt]
    out["truth_to_result_p50"] = float(np.median(d_tr[seen]))
    out["truth_to_result_p999"] = float(np.percentile(d_tr[seen], 99.9))
    out["result_to_truth_p999"] = float(np.percentile(d_rt[r_judged], 99.9)) if r_judged.any() else 0.0
    out["unseen_frac"] = float(1 - seen.mean())
    out["unseen_max_dev"] = float(d_tr[~seen].max()) if (~seen).any() else 0.0
    out["pass"] = bool(out["valid"] and out["solid"] and out["truth_to_result_p999"] < 1.0
                       and out["result_to_truth_p999"] < 1.0)
    return out


def _dist_to_mesh(Q, V, T):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(T.astype(np.uint32)))
    return sc.compute_distance(o3d.core.Tensor(Q.astype(np.float32))).numpy().astype(float)


def _area_samples(V, T, step):
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    rng = np.random.default_rng(1)
    k = np.maximum(1, np.round(area / step ** 2).astype(int))
    idx = np.repeat(np.arange(len(T)), k)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    s1 = np.sqrt(r1)
    S = (1 - s1)[:, None] * A[idx] + (s1 * (1 - r2))[:, None] * B[idx] + (s1 * r2)[:, None] * C[idx]
    nt = np.cross(B - A, C - A)
    _area_samples.normals = (nt / np.maximum(np.linalg.norm(nt, axis=1, keepdims=True), 1e-12))[idx]
    return S


def read_back(path):
    """The STEP read in like a CAD program does it (standard healing when the checker objects)."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeFix import ShapeFix_Shape
    r = STEPControl_Reader()
    r.ReadFile(str(path))
    r.TransferRoots()
    rs = r.OneShape()
    if not BRepCheck_Analyzer(rs).IsValid():
        fx = ShapeFix_Shape(rs)
        fx.SetPrecision(1e-4)
        fx.SetMaxTolerance(1e-2)
        fx.Perform()
        rs = fx.Shape()
    return rs


def run(noise=0.5, names=None, out_dir=Path("cad_gauntlet_out"), seeds=(0,)):
    out_dir.mkdir(exist_ok=True)
    results = {}
    for name in names or list(CASES):
      for seed in seeds:
        t0 = time.time()
        try:
            truth = CASES[name]()
            P, N, cams = simulate_scan(truth, noise, seed=seed, ghost=name.startswith("ghost"))
            # the pipeline's noise estimate is low; give 60% of the truth like it would
            shape, info = cadfit.reconstruct(P, cams, noise_mm=0.6 * noise, floor_z=0.0, normals=None)
            path = out_dir / f"{name}_s{seed}.step"
            ok, how, _ = cadfit.export_solid(path, name, shape, (0.6, 0.6, 0.7),
                                             ref_volume=info.get("cells_volume"))
            # grade what the user gets: the STEP as a CAD program reads it in
            g = grade(read_back(path) if ok else shape, truth, P, cams)
            g.update({k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))})
            g["export"] = how
            g["pass"] = bool(g["pass"] and ok)
        except Exception as e:  # a crash is a failed case, not a crashed gauntlet
            g = {"pass": False, "error": repr(e)}
        g["seconds"] = round(time.time() - t0, 1)
        results[f"{name}_s{seed}"] = g
        print(f"{'PASS' if g['pass'] else 'FAIL'} {name} seed {seed}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                                           for k, v in g.items()}), flush=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    noise = float(args[0]) if args and args[0].replace(".", "").isdigit() else 0.5
    names = [a for a in args if a in CASES] or None
    seeds = tuple(int(a[5:]) for a in args if a.startswith("seed=")) or (0,)
    run(noise, names, seeds=seeds)
