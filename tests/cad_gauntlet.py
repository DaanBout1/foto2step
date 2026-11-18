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


def case_bracket():
    """L-bracket with a boss and a through hole: plane + cylinder interplay."""
    s = _fuse(_box(0, 0, 0, 120, 50, 12), _box(0, 0, 0, 14, 50, 70))
    s = _fuse(s, _cyl((70, 25, 12), (0, 0, 1), 12.0, 18))
    s = _cut(s, _cyl((70, 25, -1), (0, 0, 1), 6.0, 40))
    return s


CASES = {"box": case_box, "tool_block": case_tool_block, "bracket": case_bracket}


# ---------------------------------------------------------------- scan simulation

def simulate_scan(shape, noise, density=1.6, seed=0):
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
    P = P + N * rng.normal(0, noise, (len(P), 1)) + rng.normal(0, 0.3 * noise, P.shape)
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
    return P[keep], N[keep], cams


# ---------------------------------------------------------------- grading

def grade(result, truth, scan_P):
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
    # surface distances truth <-> result (excluding the unseen floor of the truth)
    Vt, Tt = cadfit._mesh_of_shape(truth, defl=0.05)
    Vr, Tr = cadfit._mesh_of_shape(result, defl=0.05)
    St, Sr = _area_samples(Vt, Tt, 0.5), _area_samples(Vr, Tr, 0.5)
    St = St[St[:, 2] > 0.5]
    Sr = Sr[Sr[:, 2] > 0.5]
    d_tr = _dist_to_mesh(St, Vr, Tr)   # exact point-to-surface distances
    d_rt = _dist_to_mesh(Sr, Vt, Tt)
    out["truth_to_result_p50"] = float(np.median(d_tr))
    out["truth_to_result_p95"] = float(np.percentile(d_tr, 95))
    out["result_to_truth_p95"] = float(np.percentile(d_rt, 95))
    out["hausdorff"] = float(max(d_tr.max(), d_rt.max()))
    out["pass"] = bool(out["valid"] and out["solid"] and abs(out["vol_err_pct"]) < 1.0
                       and out["truth_to_result_p95"] < 0.5 and out["result_to_truth_p95"] < 0.5)
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
    return (1 - s1)[:, None] * A[idx] + (s1 * (1 - r2))[:, None] * B[idx] + (s1 * r2)[:, None] * C[idx]


def run(noise=0.5, names=None, out_dir=Path("cad_gauntlet_out")):
    out_dir.mkdir(exist_ok=True)
    results = {}
    for name in names or list(CASES):
        truth = CASES[name]()
        P, N, cams = simulate_scan(truth, noise)
        t0 = time.time()
        try:
            shape, info = cadfit.reconstruct(P, cams, noise_mm=noise, floor_z=0.0, normals=None)
            g = grade(shape, truth, P)
            g.update({k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))})
            cadfit.write_step(shape, out_dir / f"{name}.step")
        except Exception as e:  # a crash is a failed case, not a crashed gauntlet
            g = {"pass": False, "error": repr(e)}
        g["seconds"] = round(time.time() - t0, 1)
        results[name] = g
        print(f"{'PASS' if g['pass'] else 'FAIL'} {name}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                                           for k, v in g.items()}), flush=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    noise = float(args[0]) if args and args[0].replace(".", "").isdigit() else 0.5
    names = [a for a in args if a in CASES] or None
    run(noise, names)
