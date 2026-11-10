"""Developer harness: test mesh -> bodies -> parts -> STEP on simulated dense clouds from ground truth.

Skips SfM/dense (slow) so the geometry stages can be iterated quickly.
  python tests/downstream_harness.py <dataset_dir> <work_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foto2step.config import _merge, DEFAULTS  # noqa: E402
from foto2step.pipeline import Pipeline  # noqa: E402
from foto2step.synthetic import Renderer, build_scene  # noqa: E402


class PinholeCam:
    def __init__(self, K, w, h):
        self.K = K
        self.width = w
        self.height = h

    def img_from_cam(self, pc, check_cheirality=False):
        return np.c_[pc[:, 0] / pc[:, 2] * self.K[0, 0] + self.K[0, 2] + 0.5,
                     pc[:, 1] / pc[:, 2] * self.K[1, 1] + self.K[1, 2] + 0.5]


def simulate(ds: Path, work: Path, noise=0.25, step=4):
    gt = json.loads((ds / "gt.json").read_text())
    cfg = _merge(DEFAULTS, yaml.safe_load((ds / "config.yaml").read_text()))
    sc = build_scene()
    K = np.array(gt["K"])
    w, h = gt["size"]
    rng = np.random.default_rng(0)
    cams_by_state = {}
    for s, sd in gt["states"].items():
        d = work / s
        d.mkdir(parents=True, exist_ok=True)
        R = Renderer(sc, s)
        pts_all = []
        nrm_all = []
        cams = []
        for c in sd["cameras"][::2]:
            Rw = np.array(c["R"])
            C = np.array(c["C"])
            cams.append((PinholeCam(K, w, h), np.c_[Rw, -Rw @ C], C))
            v, u = np.mgrid[0:h:step, 0:w:step]
            x = np.c_[(u.ravel() - K[0, 2]) / K[0, 0], (v.ravel() - K[1, 2]) / K[1, 1], np.ones(u.size)]
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dw = x @ Rw
            rays = np.c_[np.broadcast_to(C, dw.shape), dw].astype(np.float32)
            ans = R.rc.cast_rays(o3d.core.Tensor(rays))
            t = ans["t_hit"].numpy()
            ok = np.isfinite(t)
            n = ans["primitive_normals"].numpy()[ok]
            inc = np.abs((n * dw[ok]).sum(1))
            ok2 = inc > 0.25  # grazing views are not reconstructed well
            p = C + dw[ok] * (t[ok] + rng.normal(0, noise / np.maximum(inc, 0.3)))[:, None]
            pts_all.append(p[ok2])
            nrm_all.append(n[ok2])
        P = np.vstack(pts_all)
        N = np.vstack(nrm_all)
        # outliers
        P = np.vstack([P, P[rng.choice(len(P), len(P) // 200)] + rng.normal(0, 30, (len(P) // 200, 3))])
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        o3d.io.write_point_cloud(str(d / "dense.ply"), pc)
        mk = {str(k): {"corners": v, "n_views": [9] * 4, "rms_px": [0.3] * 4} for k, v in sd["marker_corners"].items()}
        (d / "markers_world.json").write_text(json.dumps(mk))
        (d / "calibration.json").write_text(json.dumps({
            "scale": {"sigma_corner_mm": 0.1, "rel_sigma": 1e-4, "bars": []}, "frame": {"floor_plane_rms_mm": 0.0},
            "checks": [], "marker_extent_mm": 3000.0, "scale_error_at_extent_mm": 0.3}))
        cams_by_state[s] = cams
    return cfg, cams_by_state


if __name__ == "__main__":
    ds = Path(sys.argv[1])
    work = Path(sys.argv[2])
    cfg, cams = simulate(ds, work)
    cfg["project"] = "harness"
    p = Pipeline(cfg, work, work / "output")
    p._cams = lambda s: cams[s]
    p.step_state_mesh()
    p.step_bodies()
    p.step_parts()
    from foto2step.selftest import grade
    R = grade(ds, work, work / "output", T=np.eye(4), sfm=False)
    print(f"{R['n_pass']}/{R['n_total']} geslaagd")
