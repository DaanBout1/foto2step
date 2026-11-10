"""Synthetic ground-truth machine + physically scaled ray-cast 'phone' renderer.

Used by the self-test (gauntlet): every stage of the pipeline is checked against known geometry in mm.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from .geometry import to_h
from .markers import marker_image

DICT = "DICT_5X5_250"


def _box(lo, hi):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    return m


def _cyl(center_bottom, radius, height, sections=96):
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_translation(np.asarray(center_bottom, float) + [0, 0, height / 2])
    return m


def rot_z(deg, pivot):
    a = np.radians(deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    p = np.asarray(pivot, float)
    T[:3, 3] = p - T[:3, :3] @ p
    return T


def trans(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def build_scene():
    """Return scene dict with bodies, markers (in state-A world coords), states."""
    bodies = {
        "frame": [
            _box([-600, -300, 0], [600, 300, 700]),
            _box([-550, -100, 700], [-350, 100, 1600]),
            _cyl([300, 170, 700], 40, 50),
        ],
        "slide": [
            _box([-100, -125, 700], [200, 125, 850]),
            _box([0, -40, 850], [80, 40, 910]),
        ],
        "arm": [
            _box([-520, -60, 1600], [150, 60, 1680]),
            _cyl([100, 0, 1400], 30, 200),
        ],
    }
    bodies = {k: trimesh.util.concatenate(v) for k, v in bodies.items()}
    # markers: id, body, center, normal, x-dir (marker 'right'), size
    M = []

    def add(i, body, c, n, xdir, size):
        M.append({"id": i, "body": body, "center": list(map(float, c)), "normal": list(map(float, n)),
                  "xdir": list(map(float, xdir)), "size": float(size)})

    # floor (z up), x-dir +x
    floor = [(-500, -1000), (500, -1000), (1000, -500), (1000, 500), (500, 1000), (-500, 1000),
             (-1000, 500), (-1000, -500), (0, -1250), (1250, 0), (0, 1250), (-1250, 0)]
    for k, (x, y) in enumerate(floor):
        add(k, "floor", [x, y, 0.2], [0, 0, 1], [1, 0, 0], 100)
    # static machine frame
    add(20, "frame", [-350, -300.2, 450], [0, -1, 0], [1, 0, 0], 80)
    add(21, "frame", [0, -300.2, 300], [0, -1, 0], [1, 0, 0], 80)
    add(22, "frame", [400, -300.2, 450], [0, -1, 0], [1, 0, 0], 80)
    add(23, "frame", [600.2, -150, 400], [1, 0, 0], [0, 1, 0], 80)
    add(24, "frame", [600.2, 150, 400], [1, 0, 0], [0, 1, 0], 80)
    add(25, "frame", [-450, -100.2, 1250], [0, -1, 0], [1, 0, 0], 60)
    add(26, "frame", [-600.2, 0, 400], [-1, 0, 0], [0, -1, 0], 80)
    add(27, "frame", [0, 300.2, 400], [0, 1, 0], [-1, 0, 0], 80)
    # slide
    add(200, "slide", [50, -125.2, 775], [0, -1, 0], [1, 0, 0], 50)
    add(201, "slide", [-50, 0, 850.2], [0, 0, 1], [1, 0, 0], 50)
    add(202, "slide", [200.2, 0, 775], [1, 0, 0], [0, 1, 0], 50)
    add(203, "slide", [50, 125.2, 775], [0, 1, 0], [-1, 0, 0], 50)
    # arm
    add(210, "arm", [-150, -60.2, 1640], [0, -1, 0], [1, 0, 0], 50)
    add(211, "arm", [-250, 0, 1680.2], [0, 0, 1], [1, 0, 0], 60)
    add(212, "arm", [50, 0, 1680.2], [0, 0, 1], [1, 0, 0], 60)
    add(213, "arm", [-150, 60.2, 1640], [0, 1, 0], [-1, 0, 0], 50)
    states = {
        "A": {"frame": np.eye(4), "slide": np.eye(4), "arm": np.eye(4)},
        "B": {"frame": np.eye(4), "slide": trans([150, 0, 0]), "arm": rot_z(30, [-450, 0, 0])},
    }
    joints = {
        "slide": {"type": "prismatic", "axis": [1, 0, 0], "travel": 150.0},
        "arm": {"type": "revolute", "axis": [0, 0, 1], "point": [-450, 0, 0], "angle_deg": 30.0},
    }
    return {"bodies": bodies, "markers": M, "states": states, "joints": joints}


def marker_corners_world(m, T=np.eye(4)):
    c = np.asarray(m["center"])
    n = np.asarray(m["normal"]) / np.linalg.norm(m["normal"])
    x = np.asarray(m["xdir"]) / np.linalg.norm(m["xdir"])
    y = np.cross(n, x)
    h = m["size"] / 2
    pts = np.array([c - h * x + h * y, c + h * x + h * y, c + h * x - h * y, c - h * x - h * y])
    return to_h(T, pts)


# ---------------------------------------------------------------- texture

def _hash(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFF) / 65535.0


def _value_noise(u, v, cell, seed):
    x = u / cell
    y = v / cell
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a = _hash(ix, iy, seed)
    b = _hash(ix + 1, iy, seed)
    c = _hash(ix, iy + 1, seed)
    d = _hash(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def speckle(p_local, n_local, seed):
    """Triplanar speckle texture (like scan-spray/ speckle stickers) in [0,1]."""
    an = np.abs(n_local)
    axis = np.argmax(an, axis=1)
    u = np.where(axis == 0, p_local[:, 1], p_local[:, 0])
    v = np.where(axis == 2, p_local[:, 1], p_local[:, 2])
    s = seed + axis * 7
    val = 0.5 * _value_noise(u, v, 2.5, s) + 0.3 * _value_noise(u, v, 8.0, s + 1) + 0.2 * _value_noise(u, v, 30.0, s + 2)
    return np.clip((val - 0.5) * 2.2 + 0.5, 0, 1)


# ---------------------------------------------------------------- renderer

class Renderer:
    def __init__(self, scene, state, floor_size=4200.0):
        self.scene = scene
        self.state = state
        self.rc = o3d.t.geometry.RaycastingScene()
        self.geom = {}
        self.meshes = {}
        for name, mesh in scene["bodies"].items():
            T = scene["states"][state][name]
            m = mesh.copy()
            m.apply_transform(T)
            gid = self.rc.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)),
                                        o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
            self.geom[gid] = name
            self.meshes[name] = m
        f = floor_size / 2
        fv = np.array([[-f, -f, 0], [f, -f, 0], [f, f, 0], [-f, f, 0]], np.float32)
        ff = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
        gid = self.rc.add_triangles(o3d.core.Tensor(fv), o3d.core.Tensor(ff))
        self.geom[gid] = "floor"
        self.mk_imgs = {m["id"]: marker_image(DICT, m["id"], 240).astype(np.float32) / 255.0 for m in scene["markers"]}
        self.light = np.array([0.35, -0.5, 0.8])
        self.light /= np.linalg.norm(self.light)

    def render(self, K, dist, R, C, w, h, ss=2, rng=None):
        """R: world->cam rotation, C: camera centre. Returns uint8 BGR image and depth (mm, along ray)."""
        rng = np.random.default_rng() if rng is None else rng
        # supersampled pixel grid (OpenCV convention: pixel centre at integer coords)
        offs = (np.arange(ss) + 0.5) / ss - 0.5
        us = (np.arange(w)[None, :, None, None] + offs[None, None, None, :])
        vs = (np.arange(h)[:, None, None, None] + offs[None, None, :, None])
        U = np.broadcast_to(us, (h, w, ss, ss)).ravel()
        V = np.broadcast_to(vs, (h, w, ss, ss)).ravel()
        pts = np.stack([U, V], 1).astype(np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, K, dist, None, None, None,
                                      (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)).reshape(-1, 2)
        d_cam = np.c_[und, np.ones(len(und))]
        d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
        d_w = d_cam @ R  # R^T applied
        rays = np.c_[np.broadcast_to(C, d_w.shape), d_w].astype(np.float32)
        ans = self.rc.cast_rays(o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy().astype(np.float64)
        gid = ans["geometry_ids"].numpy()
        nrm = ans["primitive_normals"].numpy().astype(np.float64)
        hit = np.isfinite(t)
        col = np.full(len(t), 0.75)
        p = C + d_w * np.where(hit, t, 0)[:, None]
        # make normals face the camera
        flip = (nrm * d_w).sum(1) > 0
        nrm[flip] *= -1
        for g, name in self.geom.items():
            sel = hit & (gid == g)
            if not sel.any():
                continue
            if name == "floor":
                Tinv = np.eye(4)
                seed = 11
            else:
                Tinv = np.linalg.inv(self.scene["states"][self.state][name])
                seed = {"frame": 3, "slide": 5, "arm": 9}.get(name, 1)
            pl = to_h(Tinv, p[sel])
            nl = nrm[sel] @ Tinv[:3, :3].T
            tex = speckle(pl, nl, seed)
            base = {"floor": 0.55, "frame": 0.62, "slide": 0.70, "arm": 0.66}.get(name, 0.6)
            alb = base * 0.45 + tex * 0.55
            alb = self._apply_markers(alb, pl, name)
            shade = 0.38 + 0.62 * np.clip(nrm[sel] @ self.light, 0, 1)
            col[sel] = alb * shade
        img = col.reshape(h, w, ss, ss).mean(axis=(2, 3))
        depth = np.where(hit, t, np.inf).reshape(h, w, ss, ss)[:, :, ss // 2, ss // 2]
        img = img + rng.normal(0, 0.008, img.shape)
        img8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        # subtle colour tint so it looks like a photo; markers stay B/W
        bgr = np.clip(bgr.astype(np.float32) * np.array([0.97, 1.0, 1.03]), 0, 255).astype(np.uint8)
        return bgr, depth

    def _apply_markers(self, alb, pl, body):
        for m in self.scene["markers"]:
            if m["body"] != body:
                continue
            c = np.asarray(m["center"])
            n = np.asarray(m["normal"], float)
            n /= np.linalg.norm(n)
            x = np.asarray(m["xdir"], float)
            x /= np.linalg.norm(x)
            y = np.cross(n, x)
            d = pl - c
            off = d @ n
            u = d @ x
            v = d @ y
            h = m["size"] / 2
            q = 0.25 * m["size"]
            on = (np.abs(off) < 2.0) & (np.abs(u) <= h + q) & (np.abs(v) <= h + q)
            if not on.any():
                continue
            a = np.ones(on.sum())
            uu = u[on]
            vv = v[on]
            inside = (np.abs(uu) < h) & (np.abs(vv) < h)
            img = self.mk_imgs[m["id"]]
            N = img.shape[0]
            col = np.clip(((uu[inside] + h) / m["size"] * N).astype(int), 0, N - 1)
            row = np.clip(((h - vv[inside]) / m["size"] * N).astype(int), 0, N - 1)
            a[inside] = img[row, col] * 0.92 + 0.04
            a[~inside] = 0.96
            alb = alb.copy()
            alb[on] = a
        return alb


def look_at(C, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - C
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)  # camera y points down
    return np.vstack([r, d, f])  # rows: cam axes in world -> R world->cam


def camera_path(n_per_ring=(16, 18, 16, 10), radii=(2300, 2200, 2100, 1700), heights=(250, 950, 1750, 2700),
                target=(0, 0, 850), seed=0):
    rng = np.random.default_rng(seed)
    cams = []
    for k, (n, r, z) in enumerate(zip(n_per_ring, radii, heights)):
        for i in range(n):
            a = 2 * np.pi * (i + 0.5 * (k % 2)) / n + rng.normal(0, 0.03)
            C = np.array([r * np.cos(a), r * np.sin(a), z]) + rng.normal(0, 40, 3)
            tgt = np.asarray(target, float) + rng.normal(0, 120, 3)
            cams.append((look_at(C, tgt), C))
    # close-ups of the moving parts
    for tgt, rr in (((50, 0, 800), 1100), ((-200, 0, 1600), 1150)):
        for i in range(10):
            a = 2 * np.pi * i / 10 + 0.2
            C = np.array([tgt[0] + rr * np.cos(a), tgt[1] + rr * np.sin(a), tgt[2] + 500 + 250 * (i % 2)])
            cams.append((look_at(C, tgt), C))
    return cams


def make_dataset(out: Path, w=1600, h=1200, f=1250.0, k1=-0.06, k2=0.03, states=("A", "B"), seed=1,
                 n_per_ring=(16, 18, 16, 10)):
    """Render the synthetic machine for each state. Writes images, gt.json, gt meshes and config.yaml."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    K = np.array([[f, 0, w / 2 - 0.5 + 3.0], [0, f, h / 2 - 0.5 - 2.0], [0, 0, 1]])
    dist = np.array([k1, k2, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(seed)
    gt = {"K": K.tolist(), "dist": dist.tolist(), "size": [w, h], "states": {}, "markers": scene["markers"],
          "joints": scene["joints"]}
    for st in states:
        R = Renderer(scene, st)
        d = out / f"photos_{st}"
        d.mkdir(exist_ok=True)
        cams = camera_path(n_per_ring=n_per_ring, seed=seed + ord(st))
        gt_c = []
        for i, (Rw, C) in enumerate(cams):
            img, _ = R.render(K, dist, Rw, C, w, h, rng=rng)
            name = f"IMG_{st}_{i:03d}.jpg"
            cv2.imwrite(str(d / name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            gt_c.append({"name": name, "R": Rw.tolist(), "C": C.tolist()})
        mk = {}
        for m in scene["markers"]:
            T = np.eye(4) if m["body"] == "floor" else scene["states"][st][m["body"]]
            mk[m["id"]] = marker_corners_world(m, T).tolist()
        gt["states"][st] = {"cameras": gt_c, "marker_corners": mk,
                            "body_T": {k: v.tolist() for k, v in scene["states"][st].items()}}
        for name, mesh in R.meshes.items():
            mesh.export(out / f"gt_{st}_{name}.ply")
    (out / "gt.json").write_text(json.dumps(gt, indent=1))
    sizes = {m["id"]: m["size"] for m in scene["markers"] if m["size"] != 100}
    floor_xy = {m["id"]: m["center"][:2] for m in scene["markers"] if m["body"] == "floor"}
    import yaml
    cfg = {
        "project": "synthetic_machine",
        "markers": {"dictionary": DICT, "size_mm": 100.0, "sizes_mm": sizes,
                    "scale_bars": [{"a": 0, "b": 1, "distance_mm": 1000.0}, {"a": 8, "b": 10, "distance_mm": 2500.0}],
                    "check_distances": [{"a": 2, "b": 6, "distance_mm": float(np.linalg.norm(
                        np.subtract(floor_xy[2], floor_xy[6])))}, {"a": 9, "b": 11, "distance_mm": 2500.0}],
                    "world": {"origin_id": 8, "x_axis_id": 9}},
        "states": [{"name": s, "images": f"photos_{s}"} for s in states],
        "dense": {"backend": "openmvs", "resolution_level": 0},
        "sfm": {"max_features": 8192},
    }
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
