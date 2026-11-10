"""End-to-end pipeline orchestration. Every step is resumable (skips work that is already done)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from . import meshing
from .config import resolve
from .log import log, set_logfile

PART_COLORS = [(0.62, 0.64, 0.68), (0.90, 0.55, 0.15), (0.20, 0.50, 0.85), (0.30, 0.70, 0.35),
               (0.80, 0.25, 0.30), (0.55, 0.35, 0.75), (0.85, 0.80, 0.20), (0.25, 0.75, 0.75)]


class Pipeline:
    def __init__(self, cfg: dict, work: Path, out: Path, force: bool = False):
        self.cfg = cfg
        self.work = Path(work)
        self.out = Path(out)
        self.force = force
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        set_logfile(self.work / "foto2step.log")
        self.states = [s["name"] for s in cfg["states"]]
        self.report = {"project": cfg["project"], "states": {}, "started": time.strftime("%Y-%m-%d %H:%M")}

    def sdir(self, s):
        d = self.work / s
        d.mkdir(exist_ok=True)
        return d

    # ------------------------------------------------------------------ steps
    def step_frames(self):
        from .frames import collect_images
        for st in self.cfg["states"]:
            d = self.sdir(st["name"])
            if (d / "images_info.json").exists() and not self.force:
                continue
            log(f"[{st['name']}] beelden verzamelen uit {st['images']}")
            collect_images(resolve(self.cfg, st["images"]), d / "images", self.cfg["frames"])

    def step_sfm(self):
        from .sfm import run_sfm
        for s in self.states:
            d = self.sdir(s)
            if (d / "sparse" / "images.bin").exists() and not self.force:
                continue
            info = json.loads((d / "images_info.json").read_text())
            log(f"[{s}] Structure-from-Motion")
            run_sfm(d / "images", d, self.cfg["sfm"], info.get("focal_px_prior"))

    def step_calibrate(self):
        from .scale import calibrate_state
        for s in self.states:
            d = self.sdir(s)
            if (d / "calibration.json").exists() and (d / "sparse_world").exists() and not self.force:
                continue
            log(f"[{s}] schaal + assenstelsel uit markers")
            calibrate_state(d, d / "images", d / "sparse", self.cfg)

    def step_dense(self):
        from .dense import run_dense
        for s in self.states:
            d = self.sdir(s)
            if (d / "dense.ply").exists() and not self.force:
                continue
            log(f"[{s}] dichte reconstructie")
            run_dense(d, d / "images", self.cfg)

    def _cams(self, s):
        from .sfm import load_cameras
        _, cams = load_cameras(self.sdir(s) / "sparse_world")
        return list(cams.values())

    def _markers(self, s):
        mk = json.loads((self.sdir(s) / "markers_world.json").read_text())
        return {int(k): {"corners": np.array(v["corners"])} for k, v in mk.items()}

    def step_state_mesh(self):
        from .dense import load_dense
        cm = self.cfg["mesh"]
        for s in self.states:
            d = self.sdir(s)
            if (d / "mesh_full.ply").exists() and (d / "cloud_clean.ply").exists() and not self.force:
                continue
            log(f"[{s}] puntenwolk opschonen + volledige mesh")
            cams = self._cams(s)
            centers = np.array([c[2] for c in cams])
            pc = load_dense(d / "dense.ply", centers)
            mk = json.loads((d / "markers_world.json").read_text())
            pc = meshing.crop_cloud(pc, mk, cm)
            pc = meshing.clean_cloud(pc, cm["voxel_mm"])
            noise = meshing.noise_estimate(pc)
            log(f"  oppervlakteruis ~{noise:.3f} mm (RMS)")
            o3d.io.write_point_cloud(str(d / "cloud_clean.ply"), pc)
            mesh = meshing.poisson_mesh(pc, cm["voxel_mm"], cm["poisson_depth"], density_q=cm["density_trim_quantile"])
            mesh = meshing.cap_triangles(mesh, int(cm.get("working_triangles", 1200000)) // 2)
            o3d.io.write_triangle_mesh(str(d / "mesh_full.ply"), mesh)
            (d / "noise.json").write_text(json.dumps({"noise_mm": noise}))

    def step_bodies(self):
        """Find rigid bodies and label points of every state per body."""
        from .motion import find_bodies, joints_for_bodies
        from .segment import apply_manual, fuse_other_state, label_points, label_reference, submesh
        f = self.work / "bodies.json"
        if f.exists() and not self.force:
            return
        ref = self.states[0]
        cm = self.cfg["mesh"]
        meshA = o3d.io.read_triangle_mesh(str(self.sdir(ref) / "mesh_full.ply"))
        noise = json.loads((self.sdir(ref) / "noise.json").read_text())["noise_mm"]
        tau = float(self.cfg["motion"].get("label_tol_mm") or 0) or max(4 * noise, 2.0 * cm["voxel_mm"], 1.5)
        cal = json.loads((self.sdir(ref) / "calibration.json").read_text())
        sigma = cal["scale"]["sigma_corner_mm"]
        manual = self.cfg["motion"].get("manual_bodies") or []
        if len(self.states) > 1:
            log("Bewegende delen bepalen uit de standen")
            mks = {s: self._markers(s) for s in self.states}
            res = find_bodies(mks, self.states, float(self.cfg["motion"]["move_threshold_mm"]), sigma,
                              int(self.cfg["motion"].get("min_markers_per_body", 1)))
            bodies = res["bodies"]
            joints = joints_for_bodies(bodies, self.states)
            others = []
            for s in self.states[1:]:
                others.append({"state": s, "mesh": o3d.io.read_triangle_mesh(str(self.sdir(s) / "mesh_full.ply")),
                               "cams": self._cams(s)})
            face_lab, diag = label_reference(meshA, bodies, others, tol=max(3 * noise, 0.5), tau=tau, manual=manual)
        else:
            bodies = [{"name": "machine", "marker_ids": [], "T": {ref: np.eye(4)}, "static": True}]
            joints = []
            face_lab = np.zeros(len(meshA.triangles), int)
            names = ["machine"]
            if manual:
                V = np.asarray(meshA.vertices)
                F = np.asarray(meshA.triangles)
                face_lab = apply_manual(V, F, face_lab, manual, names)
                for n in names[1:]:
                    bodies.append({"name": n, "marker_ids": [], "T": {ref: np.eye(4)}, "static": True})
            diag = {}
        # names from config (optional rename by marker id)
        rename = {int(k): v for k, v in (self.cfg["motion"].get("names") or {}).items()} if isinstance(
            self.cfg["motion"].get("names"), dict) else {}
        for b in bodies:
            for mid in b.get("marker_ids", []):
                if mid in rename:
                    b["name"] = rename[mid]
        if len(bodies) > len(set(b["name"] for b in bodies)):
            for i, b in enumerate(bodies):
                b["name"] = f"{b['name']}_{i}"
        for b, j in zip(bodies[1:], joints):
            j["body"] = b["name"]
        # body meshes in reference pose (for fusing other states)
        body_meshes = [submesh(meshA, face_lab == h) for h in range(len(bodies))]
        pcA = o3d.io.read_point_cloud(str(self.sdir(ref) / "cloud_clean.ply"))
        ptsA = np.asarray(pcA.points)
        labA, dA = label_points(ptsA, meshA, face_lab)
        per_body = [[ptsA[(labA == h) & (dA < tau * 2)]] for h in range(len(bodies))]
        for s in self.states[1:]:
            pcS = o3d.io.read_point_cloud(str(self.sdir(s) / "cloud_clean.ply"))
            fused, _ = fuse_other_state(np.asarray(pcS.points), body_meshes, bodies, s, tau, 2.5 * cm["voxel_mm"])
            for h in range(len(bodies)):
                per_body[h].append(fused[h])
                if len(fused[h]):
                    log(f"  stand {s}: {len(fused[h]):,} punten toegevoegd aan {bodies[h]['name']}")
        bdir = self.work / "bodies"
        bdir.mkdir(exist_ok=True)
        for h, b in enumerate(bodies):
            pts = np.vstack(per_body[h])
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            o3d.io.write_point_cloud(str(bdir / f"{h:02d}_{b['name']}.ply"), pc)
        out = {"tau_mm": tau, "noise_mm": noise, "diag": diag, "joints": joints,
               "bodies": [{"name": b["name"], "marker_ids": b.get("marker_ids", []), "static": b.get("static", False),
                           "T": {s: (np.asarray(t).tolist() if t is not None else None) for s, t in b["T"].items()}}
                          for b in bodies]}
        f.write_text(json.dumps(out, indent=1))

    def step_parts(self):
        from .features import regularize_mesh
        from .stepexport import mesh_to_shape, read_step_check, write_step
        info = json.loads((self.work / "bodies.json").read_text())
        cm = self.cfg["mesh"]
        noise = info["noise_mm"]
        pdir = self.out / "onderdelen"
        pdir.mkdir(exist_ok=True)
        bdir = self.work / "bodies"
        shapes = []
        part_info = []
        for h, b in enumerate(info["bodies"]):
            name = b["name"]
            pc = o3d.io.read_point_cloud(str(bdir / f"{h:02d}_{name}.ply"))
            if len(pc.points) < 500:
                log(f"  {name}: te weinig punten ({len(pc.points)}), overgeslagen")
                continue
            log(f"[deel {name}] mesh + STEP ({len(pc.points):,} punten)")
            pc = pc.voxel_down_sample(cm["voxel_mm"] * 0.7)
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * cm["voxel_mm"], max_nn=40))
            self._orient_normals(pc, h, info)
            mesh = meshing.poisson_mesh(pc, cm["voxel_mm"], cm["poisson_depth"], density_q=cm["density_trim_quantile"])
            mesh = meshing.cap_triangles(mesh, int(cm.get("working_triangles", 1200000)))
            if b.get("static") and cm.get("remove_floor", True):
                mesh = _clamp_floor(mesh, cm["floor_clearance_mm"] + 4 * cm["voxel_mm"])
            mesh = _fill(mesh, hole_mm=float(cm.get("fill_holes_mm", 3000.0)))
            feat = {}
            if self.cfg["features"].get("enable", True):
                mesh, feat = regularize_mesh(mesh, noise, self.cfg["features"])
            ref_pts = np.asarray(pc.points)
            dec = meshing.decimate_bounded(mesh, max(cm["target_max_error_mm"], 0.05), cm["max_triangles"])
            dec = _fill(dec, hole_mm=float(cm.get("fill_holes_mm", 3000.0)))
            # accuracy: measured points -> final surface
            acc = _surface_dev(dec, ref_pts)
            o3d.io.write_triangle_mesh(str(self.work / "bodies" / f"{h:02d}_{name}_final.ply"), dec)
            V = np.asarray(dec.vertices)
            F = np.asarray(dec.triangles)
            shp, sinfo = mesh_to_shape(V, F, self.cfg["export"].get("merge_coplanar", True))
            rgb = PART_COLORS[h % len(PART_COLORS)]
            path = pdir / f"{name}.step"
            write_step([(name, shp, rgb)], path, name)
            chk = read_step_check(path)
            log(f"  {path.name}: {sinfo['kind']}, {chk['faces']} vlakken (uit {sinfo['triangles']} driehoeken), "
                f"geldig={sinfo['valid']}, afwijking punten->STEP mediaan {acc['median']:.2f} / 95% {acc['p95']:.2f} mm")
            shapes.append((name, shp, rgb))
            part_info.append({"name": name, "step": str(path.relative_to(self.out)), **sinfo, "readback": chk,
                              "deviation_mm": acc, "features": {k: v for k, v in feat.items() if k != "regions"},
                              "n_planes": sum(1 for r in feat.get("regions", []) if r["type"] == "plane"),
                              "cylinders": [{"radius": r["radius"], "diameter": 2 * r["radius"], "axis": r["axis"],
                                             "point": r["point"], "length": r["length"], "rms": r["rms"]}
                                            for r in feat.get("regions", []) if r["type"] == "cylinder"]})
        asm = self.out / f"{self.cfg['project']}_samenstelling.step"
        write_step(shapes, asm, self.cfg["project"])
        log(f"Samenstelling geschreven: {asm.name} ({len(shapes)} delen)")
        (self.work / "parts.json").write_text(json.dumps(part_info, indent=1, default=_np))
        (self.out / "bewegingen.json").write_text(json.dumps({"joints": info["joints"], "bodies": [
            {"name": b["name"], "marker_ids": b["marker_ids"], "vast": b["static"]} for b in info["bodies"]]},
            indent=1, default=_np))

    def _orient_normals(self, pc, h, info):
        # orient towards the cameras of the reference state (points are in reference pose)
        from scipy.spatial import cKDTree
        cams = self._cams(self.states[0])
        C = np.array([c[2] for c in cams])
        pts = np.asarray(pc.points)
        n = np.asarray(pc.normals).copy()
        _, ci = cKDTree(C).query(pts)
        flip = ((C[ci] - pts) * n).sum(1) < 0
        n[flip] *= -1
        pc.normals = o3d.utility.Vector3dVector(n)

    def step_extras(self):
        from .report import write_report
        from .urdf import write_urdf
        if self.cfg["export"].get("urdf", True):
            try:
                write_urdf(self.work, self.out, self.cfg["project"])
            except Exception as e:
                log(f"  URDF overgeslagen: {e}")
        write_report(self.work, self.out, self.cfg, self.states)

    def run(self, until=None):
        steps = [("frames", self.step_frames), ("sfm", self.step_sfm), ("calibrate", self.step_calibrate),
                 ("dense", self.step_dense), ("mesh", self.step_state_mesh), ("bodies", self.step_bodies),
                 ("parts", self.step_parts), ("report", self.step_extras)]
        for name, fn in steps:
            t = time.time()
            fn()
            log(f"== stap '{name}' klaar ({time.time() - t:.0f}s)")
            if until == name:
                break


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _clamp_floor(mesh, z_thr):
    """Boundary vertices just above the floor -> put exactly on the floor (z=0), so the base closes flat."""
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles)
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    bverts = np.unique(u[c == 1])
    sel = bverts[V[bverts, 2] < z_thr]
    V[sel, 2] = 0.0
    mesh.vertices = o3d.utility.Vector3dVector(V)
    return mesh


def _fill(mesh, hole_mm=400.0):
    try:
        t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        t = t.fill_holes(hole_size=hole_mm)
        m = t.to_legacy()
        m.remove_degenerate_triangles()
        m.remove_duplicated_triangles()
        m.remove_duplicated_vertices()
        m.remove_unreferenced_vertices()
        m.compute_vertex_normals()
        return m
    except Exception as e:
        log(f"  gaten vullen mislukt: {e}")
        return mesh


def _surface_dev(mesh, pts):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d = sc.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()
    return {"median": float(np.median(d)), "p95": float(np.quantile(d, 0.95)), "rms": float(np.sqrt(np.mean(d ** 2)))}
