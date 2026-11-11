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
            # this mesh is only used to decide which surface belongs to which part: 2-3 mm resolution is plenty
            mesh = meshing.poisson_mesh(pc, max(cm["voxel_mm"], 1.5), 0, density_q=cm["density_trim_quantile"], max_depth=10)
            mesh = meshing.cap_triangles(mesh, int(cm.get("working_triangles", 1200000)) // 2)
            o3d.io.write_triangle_mesh(str(d / "mesh_full.ply"), mesh)
            (d / "noise.json").write_text(json.dumps({"noise_mm": noise}))

    def _state_scale(self, s, mks):
        """Scale factor that maps state s onto the reference scale, from the static markers.

        Every state has its own SfM + marker scale; rigid transforms cannot absorb a scale ratio, so
        it is estimated from the same physical (static) markers (precision ~ sigma/extent)."""
        from .geometry import umeyama
        from .motion import group_state
        ref = self.states[0]
        sigma = json.loads((self.sdir(ref) / "calibration.json").read_text())["scale"]["sigma_corner_mm"]
        _, groups = group_state(mks[ref], mks[s], float(self.cfg["motion"]["move_threshold_mm"]), sigma)
        ids = [i for i in groups[0][0] if i in mks[s]]
        f, _, _ = umeyama(np.vstack([mks[s][i]["corners"] for i in ids]),
                          np.vstack([mks[ref][i]["corners"] for i in ids]), with_scale=True)
        return float(f)

    def _scaled(self, s):
        f = self.scale_f.get(s, 1.0)
        cams = []
        for cam, M, C in self._cams(s):
            M2 = np.array(M, float).copy()
            M2[:, 3] *= f
            cams.append((cam, M2, np.asarray(C) * f))
        mesh = o3d.io.read_triangle_mesh(str(self.sdir(s) / "mesh_full.ply"))
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * f)
        pc = o3d.io.read_point_cloud(str(self.sdir(s) / "cloud_clean.ply"))
        pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points) * f)
        return cams, mesh, pc

    def step_bodies(self):
        """Find rigid bodies, label every surface per body, fuse all states per body (in reference pose)."""
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
        self.scale_f = {ref: 1.0}
        others = []
        if len(self.states) > 1:
            log("Bewegende delen bepalen uit de standen")
            mks = {s: self._markers(s) for s in self.states}
            for s in self.states[1:]:
                fs = self._state_scale(s, mks)
                self.scale_f[s] = fs
                log(f"  stand {s}: schaalverhouding t.o.v. {ref}: {fs:.6f} ({(fs - 1) * 1e6:+.0f} ppm)")
                if abs(fs - 1) > 2e-3:
                    log("  WAARSCHUWING: grote schaalafwijking tussen standen; zijn schaallat/markers verplaatst?")
                mks[s] = {k: {"corners": v["corners"] * fs} for k, v in mks[s].items()}
            res = find_bodies(mks, self.states, float(self.cfg["motion"]["move_threshold_mm"]), sigma,
                              int(self.cfg["motion"].get("min_markers_per_body", 1)))
            bodies = res["bodies"]
            for s in self.states[1:]:
                cams, mesh, pc = self._scaled(s)
                others.append({"state": s, "mesh": mesh, "cams": cams, "pc": pc})
            face_lab, diag = label_reference(meshA, bodies, others, tol=max(3 * noise, 0.5), tau=tau, manual=manual)
            for n in diag.get("names", [])[len(bodies):]:
                bodies.append({"name": n, "marker_ids": [], "static": False,
                               "T": {ref: np.eye(4), **{s: None for s in self.states[1:]}}})
        else:
            bodies = [{"name": "machine", "marker_ids": [], "T": {ref: np.eye(4)}, "static": True}]
            face_lab = np.zeros(len(meshA.triangles), int)
            names = ["machine"]
            if manual:
                V = np.asarray(meshA.vertices)
                F = np.asarray(meshA.triangles)
                face_lab = apply_manual(V, F, face_lab, manual, names)
                for n in names[1:]:
                    bodies.append({"name": n, "marker_ids": [], "T": {ref: np.eye(4)}, "static": True})
            diag = {}
        # body meshes in reference pose (for fusing other states)
        body_meshes = [submesh(meshA, face_lab == h) for h in range(len(bodies))]
        pcA = o3d.io.read_point_cloud(str(self.sdir(ref) / "cloud_clean.ply"))
        ptsA = np.asarray(pcA.points)
        nA = np.asarray(pcA.normals) if pcA.has_normals() else np.zeros_like(ptsA)
        labA, dA = label_points(ptsA, meshA, face_lab)
        selA = [(labA == h) & (dA < tau * 2) for h in range(len(bodies))]
        per_body = [[ptsA[selA[h]]] for h in range(len(bodies))]
        per_body_n = [[nA[selA[h]]] for h in range(len(bodies))]
        icp_log = []
        for o in others:
            s = o["state"]
            ptsS = np.asarray(o["pc"].points)
            nS = np.asarray(o["pc"].normals) if o["pc"].has_normals() else None
            fused, fused_n, _ = fuse_other_state(ptsS, body_meshes, bodies, s, tau, 2.5 * cm["voxel_mm"], nS)
            # refine each body's motion with point-to-plane ICP (markers give the start; surfaces refine it)
            changed = False
            for h, b in enumerate(bodies):
                if b["T"].get(s) is None or len(fused[h]) < 2000 or len(per_body[h][0]) < 2000:
                    continue
                dT, info = _icp_refine(fused[h], per_body[h][0], per_body_n[h][0], tau)
                icp_log.append({"state": s, "body": b["name"], **info})
                if info["accepted"]:
                    b["T"][s] = b["T"][s] @ np.linalg.inv(dT)
                    changed = True
                    log(f"  ICP {b['name']} stand {s}: correctie {info['dt_mm']:.3f} mm / {info['dr_deg']:.4f}°, "
                        f"rms {info['rmse_before']:.3f} -> {info['rmse_after']:.3f} mm")
            if changed:
                fused, fused_n, _ = fuse_other_state(ptsS, body_meshes, bodies, s, tau, 2.5 * cm["voxel_mm"], nS)
            for h in range(len(bodies)):
                per_body[h].append(fused[h])
                per_body_n[h].append(fused_n[h])
                if len(fused[h]):
                    log(f"  stand {s}: {len(fused[h]):,} punten toegevoegd aan {bodies[h]['name']}")
        joints = joints_for_bodies(bodies, self.states) if len(self.states) > 1 else []
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
        bdir = self.work / "bodies"
        bdir.mkdir(exist_ok=True)
        for old in bdir.glob("*.ply"):
            old.unlink()
        for h, b in enumerate(bodies):
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(per_body[h])))
            nn = np.vstack(per_body_n[h])
            if np.any(nn):
                pc.normals = o3d.utility.Vector3dVector(nn)
            o3d.io.write_point_cloud(str(bdir / f"{h:02d}_{b['name']}.ply"), pc)
        out = {"tau_mm": tau, "noise_mm": noise, "diag": {k: v for k, v in diag.items() if k != "names"},
               "joints": joints, "scale_factors": self.scale_f, "icp": icp_log,
               "bodies": [{"name": b["name"], "marker_ids": b.get("marker_ids", []), "static": b.get("static", False),
                           "T": {s: (np.asarray(t).tolist() if t is not None else None) for s, t in b["T"].items()}}
                          for b in bodies]}
        f.write_text(json.dumps(out, indent=1, default=_np))

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
            pj = bdir / f"{h:02d}_{name}_part.json"
            path = pdir / f"{name}.step"
            if pj.exists() and path.exists() and not self.force:
                from .stepexport import read_step_shape
                rgb = PART_COLORS[h % len(PART_COLORS)]
                shapes.append((name, read_step_shape(path), rgb))
                part_info.append(json.loads(pj.read_text()))
                log(f"[deel {name}] al klaar (hervat)")
                continue
            log(f"[deel {name}] mesh + STEP ({len(pc.points):,} punten)")
            pc = pc.voxel_down_sample(cm["voxel_mm"] * 0.7)
            if pc.has_normals() and np.any(np.asarray(pc.normals)):
                n = np.asarray(pc.normals)
                pc.normals = o3d.utility.Vector3dVector(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
            else:
                pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * cm["voxel_mm"], max_nn=40))
                self._orient_normals(pc, h, info)
            mesh = meshing.poisson_mesh(pc, cm["voxel_mm"], cm["poisson_depth"], density_q=cm["density_trim_quantile"],
                                        max_depth=int(cm.get("poisson_max_depth", 11)))
            mesh = meshing.cap_triangles(mesh, int(cm.get("working_triangles", 1200000)))
            if b.get("static") and cm.get("remove_floor", True):
                mesh = _clamp_floor(mesh, float(cm.get("floor_snap_mm", 40.0)))
            _FILLED["area"] = 0.0
            mesh = _fill(mesh, hole_mm=float(cm.get("fill_holes_mm", 3000.0)))
            filled_area = _FILLED["area"]
            feat = {}
            if self.cfg["features"].get("enable", True):
                mesh, feat = regularize_mesh(mesh, noise, self.cfg["features"])
            ref_pts = np.asarray(pc.points)
            dec = meshing.decimate_bounded(mesh, max(cm["target_max_error_mm"], noise, 0.05), cm["max_triangles"])
            dec = _fill(dec, hole_mm=float(cm.get("fill_holes_mm", 3000.0)))
            dec = meshing.make_manifold(dec)
            dec = meshing.make_manifold(_fill(dec, hole_mm=float(cm.get("fill_holes_mm", 3000.0))))
            # accuracy: measured points -> final surface
            acc = _surface_dev(dec, ref_pts)
            o3d.io.write_triangle_mesh(str(self.work / "bodies" / f"{h:02d}_{name}_final.ply"), dec)
            V = np.asarray(dec.vertices)
            F = np.asarray(dec.triangles)
            shp, sinfo = mesh_to_shape(V, F, self.cfg["export"].get("merge_coplanar", True))
            rgb = PART_COLORS[h % len(PART_COLORS)]
            write_step([(name, shp, rgb)], path, name, self.cfg["export"].get("step_schema", "AP214"))
            chk = read_step_check(path)
            log(f"  {path.name}: {sinfo['kind']}, {chk['faces']} vlakken (uit {sinfo['triangles']} driehoeken), "
                f"geldig={sinfo['valid']}, afwijking punten->STEP mediaan {acc['median']:.2f} / 95% {acc['p95']:.2f} mm")
            shapes.append((name, shp, rgb))
            filled_frac = filled_area / max(dec.get_surface_area(), 1e-9)
            part_info.append({"name": name, "step": str(path.relative_to(self.out)), **sinfo, "readback": chk,
                              "filled_fraction": float(filled_frac),
                              "deviation_mm": acc, "features": {k: v for k, v in feat.items() if k != "regions"},
                              "n_planes": sum(1 for r in feat.get("regions", []) if r["type"] == "plane"),
                              "cylinders": [{"radius": r["radius"], "diameter": 2 * r["radius"], "axis": r["axis"],
                                             "point": r["point"], "length": r["length"], "rms": r["rms"]}
                                            for r in feat.get("regions", []) if r["type"] == "cylinder"]})
            pj.write_text(json.dumps(part_info[-1], indent=1, default=_np))
        if not shapes:
            raise RuntimeError("Geen enkel onderdeel kon worden opgebouwd (te weinig punten). Zie fotoprotocol.")
        asm = self.out / f"{self.cfg['project']}_samenstelling.step"
        write_step(shapes, asm, self.cfg["project"], self.cfg["export"].get("step_schema", "AP214"))
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


def _icp_refine(src_pts, dst_pts, dst_normals, tau):
    """Point-to-plane ICP src(A-pose estimate) -> dst(A). Returns (dT, info); dT maps src onto dst.

    Accepted only if it is a small correction that clearly improves the fit (markers stay the anchor)."""
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst_pts))
    if np.any(dst_normals):
        dst.normals = o3d.utility.Vector3dVector(dst_normals)
    else:
        dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    src = src.voxel_down_sample(1.0)
    reg = o3d.pipelines.registration
    ev0 = reg.evaluate_registration(src, dst, tau, np.eye(4))
    r = reg.registration_icp(src, dst, tau, np.eye(4), reg.TransformationEstimationPointToPlane(
        reg.TukeyLoss(k=tau / 2)), reg.ICPConvergenceCriteria(max_iteration=60))
    dT = np.asarray(r.transformation)
    dt = float(np.linalg.norm(dT[:3, 3]))
    dr = float(np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1))))
    ok = (r.fitness >= 0.9 * ev0.fitness and r.inlier_rmse < ev0.inlier_rmse * 0.98 and dt < 5.0 and dr < 1.0)
    return dT, {"accepted": bool(ok), "dt_mm": dt, "dr_deg": dr, "rmse_before": float(ev0.inlier_rmse),
                "rmse_after": float(r.inlier_rmse), "fitness": float(r.fitness)}


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _clamp_floor(mesh, z_band):
    """Extend the machine down to the floor: boundary loops that run along the floor (most vertices
    below z_band) get their low vertices moved straight down to z=0, so the base can be closed flat."""
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles)
    n = 0
    for lp in meshing.boundary_loops(F):
        z = V[lp, 2]
        if len(lp) < 8 or np.mean(z < z_band) < 0.8 or z.min() > z_band * 0.6:
            continue
        P = V[lp] - V[lp].mean(0)
        nrm = np.linalg.svd(P, full_matrices=False)[2][2]
        # loop must lie roughly horizontal (a contour along the floor), not a cut-out in a side wall
        if abs(nrm[2]) > 0.7 or np.ptp(P[:, 2]) < 0.2 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1])):
            sel = np.array(lp)[z < z_band]
            V[sel, 2] = 0.0
            n += len(sel)
    if n:
        log(f"  {n} randpunten naar de vloer (z=0) doorgetrokken")
    mesh.vertices = o3d.utility.Vector3dVector(V)
    return mesh


_FILLED = {"area": 0.0}


def _fill(mesh, hole_mm=3000.0):
    a0 = mesh.get_surface_area() if len(mesh.triangles) else 0.0
    mesh, n, left = meshing.fill_holes(mesh, hole_mm)
    _FILLED["area"] += max(mesh.get_surface_area() - a0, 0.0) if len(mesh.triangles) else 0.0
    if n or left:
        log(f"  gaten gedicht: {n}" + (f", {left} te groot/onregelmatig (blijft open)" if left else ""))
    return mesh


def _surface_dev(mesh, pts):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d = sc.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()
    return {"median": float(np.median(d)), "p95": float(np.quantile(d, 0.95)), "rms": float(np.sqrt(np.mean(d ** 2)))}
