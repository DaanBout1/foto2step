"""Gauntlet: run the full pipeline on a synthetic machine with exactly known geometry and grade every stage.

Ground truth: three rigid bodies (frame, linear slide, rotating arm), ArUco markers, two poses, rendered
as phone-like photos (lens distortion, noise, JPEG). Every number in the result is compared in mm.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import yaml

from .geometry import to_h
from .log import log
from .scale import world_frame


def _gt_world_T(gt: dict, cfg_m: dict, state="A"):
    mk = {int(k): {"corners": np.array(v)} for k, v in gt["states"][state]["marker_corners"].items()}
    C = np.array([c["C"] for c in gt["states"][state]["cameras"]])
    T, _ = world_frame(mk, cfg_m, C)
    return T


def _dist_to_mesh(pts, mesh_tm):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(mesh_tm.vertices, np.float32)),
                     o3d.core.Tensor(np.asarray(mesh_tm.faces, np.uint32)))
    return sc.compute_distance(o3d.core.Tensor(np.asarray(pts, np.float32))).numpy()


def grade(ds: Path, work: Path, out: Path, T=None, sfm=True) -> dict:
    gt = json.loads((ds / "gt.json").read_text())
    cfg = yaml.safe_load((ds / "config.yaml").read_text())
    from .config import _merge, DEFAULTS
    cfg = _merge(DEFAULTS, cfg)
    if T is None:
        T = _gt_world_T(gt, cfg["markers"])
    R = {"checks": []}

    def check(name, value, limit, unit="mm", higher_better=False):
        ok = (value >= limit) if higher_better else (value <= limit)
        R["checks"].append({"name": name, "value": float(value), "limit": limit, "unit": unit, "pass": bool(ok)})
        log(f"  {'PASS' if ok else 'FAIL'}  {name}: {value:.4g} {unit} (grens {'>=' if higher_better else '<='} {limit})")

    states = [s["name"] for s in cfg["states"]]
    # 1. markers
    for s in (states if sfm else []):
        est = json.loads((work / s / "markers_world.json").read_text())
        g = gt["states"][s]["marker_corners"]
        errs = []
        for k, v in est.items():
            if k in g:
                errs.append(np.linalg.norm(np.array(v["corners"]) - to_h(T, np.array(g[k])), axis=1))
        errs = np.concatenate(errs)
        check(f"[{s}] marker-hoekpunten 3D fout (95%)", np.quantile(errs, 0.95), 1.0)
        check(f"[{s}] marker-hoekpunten 3D fout (max)", errs.max(), 2.5)
        cal = json.loads((work / s / "calibration.json").read_text())
        for c in cal["checks"]:
            check(f"[{s}] controlemaat {c['a']}-{c['b']} ({c['measured_mm']:.0f} mm)", abs(c["error_mm"]), 1.0)
        # camera centres
        from .sfm import load_cameras
        _, cams = load_cameras(work / s / "sparse_world")
        gc = {c["name"]: np.array(c["C"]) for c in gt["states"][s]["cameras"]}
        info = json.loads((work / s / "images_info.json").read_text())
        name_map = {i["name"]: i["orig"] for i in info["images"]}
        ce = [np.linalg.norm(c[2] - to_h(T, gc[name_map[n]][None])[0]) for n, c in cams.items() if name_map.get(n) in gc]
        check(f"[{s}] cameraposities fout (mediaan)", np.median(ce), 3.0)
        check(f"[{s}] geregistreerde beelden", len(cams) / len(gc), 0.95, unit="fractie", higher_better=True)
    # 2. bodies / joints
    bodies = json.loads((work / "bodies.json").read_text())
    joints = bodies["joints"]
    gtj = gt["joints"]
    n_moving = len(bodies["bodies"]) - 1
    check("aantal bewegende delen gevonden", abs(n_moving - len(gtj)), 0, unit="verschil")
    # match bodies to gt by markers
    gt_mk_body = {m["id"]: m["body"] for m in gt["markers"]}
    body_map = {}
    for b in bodies["bodies"]:
        names = [gt_mk_body.get(i) for i in b["marker_ids"]]
        names = [n for n in names if n]
        if b["static"]:
            body_map[b["name"]] = "frame"
        elif names:
            body_map[b["name"]] = max(set(names), key=names.count)
    for j in joints:
        g = gtj.get(body_map.get(j["body"]))
        if not g:
            continue
        gax = T[:3, :3] @ np.array(g["axis"], float)
        ax = np.array(j["axis"])
        ang = np.degrees(np.arccos(min(1, abs(ax @ gax))))
        check(f"as-richting {j['body']} ({g['type']})", ang, 0.5, unit="graden")
        check(f"type {j['body']}", 0 if j["type"] == g["type"] else 1, 0, unit="")
        if g["type"] == "prismatic" and j["type"] == "prismatic":
            check(f"slag {j['body']}", abs(abs(j["positions_mm"]["B"]) - g["travel"]), 0.5)
        if g["type"] == "revolute" and j["type"] == "revolute":
            check(f"hoek {j['body']}", abs(abs(j["angles_deg"]["B"]) - g["angle_deg"]), 0.1, unit="graden")
            p = np.array(j["point"])
            gp_ = to_h(T, np.array(g["point"], float)[None])[0]
            d = p - gp_
            d -= (d @ gax) * gax
            check(f"draaias-positie {j['body']}", np.linalg.norm(d), 1.0)
    # 3. parts vs ground truth
    parts = json.loads((work / "parts.json").read_text())
    gtm = {}
    for name in ("frame", "slide", "arm"):
        m = trimesh.load(ds / f"gt_A_{name}.ply")
        m.apply_transform(T)
        gtm[name] = m
    for h, b in enumerate(bodies["bodies"]):
        f = work / "bodies" / f"{h:02d}_{b['name']}_final.ply"
        if not f.exists():
            continue
        rm = o3d.io.read_triangle_mesh(str(f))
        pts = np.asarray(rm.sample_points_uniformly(60000).points)
        D = np.stack([_dist_to_mesh(pts, gtm[n]) for n in gtm], 1)
        names = list(gtm)
        nearest = np.array(names)[np.argmin(D, 1)]
        target = body_map.get(b["name"])
        if target:
            frac = float(np.mean(nearest == target))
            check(f"segmentatie {b['name']} -> {target} (oppervlak juist)", frac, 0.97, unit="fractie", higher_better=True)
            d = D[:, names.index(target)]
            # exclude floor-closing faces and contact areas from accuracy (not observable)
            sel = (nearest == target) & (pts[:, 2] > 5)
            check(f"oppervlak {b['name']} afwijking mediaan", np.median(d[sel]), 0.35)
            check(f"oppervlak {b['name']} afwijking 95%", np.quantile(d[sel], 0.95), 1.5)
            # completeness: visible gt surface covered by reconstruction
            gs = gtm[target].sample(40000)
            gs = gs[gs[:, 2] > 5]
            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(rm))
            dd = sc.compute_distance(o3d.core.Tensor(gs.astype(np.float32))).numpy()
            check(f"volledigheid {b['name']} (GT-oppervlak binnen 2 mm)", float(np.mean(dd < 2.0)), 0.85,
                  unit="fractie", higher_better=True)
    for p in parts:
        check(f"STEP {p['name']} geldig + teruggelezen", 0 if (p["valid"] and p["readback"]["faces"] > 0) else 1, 0, unit="")
        check(f"STEP {p['name']} is een solid", 0 if p["readback"]["solids"] >= 1 else 1, 0, unit="")
    asm = list(out.glob("*_samenstelling.step"))
    check("samenstelling-STEP aanwezig", 0 if asm else 1, 0, unit="")
    R["passed"] = all(c["pass"] for c in R["checks"])
    R["n_pass"] = sum(c["pass"] for c in R["checks"])
    R["n_total"] = len(R["checks"])
    return R


def run_selftest(root: Path, quick=False, openmvs_bin="") -> bool:
    from .config import load_config
    from .pipeline import Pipeline
    from .synthetic import make_dataset
    root = Path(root)
    ds = root / "dataset"
    if not (ds / "gt.json").exists():
        log("Synthetische machine renderen (eenmalig)...")
        if quick:
            make_dataset(ds, w=1200, h=900, f=940.0, n_per_ring=(12, 14, 12, 8))
        else:
            make_dataset(ds)
    cfgp = ds / "config.yaml"
    c = yaml.safe_load(cfgp.read_text())
    c.setdefault("dense", {})
    c["dense"]["backend"] = "auto"
    if openmvs_bin:
        c["dense"]["openmvs_bin"] = openmvs_bin
    cfgp.write_text(yaml.safe_dump(c, sort_keys=False))
    cfg = load_config(cfgp)
    Pipeline(cfg, root / "work", root / "output").run()
    log("GAUNTLET: resultaten vergelijken met de exacte maten")
    R = grade(ds, root / "work", root / "output")
    (root / "gauntlet.json").write_text(json.dumps(R, indent=1))
    log(f"GAUNTLET {'GESLAAGD' if R['passed'] else 'NIET GESLAAGD'}: {R['n_pass']}/{R['n_total']} controles")
    return R["passed"]
