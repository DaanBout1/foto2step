"""Metric scale + world frame from ArUco markers triangulated in the SfM model, with honest accuracy stats."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import fit_plane, make_T, sim_T, to_h, triangulate_dlt, triangulate_refine, umeyama
from .log import log
from .markers import detect, make_detector, marker_object_corners


def detect_all(image_dir: Path, names, dict_name: str) -> dict:
    det = make_detector(dict_name)
    out = {}
    for n in names:
        img = cv2.imread(str(image_dir / n), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out[n] = detect(img, det)
    return out


def triangulate_markers(recon, cams: dict, detections: dict, min_views: int = 3, max_err_px: float = 2.0) -> dict:
    """Triangulate each marker corner in model coordinates.

    Returns {id: {"corners": 4x3, "n_views": [..4], "rms_px": [..4]}} for markers with all 4 corners.
    """
    obs = {}  # (id, k) -> list of (P, xn, f)
    for name, dets in detections.items():
        if name not in cams:
            continue
        cam, M, _ = cams[name]
        f = float(np.mean([cam.focal_length_x, cam.focal_length_y]))
        for mid, c in dets.items():
            xn = np.asarray(cam.cam_from_img(c + 0.5))  # OpenCV -> COLMAP pixel convention
            if xn.shape != (4, 2) or not np.all(np.isfinite(xn)):
                continue
            for k in range(4):
                obs.setdefault((mid, k), []).append((M, xn[k], f))
    res = {}
    for (mid, k), lst in obs.items():
        if len(lst) < min_views:
            continue
        P = [o[0] for o in lst]
        x = [o[1] for o in lst]
        f = np.array([o[2] for o in lst])
        keep = np.ones(len(lst), bool)
        X = None
        for _ in range(3):
            idx = np.where(keep)[0]
            if len(idx) < min_views:
                X = None
                break
            X0 = triangulate_dlt([P[i] for i in idx], [x[i] for i in idx])
            X = triangulate_refine([P[i] for i in idx], [x[i] for i in idx], X0, sigmas=1.0 / f[idx])
            err = _reproj_px(P, x, f, X)
            # also reject points behind cameras
            depth = np.array([(Pi @ np.append(X, 1))[2] for Pi in P])
            new_keep = (err < max_err_px) & (depth > 0)
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
        if X is None or keep.sum() < min_views:
            continue
        err = _reproj_px(P, x, f, X)[keep]
        res.setdefault(mid, {})[k] = (X, int(keep.sum()), float(np.sqrt(np.mean(err ** 2))))
    out = {}
    for mid, d in res.items():
        if len(d) == 4:
            out[int(mid)] = {"corners": np.array([d[k][0] for k in range(4)]),
                             "n_views": [d[k][1] for k in range(4)],
                             "rms_px": [d[k][2] for k in range(4)]}
    return out


def _reproj_px(P, x, f, X):
    Xh = np.append(X, 1.0)
    e = []
    for Pi, xi, fi in zip(P, x, f):
        pc = Pi @ Xh
        e.append(np.linalg.norm(pc[:2] / pc[2] - xi) * fi)
    return np.array(e)


def marker_size(cfg_m: dict, mid: int) -> float:
    s = cfg_m.get("sizes_mm") or {}
    return float(s.get(mid, s.get(str(mid), cfg_m["size_mm"])))


def shape_fit(corners_model: np.ndarray, size_mm: float):
    """Fit ideal square (similarity) to triangulated corners. Returns (scale_model_per_mm^-1, rms residual model)."""
    obj = marker_object_corners(size_mm)
    s, R, t = umeyama(obj, corners_model, with_scale=True)
    fit = (obj @ R.T) * s + t
    rms = float(np.sqrt(np.mean(np.sum((fit - corners_model) ** 2, 1))))
    return s, rms, R, t


def estimate_scale(markers: dict, cfg_m: dict) -> dict:
    """Return dict with scale (mm per model unit), sigma estimates and per-source details."""
    per_marker = {}
    for mid, m in markers.items():
        size = marker_size(cfg_m, mid)
        s, rms, R, t = shape_fit(m["corners"], size)
        per_marker[mid] = {"mm_per_unit": 1.0 / s, "shape_rms_model": rms, "size": size}
    if not per_marker:
        raise RuntimeError("Geen enkele marker volledig getrianguleerd. Staan de markers op minstens 3 foto's scherp in beeld?")
    k_mk = np.array([v["mm_per_unit"] for v in per_marker.values()])
    k_med = float(np.median(k_mk))
    # corner precision in mm (robust)
    sig_c = float(np.median([v["shape_rms_model"] * k_med for v in per_marker.values()]))
    sig_c = max(sig_c, 0.02)
    mad = float(np.median(np.abs(k_mk - k_med)) * 1.4826)
    inl = np.abs(k_mk - k_med) < max(4 * mad, 1e-12) if mad > 0 else np.ones(len(k_mk), bool)
    k_markers = float(np.mean(k_mk[inl]))
    se_markers = float(np.std(k_mk[inl], ddof=1) / np.sqrt(inl.sum())) if inl.sum() > 2 else abs(k_markers) * 0.01
    out = {"sigma_corner_mm": sig_c, "markers": {"mm_per_unit": k_markers, "rel_se": se_markers / k_markers,
                                                  "n": int(inl.sum()), "n_rejected": int((~inl).sum()),
                                                  "rel_spread": float(np.std(k_mk[inl]) / k_markers) if inl.sum() > 1 else 0.0}}
    bars = []
    for sb in cfg_m.get("scale_bars", []) or []:
        a, b = int(sb["a"]), int(sb["b"])
        if a in markers and b in markers:
            d_model = np.linalg.norm(markers[a]["corners"].mean(0) - markers[b]["corners"].mean(0))
            L = float(sb["distance_mm"])
            k = L / d_model
            rel_sig = np.sqrt(2) * (sig_c / 2) / L
            bars.append({"a": a, "b": b, "distance_mm": L, "mm_per_unit": k, "rel_sigma": float(rel_sig)})
        else:
            log(f"  WAARSCHUWING: schaallat {a}-{b}: marker(s) niet gevonden")
    out["bars"] = bars
    # Scale bars are measured independently and are long -> they define the scale when present.
    # Marker sizes then only serve as a check (printers often scale by 0.5-2%).
    if bars:
        ws = np.array([1.0 / max(b["rel_sigma"], 1e-6) ** 2 for b in bars])
        vals = np.array([b["mm_per_unit"] for b in bars])
        k = float((vals * ws).sum() / ws.sum())
        rel = float(1 / np.sqrt(ws.sum()))
        if len(vals) > 1:
            chi2 = float((ws * ((vals - k) / k) ** 2).sum() / (len(vals) - 1))
            if chi2 > 1:
                rel *= np.sqrt(chi2)
            out["consistency_chi2"] = chi2
        out["source"] = "schaallatten"
        out["marker_size_factor"] = float(k_markers / k)  # >1: markers kleiner geprint dan opgegeven
        if abs(out["marker_size_factor"] - 1) > 3 * max(out["markers"]["rel_se"], 1e-4) and abs(out["marker_size_factor"] - 1) > 0.002:
            log(f"  LET OP: markers zijn {abs(1 - 1 / out['marker_size_factor']) * 100:.2f}% "
                f"{'groter' if out['marker_size_factor'] < 1 else 'kleiner'} dan opgegeven in de config "
                "(printerschaal?). Schaal komt van de schaallatten, dus geen probleem, maar controleer size_mm.")
    else:
        k = k_markers
        rel = out["markers"]["rel_se"]
        # systematic print/measure uncertainty of marker size (caliper ~0.05 mm)
        sizes = [v["size"] for v in per_marker.values()]
        rel = float(np.sqrt(rel ** 2 + (0.05 / np.median(sizes)) ** 2))
        out["source"] = "markergrootte"
    out["mm_per_unit"] = k
    out["rel_sigma"] = rel
    out["markers_vs_bars_rel_diff"] = out.get("marker_size_factor", 1.0) - 1.0
    out["per_marker"] = {int(mid): {"mm_per_unit": float(v["mm_per_unit"]), "shape_rms_mm": float(v["shape_rms_model"] * k)}
                         for mid, v in per_marker.items()}
    return out


def world_frame(markers_mm: dict, cfg_m: dict, cam_centers_mm: np.ndarray):
    """Return 4x4 T_world_from_scaled given marker corners already in (scaled) mm units."""
    w = cfg_m.get("world") or {}
    centers = {mid: m["corners"].mean(0) for mid, m in markers_mm.items()}
    oid = w.get("origin_id")
    xid = w.get("x_axis_id")
    if oid is None or int(oid) not in centers:
        if oid is not None:
            log(f"  WAARSCHUWING: origin-marker {oid} niet gevonden; automatisch assenstelsel")
        oid = None
    # floor plane: RANSAC over marker centers whose own normal is ~parallel
    ids = list(centers)
    P = np.array([centers[i] for i in ids])
    N = []
    for i in ids:
        n, _, _ = fit_plane(markers_mm[i]["corners"])
        N.append(n)
    N = np.array(N)
    best = None
    seed_ids = [ids.index(int(oid))] if oid is not None else range(len(ids))
    for si in seed_ids:
        n0 = N[si]
        d = np.abs((P - P[si]) @ n0)
        par = np.abs(N @ n0) > np.cos(np.radians(10))
        inl = (d < 15.0) & par
        if best is None or inl.sum() > best.sum():
            best = inl
    inl_ids = [ids[i] for i in np.where(best)[0]]
    if len(inl_ids) >= 3:
        pts = np.vstack([markers_mm[i]["corners"] for i in inl_ids])
        n, d, rms = fit_plane(pts)
    else:
        n, d, rms = fit_plane(markers_mm[ids[0] if oid is None else int(oid)]["corners"])
        log("  WAARSCHUWING: <3 vloermarkers; vloervlak uit 1 marker (minder nauwkeurige Z-richting)")
    # z up: cameras above
    if np.median((cam_centers_mm @ n) + d) < 0:
        n, d = -n, -d
    origin = centers[int(oid)] if oid is not None else pts.mean(0) if len(inl_ids) >= 3 else centers[ids[0]]
    origin = origin - ((origin @ n) + d) * n  # project to plane
    if xid is not None and int(xid) in centers:
        xv = centers[int(xid)] - origin
    else:
        if xid is not None:
            log(f"  WAARSCHUWING: x-as marker {xid} niet gevonden; X uit marker-oriëntatie")
        xv = markers_mm[ids[0] if oid is None else int(oid)]["corners"][1] - markers_mm[ids[0] if oid is None else int(oid)]["corners"][0]
    xv = xv - (xv @ n) * n
    xv /= np.linalg.norm(xv)
    yv = np.cross(n, xv)
    R = np.vstack([xv, yv, n])  # rows = world axes in scaled coords
    T = make_T(R, -R @ origin)
    return T, {"floor_marker_ids": [int(i) for i in inl_ids], "floor_plane_rms_mm": float(rms)}


def calibrate_state(state_dir: Path, image_dir: Path, sparse_dir: Path, cfg: dict) -> dict:
    """Detect + triangulate markers, estimate scale & world frame, write sparse_world model. Returns stats."""
    from .sfm import load_cameras
    import pycolmap

    cfg_m = cfg["markers"]
    recon, cams = load_cameras(sparse_dir)
    log(f"  markers detecteren in {len(cams)} beelden")
    dets = detect_all(image_dir, list(cams), cfg_m["dictionary"])
    n_det = sum(len(d) for d in dets.values())
    mk = triangulate_markers(recon, cams, dets, int(cfg_m.get("min_views", 3)))
    log(f"  {n_det} marker-detecties, {len(mk)} markers volledig getrianguleerd")
    sc = estimate_scale(mk, cfg_m)
    k = sc["mm_per_unit"]
    mk_mm = {mid: {**m, "corners": m["corners"] * k} for mid, m in mk.items()}
    centers = np.array([c[2] for c in cams.values()]) * k
    Tw, finfo = world_frame(mk_mm, cfg_m, centers)
    S = np.eye(4)
    S[:3, :3] *= k
    T_world_from_model = Tw @ S
    # apply to reconstruction
    sim = pycolmap.Sim3d(k, pycolmap.Rotation3d(Tw[:3, :3]), Tw[:3, 3])
    recon.transform(sim)
    out_dir = state_dir / "sparse_world"
    out_dir.mkdir(exist_ok=True)
    recon.write(str(out_dir))
    markers_world = {int(mid): {"corners": to_h(T_world_from_model, m["corners"]).tolist(),
                                "n_views": m["n_views"], "rms_px": m["rms_px"]} for mid, m in mk.items()}
    # independent checks
    checks = []
    for cd in cfg_m.get("check_distances", []) or []:
        a, b = int(cd["a"]), int(cd["b"])
        if a in markers_world and b in markers_world:
            d = np.linalg.norm(np.mean(markers_world[a]["corners"], 0) - np.mean(markers_world[b]["corners"], 0))
            checks.append({"a": a, "b": b, "measured_mm": float(cd["distance_mm"]), "photo_mm": float(d),
                           "error_mm": float(d - cd["distance_mm"])})
    # leave-one-out on scale bars
    loo = []
    bars = sc.get("bars", [])
    if len(bars) >= 2:
        for i, b in enumerate(bars):
            others = [o for j, o in enumerate(bars) if j != i]
            k_o = np.average([o["mm_per_unit"] for o in others], weights=[1 / o["rel_sigma"] ** 2 for o in others])
            loo.append({"a": b["a"], "b": b["b"], "error_mm": float(b["distance_mm"] * (k_o / b["mm_per_unit"] - 1))})
    sc["loo_bars"] = loo
    bbox = np.array([c for m in markers_world.values() for c in m["corners"]])
    extent = float(np.linalg.norm(bbox.max(0) - bbox.min(0))) if len(bbox) else 0.0
    stats = {"scale": sc, "frame": finfo, "checks": checks, "T_world_from_model": T_world_from_model.tolist(),
             "marker_extent_mm": extent,
             "scale_error_at_extent_mm": float(sc["rel_sigma"] * extent)}
    (state_dir / "markers_world.json").write_text(json.dumps(markers_world, indent=1))
    (state_dir / "calibration.json").write_text(json.dumps(stats, indent=1, default=float))
    log(f"  schaal: {k:.4f} mm/eenheid  (rel. onzekerheid {sc['rel_sigma'] * 1e6:.0f} ppm = "
        f"{sc['rel_sigma'] * 1000:.3f} mm per meter), hoekpunt-precisie ~{sc['sigma_corner_mm']:.3f} mm")
    for c in checks:
        log(f"  CONTROLE {c['a']}-{c['b']}: gemeten {c['measured_mm']:.1f}, foto {c['photo_mm']:.2f} -> fout {c['error_mm']:+.2f} mm")
    return stats
