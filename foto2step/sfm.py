"""Structure-from-Motion via COLMAP (pycolmap or colmap CLI)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .log import log


def _pycolmap():
    import pycolmap
    return pycolmap


def run_sfm(image_dir: Path, work: Path, cfg_sfm: dict, focal_prior: float | None = None, groups: dict | None = None):
    """Run SfM, return path to best sparse model directory.

    groups: {subfolder: {"focal_px_prior": ...}}; with more than one lens group every subfolder gets its
    own camera calibration (PER_FOLDER)."""
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    sparse = work / "sparse_raw"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    n_img = len([p for p in image_dir.rglob("*") if p.is_file()])
    multi = bool(groups) and len(groups) > 1
    matcher = cfg_sfm.get("matcher", "exhaustive")
    if matcher == "exhaustive" and n_img > 400:
        log(f"  {n_img} beelden: overschakelen op sequential matching")
        matcher = "sequential"
    backend = cfg_sfm.get("backend", "auto")
    if backend == "auto":
        backend = "pycolmap"
    if backend == "colmap_cli":
        _run_cli(image_dir, db, sparse, cfg_sfm, matcher, None if multi else focal_prior, multi, groups)
    else:
        _run_pycolmap(image_dir, db, sparse, cfg_sfm, matcher, None if multi else focal_prior, multi, groups)
    return _select_best(sparse, work / "sparse", n_img)


def _camera_params(cfg_sfm, focal_prior, image_dir):
    if not focal_prior:
        return ""
    import cv2
    first = sorted(p for p in image_dir.rglob("*") if p.is_file())[0]
    h, w = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE).shape
    model = cfg_sfm.get("camera_model", "OPENCV")
    if model == "OPENCV":
        return f"{focal_prior},{focal_prior},{w / 2},{h / 2},0,0,0,0"
    if model == "SIMPLE_RADIAL":
        return f"{focal_prior},{w / 2},{h / 2},0"
    if model == "RADIAL":
        return f"{focal_prior},{w / 2},{h / 2},0,0"
    return ""


def _set_group_priors(db, groups):
    """Write per-lens focal priors into the COLMAP database (one camera per subfolder)."""
    import pycolmap
    if not groups:
        return
    try:
        d = pycolmap.Database(str(db)) if not hasattr(pycolmap.Database, "open") else pycolmap.Database.open(str(db))
    except Exception:
        try:
            d = pycolmap.Database()
            d.open(str(db))
        except Exception as e:
            log(f"  (lens-startwaarden niet gezet: {e})")
            return
    try:
        cam_group = {}
        for im in d.read_all_images():
            cam_group.setdefault(im.camera_id, im.name.split("/")[0])
        for cid, g in cam_group.items():
            fp = (groups.get(g) or {}).get("focal_px_prior")
            if not fp:
                continue
            cam = d.read_camera(cid)
            p = list(cam.params)
            for i in cam.focal_length_idxs():
                p[i] = fp
            cam.params = p
            cam.has_prior_focal_length = True
            d.update_camera(cam)
        log(f"  {len(cam_group)} lensgroepen, elk met eigen kalibratie")
    finally:
        try:
            d.close()
        except Exception:
            pass


def _run_pycolmap(image_dir, db, sparse, cfg_sfm, matcher, focal_prior, multi=False, groups=None):
    pycolmap = _pycolmap()
    log("  SfM (pycolmap): features extraheren")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = cfg_sfm.get("camera_model", "OPENCV")
    cp = _camera_params(cfg_sfm, focal_prior, image_dir)
    if cp:
        reader.camera_params = cp
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = int(cfg_sfm.get("max_features", 16384))
    ext.max_image_size = 10000  # volle resolutie
    gpu = bool(cfg_sfm.get("use_gpu", True)) and pycolmap.has_cuda
    ext.use_gpu = gpu
    if not gpu:
        ext.sift.estimate_affine_shape = True
        ext.sift.domain_size_pooling = True
    if multi:
        mode = pycolmap.CameraMode.PER_FOLDER
    else:
        mode = pycolmap.CameraMode.SINGLE if cfg_sfm.get("single_camera", True) else pycolmap.CameraMode.AUTO
    pycolmap.extract_features(db, image_dir, camera_mode=mode, reader_options=reader, extraction_options=ext)
    if multi:
        _set_group_priors(db, groups)
    log(f"  SfM: matching ({matcher})")
    mo = pycolmap.FeatureMatchingOptions()
    mo.use_gpu = gpu
    g = cfg_sfm.get("guided_matching", "auto")
    mo.guided_matching = gpu if g == "auto" else bool(g)  # ~10x trager op CPU, kleine winst
    if matcher == "sequential":
        po = pycolmap.SequentialPairingOptions()
        po.overlap = 30
        po.quadratic_overlap = True
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=po)
    else:
        pycolmap.match_exhaustive(db, matching_options=mo)
    log("  SfM: incrementele reconstructie + bundle adjustment")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_principal_point = False
    opts.multiple_models = True
    pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)


def _cli_help(bin_, cmd):
    try:
        return subprocess.run([bin_, cmd, "-h"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def _run_cli(image_dir, db, sparse, cfg_sfm, matcher, focal_prior, multi=False, groups=None):
    b = cfg_sfm.get("colmap_bin", "colmap")
    h = _cli_help(b, "feature_extractor")
    gpu = "1" if cfg_sfm.get("use_gpu", True) else "0"
    ext_prefix = "FeatureExtraction" if "FeatureExtraction.use_gpu" in h else "SiftExtraction"
    cmd = [b, "feature_extractor", "--database_path", str(db), "--image_path", str(image_dir),
           "--ImageReader.camera_model", cfg_sfm.get("camera_model", "OPENCV"),
           "--ImageReader.single_camera", "0" if multi else ("1" if cfg_sfm.get("single_camera", True) else "0"),
           "--ImageReader.single_camera_per_folder", "1" if multi else "0",
           "--SiftExtraction.max_num_features", str(int(cfg_sfm.get("max_features", 16384))),
           f"--{ext_prefix}.use_gpu", gpu, f"--{ext_prefix}.max_image_size", "10000"]
    cp = _camera_params(cfg_sfm, focal_prior, image_dir)
    if cp:
        cmd += ["--ImageReader.camera_params", cp]
    if gpu == "0":
        cmd += ["--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"]
    _sh(cmd)
    if multi:
        _set_group_priors(db, groups)
    hm = _cli_help(b, "exhaustive_matcher")
    mprefix = "FeatureMatching" if "FeatureMatching.use_gpu" in hm else "SiftMatching"
    if matcher == "sequential":
        _sh([b, "sequential_matcher", "--database_path", str(db), f"--{mprefix}.use_gpu", gpu,
             "--SequentialMatching.overlap", "30", f"--{mprefix}.guided_matching", _guided(cfg_sfm, gpu)])
    else:
        _sh([b, "exhaustive_matcher", "--database_path", str(db), f"--{mprefix}.use_gpu", gpu,
             f"--{mprefix}.guided_matching", _guided(cfg_sfm, gpu)])
    _sh([b, "mapper", "--database_path", str(db), "--image_path", str(image_dir), "--output_path", str(sparse)])


def _guided(cfg_sfm, gpu):
    g = cfg_sfm.get("guided_matching", "auto")
    return ("1" if gpu == "1" else "0") if g == "auto" else ("1" if g else "0")


def _sh(cmd):
    log("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Commando faalde ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stderr[-3000:]}")
    return r


def _select_best(sparse_raw: Path, sparse_out: Path, n_img: int) -> Path:
    pycolmap = _pycolmap()
    models = []
    for d in sorted(sparse_raw.iterdir()):
        if d.is_dir() and ((d / "images.bin").exists() or (d / "images.txt").exists()):
            r = pycolmap.Reconstruction(str(d))
            models.append((r.num_reg_images(), d, r))
    if not models:
        raise RuntimeError("SfM gaf geen reconstructie. Te weinig overlap/textuur? Zie fotoprotocol in README.")
    models.sort(key=lambda m: -m[0])
    n, d, r = models[0]
    if len(models) > 1:
        log(f"  WAARSCHUWING: {len(models)} losse deelmodellen ({[m[0] for m in models]} beelden). "
            "Alleen het grootste wordt gebruikt -> maak meer overlappende foto's tussen de delen.")
    frac = n / max(n_img, 1)
    log(f"  SfM: {n}/{n_img} beelden geregistreerd ({frac:.0%}), {r.num_points3D()} punten, "
        f"reprojectiefout {r.compute_mean_reprojection_error():.3f} px")
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    shutil.copytree(d, sparse_out)
    cam = next(iter(r.cameras.values()))
    stats = {"registered": n, "total": n_img, "points3D": r.num_points3D(),
             "mean_reproj_px": r.compute_mean_reprojection_error(),
             "mean_track_length": r.compute_mean_track_length(),
             "camera_model": cam.model_name if hasattr(cam, "model_name") else str(cam.model),
             "camera_params": list(map(float, cam.params)), "n_submodels": len(models)}
    (sparse_out.parent / "sfm_stats.json").write_text(json.dumps(stats, indent=1))
    return sparse_out


def load_cameras(sparse_dir: Path):
    """Return dict name -> (camera, cam_from_world 3x4, center)."""
    pycolmap = _pycolmap()
    r = pycolmap.Reconstruction(str(sparse_dir))
    out = {}
    for iid, im in r.images.items():
        if not im.has_pose:
            continue
        M = np.asarray(im.cam_from_world().matrix() if callable(im.cam_from_world) else im.cam_from_world.matrix())
        out[im.name] = (r.cameras[im.camera_id], M, np.asarray(im.projection_center()))
    return r, out
