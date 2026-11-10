"""Dense reconstruction: COLMAP PatchMatch (NVIDIA/CUDA) or OpenMVS (CPU). Output in world mm."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

from .log import log


def _which(name, folder=""):
    if folder:
        for ext in ("", ".exe"):
            p = Path(folder) / (name + ext)
            if p.exists():
                return str(p)
    return shutil.which(name)


def _sh(cmd, cwd=None):
    log("  $ " + " ".join(map(str, cmd)))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"Commando faalde ({r.returncode}): {' '.join(map(str, cmd))}\n{(r.stdout + r.stderr)[-3000:]}")
    return r


def colmap_cuda_available(colmap_bin="colmap") -> bool:
    try:
        import pycolmap
        if pycolmap.has_cuda:
            return True
    except Exception:
        pass
    b = shutil.which(colmap_bin) or (colmap_bin if Path(colmap_bin).exists() else None)
    if not b:
        return False
    try:
        out = subprocess.run([b, "-h"], capture_output=True, text=True, timeout=60).stdout
        return "with CUDA" in out and "without CUDA" not in out
    except Exception:
        return False


def choose_backend(cfg_d: dict, cfg_sfm: dict) -> str:
    b = cfg_d.get("backend", "auto")
    if b != "auto":
        return b
    if colmap_cuda_available(cfg_sfm.get("colmap_bin", "colmap")):
        return "colmap"
    if _which("DensifyPointCloud", cfg_d.get("openmvs_bin", "")):
        return "openmvs"
    raise RuntimeError("Geen dense-backend gevonden. Installeer COLMAP (met CUDA) of OpenMVS; zie README.")


def run_dense(state_dir: Path, image_dir: Path, cfg: dict) -> Path:
    import pycolmap

    cfg_d = cfg["dense"]
    backend = choose_backend(cfg_d, cfg["sfm"])
    dense_dir = state_dir / "dense"
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    log(f"  undistort beelden -> {dense_dir.name} (backend: {backend})")
    pycolmap.undistort_images(str(dense_dir), str(state_dir / "sparse_world"), str(image_dir))
    out_ply = state_dir / "dense.ply"
    if backend == "colmap":
        _dense_colmap(dense_dir, out_ply, cfg)
    elif backend == "openmvs":
        _dense_openmvs(dense_dir, out_ply, cfg)
    else:
        raise ValueError(f"Onbekende dense backend {backend}")
    pc = o3d.io.read_point_cloud(str(out_ply))
    log(f"  dense puntenwolk: {len(pc.points):,} punten")
    return out_ply


def _dense_colmap(dense_dir, out_ply, cfg):
    try:
        import pycolmap
        if pycolmap.has_cuda:
            po = pycolmap.PatchMatchOptions()
            po.geom_consistency = True
            po.max_image_size = int(cfg["dense"].get("max_resolution", 4096))
            pycolmap.patch_match_stereo(str(dense_dir), options=po)
            fo = pycolmap.StereoFusionOptions()
            fo.min_num_pixels = int(cfg["dense"].get("number_views_fuse", 3))
            pycolmap.stereo_fusion(str(out_ply), str(dense_dir), input_type="geometric", options=fo, output_type="ply")
            return
    except (ImportError, AttributeError, TypeError) as e:
        log(f"  pycolmap dense niet beschikbaar ({e}), colmap CLI wordt gebruikt")
    b = cfg["sfm"].get("colmap_bin", "colmap")
    _sh([b, "patch_match_stereo", "--workspace_path", dense_dir, "--PatchMatchStereo.geom_consistency", "true",
         "--PatchMatchStereo.max_image_size", str(int(cfg["dense"].get("max_resolution", 4096)))])
    _sh([b, "stereo_fusion", "--workspace_path", dense_dir, "--output_path", out_ply,
         "--StereoFusion.min_num_pixels", str(int(cfg["dense"].get("number_views_fuse", 3)))])


def _dense_openmvs(dense_dir, out_ply, cfg):
    import pycolmap
    folder = cfg["dense"].get("openmvs_bin", "")
    interface = _which("InterfaceCOLMAP", folder)
    densify = _which("DensifyPointCloud", folder)
    if not interface or not densify:
        raise RuntimeError("OpenMVS (InterfaceCOLMAP/DensifyPointCloud) niet gevonden. Zet dense.openmvs_bin in config.")
    # OpenMVS reads the text model
    sp = dense_dir / "sparse"
    r = pycolmap.Reconstruction(str(sp))
    r.write_text(str(sp))
    mvs = dense_dir / "scene.mvs"
    _sh([interface, "-i", dense_dir, "-o", mvs, "--image-folder", dense_dir / "images", "-w", dense_dir])
    lvl = int(cfg["dense"].get("resolution_level", 0))
    cmd = [densify, "-i", mvs, "-o", dense_dir / "scene_dense.mvs", "-w", dense_dir,
           "--resolution-level", str(lvl), "--max-resolution", str(int(cfg["dense"].get("max_resolution", 4096))),
           "--number-views-fuse", str(int(cfg["dense"].get("number_views_fuse", 3))),
           "--estimate-normals", "2"]
    _sh(cmd)
    ply = dense_dir / "scene_dense.ply"
    if not ply.exists():
        cands = sorted(dense_dir.glob("*dense*.ply"))
        if not cands:
            raise RuntimeError("OpenMVS gaf geen dense .ply")
        ply = cands[0]
    shutil.copy(ply, out_ply)


def load_dense(ply: Path, cam_centers: np.ndarray):
    pc = o3d.io.read_point_cloud(str(ply))
    pts = np.asarray(pc.points)
    if not pc.has_normals():
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6.0, max_nn=40))
    nrm = np.asarray(pc.normals).copy()
    # orient each normal towards nearest camera
    from scipy.spatial import cKDTree
    _, ci = cKDTree(cam_centers).query(pts)
    v = cam_centers[ci] - pts
    flip = (nrm * v).sum(1) < 0
    nrm[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(nrm)
    return pc
