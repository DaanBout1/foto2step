"""Installation check."""
from __future__ import annotations

import shutil


def doctor() -> bool:
    ok = True

    def line(name, good, msg=""):
        nonlocal ok
        print(f"  [{'OK' if good else '!!'}] {name} {msg}")

    print("foto2step installatiecheck")
    for mod in ("numpy", "scipy", "cv2", "open3d", "trimesh", "pycolmap", "OCP", "yaml", "reportlab"):
        try:
            m = __import__(mod)
            line(mod, True, getattr(m, "__version__", ""))
        except Exception as e:
            line(mod, False, f"ontbreekt ({e})")
            ok = False
    try:
        import cv2
        line("OpenCV ArUco", hasattr(cv2, "aruco"))
    except Exception:
        pass
    try:
        import pycolmap
        line("pycolmap CUDA (snelle SfM + COLMAP dense)", pycolmap.has_cuda, "" if pycolmap.has_cuda else "(niet nodig als OpenMVS er is)")
    except Exception:
        pass
    from .dense import colmap_cuda_available
    col = shutil.which("colmap")
    line("colmap CLI", bool(col), col or "(optioneel)")
    cuda = colmap_cuda_available()
    from .dense import _conda_roots, find_tool
    mvs = find_tool("DensifyPointCloud")
    line("OpenMVS DensifyPointCloud", bool(mvs), mvs or "")
    if not mvs:
        roots = _conda_roots()
        print("     gezocht in PATH en: " + (", ".join(str(r) for r in roots) if roots else "(geen Miniforge/conda-map gevonden)"))
    if not (cuda or mvs):
        print("  !! Geen dense-backend: installeer OpenMVS (conda-forge) of COLMAP met CUDA. Zie README.")
        ok = False
    print("Alles klaar." if ok else "Er ontbreekt iets, zie hierboven.")
    return ok
