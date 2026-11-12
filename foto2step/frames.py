"""Collect input images: photos (EXIF-orientation normalised) or video frames (sharpest, with motion filter)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .log import log

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic", ".heif", ".webp", ".dng"}
VID_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def sharpness(gray: np.ndarray) -> float:
    g = gray
    if max(g.shape) > 1600:
        f = 1600 / max(g.shape)
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _resize(img, max_side):
    if max_side and max(img.shape[:2]) > max_side:
        f = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def collect_images(src: Path, out_dir: Path, cfg_frames: dict) -> dict:
    """Write normalised images into out_dir. Returns info dict."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"source": str(src), "images": [], "focal_px_prior": None, "rejected_blurry": [], "groups": {},
            "device": None}
    max_side = int(cfg_frames.get("max_side", 0) or 0)
    if src.is_file() and src.suffix.lower() in VID_EXT:
        _from_video(src, out_dir / "video", cfg_frames, info, max_side, prefix="video/")
    else:
        files = []
        if src.is_dir():
            files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
            vids = sorted(p for p in src.rglob("*") if p.suffix.lower() in VID_EXT)
        else:
            raise FileNotFoundError(f"Invoer niet gevonden: {src}")
        if not files and vids:
            for k, v in enumerate(vids):
                _from_video(v, out_dir / "video", cfg_frames, info, max_side, prefix=f"video/v{k:02d}_")
        else:
            _from_photos(files, out_dir, info, max_side)
    if len(info["images"]) < 8:
        raise RuntimeError(f"Te weinig bruikbare beelden ({len(info['images'])}) in {src}")
    (out_dir.parent / "images_info.json").write_text(json.dumps(info, indent=1))
    log(f"  {len(info['images'])} beelden klaar ({len(info['rejected_blurry'])} te vaag verworpen)")
    return info


def _read_photo(p: Path):
    """Read in *sensor* orientation (EXIF rotation ignored) so all photos of one lens share one calibration.
    HEIC (iPhone) via pillow-heif, ProRAW/DNG via rawpy (LibRaw: camera white balance, no rotation)."""
    suf = p.suffix.lower()
    if suf == ".dng":
        try:
            import rawpy
            with rawpy.imread(str(p)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, user_flip=0, output_bps=8, no_auto_bright=False,
                                      bright=1.0, half_size=False)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            log("  WAARSCHUWING: DNG gevonden maar 'rawpy' ontbreekt (pip install rawpy)")
            return None
        except Exception as e:
            log(f"  WAARSCHUWING: DNG {p.name} niet leesbaar: {e}")
            return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None and suf in (".heic", ".heif"):
        try:
            import pillow_heif
            from PIL import Image
            pillow_heif.register_heif_opener()
            with Image.open(p) as im:
                img = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    return img


def _from_photos(files, out_dir, info, max_side):
    from .device import LENS_NL, assess, focal_prior_px, group_key, lens_class, read_exif
    # if a photo exists as both DNG and JPEG/HEIC (same base name), use the DNG
    by_stem = {}
    for p in files:
        k = (p.parent, p.stem.lower())
        if k not in by_stem or p.suffix.lower() == ".dng":
            by_stem[k] = p
    files = sorted(by_stem.values())
    entries = []
    for p in files:
        img = _read_photo(p)
        if img is None:
            log(f"  WAARSCHUWING: kan {p.name} niet lezen (overgeslagen)")
            continue
        ex = read_exif(p)
        entries.append({"path": p, "shape": img.shape[:2], "exif": ex})
    if not entries:
        return
    dev = assess([e["exif"] for e in entries])
    info["device"] = dev
    for w in dev["warnings"]:
        log(f"  LET OP: {w}")
    for a in dev["advice"]:
        log(f"  ADVIES: {a}")
    # group per lens / crop / format; transposed images (rotated by an app) are rotated back to the group shape
    groups = {}
    for e in entries:
        hw = tuple(sorted(e["shape"], reverse=True))  # landscape-normalised size for grouping
        key = group_key(e["exif"], hw)
        groups.setdefault(key, []).append(e)
    small = [k for k, v in groups.items() if len(v) < 8]
    for k in small:
        log(f"  WAARSCHUWING: lensgroep '{k}' heeft maar {len(groups[k])} foto's (min. 8) -> overgeslagen")
        del groups[k]
    sharp = []
    k_img = 0
    for key, ents in sorted(groups.items()):
        shapes = [e["shape"] for e in ents]
        main = max(set(shapes), key=shapes.count)
        (out_dir / key).mkdir(parents=True, exist_ok=True)
        n_rot = 0
        focal = []
        for e in ents:
            img = _read_photo(e["path"])
            if e["shape"] != main:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                n_rot += 1
            img = _resize(img, max_side)
            fp = focal_prior_px(e["exif"], img.shape[1], img.shape[0])
            if fp:
                focal.append(fp)
            s = sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            name = f"{key}/img_{k_img:04d}.jpg"
            k_img += 1
            sharp.append((key, name, s, e["path"].name))
            cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        ex0 = ents[0]["exif"]
        info["groups"][key] = {"count": len(ents), "model": ex0.get("model"), "lens": ex0.get("lens_model"),
                               "lens_class": lens_class(ex0), "focal_35mm": ex0.get("focal_35mm"),
                               "raw": bool(ex0.get("raw")), "size": [int(img.shape[1]), int(img.shape[0])],
                               "focal_px_prior": float(np.median(focal)) if focal else None, "rotated": n_rot}
        log(f"  lensgroep {key}: {len(ents)} foto's, {LENS_NL.get(lens_class(ex0), '')}"
            f"{' (ProRAW)' if ex0.get('raw') else ''}" + (f", {n_rot} teruggedraaid" if n_rot else ""))
    # blur rejection per group
    for key in groups:
        vals = [s for g, _, s, _ in sharp if g == key]
        med = np.median(vals) if vals else 0
        for g, name, s, orig in sharp:
            if g != key:
                continue
            if s < 0.3 * med:
                (out_dir / name).unlink()
                info["rejected_blurry"].append(orig)
            else:
                info["images"].append({"name": name, "orig": orig, "sharpness": s, "group": g})
    priors = [g["focal_px_prior"] for g in info["groups"].values() if g["focal_px_prior"]]
    if len(info["groups"]) == 1 and priors:
        info["focal_px_prior"] = priors[0]


def _from_video(path, out_dir, cfg, info, max_side, prefix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir.parent  # names below are relative to the images root (prefix holds the subfolder)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / float(cfg.get("video_fps", 2.0)))))
    win = []
    idx = 0
    last_kept = None
    min_motion = float(cfg.get("min_motion_px", 30.0))
    kept = 0
    scores = []

    def flush():
        nonlocal last_kept, kept
        if not win:
            return
        best = max(win, key=lambda w: w[1])
        img, s, fi = best
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if last_kept is not None and min_motion > 0:
            small = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1])))
            if _motion(last_kept, small) < min_motion * 320 / g.shape[1]:
                win.clear()
                return
            last_kept = small
        else:
            last_kept = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1])))
        name = f"{prefix}frame_{fi:06d}.jpg"
        cv2.imwrite(str(base / name), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        scores.append((name, s))
        kept += 1
        win.clear()

    while True:
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # sensor is landscape
        img = _resize(img, max_side)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        win.append((img, sharpness(g), idx))
        idx += 1
        if len(win) >= step:
            flush()
    flush()
    cap.release()
    if scores:
        q = np.quantile([s for _, s in scores], float(cfg.get("blur_reject_quantile", 0.25)))
        for name, s in scores:
            if s < q:
                (base / name).unlink()
                info["rejected_blurry"].append(name)
            else:
                info["images"].append({"name": name, "orig": f"{path.name}", "sharpness": s, "group": "video"})
        info["groups"].setdefault("video", {"count": 0, "model": None, "lens": None, "lens_class": "video",
                                            "raw": False, "size": None, "focal_px_prior": None})
        info["groups"]["video"]["count"] += sum(1 for _ in scores)
    log(f"  video {path.name}: {idx} frames gelezen, {kept} gekozen")


def _motion(a, b):
    try:
        p0 = cv2.goodFeaturesToTrack(a, 200, 0.01, 8)
        if p0 is None:
            return 1e9
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
        d = np.linalg.norm((p1 - p0).reshape(-1, 2)[st.ravel() == 1], axis=1)
        return float(np.median(d)) if len(d) else 1e9
    except cv2.error:
        return 1e9


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
