"""Collect input images: photos (EXIF-orientation normalised) or video frames (sharpest, with motion filter)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .log import log

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic", ".webp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def sharpness(gray: np.ndarray) -> float:
    g = gray
    if max(g.shape) > 1600:
        f = 1600 / max(g.shape)
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _exif_focal_px(path: Path, width: int, height: int):
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
            f35 = sub.get(0xA405) or ex.get(0xA405)
            if f35:
                return float(f35) / 36.0 * max(width, height)
    except Exception:
        pass
    return None


def _resize(img, max_side):
    if max_side and max(img.shape[:2]) > max_side:
        f = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def collect_images(src: Path, out_dir: Path, cfg_frames: dict) -> dict:
    """Write normalised images into out_dir. Returns info dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*"):
        f.unlink()
    info = {"source": str(src), "images": [], "focal_px_prior": None, "rejected_blurry": []}
    max_side = int(cfg_frames.get("max_side", 0) or 0)
    if src.is_file() and src.suffix.lower() in VID_EXT:
        _from_video(src, out_dir, cfg_frames, info, max_side)
    else:
        files = []
        if src.is_dir():
            files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
            vids = sorted(p for p in src.rglob("*") if p.suffix.lower() in VID_EXT)
        else:
            raise FileNotFoundError(f"Invoer niet gevonden: {src}")
        if not files and vids:
            for k, v in enumerate(vids):
                _from_video(v, out_dir, cfg_frames, info, max_side, prefix=f"v{k:02d}_")
        else:
            _from_photos(files, out_dir, info, max_side)
    if len(info["images"]) < 8:
        raise RuntimeError(f"Te weinig bruikbare beelden ({len(info['images'])}) in {src}")
    (out_dir.parent / "images_info.json").write_text(json.dumps(info, indent=1))
    log(f"  {len(info['images'])} beelden klaar ({len(info['rejected_blurry'])} te vaag verworpen)")
    return info


def _read_photo(p: Path):
    """Read in *sensor* orientation (EXIF rotation ignored) so all photos share one intrinsic calibration.
    HEIC (iPhone) is supported via pillow-heif when installed."""
    img = cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None and p.suffix.lower() in (".heic", ".heif"):
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
    imgs = []
    for p in files:
        img = _read_photo(p)
        if img is None:
            log(f"  WAARSCHUWING: kan {p.name} niet lezen (overgeslagen)")
            continue
        imgs.append((p, img.shape[:2]))
    if not imgs:
        return
    # majority shape; transposed images (rotated by the phone app) are rotated back
    shapes = [s for _, s in imgs]
    main = max(set(shapes), key=shapes.count)
    sharp = []
    focal = []
    n_rot = n_skip = 0
    for k, (p, shp) in enumerate(imgs):
        img = _read_photo(p)
        if shp != main:
            if shp == main[::-1]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                n_rot += 1
            else:
                log(f"  WAARSCHUWING: {p.name} heeft afwijkend formaat {shp} (andere lens/zoom?) -> overgeslagen")
                n_skip += 1
                continue
        img = _resize(img, max_side)
        fp = _exif_focal_px(p, img.shape[1], img.shape[0])
        if fp:
            focal.append(fp)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = sharpness(g)
        name = f"img_{k:04d}.jpg"
        sharp.append((name, s, p.name))
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    if n_rot:
        log(f"  {n_rot} staande foto's teruggedraaid naar sensororiëntatie")
    if n_skip:
        log(f"  {n_skip} foto's met ander formaat overgeslagen: gebruik 1 lens en vaste zoom!")
    med = np.median([s for _, s, _ in sharp]) if sharp else 0
    for name, s, orig in sharp:
        if s < 0.3 * med:
            (out_dir / name).unlink()
            info["rejected_blurry"].append(orig)
        else:
            info["images"].append({"name": name, "orig": orig, "sharpness": s})
    if focal:
        info["focal_px_prior"] = float(np.median(focal))


def _from_video(path, out_dir, cfg, info, max_side, prefix=""):
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
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
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
                (out_dir / name).unlink()
                info["rejected_blurry"].append(name)
            else:
                info["images"].append({"name": name, "orig": f"{path.name}", "sharpness": s})
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
