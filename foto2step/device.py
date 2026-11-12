"""Phone / lens recognition from EXIF, per-lens camera groups, and capture-quality advice.

Each lens (and each crop mode, e.g. the 2x crop of the main camera) has its own optics, so every
group gets its own calibration in SfM. Settings that matter for accuracy (aperture, shutter, ISO,
RAW) are checked and turned into concrete advice.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# EXIF tags
_MAKE, _MODEL = 0x010F, 0x0110
_EXIF_IFD = 0x8769
_EXPOSURE, _FNUMBER, _ISO = 0x829A, 0x829D, 0x8827
_FOCAL, _FOCAL35 = 0x920A, 0xA405
_LENS_MAKE, _LENS_MODEL = 0xA433, 0xA434


def _num(v):
    try:
        if isinstance(v, tuple) and len(v) == 2:
            return float(v[0]) / float(v[1]) if v[1] else None
        return float(v)
    except Exception:
        return None


def read_exif(path: Path) -> dict:
    """Best-effort EXIF for JPEG/HEIC/DNG. Missing values are None."""
    out = {"make": None, "model": None, "lens_model": None, "f_number": None, "exposure_s": None,
           "iso": None, "focal_mm": None, "focal_35mm": None, "raw": Path(path).suffix.lower() == ".dng"}
    try:
        from PIL import Image
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(_EXIF_IFD) if hasattr(ex, "get_ifd") else {}

            def g(tag):
                return sub.get(tag, ex.get(tag))
            out["make"] = (g(_MAKE) or None) and str(g(_MAKE)).strip("\x00 ")
            out["model"] = (g(_MODEL) or None) and str(g(_MODEL)).strip("\x00 ")
            out["lens_model"] = (g(_LENS_MODEL) or None) and str(g(_LENS_MODEL)).strip("\x00 ")
            out["f_number"] = _num(g(_FNUMBER))
            out["exposure_s"] = _num(g(_EXPOSURE))
            iso = g(_ISO)
            if isinstance(iso, (list, tuple)) and iso and not isinstance(iso[0], (tuple, list)):
                iso = iso[0]
            out["iso"] = _num(iso)
            out["focal_mm"] = _num(g(_FOCAL))
            out["focal_35mm"] = _num(g(_FOCAL35))
    except Exception:
        pass
    if out["model"] is None:  # DNG / TIFF: Pillow often cannot read these tags, exifread can
        try:
            import exifread
            with open(path, "rb") as fh:
                t = exifread.process_file(fh, details=False)

            def val(k):
                v = t.get(k)
                if v is None:
                    return None
                vv = v.values
                if isinstance(vv, list) and vv:
                    vv = vv[0]
                if hasattr(vv, "num"):
                    return vv.num / vv.den if vv.den else None
                return vv
            out["make"] = str(t["Image Make"]).strip() if "Image Make" in t else None
            out["model"] = str(t["Image Model"]).strip() if "Image Model" in t else None
            out["lens_model"] = str(t["EXIF LensModel"]).strip() if "EXIF LensModel" in t else None
            out["f_number"] = _num(val("EXIF FNumber"))
            out["exposure_s"] = _num(val("EXIF ExposureTime"))
            out["iso"] = _num(val("EXIF ISOSpeedRatings"))
            out["focal_mm"] = _num(val("EXIF FocalLength"))
            out["focal_35mm"] = _num(val("EXIF FocalLengthIn35mmFilm"))
        except Exception:
            pass
    return out


def lens_class(ex: dict) -> str:
    """'ultra' | 'main' | 'main2x' | 'tele' | 'tele2x' | 'front' | 'unknown' from 35 mm-equivalent focal / lens name."""
    lm = (ex.get("lens_model") or "").lower()
    if "front" in lm:
        return "front"
    f35 = ex.get("focal_35mm")
    if not f35 and ex.get("focal_mm"):
        # iPhone lens names contain the real focal; without 35mm-equivalent we can only guess crudely
        f35 = None
    if f35:
        if f35 < 18:
            return "ultra"
        if f35 < 36:
            return "main"
        if f35 < 70:
            return "main2x"
        if f35 < 150:
            return "tele"
        return "tele2x"
    if "ultra" in lm or "triple camera 2." in lm:
        return "ultra"
    return "unknown"


LENS_NL = {"ultra": "ultragroothoek", "main": "hoofdcamera 1x", "main2x": "hoofdcamera 2x (uitsnede)",
           "tele": "telelens", "tele2x": "telelens uitsnede", "front": "voorcamera", "unknown": "camera"}


def group_key(ex: dict, shape) -> str:
    h, w = shape[:2]
    mp = int(round(w * h / 1e6))
    cls = lens_class(ex)
    name = cls if cls != "unknown" else "camera"
    if ex.get("raw"):
        name += "_raw"
    return f"{name}_{mp}mp"


def phone_tier(model: str | None) -> str:
    """'pro_var_aperture' (iPhone 18 Pro/Pro Max and newer Pro), 'pro', 'air', 'iphone', 'other', 'unknown'."""
    if not model:
        return "unknown"
    m = model.lower()
    if "iphone" not in m:
        return "other"
    if "air" in m:
        return "air"
    num = re.search(r"iphone\s*(\d+)", m)
    gen = int(num.group(1)) if num else 0
    if "pro" in m:
        return "pro_var_aperture" if gen >= 18 else "pro"
    return "iphone"


def assess(exifs: list[dict]) -> dict:
    """Summarise phones/settings and produce warnings + advice (Dutch)."""
    models = sorted({e["model"] for e in exifs if e.get("model")})
    tiers = {phone_tier(m) for m in models} or {"unknown"}
    fn = np.array([e["f_number"] for e in exifs if e.get("f_number")], float)
    ex = np.array([e["exposure_s"] for e in exifs if e.get("exposure_s")], float)
    iso = np.array([e["iso"] for e in exifs if e.get("iso")], float)
    n_raw = sum(1 for e in exifs if e.get("raw"))
    n = max(len(exifs), 1)
    warn, advice = [], []
    if "air" in tiers:
        warn.append("iPhone Air herkend: dit werkt, maar verwacht een lagere nauwkeurigheid dan met een iPhone 18 Pro (Max). "
                    "De Air heeft een vast diafragma ƒ/1.6 (weinig scherptediepte), geen ProRAW en geen handmatige sluitertijd.")
        advice.append("iPhone Air: fotografeer van iets verder weg (meer scherptediepte), in goed licht (korte sluitertijd) en "
                      "zet focus en belichting vast (lang tikken, AE/AF-vergrendeling).")
    if tiers & {"iphone", "pro"}:
        warn.append(f"{', '.join(models)}: werkt; met een iPhone 18 Pro (Max) haal je meer nauwkeurigheid (variabel diafragma tot ƒ/4).")
    if "unknown" in tiers and not models:
        warn.append("Toestel onbekend (geen EXIF in de beelden). Advies over instellingen niet mogelijk.")
    if "pro_var_aperture" in tiers:
        if len(fn) and np.median(fn) < 2.8:
            advice.append(f"Diafragma was ƒ/{np.median(fn):.1f}. Zet in de Pro-bediening ƒ/4.0: veel meer scherptediepte, scherpere randen.")
        if n_raw < 0.5 * n:
            advice.append("Fotografeer in ProRAW (DNG, 48MP): minder kunstmatige verscherping en ruisonderdrukking.")
    if len(ex) and np.median(ex) > 1 / 125:
        advice.append(f"Sluitertijd was ~1/{1 / np.median(ex):.0f} s: kans op bewegingsonscherpte. Gebruik ≤ 1/250 s of een statief.")
    if len(iso) and np.median(iso) > 1600:
        advice.append(f"ISO was ~{np.median(iso):.0f}: veel ruis. Meer licht (bouwlampen) of statief met langere sluitertijd.")
    return {"models": models, "tiers": sorted(tiers), "f_number_median": float(np.median(fn)) if len(fn) else None,
            "exposure_median_s": float(np.median(ex)) if len(ex) else None, "iso_median": float(np.median(iso)) if len(iso) else None,
            "raw_fraction": n_raw / n, "warnings": warn, "advice": advice}


def focal_prior_px(ex: dict, w: int, h: int):
    f35 = ex.get("focal_35mm")
    if f35:
        return float(f35) / 43.27 * float(np.hypot(w, h))  # 35 mm equivalent is defined on the diagonal
    return None
