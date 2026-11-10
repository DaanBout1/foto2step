"""ArUco marker generation (printable PDF at exact scale) and sub-pixel detection."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def get_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Onbekende ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_image(dict_name: str, marker_id: int, px: int = 600) -> np.ndarray:
    d = get_dictionary(dict_name)
    return cv2.aruco.generateImageMarker(d, int(marker_id), px, borderBits=1)


def make_detector(dict_name: str):
    d = get_dictionary(dict_name)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 7
    p.cornerRefinementMaxIterations = 60
    p.cornerRefinementMinAccuracy = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.minMarkerPerimeterRate = 0.01
    p.errorCorrectionRate = 0.4  # strenger dan standaard -> minder valse ID's
    return cv2.aruco.ArucoDetector(d, p)


def detect(image_bgr_or_gray: np.ndarray, detector) -> dict[int, np.ndarray]:
    """Return {id: 4x2 corners (pixel, OpenCV convention: pixel centre = integer)}.

    Corners order: top-left, top-right, bottom-right, bottom-left of the marker as printed.
    Markers seen twice in one image (ambiguous) are dropped.
    """
    img = image_bgr_or_gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(img)
    out: dict[int, np.ndarray] = {}
    if ids is None:
        return out
    seen = {}
    for c, i in zip(corners, ids.ravel()):
        seen.setdefault(int(i), []).append(c.reshape(4, 2).astype(np.float64))
    for i, cs in seen.items():
        if len(cs) == 1:
            # extra sub-pixel refinement on the full-resolution image
            c = cs[0].astype(np.float32).reshape(-1, 1, 2)
            side = np.linalg.norm(cs[0][0] - cs[0][1])
            win = int(np.clip(side / 12, 3, 11))
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.005)
            try:
                cv2.cornerSubPix(img, c, (win, win), (-1, -1), crit)
            except cv2.error:
                pass
            out[i] = c.reshape(4, 2).astype(np.float64)
    return out


def marker_object_corners(size_mm: float) -> np.ndarray:
    """Corners in marker frame (x right, y up on the printed sheet, z out of paper)."""
    h = size_mm / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def write_marker_pdf(path: str | Path, dict_name: str, ids, size_mm: float, title: str = "") -> Path:
    """Write an A4 PDF with markers at exact physical size (print at 100%, no fit-to-page)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from PIL import Image

    path = Path(path)
    page_w, page_h = A4
    margin = 12 * mm
    quiet = max(size_mm * 0.25, 8.0)  # witte rand rondom (mm)
    cell = size_mm + 2 * quiet
    label_h = 7.0
    cols = max(1, int((page_w / mm - 2 * 12) // cell))
    rows = max(1, int((page_h / mm - 2 * 12 - 30) // (cell + label_h)))
    per_page = cols * rows
    c = canvas.Canvas(str(path), pagesize=A4)
    ids = list(ids)
    for p0 in range(0, len(ids), per_page):
        chunk = ids[p0:p0 + per_page]
        c.setFont("Helvetica", 8)
        c.drawString(margin, page_h - margin + 2, f"foto2step markers {dict_name}  zijde zwart vierkant = {size_mm:.1f} mm   {title}")
        c.drawString(margin, page_h - margin - 8,
                     "PRINT OP 100% (geen 'passend maken'). Controleer de 100 mm lijn hieronder en meet elke marker na.")
        # 100 mm reference
        y_ref = page_h - margin - 18
        c.setLineWidth(0.6)
        c.line(margin, y_ref, margin + 100 * mm, y_ref)
        for k in range(0, 101, 10):
            c.line(margin + k * mm, y_ref - (3 if k % 50 == 0 else 1.5), margin + k * mm, y_ref + (3 if k % 50 == 0 else 1.5))
        c.drawString(margin + 102 * mm, y_ref - 3, "100 mm")
        top = page_h - margin - 30
        for k, mid in enumerate(chunk):
            r, col = divmod(k, cols)
            x0 = margin + col * cell * mm
            y0 = top - (r + 1) * (cell + label_h) * mm
            img = marker_image(dict_name, mid, 800)
            pil = Image.fromarray(img)
            c.drawImage(ImageReader(pil), x0 + quiet * mm, y0 + (quiet + label_h) * mm, size_mm * mm, size_mm * mm)
            c.setFont("Helvetica", 7)
            c.drawString(x0 + quiet * mm, y0 + (label_h - 1) * mm, f"ID {mid}  ({size_mm:.1f} mm)")
            # cut marks
            c.setLineWidth(0.2)
            c.rect(x0, y0, cell * mm, (cell + label_h) * mm)
        c.showPage()
    c.save()
    return path


def refine_edges(gray: np.ndarray, corners_px: np.ndarray, to_norm=None, min_contrast=25.0):
    """Unbiased marker corners from the four straight edges of the black square.

    For each side, the edge is located at many points along it as the 50% intensity crossing between
    the dark border (inside) and the white quiet zone (outside); for a symmetric blur this is unbiased,
    unlike corner detectors on L-shaped corners (which pull inward). The edge points are converted with
    `to_norm` (pixel -> undistorted normalised coords, e.g. COLMAP cam_from_img) so lens distortion is
    removed before straight lines are fitted. Corners = intersections of adjacent lines.

    Returns 4x2 corners in the coordinate system of to_norm (pixels if to_norm is None) or None.
    """
    from scipy.ndimage import map_coordinates
    c = np.asarray(corners_px, float)
    center = c.mean(0)
    g = gray.astype(np.float32)
    lines = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        if L < 12:
            return None
        u = d / L
        n = np.array([u[1], -u[0]])
        if n @ ((a + b) / 2 - center) < 0:
            n = -n  # outward
        w = max(2.0, 0.8 * L / 7.0)  # stay within the one-bit black border inside
        ns = int(np.clip(L / 2, 8, 80))
        ts = np.linspace(0.18, 0.82, ns)
        offs = np.linspace(-w, w, int(np.ceil(2 * w * 4)) + 1)
        P = a[None, :] + ts[:, None] * d[None, :]
        S = P[:, None, :] + offs[None, :, None] * n[None, None, :]
        prof = map_coordinates(g, [S[..., 1].ravel(), S[..., 0].ravel()], order=1, mode="nearest").reshape(S.shape[:2])
        m = max(2, len(offs) // 6)
        dark = np.median(prof[:, :m], axis=1)
        light = np.median(prof[:, -m:], axis=1)
        good = (light - dark) > min_contrast
        pts = []
        for i in np.where(good)[0]:
            thr = 0.5 * (dark[i] + light[i])
            pr = prof[i]
            idx = np.where((pr[:-1] < thr) & (pr[1:] >= thr))[0]
            if len(idx) != 1:
                continue
            j = idx[0]
            f = (thr - pr[j]) / max(pr[j + 1] - pr[j], 1e-6)
            o = offs[j] + f * (offs[j + 1] - offs[j])
            pts.append(P[i] + o * n)
        if len(pts) < 5:
            return None
        pts = np.asarray(pts)
        if to_norm is not None:
            pts = np.asarray(to_norm(pts))
            if pts.shape[0] < 5 or not np.all(np.isfinite(pts)):
                return None
        mu = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - mu)
        dirv = Vt[0]
        res = (pts - mu) @ Vt[1]
        # one robust reweighting pass
        s = 1.4826 * np.median(np.abs(res)) + 1e-12
        keep = np.abs(res) < 3 * s
        if keep.sum() >= 5:
            mu = pts[keep].mean(0)
            _, _, Vt = np.linalg.svd(pts[keep] - mu)
            dirv = Vt[0]
        lines.append((mu, dirv))
    out = []
    for k in range(4):
        (p1, d1), (p2, d2) = lines[(k - 1) % 4], lines[k]
        A = np.c_[d1, -d2]
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t[0] * d1)
    return np.array(out)
