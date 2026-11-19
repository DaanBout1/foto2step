"""Strict re-fusion of the OpenMVS depth maps for one part.

OpenMVS fuses a point as soon as 3 views roughly agree. On shiny metal that lets through
reflection ghosts (a second layer in front of the real surface), fans of points along viewing rays
at depth edges, and double layers. Here the depth maps (depthXXXX.dmap, kept by DensifyPointCloud)
are fused again, only inside the part's bounding box, with stricter rules:

* a point counts only if >= min_views OTHER views measured the same surface: depth within
  rel_depth (relative), normals within max_angle_deg, back-projection within 1.5 px;
* only views with a real baseline (>= min_baseline_deg) count: almost identical views repeat the
  same mistake;
* free-space test: a view that measured a surface clearly BEHIND the point looked straight
  through it, so the point is a ghost (at most max_violation_frac of the agreeing views);
* the fused point is the mean of the agreeing measurements (less noise).
On the test_01 block this halved the plane noise (0.48 -> 0.25-0.29 mm) and removed the reflection
layer and the fans.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .log import log

DEFAULTS = {"min_views": 3, "rel_depth": 0.004, "max_angle_deg": 20.0, "views": 30, "min_baseline_deg": 8.0,
            "max_violation_frac": 0.3, "stride": 2, "voxel_mm": 0.3, "margin_mm": 8.0}


def read_dmap_header(fn):
    """OpenMVS raw depth-map header: camera (K, R, C: world->camera x = R (X - C)), sizes, view IDs."""
    with open(fn, "rb") as f:
        name, typ, _pad, iw, ih, dw, dh, dmin, dmax = struct.unpack("<HBBIIIIff", f.read(28))
        if name != 0x5244:
            raise ValueError(f"{fn}: geen OpenMVS depth map")
        n = struct.unpack("<H", f.read(2))[0]
        img = f.read(n).decode(errors="replace")
        nid = struct.unpack("<I", f.read(4))[0]
        ids = np.array(struct.unpack("<%dI" % nid, f.read(4 * nid)))
        K = np.frombuffer(f.read(72), "<f8").reshape(3, 3).copy()
        R = np.frombuffer(f.read(72), "<f8").reshape(3, 3).copy()
        C = np.frombuffer(f.read(24), "<f8").copy()
        off = f.tell()
    return dict(type=typ, iw=iw, ih=ih, dw=dw, dh=dh, img=img, ids=ids, K=K, R=R, C=C, offset=off)


def crop_depthmaps(dense_dir: Path, lo, hi, cache: Path):
    """Cut every depth map down to the projection of the box [lo, hi] (depth, normal f16)."""
    cache.mkdir(parents=True, exist_ok=True)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    n_ok = 0
    for fn in sorted(Path(dense_dir).glob("depth*.dmap")):
        o = cache / (fn.stem + ".npz")
        if o.exists():
            n_ok += 1
            continue
        h = read_dmap_header(fn)
        xc = (corners - h["C"]) @ h["R"].T
        dw, dh = h["dw"], h["dh"]
        if np.any(xc[:, 2] <= 0):
            r0, r1, c0, c1 = 0, dh, 0, dw
        else:
            uv = xc @ h["K"].T
            uv = uv[:, :2] / uv[:, 2:]
            sx, sy = dw / h["iw"], dh / h["ih"]
            c0 = int(max(0, np.floor(uv[:, 0].min() * sx) - 2))
            c1 = int(min(dw, np.ceil(uv[:, 0].max() * sx) + 3))
            r0 = int(max(0, np.floor(uv[:, 1].min() * sy) - 2))
            r1 = int(min(dh, np.ceil(uv[:, 1].max() * sy) + 3))
        if r1 <= r0 or c1 <= c0:
            np.savez(o, empty=1)
            continue
        mm = np.memmap(fn, dtype="<f4", mode="r", offset=h["offset"])
        N = dw * dh
        D = np.array(mm[:N].reshape(dh, dw)[r0:r1, c0:c1])
        Nr = (np.array(mm[N:4 * N].reshape(dh, dw, 3)[r0:r1, c0:c1]).astype(np.float16)
              if h["type"] & 2 else np.zeros((0,), np.float16))
        del mm
        # depth-map pixels may be scaled w.r.t. the image (resolution level): scale K to the map
        K = h["K"].copy()
        K[0] *= dw / h["iw"]
        K[1] *= dh / h["ih"]
        np.savez(o, D=D, N=Nr, K=K, R=h["R"], C=h["C"], ids=h["ids"], roi=np.array([r0, r1, c0, c1]))
        n_ok += 1
    return n_ok


def fuse_box(cache: Path, lo, hi, p=None):
    """Strict fusion of the cropped depth maps; returns (points, weights) inside [lo, hi]."""
    p = {**DEFAULTS, **(p or {})}
    files = sorted(Path(cache).glob("depth*.npz"))
    cams = {}
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        if "empty" in z:
            continue
        cams[fn] = dict(K=z["K"], R=z["R"], C=z["C"], ids=z["ids"], roi=z["roi"])
    names = list(cams)
    if not names:
        return np.zeros((0, 3)), np.zeros(0)
    Cs = np.array([cams[n]["C"] for n in names])
    idnum = {int(n.stem[5:]): n for n in names}
    cache_d = {}

    def load(n):
        if n not in cache_d:
            if len(cache_d) > 40:
                cache_d.pop(next(iter(cache_d)))
            z = np.load(n, allow_pickle=True)
            cache_d[n] = (z["D"], z["N"].astype(np.float32) if z["N"].ndim == 3 else None)
        return cache_d[n]

    cos_a = np.cos(np.radians(p["max_angle_deg"]))
    cos_b = np.cos(np.radians(p["min_baseline_deg"]))
    tau = p["rel_depth"]
    st = int(p["stride"])
    out_p, out_w = [], []
    for n in names:
        c = cams[n]
        D, Nm = load(n)
        r0, r1, c0, c1 = c["roi"]
        vv, uu = np.mgrid[r0:r1:st, c0:c1:st]
        d = D[vv - r0, uu - c0]
        m = d > 0
        vv, uu, d = vv[m], uu[m], d[m]
        ray = np.c_[uu, vv, np.ones_like(uu)].astype(np.float64) @ np.linalg.inv(c["K"]).T
        X = c["C"] + (ray * d[:, None]) @ c["R"]
        nw = Nm[vv - r0, uu - c0] @ c["R"] if Nm is not None else None
        ib = np.all((X > lo) & (X < hi), 1)
        X = X[ib]
        nw = nw[ib] if nw is not None else None
        if len(X) == 0:
            continue
        nb = [idnum[k] for k in c["ids"][1:] if k in idnum and idnum[k] != n]
        for k in np.argsort(np.linalg.norm(Cs - c["C"], axis=1)):
            if len(nb) >= p["views"]:
                break
            if names[k] != n and names[k] not in nb:
                nb.append(names[k])
        S = X.copy()
        cnt = np.ones(len(X))
        viol = np.zeros(len(X))
        ri = X - c["C"]
        ri_n = ri / np.linalg.norm(ri, axis=1, keepdims=True)
        pi = (ri @ c["R"].T) @ c["K"].T
        pi = pi[:, :2] / pi[:, 2:]
        for nj in nb[: p["views"]]:
            cj = cams[nj]
            Dj, Nj = load(nj)
            a0, a1, b0, b1 = cj["roi"]
            rj = X - cj["C"]
            cang = np.sum(ri_n * rj, 1) / np.linalg.norm(rj, axis=1)
            xc = rj @ cj["R"].T
            z = xc[:, 2]
            uv = xc @ cj["K"].T
            ui = np.round(uv[:, 0] / uv[:, 2]).astype(int)
            vi = np.round(uv[:, 1] / uv[:, 2]).astype(int)
            ok = (z > 0) & (ui >= b0) & (ui < b1) & (vi >= a0) & (vi < a1) & (cang < cos_b)
            idx = np.where(ok)[0]
            dj = Dj[vi[idx] - a0, ui[idx] - b0]
            g = dj > 0
            idx, dj = idx[g], dj[g]
            viol[idx[dj > z[idx] * (1 + 3 * tau)]] += 1
            g = np.abs(dj - z[idx]) < tau * z[idx]
            idx, dj = idx[g], dj[g]
            rr = np.c_[ui[idx], vi[idx], np.ones(len(idx))] @ np.linalg.inv(cj["K"]).T
            Xj = cj["C"] + (rr * dj[:, None]) @ cj["R"]
            ur = ((Xj - c["C"]) @ c["R"].T) @ c["K"].T
            ur = ur[:, :2] / ur[:, 2:]
            g = np.linalg.norm(ur - pi[idx], axis=1) < 1.5
            if nw is not None and Nj is not None:
                g &= np.sum((Nj[vi[idx] - a0, ui[idx] - b0] @ cj["R"]) * nw[idx], 1) > cos_a
            idx, Xj = idx[g], Xj[g]
            S[idx] += Xj
            cnt[idx] += 1
        keep = (cnt >= p["min_views"] + 1) & (viol <= np.maximum(1, p["max_violation_frac"] * (cnt - 1)))
        out_p.append(S[keep] / cnt[keep, None])
        out_w.append(cnt[keep])
    P = np.concatenate(out_p) if out_p else np.zeros((0, 3))
    W = np.concatenate(out_w) if out_w else np.zeros(0)
    # merge the per-view duplicates on a fine grid (weighted mean)
    vox = float(p["voxel_mm"])
    if len(P):
        k = np.floor((P - P.min(0)) / vox).astype(np.int64)
        key = (k[:, 0] * 200003 + k[:, 1]) * 200003 + k[:, 2]
        u, inv = np.unique(key, return_inverse=True)
        Ssum = np.zeros((len(u), 3))
        np.add.at(Ssum, inv, P * W[:, None])
        Wsum = np.bincount(inv, W)
        P, W = Ssum / Wsum[:, None], Wsum
    return P, W


def strict_points(dense_dir: Path, body_pts: np.ndarray, cache: Path, params=None, zmin=None):
    """Strictly fused points for one part: inside its box, and on (near) its own surface."""
    from scipy.spatial import cKDTree
    p = {**DEFAULTS, **(params or {})}
    if not any(Path(dense_dir).glob("depth*.dmap")):
        return None
    lo = body_pts.min(0) - p["margin_mm"]
    hi = body_pts.max(0) + p["margin_mm"]
    n = crop_depthmaps(dense_dir, lo, hi, cache)
    P, W = fuse_box(cache, lo, hi, p)
    if len(P) == 0:
        return None
    # keep what lies on this part (not the floor, not a neighbour part)
    d, _ = cKDTree(body_pts).query(P, distance_upper_bound=float(p.get("near_mm", 4.0)))
    keep = np.isfinite(d)
    if zmin is not None:  # floor clearance as for the part's own points
        keep &= P[:, 2] >= zmin
    P = P[keep]
    log(f"  strikte fusie: {n} dieptekaarten, {len(P):,} punten op het deel (was {len(body_pts):,})")
    return P
