"""Profile (extrusion) recognition: the first "finding" about a part.

Many machine parts are, wholly or partly, one 2D cross-section pushed along a straight line: bar
stock, rails and T-slot profiles, beams, milled blocks with grooves along their length, guides. Then
every surface with its normal perpendicular to that direction belongs to the profile, and the
profile is the same at every position along it.

Classical reverse engineering recognises such "translational" surfaces from the normals (Pottmann et
al., line geometry for kinematic surfaces): the normals of an extrusion all lie perpendicular to
its direction. Here:

1. find_extrusions: for each axis of the part's frame, take the points whose normal is
   perpendicular to it and test that their 2D cross-section is the same in slabs along the axis;
2. fit_profile: collapse those points onto one 2D section (all positions along the length
   together: a cove seen only on its lower half anywhere along the 100 mm still gives one clean
   arc) and fit lines and arcs (RANSAC with normals, line-vs-arc model selection);
3. profile_primitives: each 2D line becomes a plane and each arc a cylinder along the direction.
   They go into the same cell-complex solid builder as all other surfaces, so a part can be an
   extrusion in one direction and have pockets, holes or lugs in another.
"""
from __future__ import annotations

import numpy as np

from .log import log

# Let a clear profile overrule the general segmentation along its direction. Off by default: in the
# gauntlet it fixed the T-slot rail but cost detail on prismatic parts (bracket, shallow steps).
import os as _os
AUTHORITY = _os.environ.get("FOTO2STEP_PROFILE_AUTHORITY", "0") == "1"
# free-form profile curves (several arcs in a row) as one extruded spline surface
SPLINES = _os.environ.get("FOTO2STEP_PROFILE_SPLINES", "1") == "1"


def _raster(X, cell, lo, shape):
    ij = np.floor((X - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    R = np.zeros(shape, bool)
    R[ij[ok, 0], ij[ok, 1]] = True
    return R


def _dilate(R):
    D = R.copy()
    D[1:] |= R[:-1]
    D[:-1] |= R[1:]
    D[:, 1:] |= R[:, :-1]
    D[:, :-1] |= R[:, 1:]
    return D


def find_extrusions(P, N, axes, noise_mm, min_frac=0.25, min_len=20.0, n_slabs=4, cell=1.5):
    """Directions along which (part of) the part is a constant profile.
    Returns [{axis, mask, frac, length, consistency, t0, t1}] sorted by support."""
    from .cadfit import _perp_basis
    out = []
    for a in axes:
        m = np.abs(N @ a) < np.sin(np.radians(10))
        if m.mean() < min_frac:
            continue
        # the true extrusion direction is the one most perpendicular to all these normals (the
        # part may stand a little tilted w.r.t. the frame): smallest eigenvector of sum(n n^T).
        # 1 deg off smears a 100 mm long profile by 1.7 mm.
        a0 = a.copy()
        for _ in range(3):
            Nm = N[m]
            ev, evec = np.linalg.eigh(Nm.T @ Nm)
            a_new = evec[:, 0] * np.sign(evec[:, 0] @ a0)
            if a_new @ a0 < np.cos(np.radians(5)):
                break  # not near this axis: keep the frame axis
            a = a_new
            m = np.abs(N @ a) < np.sin(np.radians(10))
        t = P[m] @ a
        t0, t1 = np.percentile(t, [1, 99])
        if t1 - t0 < min_len:
            continue
        u, v = _perp_basis(a)
        X = np.c_[P[m] @ u, P[m] @ v]
        lo = X.min(0) - cell
        shape = tuple(np.ceil((X.max(0) - lo) / cell).astype(int) + 2)
        edges = np.linspace(t0, t1, n_slabs + 1)
        R = [_raster(X[(t >= edges[k]) & (t < edges[k + 1])], cell, lo, shape) for k in range(n_slabs)]
        # how much of each slab's section is also present (within one cell) in the other slabs
        sc = []
        for i in range(n_slabs):
            for j in range(n_slabs):
                if i != j and R[i].sum() > 0:
                    sc.append((R[i] & _dilate(R[j])).sum() / R[i].sum())
        cons = float(np.median(sc)) if sc else 0.0
        out.append({"axis": a, "mask": m, "frac": float(m.mean()), "length": float(t1 - t0),
                    "consistency": cons, "t0": float(t0), "t1": float(t1)})
    out.sort(key=lambda e: -e["frac"] * e["consistency"])
    return out


def _grid_mean(X, NX, cell):
    k = np.floor(X / cell).astype(np.int64)
    key = k[:, 0] * 10_000_019 + k[:, 1]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    Xs = np.zeros((len(u), 2))
    Ns = np.zeros((len(u), 2))
    np.add.at(Xs, inv, X)
    np.add.at(Ns, inv, NX)
    Xs /= cnt[:, None]
    Ns /= np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-12)
    return Xs, Ns, cnt.astype(float)


def _curve_samples(X, NX, noise_mm, h=None, cell=0.4, max_pts=250_000, seed=0):
    """Thin, precise samples of the section curve (moving least squares).

    After collapsing the extrusion every spot of the section holds many points. Each sample is
    projected onto the local weighted best-fit line of its neighbours (twice, as MLS does), which
    puts it on the curve to a fraction of the scan noise. Direction and curvature then come from
    the neighbouring projected samples (a small circle/line fit along the thin curve).
    Returns Y (on-curve points), T (tangents), Nn (normals, oriented like the data), K (signed
    curvature, + = convex) and S (support counts)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(X) > max_pts:
        sel = rng.choice(len(X), max_pts, replace=False)
        X, NX = X[sel], NX[sel]
    h = h or max(1.5, 3.0 * noise_mm)
    Xs, Ns, W = _grid_mean(X, NX, cell)
    tree = cKDTree(X)
    Y = Xs.copy()
    Nn = Ns.copy()
    S = np.zeros(len(Xs))
    good = np.ones(len(Xs), bool)
    for _it in range(2):
        nbs = tree.query_ball_point(Y, h)
        for i, nb in enumerate(nbs):
            if not good[i]:
                continue
            nb = np.asarray(nb, int)
            if len(nb):
                nb = nb[NX[nb] @ Ns[i] > 0.5]  # same side of a thin wall only
            if len(nb) < 30:
                good[i] = False
                continue
            D = X[nb] - Y[i]
            w = np.exp(-0.5 * (np.linalg.norm(D, axis=1) / (0.5 * h)) ** 2)
            mu = (w[:, None] * D).sum(0) / w.sum()
            C = ((D - mu) * w[:, None]).T @ (D - mu)
            ev, evec = np.linalg.eigh(C)
            n = evec[:, 0]
            if n @ Ns[i] < 0:
                n = -n
            Y[i] = Y[i] + (mu @ n) * n  # onto the local line, along its normal
            Nn[i] = n
            S[i] = len(nb)
    Y, Nn, S = Y[good], Nn[good], S[good]
    if len(Y) < 5:
        z = np.zeros((0, 2))
        return z, z, z, np.zeros(0), np.zeros(0)
    # thin out duplicates, then curvature along the thin curve
    k = np.floor(Y / (0.5 * cell)).astype(np.int64)
    _, first = np.unique(k[:, 0] * 10_000_019 + k[:, 1], return_index=True)
    Y, Nn, S = Y[first], Nn[first], S[first]
    ty = cKDTree(Y)
    K = np.zeros(len(Y))
    for i, nb in enumerate(ty.query_ball_point(Y, max(1.5, h))):
        nb = np.asarray(nb, int)
        nb = nb[Nn[nb] @ Nn[i] > 0.0]
        if len(nb) < 5:
            continue
        D = Y[nb] - Y[i]
        t = np.array([Nn[i][1], -Nn[i][0]])
        sx, o = D @ t, D @ Nn[i]
        A = np.c_[np.ones_like(sx), sx, sx ** 2]
        c = np.linalg.lstsq(A, o, rcond=None)[0]
        K[i] = -2 * c[2] / (1 + c[1] ** 2) ** 1.5  # convex (normal away from the centre) -> positive
    T = np.c_[Nn[:, 1], -Nn[:, 0]]
    return Y, T, Nn, K, S


def _axis_lines(X, NX, noise_mm, min_support, bin_w=0.1):
    """Straight profile edges square to the part's axes (nearly all of them on machined parts and
    extrusions): for each axis direction and facing side, the histogram of the points' coordinate
    has a peak at every such edge. Over the whole extrusion length that peak holds thousands of
    points, so even a 3 mm wall is found and placed to a small fraction of the noise."""
    from scipy.ndimage import gaussian_filter1d
    eps = max(2.0 * noise_mm, 0.4)
    out = []
    for e in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        for sg in (1.0, -1.0):
            n = sg * e
            m = NX @ n > np.cos(np.radians(35))  # loose: short walls get blended normals; curved pieces are rejected below
            if m.sum() < min_support:
                continue
            c = X[m] @ n
            lo, hi = c.min() - 1, c.max() + 1
            h, edges = np.histogram(c, bins=int(np.ceil((hi - lo) / bin_w)), range=(lo, hi))
            hs = gaussian_filter1d(h.astype(float), max(1.0, 0.5 * noise_mm / bin_w))
            ctr = 0.5 * (edges[:-1] + edges[1:])
            # local maxima, strongest first, at least eps apart
            pk = np.where((hs[1:-1] >= hs[:-2]) & (hs[1:-1] > hs[2:]))[0] + 1
            pk = pk[np.argsort(-hs[pk])]
            taken = []
            for p in pk:
                pos = ctr[p]
                if any(abs(pos - t) < eps for t in taken):
                    continue
                sel = np.abs(c - pos) < eps
                if sel.sum() < min_support:
                    continue
                pos = float(np.median(c[sel]))
                # the points must form a real piece of edge (>= 1.5 mm without big gaps)
                tdir = np.array([-n[1], n[0]])
                tq = X[m][sel] @ tdir
                oq = c[sel] - pos
                order = np.argsort(tq)
                tq, oq = tq[order], oq[order]
                cuts = np.where(np.diff(tq) > 1.0)[0] + 1
                spans = []
                for ts, os_ in zip(np.split(tq, cuts), np.split(oq, cuts)):
                    if len(ts) < 8 or ts[-1] - ts[0] < 3.0:
                        continue
                    # a straight piece: no curvature along it (the flattest part of a bore is curved).
                    # Judged without its ends: where a groove starts, its first mm stays within the
                    # band of the face and bends the ends of an otherwise straight piece
                    trim = min(1.5, 0.2 * (ts[-1] - ts[0]))
                    mi = (ts > ts[0] + trim) & (ts < ts[-1] - trim)
                    if mi.sum() < 8:
                        continue
                    tc = ts[mi] - ts[mi].mean()
                    A = np.c_[np.ones_like(tc), tc, tc ** 2]
                    coef, *_ = np.linalg.lstsq(A, os_[mi], rcond=None)
                    res = os_[mi] - A @ coef
                    s2 = float(res @ res) / max(1, len(tc) - 3)
                    try:
                        sd_k2 = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[2, 2]))
                    except np.linalg.LinAlgError:
                        continue
                    if abs(2 * coef[2]) > 0.03 or (abs(2 * coef[2]) > 0.01 and abs(coef[2]) > 3 * sd_k2):
                        continue  # curved (the flank of a bore) or significantly curved (a gentle cove)
                    if abs(coef[1]) > np.tan(np.radians(8)):
                        continue  # runs across the band at an angle: a slope, not this edge
                    spans.append((ts[0], ts[-1]))
                if not spans:
                    continue
                taken.append(pos)
                out.append((("line", n.copy(), -pos), float(sel.sum()), spans))
    return out


def _chains(Y, Nn, idx, link=0.9):
    """Order curve samples (thin) into chains: per connected piece, the longest path of its
    minimum spanning tree (branches dropped)."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, breadth_first_order, dijkstra
    if len(idx) < 3:
        return []
    Z = Y[idx]
    pairs = cKDTree(Z).query_pairs(link, output_type="ndarray")
    if len(pairs) == 0:
        return []
    ok = np.sum(Nn[idx][pairs[:, 0]] * Nn[idx][pairs[:, 1]], 1) > np.cos(np.radians(40))
    pairs = pairs[ok]
    w = np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1) + 1e-6
    G = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(len(Z), len(Z))).tocsr()
    nc, lab = connected_components(G, directed=False)
    out = []
    T = minimum_spanning_tree(G)
    T = T + T.T
    for k in range(nc):
        comp = np.where(lab == k)[0]
        if len(comp) < 4:
            continue
        d0 = dijkstra(T, indices=comp[0])
        a = comp[np.argmax(np.where(np.isfinite(d0[comp]), d0[comp], -1))]
        da, pred = dijkstra(T, indices=a, return_predecessors=True)
        b = comp[np.argmax(np.where(np.isfinite(da[comp]), da[comp], -1))]
        path = [b]
        while path[-1] != a and pred[path[-1]] >= 0:
            path.append(pred[path[-1]])
        if len(path) >= 4:
            out.append(idx[np.array(path[::-1])])
    return out


def _smooth_polyline(Z, Nz, step=0.25, win=2.0):
    """Ordered thin samples -> smooth, evenly resampled polyline with normals (oriented like Nz)."""
    if len(Z) < 5:
        return None
    seg = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 2.0:
        return None
    ss = np.arange(0, s[-1] + 1e-9, step)
    P = np.c_[np.interp(ss, s, Z[:, 0]), np.interp(ss, s, Z[:, 1])]
    k = max(1, int(round(win / step)))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    pad = np.r_[np.repeat(P[:1], k, 0), P, np.repeat(P[-1:], k, 0)]
    P = np.c_[np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")]
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    N = np.c_[-T[:, 1], T[:, 0]]
    nz = np.c_[np.interp(ss, s, Nz[:, 0]), np.interp(ss, s, Nz[:, 1])]
    if np.sum(N * nz) < 0:
        N = -N
    return P, N


def _arc_spline(Y, chain, tol):
    """Cover an ordered chain of thin samples with as few arcs as possible (greedy, each arc as
    long as a circle stays within tol of its samples). Returns [(c, r, i0, i1)]."""
    arcs = []
    i0 = 0
    n = len(chain)
    while i0 < n - 3:
        j = i0 + 3
        best = None
        while j <= n:
            Z = Y[chain[i0:j]]
            A = np.c_[2 * Z, np.ones(len(Z))]
            sol = np.linalg.lstsq(A, np.sum(Z ** 2, 1), rcond=None)[0]
            c = sol[:2]
            r = float(np.sqrt(max(sol[2] + c @ c, 1e-12)))
            res = np.abs(np.linalg.norm(Z - c, axis=1) - r)
            if res.max() > tol or r > 400:
                break
            best = (c, r, i0, j)
            j += 1
        if best is None:
            i0 += 1
            continue
        arcs.append(best)
        i0 = best[3] - 1
    return arcs


def _g1_runs(Y, chains, noise_mm, min_arc_mm=4.0):
    """Parts of the chains that are smooth free-form curves: split at sharp corners and keep a part
    only if it needs two or more different curved arcs. A groove (one arc between two flat faces,
    sharp edges) or a bore stays an arc; an ogee or a cove with a lip (arcs flowing into each
    other) becomes a spline.
    On the smoothed curve a sharp corner becomes a short, tight arc (about the smoothing window
    long); a real bend of the part is longer than that. (Corners next to a straight face are cut
    off before: the chains stop at the verified straight pieces.)"""
    tol = max(0.25, 0.6 * noise_mm)
    r_min = max(2.5, 8.0 * noise_mm)  # smoothing window + noise round a sharp corner to about this
    sag_min = max(0.3, noise_mm)  # on the smoothed curve: the noise is averaged out
    out = []
    for ch in chains:
        # on the smoothed curve (the thin samples of a real scan still zigzag across the band)
        poly = _smooth_polyline(Y[ch], np.ones((len(ch), 2)))
        if poly is None:
            continue
        Q = poly[0]
        s_ch = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Y[ch], axis=0), axis=1))]
        runs, cur = [], []
        for c, r, i0, i1 in _arc_spline(Q, np.arange(len(Q)), tol):
            Ls = 0.25 * (i1 - i0 - 1)
            bent = r < 150.0 and r * (1 - np.cos(min(Ls / r, np.pi) / 2)) >= sag_min
            if bent and (Ls < min_arc_mm or r < r_min):
                if cur:
                    runs.append(cur)  # a corner
                cur = []
                continue
            cur.append((c, r, i0, i1, bent))
        if cur:
            runs.append(cur)
        for run in runs:
            curved = [(c, r) for c, r, _i0, _i1, bent in run if bent]
            if len(curved) < 2:
                continue
            c0, r0 = curved[0]
            if all(np.linalg.norm(c_ - c0) < max(1.0, 2 * noise_mm) and abs(r_ - r0) < 0.15 * r0
                   for c_, r_ in curved[1:]):
                continue  # one circle split by the greedy cover: an arc
            ext_ = np.ptp(Q[run[0][2]:run[-1][3]], axis=0)
            if ext_.min() < max(3.0, 6.0 * noise_mm):
                continue  # a low step (2 mm pad, 3 mm notch) rounded by the smoothing, not a curve
            # back to the chain samples (the smoothed curve is resampled every 0.25 mm)
            j0 = int(np.searchsorted(s_ch, 0.25 * run[0][2]))
            j1 = int(np.searchsorted(s_ch, 0.25 * (run[-1][3] - 1))) + 1
            if j1 - j0 >= 5:
                out.append(ch[j0:j1])
    return out


def _spline_stretches(X, NX, noise_mm, min_support, lines3, near_line, expl, eps, seed=0,
                      long_mm=6.0, min_len=12.0, min_turn_deg=45.0):
    """Free-form stretches of the section, seen whole: the section is ordered into chains except
    where it runs along a long straight edge (a real face); a chain that turns a lot and is not
    explained by the straight pieces found is one smooth spline curve (a cove, an ogee). Fitting
    it piece by piece would cut it into short lines and arcs that each fit only a part.
    Marks the points it explains in expl. Returns [(("poly", P2, N2), weight, None)]."""
    from scipy.spatial import cKDTree
    Y, T, Nn, K, S = _curve_samples(X, NX, noise_mm, seed=seed)
    if len(Y) < 5:
        return []
    on_long = np.zeros(len(Y), bool)
    for (_, n, d), _w, spans in lines3:
        tdir = np.array([-n[1], n[0]])
        tq = Y @ tdir
        for a0, a1 in spans:
            # every verified straight piece is a face of its own (a groove between two flat
            # stretches is an arc with sharp edges, not part of a free-form curve)
            if a1 - a0 >= long_mm:
                on_long |= (tq > a0 - 0.5) & (tq < a1 + 0.5) & (np.abs(Y @ n + d) < eps)
    # distance to the straight pieces where they exist (not their infinite lines: a line through a
    # cove bottom would claim the whole cove)
    near_span = np.full(len(X), np.inf)
    for (_, n, d), _w, spans in lines3:
        tdir = np.array([-n[1], n[0]])
        tq = X @ tdir
        ins = np.zeros(len(X), bool)
        for a0, a1 in spans:
            ins |= (tq > a0 - 1.0) & (tq < a1 + 1.0)
        near_span = np.minimum(near_span, np.where(ins, np.abs(X @ n + d), np.inf))
    tx = cKDTree(X)
    out = []
    for ch in _g1_runs(Y, _chains(Y, Nn, np.where(~on_long)[0]), noise_mm):
        L = float(np.sum(np.linalg.norm(np.diff(Y[ch], axis=0), axis=1)))
        if L < min_len:
            continue
        poly = _smooth_polyline(Y[ch], Nn[ch])
        if poly is None:
            continue
        P2, N2 = poly
        ang = np.arctan2(N2[:, 1], N2[:, 0])
        step = max(1, int(round(1.0 / 0.25)))
        dth = np.diff(np.unwrap(ang[::step]))
        if np.degrees(np.sum(np.abs(dth))) < min_turn_deg:
            continue
        near = np.unique(np.concatenate(tx.query_ball_point(Y[ch], 0.6)))
        if len(near) < min_support or np.mean(near_span[near] < eps) > 0.6:
            continue  # mostly the straight edges already found (a pocket, a step)
        _, jj = cKDTree(P2).query(X[near])
        res = np.sum((X[near] - P2[jj]) * N2[jj], 1)
        # (the smoothing window rounds the sharp bends of the stretch a little: some slack)
        if np.sqrt(np.mean(res ** 2)) > 2.0 * max(noise_mm, 0.15):
            continue
        out.append((("poly", P2, N2), float(len(near)), None))
        expl[near[np.abs(res) < eps]] = True
    return out


def fit_profile(X, NX, noise_mm, min_support, seed=0):
    """Lines and arcs of a 2D section.

    1. straight edges square to the part's axes from coordinate histograms (all points of the
       length together; every piece must be straight: the flat part of a curve is rejected);
    2. on the points those edges do not explain: thin curve samples with curvature (moving least
       squares); curved stretches are ordered into chains and covered by as few arcs as possible
       (an S-shaped cove becomes 2-3 arcs, a bore one arc). An arc that mostly sits on the edges
       found in 1 is a rounded view of their corner and is dropped;
    3. remaining straight stretches off the axes (chamfers, slopes) by region growing.
    Returns [(model, weight, spans)]."""
    from scipy.spatial import cKDTree
    eps = max(2.0 * noise_mm, 0.4)
    lines3 = _axis_lines(X, NX, noise_mm, min_support)
    expl = np.zeros(len(X), bool)
    near_line = np.full(len(X), np.inf)
    for (_, n, d), _w, spans in lines3:
        tdir = np.array([-n[1], n[0]])
        tq = X @ tdir
        inside = np.zeros(len(X), bool)
        for a0, a1 in spans:
            inside |= (tq > a0 - 1.0) & (tq < a1 + 1.0)
        dl = np.abs(X @ n + d)
        expl |= inside & (dl < eps) & (NX @ n > np.cos(np.radians(30)))
        near_line = np.minimum(near_line, np.where(inside | True, dl, np.inf))
    out = [(m, w, sp) for m, w, sp in lines3]
    if SPLINES:
        out += _spline_stretches(X, NX, noise_mm, min_support, lines3, near_line, expl, eps, seed)
    Xr, NXr = X[~expl], NX[~expl]
    if len(Xr) > 50:
        Y, T, Nn, K, S = _curve_samples(Xr, NXr, noise_mm, seed=seed)
        if len(Y) >= 5:
            tx = cKDTree(X)
            tol_s = max(0.25, 0.6 * noise_mm)
            # every remaining stretch of the section, ordered along the curve, covered by arcs
            # (an arc with a huge radius is a straight piece; free-form stretches were taken whole
            # before)
            polys = [m_[1] for m_, _w, _s in out if m_[0] == "poly"]
            tpoly = cKDTree(np.vstack(polys)) if polys else None
            for ch in _chains(Y, Nn, np.arange(len(Y))):
                if np.linalg.norm(Y[ch[-1]] - Y[ch[0]]) < 2.0:
                    continue
                if tpoly is not None and np.median(tpoly.query(Y[ch])[0]) < 2.0 * eps:
                    continue  # the left-overs along a spline stretch found whole
                arcs = [a_ for a_ in _arc_spline(Y, ch, tol_s) if a_[1] < 150.0]
                for c, r, i0, i1 in arcs:
                    seg = ch[i0:i1]
                    L = np.sum(np.linalg.norm(np.diff(Y[seg], axis=0), axis=1))
                    if L < max(3.0, 1.5 * max(1.5, 3.0 * noise_mm)) or r < max(2.5, 4.0 * noise_mm):
                        continue
                    if L / r < np.radians(30) or r * (1 - np.cos(min(L / r, np.pi) / 2)) < max(1.0, 4 * noise_mm):
                        continue  # too flat (or too small a bend) to be sure it is curved
                    near = np.unique(np.concatenate(tx.query_ball_point(Y[seg], 0.6)))
                    if len(near) < min_support:
                        continue
                    if np.mean(near_line[near] < eps) > 0.6:
                        continue  # a corner of two found edges, rounded by the smoothing window
                    arc = _fit_arc(X[near], NX[near], np.ones(len(near)), eps)
                    if arc is None or _rms(arc, X[near], np.ones(len(near))) > 1.3 * max(noise_mm, 0.15):
                        continue
                    out.append((arc, float(len(near)), None))
    rest = _fit_curves(Xr, NXr, noise_mm, min_support, seed=seed) if len(Xr) > 50 else []
    for m, w in rest:
        if m[0] != "line":
            continue  # curves come from the arc chains above
        sel = np.abs(Xr @ m[1] + m[2]) < eps
        tdir = np.array([-m[1][1], m[1][0]])
        if sel.sum() < 10 or np.ptp(Xr[sel] @ tdir) < 8.0:
            continue
        if max(abs(m[1][0]), abs(m[1][1])) > np.cos(np.radians(10)):
            continue
        out.append((m, w, None))
    return out


def _fit_curves(X, NX, noise_mm, min_support, seed=0):
    """Lines and arcs in a 2D section (points X with oriented 2D normals NX).

    1. precise curve samples with direction and curvature (moving least squares);
    2. region growing along the curve: neighbouring samples with the same direction trend and the
       same curvature; a sharp corner breaks a region (curvature spike, direction jump);
    3. a region with ~zero curvature is a line, constant curvature an arc (r = 1/k); each is refitted
       on the original points along it and must fit at the noise level;
    4. collinear lines / co-circular arcs are merged (one infinite plane / cylinder in 3D).
    Returns [(("line", n, d), weight)] with n.x + d = 0 (n outward) and [(("arc", c, r, s), weight)]."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    eps = max(2.0 * noise_mm, 0.4)
    tol = 1.3 * max(noise_mm, 0.15)
    Y, T, Nn, K, S = _curve_samples(X, NX, noise_mm, seed=seed)
    if len(Y) < 5:
        return []
    k_line = 1.0 / 400.0
    pairs = cKDTree(Y).query_pairs(0.9, output_type="ndarray")
    if len(pairs) == 0:
        return []
    i, j = pairs[:, 0], pairs[:, 1]
    dang = np.abs(np.sum(Nn[i] * Nn[j], 1))
    dk = np.abs(K[i] - K[j])
    dist = np.linalg.norm(Y[i] - Y[j], axis=1)
    # flat and curved samples are grown separately: a corner (rounded by the MLS window into a
    # short, strongly curved piece) must not glue two faces into one region
    k_flat = 0.04
    flat = np.abs(K) < k_flat
    kmax = np.maximum(np.abs(K[i]), np.abs(K[j]))
    allowed = np.radians(12) + kmax * dist  # on an arc the normal turns by k * distance
    ok = (np.arccos(np.clip(np.sum(Nn[i] * Nn[j], 1), -1, 1)) < allowed) & (flat[i] == flat[j])
    ok &= flat[i] | ((np.sign(K[i]) == np.sign(K[j])) & (dk < 0.3 * kmax))
    G = coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(len(Y), len(Y)))
    nc, lab = connected_components(G, directed=False)
    treeX = cKDTree(X)
    models = []
    for r in range(nc):
        idx = np.where(lab == r)[0]
        if len(idx) < 4 or np.ptp(Y[idx], axis=0).max() < 1.5:
            continue
        kk = float(np.median(K[idx]))
        if not flat[idx[0]]:
            # a curved piece must be longer than the smoothing window and its radius larger than it:
            # otherwise it is a sharp corner seen through the window
            arclen = np.ptp(Y[idx], axis=0).max()
            if arclen < max(3.0, 1.5 * max(1.5, 3.0 * noise_mm)) or 1.0 / max(abs(kk), 1e-9) < max(2.0, 4.0 * noise_mm):
                continue
        # original points along this piece of curve
        near = np.unique(np.concatenate(treeX.query_ball_point(Y[idx], 0.5)))
        if len(near) < min_support:
            continue
        Xr, Nr = X[near], NX[near]
        Wr = np.ones(len(near))
        cand = []
        n, d = _line_fit(Xr, Wr)
        if np.sum(Nr @ n) < 0:
            n, d = -n, -d
        cand.append(("line", n, d))
        if abs(kk) > k_line:
            arc = _fit_arc(Xr, Nr, Wr, eps)
            if arc is not None:
                cand.append(arc)
        best = None
        for m in cand:
            rms = _rms(m, Xr, Wr)
            if best is None or rms < best[1] * (0.8 if m[0] == "arc" else 1.0):
                best = (m, rms)
        m, rms = best
        if rms > tol:
            continue
        if m[0] == "arc":
            from .cadfit import _angular_span
            c, rr = m[1], m[2]
            ang = np.arctan2(Xr[:, 1] - c[1], Xr[:, 0] - c[0])
            sag = rr * (1 - np.cos(min(_angular_span(ang), np.pi) / 2))
            if sag < 2 * noise_mm or not (0.8 < rr < 400):
                n, d = cand[0][1], cand[0][2]
                m = ("line", n, d)
                if _rms(m, Xr, Wr) > tol:
                    continue
        if m[0] == "line":
            # straight faces of a machined part are nearly always on its own axes: snap within 3 deg
            for ax in (np.array([1.0, 0]), np.array([0, 1.0])):
                if abs(m[1] @ ax) > np.cos(np.radians(3)):
                    nn = ax * np.sign(m[1] @ ax)
                    m = ("line", nn, -float(np.median(Xr @ nn)))
                    break
            # the points must cover the segment between its ends (not two blobs far apart)
            tdir = np.array([-m[1][1], m[1][0]])
            pr = np.sort(Xr[np.abs(Xr @ m[1] + m[2]) < eps] @ tdir)
            if len(pr) < 10:
                continue
            span = pr[-1] - pr[0]
            gaps = np.diff(pr)
            if span < 2.0 or gaps.max() > max(3.0, 0.3 * span):
                continue
            m = (*m, abs(m[1][0]) > 0.999 or abs(m[1][1]) > 0.999)  # snapped flag
        models.append((m, near))
    # a line off the axes whose points the axis-aligned lines already explain is a duplicate
    snapped = [m for m, _ in models if m[0] == "line" and m[3]]
    keep = []
    for m, near in models:
        if m[0] == "line" and not m[3] and snapped:
            dmin = np.min(np.stack([np.abs(X[near] @ q[1] + q[2]) for q in snapped], 1), 1)
            if np.mean(dmin < eps) > 0.7:
                continue
        keep.append((m[:3] if m[0] == "line" else m, near))
    W = np.ones(len(X))
    return _merge_models(keep, X, W, eps)


def _classify_region(Xs, Ns, W, idx, eps, tol, min_support, noise_mm, seed, depth=0):
    """[(model, inlier idx)] for one smooth region."""
    from .cadfit import _angular_span
    ext = np.ptp(Xs[idx], axis=0).max() if len(idx) else 0.0
    if len(idx) < 8 or ext < 2.0:  # too small to say anything about its shape
        return []
    n, d = _line_fit(Xs[idx], W[idx])
    if np.sum(W[idx] * (Ns[idx] @ n)) < 0:
        n, d = -n, -d
    line = ("line", n, d)
    rl = _rms(line, Xs[idx], W[idx])
    if rl <= tol:
        return [(line, idx)]
    arc = _fit_arc(Xs[idx], Ns[idx], W[idx], eps)
    if arc is not None:
        ra = _rms(arc, Xs[idx], W[idx])
        c, r = arc[1], arc[2]
        ang = np.arctan2(Xs[idx, 1] - c[1], Xs[idx, 0] - c[0])
        span = _angular_span(ang)
        sag = r * (1 - np.cos(min(span, np.pi) / 2))
        if ra <= tol and sag >= 3 * noise_mm and 1.0 < r < 300:
            return [(arc, idx)]
    if depth >= 3 or len(idx) < 20:
        return []
    # mixed region: take the best line or arc inside it, then the rest
    rng = np.random.default_rng(seed + depth)
    best = None
    for i in rng.choice(idx, min(150, len(idx)), replace=False):
        nn = Ns[i]
        dd = -nn @ Xs[i]
        inl = idx[(np.abs(Xs[idx] @ nn + dd) < eps) & (Ns[idx] @ nn > np.cos(np.radians(15)))]
        w = W[inl].sum()
        if best is None or w > best[0]:
            best = (w, inl)
    out = []
    if best is not None and best[0] >= min_support:
        inl = best[1]
        rest = np.setdiff1d(idx, inl)
        # both parts again as contiguous pieces only: a line through two far-apart blobs fits anything
        for part in (inl, rest):
            for comp in _components(Xs, Ns, part, 3 * max(0.5 * noise_mm, 0.15)):
                if W[comp].sum() >= min_support:
                    out += _classify_region(Xs, Ns, W, comp, eps, tol, min_support, noise_mm, seed, depth + 1)
    return out


def _components(Xs, Ns, idx, link, max_angle=25.0):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(idx) < 2:
        return [idx]
    pairs = cKDTree(Xs[idx]).query_pairs(max(link, 0.6), output_type="ndarray")
    if len(pairs):
        pairs = pairs[np.sum(Ns[idx][pairs[:, 0]] * Ns[idx][pairs[:, 1]], 1) > np.cos(np.radians(max_angle))]
    G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
                   shape=(len(idx), len(idx)))
    nc, lab = connected_components(G, directed=False)
    return [idx[lab == k] for k in range(nc)]


def _fit_arc(X, Nx, W, eps):
    """Circle through weighted points (algebraic start, robust refinement); s from the normals."""
    from scipy.optimize import least_squares
    A = np.c_[2 * X, np.ones(len(X))]
    sw = np.sqrt(W)
    try:
        sol = np.linalg.lstsq(A * sw[:, None], np.sum(X ** 2, 1) * sw, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 1e-12)))
    if not np.isfinite(r) or r > 1e4:
        return None
    res = least_squares(lambda x: sw * (np.linalg.norm(X - x[:2], axis=1) - x[2]), np.r_[c, r],
                        loss="soft_l1", f_scale=eps / 2)
    c, r = res.x[:2], float(abs(res.x[2]))
    rad = X - c
    s = 1.0 if np.sum(W * np.sum(rad * Nx, 1)) >= 0 else -1.0
    return ("arc", c, r, s)


def _merge_models(models, Xs, W, eps):
    out = []
    for m, idx in models:
        w = float(W[idx].sum())
        for k, (m2, w2, idx2) in enumerate(out):
            if m[0] != m2[0]:
                continue
            if m[0] == "line" and m[1] @ m2[1] > np.cos(np.radians(2)) and abs(m[2] - m2[2]) < eps:
                idx_m = np.r_[idx2, idx]
                n, d = _line_fit(Xs[idx_m], W[idx_m])
                if n @ m2[1] < 0:
                    n, d = -n, -d
                out[k] = (("line", n, d), w + w2, idx_m)
                break
            if (m[0] == "arc" and m[3] == m2[3] and np.linalg.norm(m[1] - m2[1]) < eps
                    and abs(m[2] - m2[2]) < eps):
                out[k] = (m2 if w2 >= w else m, w + w2, np.r_[idx2, idx])
                break
        else:
            out.append((m, w, idx))
    return [(m, w) for m, w, _ in out]


def _line_fit(X, W):
    m = (W[:, None] * X).sum(0) / W.sum()
    C = ((X - m) * W[:, None]).T @ (X - m)
    ev, evec = np.linalg.eigh(C)
    n = evec[:, 0]
    return n, -n @ m


def _rms(model, X, W):
    if model[0] == "line":
        r = X @ model[1] + model[2]
    else:
        r = np.linalg.norm(X - model[1], axis=1) - model[2]
    return float(np.sqrt((W * r ** 2).sum() / W.sum()))


def _refine(model, Xs, Ns, W, idx, eps, cos30, iters=3):
    from scipy.optimize import least_squares
    inl = None
    for _ in range(iters):
        if model[0] == "line":
            n, d = model[1], model[2]
            r = Xs[idx] @ n + d
            ok = (np.abs(r) < eps) & (Ns[idx] @ n > cos30)
            inl = idx[ok]
            if len(inl) < 3:
                return None, inl
            n2, d2 = _line_fit(Xs[inl], W[inl])
            if n2 @ n < 0:
                n2, d2 = -n2, -d2
            model = ("line", n2, d2)
        else:
            c, r, s = model[1], model[2], model[3]
            rad = Xs[idx] - c
            rn = np.maximum(np.linalg.norm(rad, axis=1), 1e-9)
            ok = (np.abs(rn - r) < eps) & (np.sum(Ns[idx] * rad, 1) / rn * s > cos30)
            inl = idx[ok]
            if len(inl) < 5:
                return None, inl
            sw = np.sqrt(W[inl])
            sol = least_squares(lambda x: sw * (np.linalg.norm(Xs[inl] - x[:2], axis=1) - x[2]),
                                np.r_[c, r], loss="soft_l1", f_scale=eps / 2)
            model = ("arc", sol.x[:2], float(abs(sol.x[2])), s)
    return model, inl


def _close_end(C, lines, step=0.25, overshoot=4.0, free_len=12.0):
    """Extension beyond the last sample of the ordered 2D curve C, as samples (step apart), until it
    has crossed the profile line it joins. lines: [(n, d, spans)] with n.x + d = 0.
    A curve meeting a face at an angle: straight on along its end tangent. A curve that runs
    tangentially into a face (it never reaches it along its tangent): toward the point where a
    tangent join lies, about twice the remaining gap along the face. No face near: the tangent."""
    E = C[-1]
    k = min(len(C) - 1, max(4, int(round(3.0 / step))))
    T = E - C[-1 - k]
    T /= max(np.linalg.norm(T), 1e-12)
    best = hit = None
    for n, d, spans in lines:
        n = np.asarray(n, float)
        h = float(n @ E + d)
        if abs(h) > 6.0:
            continue
        t = np.array([-n[1], n[0]])
        cosT = float(T @ n)
        # along the tangent the line is reached after L0 = -h / (T.n)
        L0 = -h / cosT if abs(cosT) > 1e-6 else np.inf
        if 0 <= L0 <= 4.0 * abs(h) + 3.0:
            H = E + L0 * T
            if not spans or any(a0 - 3.0 <= float(H @ t) <= a1 + 3.0 for a0, a1 in spans):
                if hit is None or L0 < hit:
                    hit = L0  # the face the curve runs into
                continue
        F = E - h * n  # foot on the line
        if spans:
            tf = float(F @ t)
            if not any(a0 - 3.0 <= tf <= a1 + 3.0 for a0, a1 in spans):
                continue
        if best is None or abs(h) < abs(best[0]):
            best = (h, n, d, F, t, spans)
    if hit is not None:
        D, L = T, hit + overshoot
    elif best is None:
        D, L = T, free_len
    else:
        h, n, d, F, t, spans = best
        if True:
            # tangent join: aim at F + 2|h| along the line, on the side the curve was heading
            if t @ T < 0:
                t = -t
            G = F + 2.0 * abs(h) * t
            D = G - E
            L = float(np.linalg.norm(D)) + overshoot
            D /= max(np.linalg.norm(D), 1e-12)
    s_ = np.arange(step, L + 1e-9, step)
    return E + s_[:, None] * D


def _absorb_facets(pr, kp, prims, labels, r_cur, P, N, a, d, eps, noise_mm):
    """A curved profile surface replaces the flat facets the segmentation used to approximate it:
    a plane running along the axis whose points nearly all lie on the curve (and face the same
    way) is a facet, not a face. A real face is not explained by the curve and stays."""
    from .cadfit import prim_normals
    n_drop = 0
    for k, q in enumerate(prims):
        if q is None or k == kp or q.get("profile") or q["type"] != "plane":
            continue
        if abs(q["n"] @ a) > np.sin(np.radians(5)):
            continue
        mk = np.where(labels == k)[0]
        if len(mk) == 0:
            continue
        ok = (d[mk] < eps) & (np.sum(prim_normals(pr, P[mk]) * N[mk], axis=1) > np.cos(np.radians(30)))
        if ok.mean() < 0.85:
            continue
        # the curve must fit at least as well as the facet itself (no 1 mm offset fits)
        r_own = np.abs(r_cur[mk])
        # (the facet's own rms is biased low: it only kept the points near it)
        if np.sqrt(np.mean(d[mk][ok] ** 2)) > max(1.1 * np.sqrt(np.mean(r_own[ok] ** 2)) + 0.1,
                                                  1.2 * noise_mm):
            continue
        labels[mk[ok]] = kp
        r_cur[mk[ok]] = d[mk[ok]]
        labels[mk[~ok]] = -1
        r_cur[mk[~ok]] = np.inf
        prims[k] = None
        n_drop += 1
    return n_drop


def profile_primitives(P, N, prims, labels, axes, noise_mm, min_pts, P_full=None, N_full=None):
    """Recognise extrusions and add the surfaces of their profiles to the primitive set.
    Returns (prims, labels, findings)."""
    from .cadfit import _perp_basis, _same_surface, prim_dist, signed_cos, _unit, extr_resample
    findings = []
    ex = find_extrusions(P, N, axes, noise_mm)
    eps = max(2.0 * noise_mm, 0.8)
    labels = labels.copy()
    prims = list(prims)
    r_cur = np.full(len(P), np.inf)
    for k, p in enumerate(prims):
        mk = labels == k
        if mk.any():
            r_cur[mk] = np.abs(prim_dist(p, P[mk]))
    best_auth = None  # only the clearest extrusion becomes the authority for its direction
    for e in ex:
        if e["consistency"] >= 0.97 and e["frac"] >= 0.5:
            best_auth = id(e)
            break
    for e in ex:
        if e["consistency"] < 0.5:
            continue
        a = _unit(e["axis"])
        u, v = _perp_basis(a)
        m = e["mask"]
        if P_full is not None:
            mf = np.abs(N_full @ a) < np.sin(np.radians(10))
            X = np.c_[P_full[mf] @ u, P_full[mf] @ v]
            NX = np.c_[N_full[mf] @ u, N_full[mf] @ v]
        else:
            X = np.c_[P[m] @ u, P[m] @ v]
            NX = np.c_[N[m] @ u, N[m] @ v]
        NX /= np.maximum(np.linalg.norm(NX, axis=1, keepdims=True), 1e-12)
        # support threshold: a profile element must be seen over a real length (a third of the part
        # at min_pts density) -> in 2D cell counts
        p2d = fit_profile(X, NX, noise_mm, min_support=max(min_pts, 30) * len(X) / max(1, m.sum()))
        n_new = n_arc = n_drop = 0
        new3d = []
        for model, w, spans in p2d:
            if model[0] == "line":
                n3 = model[1][0] * u + model[1][1] * v
                tdir2 = np.array([-model[1][1], model[1][0]])
                new3d.append({"type": "plane", "n": _unit(n3), "d": float(model[2]) / np.linalg.norm(n3),
                              "_t3": tdir2[0] * u + tdir2[1] * v, "_spans": spans})
            elif model[0] == "poly":
                P2, N2 = model[1], model[2]
                # the smoothing window bends the last ~2 mm of each end: drop them, the closing
                # extension takes over there
                cut = int(round(2.0 / 0.25))
                if len(P2) > 4 * cut:
                    P2, N2 = P2[cut:-cut], N2[cut:-cut]
                lines2 = [(m_[1], m_[2], s_) for m_, _w, s_ in p2d if m_[0] == "line"]
                # each end runs on until it crosses the flat face it joins (the lines took the last
                # millimetres of a curve that runs tangentially into a face); a dangling end would
                # leave the curved surface unable to cut the part
                e0 = _close_end(P2[::-1], lines2)
                e1 = _close_end(P2, lines2)
                ext = np.r_[e0[::-1][::4], P2[::2], e1[::4]]
                Pd = np.r_[e0[::-1], P2, e1]
                Nd = np.r_[np.repeat(N2[:1], len(e0), 0), N2, np.repeat(N2[-1:], len(e1), 0)]
                pr_e = {"type": "extrusion", "a": a.copy(), "u": u.copy(), "v": v.copy(),
                        "poly": Pd, "pnrm": Nd, "poly_ext": ext}
                new3d.append(extr_resample(pr_e))  # distances measured to the very curve of the face
            else:
                c = model[1]
                new3d.append({"type": "cylinder", "a": a.copy(), "p": c[0] * u + c[1] * v + a * e["t0"],
                              "r": model[2]})
        # authority only for a real profile (with a curve in it): for a prismatic part the general
        # segmentation already finds the flat faces, and dropping its surfaces only loses detail
        authority = (AUTHORITY and id(e) == best_auth and any(m_[0] == "arc" for m_, _w, _s in p2d))
        if authority:
            # a clear profile is the authority for every surface running along it: surfaces of the
            # general segmentation that run along the axis but are not in the profile are dropped
            # (their points go to the profile surfaces below, or stay unexplained)
            for k, q in enumerate(prims):
                if q is None or q.get("profile"):
                    continue
                along = (q["type"] == "plane" and abs(q["n"] @ a) < np.sin(np.radians(2))) or \
                        (q["type"] == "cylinder" and abs(q["a"] @ a) > np.cos(np.radians(2)))
                if along and not any(_same_surface(q, p3, 2 * eps) for p3 in new3d):
                    mk = labels == k
                    labels[mk] = -1
                    r_cur[mk] = np.inf
                    prims[k] = None
                    n_drop += 1
        dominant = e is ex[0]
        for pr in new3d:
            t3, spans = pr.pop("_t3", None), pr.pop("_spans", None)
            if pr["type"] == "plane" and not dominant:
                continue  # flat faces only from the main profile direction (the others re-cut edges)
            pr.update(snapped=True, profile=True)
            if any(q is not None and _same_surface(pr, q, eps) for q in prims):
                continue
            d = np.abs(prim_dist(pr, P))
            near = m & (d < eps)
            if spans:
                # only where the edge exists in the section (the infinite plane also grazes a bore)
                tx = P @ t3
                ins = np.zeros(len(P), bool)
                for a0, a1 in spans:
                    ins |= (tx > a0 - 1.0) & (tx < a1 + 1.0)
                near &= ins
            cand = np.where(near)[0]
            if len(cand) == 0:
                continue
            cs = signed_cos(pr, P[cand], N[cand])  # sets pr["s"] from the majority
            cand = cand[cs > np.cos(np.radians(30))]
            win = cand[d[cand] < r_cur[cand]]
            if len(win) < min_pts:
                continue
            # a new surface must explain points the existing ones explain badly (or not at all):
            # otherwise it only re-cuts what is already right
            rc = r_cur[win]
            bad = ~np.isfinite(rc) | (rc > d[win] + noise_mm)
            if bad.sum() < min_pts:
                continue
            pr.update(rms=float(np.sqrt(np.mean(d[win] ** 2))), n_pts=int(len(win)))
            labels[win] = len(prims)
            r_cur[win] = d[win]
            prims.append(pr)
            n_new += 1
            n_arc += pr["type"] in ("cylinder", "extrusion")
            if pr["type"] in ("cylinder", "extrusion"):
                n_drop += _absorb_facets(pr, len(prims) - 1, prims, labels, r_cur, P, N, a, d, eps, noise_mm)
        names = "xyz"
        ai = int(np.argmax(np.abs(a)))
        findings.append({"axis": a.tolist(), "length_mm": e["length"], "fraction": e["frac"],
                         "consistency": e["consistency"], "profile_elements": len(p2d), "new_surfaces": n_new,
                         "authority": bool(authority)})
        if n_drop:
            log(f"  CAD: {n_drop} oppervlak(ken) langs het profiel die niet in de doorsnede passen vervallen")
        log(f"  CAD: bevinding: profiel langs {names[ai]} over {e['length']:.0f} mm "
            f"({e['frac'] * 100:.0f}% van de oppervlakte, doorsnede {e['consistency'] * 100:.0f}% gelijk over de lengte); "
            f"{len(p2d)} lijnen/bogen in de doorsnede, {n_new} nieuwe oppervlakken ({n_arc} gebogen)")
    # remove the dropped primitives, renumber the labels
    keep = [k for k, q in enumerate(prims) if q is not None]
    remap = -np.ones(len(prims) + 1, int)
    remap[keep] = np.arange(len(keep))
    labels = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1)
    prims = [prims[k] for k in keep]
    return prims, labels, findings
