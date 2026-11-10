"""Moving parts from multiple machine poses ('states'): marker grouping, rigid bodies and joint estimation."""
from __future__ import annotations

import numpy as np

from .geometry import ransac_rigid, rigid_T, screw_from_T, to_h
from .log import log


def _marker_T(cA, cB):
    return rigid_T(np.asarray(cA), np.asarray(cB))


def group_state(mA: dict, mS: dict, thresh: float, sigma: float):
    """Partition common markers into rigid groups between reference state A and state S.

    Returns (static_T, groups) where groups is list of (ids, T_A_to_S). groups[0] is the static group.
    """
    common = sorted(set(mA) & set(mS))
    if len(common) < 3:
        raise RuntimeError("Te weinig gemeenschappelijke markers tussen standen (min. 3 vaste markers nodig)")
    tol = max(4 * sigma, thresh / 2)
    remaining = list(common)
    groups = []
    while remaining:
        best = None
        for i in remaining:
            T = _marker_T(mA[i]["corners"], mS[i]["corners"])
            mem = []
            for j in remaining:
                r = np.linalg.norm(to_h(T, np.asarray(mA[j]["corners"])) - np.asarray(mS[j]["corners"]), axis=1)
                if r.max() < tol * 2.5:
                    mem.append(j)
            if best is None or len(mem) > len(best[1]):
                best = (i, mem)
        ids = best[1]
        # refine with all corners, then re-check membership
        src = np.vstack([mA[j]["corners"] for j in ids])
        dst = np.vstack([mS[j]["corners"] for j in ids])
        T, inl = ransac_rigid(src, dst, tol * 2.5, iters=300)
        if T is None:
            T = rigid_T(src, dst)
        ids2 = [j for j in remaining
                if np.linalg.norm(to_h(T, np.asarray(mA[j]["corners"])) - np.asarray(mS[j]["corners"]), axis=1).max() < tol * 2.5]
        if ids2:
            ids = ids2
            T = rigid_T(np.vstack([mA[j]["corners"] for j in ids]), np.vstack([mS[j]["corners"] for j in ids]))
        groups.append((ids, T))
        remaining = [j for j in remaining if j not in ids]
    # the static group is the largest one that contains floor/origin markers -> choose largest
    groups.sort(key=lambda g: -len(g[0]))
    static_T = groups[0][1]
    # sanity: static motion should be tiny (world frames come from the same floor markers)
    ang = np.degrees(np.arccos(np.clip((np.trace(static_T[:3, :3]) - 1) / 2, -1, 1)))
    if np.linalg.norm(static_T[:3, 3]) > 20 or ang > 1.0:
        log(f"  WAARSCHUWING: vaste markers verschoven tussen standen ({np.linalg.norm(static_T[:3, 3]):.1f} mm, "
            f"{ang:.2f} graden). Is de vloer/oorsprongmarker verschoven?")
    # merge 'moving' groups whose motion relative to static is below threshold into static
    out = [groups[0]]
    for ids, T in groups[1:]:
        rel = np.linalg.inv(static_T) @ T
        pts = np.vstack([mA[j]["corners"] for j in ids])
        mv = np.linalg.norm(to_h(rel, pts) - pts, axis=1).max()
        if mv < thresh:
            out[0] = (out[0][0] + ids, out[0][1])
        else:
            out.append((ids, T))
    return static_T, out


def find_bodies(markers_by_state: dict, state_names: list, thresh: float, sigma: float,
                min_markers: int = 1):
    """Return dict with bodies [{name, marker_ids, T: {state: 4x4 A->state (world)}}] and per-state static T."""
    ref = state_names[0]
    mA = markers_by_state[ref]
    partitions = []
    statics = {ref: np.eye(4)}
    for s in state_names[1:]:
        sT, groups = group_state(mA, markers_by_state[s], thresh, sigma)
        statics[s] = sT
        lab = {}
        for gi, (ids, _) in enumerate(groups):
            for i in ids:
                lab[i] = gi
        partitions.append(lab)
        log(f"  stand {s}: {len(groups) - 1} bewegende groep(en): " +
            ", ".join(f"{sorted(ids)}" for ids, _ in groups[1:]) + f"; vast: {len(groups[0][0])} markers")
    # common refinement of all partitions
    ids_all = sorted(mA)
    key = {}
    for i in ids_all:
        key[i] = tuple(p.get(i, -1) for p in partitions)
    # markers missing in some state get key -1 there: merge them with the group that matches on known states
    groups = {}
    for i, k in key.items():
        groups.setdefault(k, []).append(i)
    # resolve -1 entries
    resolved = {}
    for k, ids in groups.items():
        if -1 in k:
            match = [k2 for k2 in groups if -1 not in k2 and all(a == b or a == -1 for a, b in zip(k, k2))]
            if len(match) == 1:
                resolved.setdefault(match[0], []).extend(ids)
                continue
        resolved.setdefault(k, []).extend(ids)
    bodies = []
    static_key = tuple(0 for _ in partitions)
    for k, ids in sorted(resolved.items(), key=lambda kv: (kv[0] != static_key, -len(kv[1]))):
        is_static = (k == static_key)
        if not is_static and len(ids) < min_markers:
            log(f"  markers {ids}: te weinig markers voor een apart deel, genegeerd")
            continue
        T = {ref: np.eye(4)}
        for s in state_names[1:]:
            mS = markers_by_state[s]
            vis = [i for i in ids if i in mS]
            if is_static:
                T[s] = statics[s]
            elif vis:
                T[s] = rigid_T(np.vstack([mA[i]["corners"] for i in vis]), np.vstack([mS[i]["corners"] for i in vis]))
            else:
                T[s] = None
        name = "basis" if is_static else f"deel_{len(bodies)}"
        bodies.append({"name": name, "marker_ids": sorted(int(i) for i in ids), "T": T, "static": is_static})
    if not bodies or not bodies[0]["static"]:
        raise RuntimeError("Kon het vaste deel van de machine niet bepalen")
    return {"bodies": bodies, "statics": statics}


def joints_for_bodies(bodies: list, state_names: list) -> list:
    """Estimate joint (relative to static frame) for each moving body."""
    out = []
    static = bodies[0]
    for b in bodies[1:]:
        motions = []
        for s in state_names[1:]:
            if b["T"].get(s) is None:
                continue
            rel = np.linalg.inv(static["T"][s]) @ b["T"][s]  # motion of body in A world frame
            motions.append((s, rel))
        if not motions:
            out.append({"body": b["name"], "type": "unknown"})
            continue
        descs = [(s, screw_from_T(T)) for s, T in motions]
        types = {d["type"] for _, d in descs}
        if types == {"prismatic"}:
            ax = np.array([d["axis"] for _, d in descs])
            ref = ax[0]
            ax = np.array([a if a @ ref >= 0 else -a for a in ax])
            w = np.array([d["travel"] for _, d in descs])
            axis = (ax * w[:, None]).sum(0)
            axis /= np.linalg.norm(axis)
            pos = {s: float(T[:3, 3] @ axis) for s, T in motions}
            spread = float(max(np.degrees(np.arccos(np.clip(a @ axis, -1, 1))) for a in ax))
            pos[state_names[0]] = 0.0
            out.append({"body": b["name"], "type": "prismatic", "axis": axis.tolist(), "positions_mm": pos,
                        "axis_spread_deg": spread,
                        "residual_rotation_deg": float(max(d["residual_rotation_deg"] for _, d in descs))})
        elif "revolute" in types:
            rv = [(s, d) for s, d in descs if d["type"] == "revolute"]
            ax = np.array([d["axis"] for _, d in rv])
            ref = ax[0]
            sgn = np.array([1 if a @ ref >= 0 else -1 for a in ax])
            ax = ax * sgn[:, None]
            w = np.array([d["angle_deg"] for _, d in rv])
            axis = (ax * w[:, None]).sum(0)
            axis /= np.linalg.norm(axis)
            pts = np.array([d["point"] for _, d in rv])
            point = pts.mean(0)
            point = point - (point @ axis) * axis
            ang = {s: float(d["angle_deg"] * g) for (s, d), g in zip(rv, sgn)}
            ang[state_names[0]] = 0.0
            pitch = float(max(abs(d["pitch_mm"]) for _, d in rv))
            out.append({"body": b["name"], "type": "revolute", "axis": axis.tolist(), "point": point.tolist(),
                        "angles_deg": ang, "pitch_mm": pitch,
                        "note": "schroefbeweging (rotatie+translatie)" if pitch > 2 else ""})
        else:
            out.append({"body": b["name"], "type": "unknown", "detail": [d for _, d in descs]})
    return out
