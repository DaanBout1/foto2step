"""Fast unit tests (run: pytest -q). The full end-to-end gauntlet is `python -m foto2step selftest`."""
import numpy as np
import pytest
import trimesh

from foto2step import geometry as G


def rand_T(rng, ang=30.0, t=500.0):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    a = np.radians(ang)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    return G.make_T(R, rng.normal(size=3) * t)


def test_umeyama_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3)) * 100
    T = rand_T(rng)
    T[:3, :3] *= 2.5
    dst = G.to_h(T, src)
    s, R, t = G.umeyama(src, dst, True)
    assert abs(s - 2.5) < 1e-9
    assert np.allclose(G.to_h(G.make_T(R, t, s), src), dst, atol=1e-8)


def test_screw_prismatic_and_revolute():
    T = G.make_T(np.eye(3), [150, 0, 0])
    d = G.screw_from_T(T)
    assert d["type"] == "prismatic" and abs(d["travel"] - 150) < 1e-9
    from foto2step.synthetic import rot_z
    d = G.screw_from_T(rot_z(30, [-450, 0, 0]))
    assert d["type"] == "revolute"
    assert abs(d["angle_deg"] - 30) < 1e-6
    assert np.allclose(np.abs(d["axis"]), [0, 0, 1], atol=1e-9)
    assert np.allclose(d["point"][:2], [-450, 0], atol=1e-6)


def test_triangulation():
    rng = np.random.default_rng(0)
    X = np.array([100.0, -50, 2000])
    Ps, xs = [], []
    for _ in range(6):
        T = rand_T(rng, 10, 200)
        P = T[:3]
        pc = P @ np.append(X, 1)
        Ps.append(P)
        xs.append(pc[:2] / pc[2] + rng.normal(0, 1e-5, 2))
    X0 = G.triangulate_dlt(Ps, xs)
    X1 = G.triangulate_refine(Ps, xs, X0)
    assert np.linalg.norm(X1 - X) < 0.5


def test_fit_cylinder():
    rng = np.random.default_rng(3)
    c = trimesh.creation.cylinder(radius=25, height=100, sections=128)
    pts, fi = trimesh.sample.sample_surface(c, 5000, seed=3)
    n = c.face_normals[fi]
    side = np.abs(n[:, 2]) < 0.5
    pts = pts[side] + rng.normal(0, 0.05, (side.sum(), 3))
    r = G.fit_cylinder(pts, n[side])
    assert abs(r["radius"] - 25) < 0.05
    assert abs(abs(r["axis"][2]) - 1) < 1e-3


def test_marker_pdf_and_detection(tmp_path):
    import cv2
    from foto2step.markers import detect, make_detector, marker_image, write_marker_pdf
    p = write_marker_pdf(tmp_path / "m.pdf", "DICT_5X5_250", range(0, 6), 80)
    assert p.exists() and p.stat().st_size > 1000
    img = np.full((800, 800), 255, np.uint8)
    img[200:600, 200:600] = cv2.resize(marker_image("DICT_5X5_250", 7, 400), (400, 400), interpolation=cv2.INTER_NEAREST)
    d = detect(img, make_detector("DICT_5X5_250"))
    assert 7 in d
    # OpenCV corner convention: black square spans pixels 200..599 -> corner at 199.5
    assert np.allclose(d[7][0], [199.5, 199.5], atol=0.6)


def test_step_box_roundtrip(tmp_path):
    from foto2step.stepexport import mesh_to_shape, read_step_check, write_step
    m = trimesh.creation.box(extents=[100, 50, 30]).subdivide().subdivide()
    shp, info = mesh_to_shape(m.vertices, m.faces)
    assert info["kind"] == "solid" and info["valid"] and info["failed_triangles"] == 0
    assert info["brep_faces"] == 6
    write_step([("blok", shp, (0.5, 0.5, 0.5))], tmp_path / "b.step")
    c = read_step_check(tmp_path / "b.step")
    assert abs(c["volume_mm3"] - 150000) < 1 and c["solids"] == 1


def test_features_snap_noisy_box():
    from foto2step.features import regularize_mesh
    import open3d as o3d
    rng = np.random.default_rng(0)
    m = trimesh.creation.box(extents=[200, 100, 80])
    for _ in range(5):
        m = m.subdivide()
    V = m.vertices + rng.normal(0, 0.15, m.vertices.shape)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(m.faces))
    out, rep = regularize_mesh(om, 0.15, {"angle_deg": 12, "min_region_area_mm2": 100, "cylinders": False,
                                          "sharpen_edges": True})
    planes = [r for r in rep["regions"] if r["type"] == "plane"]
    assert len(planes) == 6
    W = np.asarray(out.vertices)
    # distance of snapped vertices to true box surface
    d = np.min(np.abs(np.c_[np.abs(W[:, 0]) - 100, np.abs(W[:, 1]) - 50, np.abs(W[:, 2]) - 40]), axis=1)
    assert np.median(d) < 0.03


def test_motion_grouping():
    from foto2step.motion import find_bodies, joints_for_bodies
    from foto2step.synthetic import build_scene, marker_corners_world
    sc = build_scene()
    rng = np.random.default_rng(0)
    mks = {}
    for s in ("A", "B"):
        d = {}
        for m in sc["markers"]:
            T = np.eye(4) if m["body"] == "floor" else sc["states"][s][m["body"]]
            d[m["id"]] = {"corners": marker_corners_world(m, T) + rng.normal(0, 0.05, (4, 3))}
        mks[s] = d
    res = find_bodies(mks, ["A", "B"], 3.0, 0.05)
    assert len(res["bodies"]) == 3
    j = joints_for_bodies(res["bodies"], ["A", "B"])
    types = sorted(x["type"] for x in j)
    assert types == ["prismatic", "revolute"]
    for x in j:
        if x["type"] == "prismatic":
            assert abs(abs(x["positions_mm"]["B"]) - 150) < 0.3
        else:
            assert abs(abs(x["angles_deg"]["B"]) - 30) < 0.1
            assert np.linalg.norm(np.array(x["point"][:2]) - [-450, 0]) < 1.0


def test_fill_holes_open_box():
    import open3d as o3d
    from foto2step.meshing import fill_holes
    m = trimesh.creation.box(extents=[100, 60, 40]).subdivide().subdivide()
    keep = m.triangles_center[:, 2] > -19.9  # remove bottom
    F = m.faces[keep]
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(F))
    om.remove_unreferenced_vertices()
    assert not om.is_watertight()
    out, n, left = fill_holes(om, 1000)
    assert n == 1 and left == 0
    assert out.is_watertight()
    assert abs(out.get_volume() - 100 * 60 * 40) < 1e-3
    assert np.abs(np.asarray(out.vertices)).max() <= 50 + 1e-9


def test_marker_edge_refinement_unbiased():
    """Edge-based corners must not shrink the marker (corner detectors on L-corners pull inward)."""
    import cv2
    from foto2step.markers import detect, make_detector, marker_image, refine_edges
    rng = np.random.default_rng(0)
    S, N = 8, 700
    shrink = []
    for _ in range(6):
        mk = (marker_image("DICT_5X5_250", 7, 300 * S).astype(np.float32) / 255 * 200 + 20).astype(np.uint8)
        src = np.float32([[-0.5, -0.5], [300 * S - 0.5, -0.5], [300 * S - 0.5, 300 * S - 0.5], [-0.5, 300 * S - 0.5]])
        c0 = np.float32([[200, 180], [500, 210], [480, 510], [190, 470]]) + rng.normal(0, 15, (4, 2)).astype(np.float32)
        H = cv2.getPerspectiveTransform(src, (c0 + 0.5) * S - 0.5)
        big = cv2.warpPerspective(mk, H, (N * S, N * S), borderValue=235, flags=cv2.INTER_LINEAR)
        img = cv2.resize(big, (N, N), interpolation=cv2.INTER_AREA).astype(np.float32)
        img = np.clip(cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        d = detect(img, make_detector("DICT_5X5_250"))
        e = refine_edges(img, d[7])
        cen = c0.mean(0)
        shrink += list(np.linalg.norm(c0 - cen, axis=1) - np.linalg.norm(e - cen, axis=1))
    assert abs(np.mean(shrink)) < 0.05  # pixels


def test_fill_holes_slit_and_wiggly():
    import open3d as o3d
    from foto2step.meshing import fill_holes
    m = trimesh.creation.box(extents=[100, 60, 40])
    for _ in range(4):
        m = m.subdivide()
    c = m.triangles_center
    # thin slit on a side face + a big irregular hole on the top
    keep = ~((np.abs(c[:, 1] + 30) < 1e-6) & (np.abs(c[:, 0]) < 30) & (np.abs(c[:, 2]) < 2.5))
    keep &= ~((c[:, 2] > 19.9) & ((c[:, 0] ** 2 / 40 ** 2 + c[:, 1] ** 2 / 20 ** 2) < 1) & (c[:, 0] < 10 + 5 * np.sin(c[:, 1])))
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces[keep]))
    om.remove_unreferenced_vertices()
    from foto2step.meshing import close_mesh
    from foto2step.stepexport import _is_closed
    out, n, nopen = close_mesh(om, 1000)
    assert nopen == 0 and _is_closed(np.asarray(out.triangles))
    t = trimesh.Trimesh(np.asarray(out.vertices), np.asarray(out.triangles), process=False)
    assert abs(t.volume - 100 * 60 * 40) < 50


def test_stepwriter_part_and_assembly(tmp_path):
    from foto2step.stepexport import read_step_check
    from foto2step.stepwriter import write_assembly, write_part
    a = trimesh.creation.box(extents=[100, 50, 30]).subdivide()
    b = trimesh.creation.icosphere(subdivisions=2, radius=20)
    b.apply_translation([200, 0, 0])
    info = write_part(tmp_path / "a.step", "blok", a.vertices, a.faces, (0.5, 0.5, 0.5))
    assert info["kind"] == "solid" and info["brep_faces"] == 6
    c = read_step_check(tmp_path / "a.step")
    assert c["solids"] == 1 and abs(c["volume_mm3"] - 150000) < 1e-3
    write_assembly(tmp_path / "s.step", "machine", [("blok", a.vertices, a.faces, None), ("bol", b.vertices, b.faces, (0.1, 0.4, 0.8))])
    c = read_step_check(tmp_path / "s.step")
    assert c["solids"] == 2 and abs(c["volume_mm3"] - (150000 + b.volume)) < 1e-2
    txt = (tmp_path / "s.step").read_text()
    assert "NEXT_ASSEMBLY_USAGE_OCCURRENCE" in txt and "'bol'" in txt


def test_cylinder_from_plane_strips():
    from foto2step.features import regularize_mesh
    import open3d as o3d
    rng = np.random.default_rng(1)
    c = trimesh.creation.cylinder(radius=30, height=200, sections=128).subdivide().subdivide()
    V = c.vertices + rng.normal(0, 0.03, c.vertices.shape)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(c.faces))
    out, rep = regularize_mesh(om, 0.1, {"angle_deg": 12, "min_region_area_mm2": 100, "cylinders": True, "sharpen_edges": True})
    cyl = [r for r in rep["regions"] if r["type"] == "cylinder"]
    assert len(cyl) >= 1
    assert abs(max(cyl, key=lambda r: r["area"])["radius"] - 30) < 0.1


def test_device_profiles_and_advice():
    from foto2step.device import assess, group_key, lens_class, phone_tier
    assert phone_tier("iPhone Air") == "air"
    assert phone_tier("iPhone 18 Pro Max") == "pro_var_aperture"
    assert phone_tier("iPhone 19 Pro") == "pro_var_aperture"
    assert phone_tier("iPhone 16 Pro") == "pro"
    assert lens_class({"focal_35mm": 13}) == "ultra"
    assert lens_class({"focal_35mm": 24}) == "main"
    assert lens_class({"focal_35mm": 48}) == "main2x"
    assert lens_class({"focal_35mm": 100}) == "tele"
    assert group_key({"focal_35mm": 24, "raw": True}, (6048, 8064)) == "main_raw_49mp"
    air = assess([{"model": "iPhone Air", "f_number": 1.6, "exposure_s": 1 / 60, "iso": 200, "raw": False}] * 10)
    assert any("iPhone Air" in w and "lagere nauwkeurigheid" in w for w in air["warnings"])
    pro = assess([{"model": "iPhone 18 Pro Max", "f_number": 1.48, "exposure_s": 1 / 500, "iso": 400, "raw": False}] * 10)
    assert not pro["warnings"]
    assert any("ƒ/4.0" in a for a in pro["advice"]) and any("ProRAW" in a for a in pro["advice"])
    good = assess([{"model": "iPhone 18 Pro Max", "f_number": 4.0, "exposure_s": 1 / 500, "iso": 400, "raw": True}] * 10)
    assert not good["warnings"] and not good["advice"]


def test_isolate_objects_drops_floor_clutter():
    import open3d as o3d
    from foto2step.meshing import isolate_objects
    rng = np.random.default_rng(0)
    # 80 mm cube standing on the floor (surface samples) + a curled sheet of paper 300 mm away (z 3..8 mm)
    g = np.stack(np.meshgrid(np.arange(0, 80, 1.0), np.arange(0, 80, 1.0)), -1).reshape(-1, 2)
    faces = [np.c_[g, np.full(len(g), 80.0)], np.c_[g[:, 0], np.zeros(len(g)), g[:, 1]],
             np.c_[g[:, 0], np.full(len(g), 80.0), g[:, 1]], np.c_[np.zeros(len(g)), g]]
    cube = np.vstack(faces)
    cube = cube[cube[:, 2] > 3]
    paper = np.c_[rng.uniform(300, 510, 6000), rng.uniform(0, 297, 6000), rng.uniform(3, 8, 6000)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([cube, paper])))
    out = np.asarray(isolate_objects(pc, {"voxel_mm": 1.0}).points)
    assert len(out) == len(cube)
    assert out[:, 0].max() < 100


def test_crop_fits_tilted_floor():
    import open3d as o3d
    from foto2step.meshing import crop_cloud
    rng = np.random.default_rng(1)
    # floor tilted 8 mm per metre plus 0.5 mm noise, cube 80 mm on top near the markers
    xy = rng.uniform(-500, 500, (200_000, 2))
    floor = np.c_[xy, 0.008 * xy[:, 0] + 0.002 * xy[:, 1] + rng.normal(0, 0.5, len(xy))]
    g = rng.uniform(0, 80, (20_000, 2))
    cube = np.c_[g[:, 0], np.zeros(len(g)), g[:, 1]]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([floor, cube])))
    mw = {"0": {"corners": [[-30, -30, 0], [30, -30, 0], [30, 30, 0], [-30, 30, 0]]}}
    out = np.asarray(crop_cloud(pc, mw, {"crop_margin_mm": 600, "floor_clearance_mm": 3.0}).points)
    assert (out[:, 2] > 3).all()
    n_floor_left = (np.abs(out[:, 1]) > 1).sum()   # anything not on the cube face
    assert n_floor_left < 50, n_floor_left
    assert len(out) > 0.95 * (cube[:, 2] > 3.5).sum()


def test_cadfit_noisy_box_becomes_6_face_solid():
    """Experimental CAD reconstruction: noisy scan of a box (open bottom) -> valid 6-face solid."""
    import open3d as o3d
    from foto2step.cadfit import segment_primitives, merge_similar, label_mesh, snap_vertices, mesh_brep, shape_report
    rng = np.random.default_rng(0)
    m = o3d.geometry.TriangleMesh.create_box(100, 60, 80).subdivide_midpoint(5)
    m.remove_duplicated_vertices()
    V = np.asarray(m.vertices) + rng.normal(0, 0.3, (len(m.vertices), 3))
    F = np.asarray(m.triangles)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=40))
    sel = V[:, 2] > 3
    P, N = V[sel], np.asarray(pc.normals)[sel]
    prims, lp = segment_primitives(P, N, 0.3)
    prims, lp = merge_similar(prims, lp, P, 0.3)
    assert len(prims) == 5
    lab, _, allp, _ = label_mesh(V, F, prims, lp, P, N, 0.3, floor_z=0.0)
    W, _, _ = snap_vertices(V, F, lab, allp, 0.3)
    shape, info = mesh_brep(W, F, lab, allp)
    rep = shape_report(shape)
    assert info["closed"] and rep["valid"] and rep["solid"] and rep["faces"] == 6
    assert abs(rep["volume_mm3"] - 100 * 60 * 80) / (100 * 60 * 80) < 0.005


def _write_dmap(fn, K, R, C, D, Nrm, ids, w, h):
    import struct
    with open(fn, "wb") as f:
        f.write(struct.pack("<HBBIIIIff", 0x5244, 1 | 2 | 4, 0, w, h, w, h, 1.0, 1e4))
        name = b"img.jpg"
        f.write(struct.pack("<H", len(name)) + name)
        f.write(struct.pack("<I", len(ids)) + struct.pack("<%dI" % len(ids), *ids))
        f.write(K.astype("<f8").tobytes() + R.astype("<f8").tobytes() + C.astype("<f8").tobytes())
        f.write(D.astype("<f4").tobytes() + Nrm.astype("<f4").tobytes() + np.ones_like(D, "<f4").tobytes())


def test_strict_fusion_removes_reflection_layer(tmp_path):
    """Plane z=0 seen by 8 cameras; in 2 of them the depth shows a reflection 4 mm in front of it.
    Strict fusion keeps the plane (with less noise than one depth map) and drops the ghost."""
    from foto2step.fusion import strict_points
    rng = np.random.default_rng(0)
    w, h, f = 160, 120, 300.0
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    dd = tmp_path / "dense"
    dd.mkdir()
    for i in range(8):
        a = 2 * np.pi * i / 8
        C = np.array([150 * np.cos(a), 150 * np.sin(a), 300.0])
        zc = -C / np.linalg.norm(C)
        xc = np.cross([0, 0, 1.0], zc); xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        R = np.stack([xc, yc, zc])
        vv, uu = np.mgrid[0:h, 0:w]
        ray = np.c_[uu.ravel(), vv.ravel(), np.ones(uu.size)] @ np.linalg.inv(K).T @ R  # world dirs (unnorm, z_cam=1)
        t = -C[2] / ray[:, 2]                          # hit z = 0; t equals camera depth (z_cam scale 1)
        D = (t + rng.normal(0, 0.3, t.size)).reshape(h, w)
        if i in (0, 4):                                # reflection: 4 mm closer over the centre patch
            X = C + ray * t[:, None]
            ghost = (np.abs(X[:, 0]) < 15) & (np.abs(X[:, 1]) < 15)
            Df = D.ravel()
            Df[ghost] -= 4.0                           # 4 mm closer to the camera
            D = Df.reshape(h, w)
        Nrm = np.tile(R @ np.array([0, 0, 1.0]), (h * w, 1)).reshape(h, w, 3)
        ids = [i + 1] + [j + 1 for j in range(8) if j != i]
        _write_dmap(dd / f"depth{i + 1:04d}.dmap", K, R, C, D, Nrm, ids, w, h)
    body = np.c_[rng.uniform(-30, 30, (4000, 2)), np.zeros(4000)]
    P = strict_points(dd, body, tmp_path / "cache", params={"min_baseline_deg": 5.0, "voxel_mm": 1.0})
    assert P is not None and len(P) > 1000
    assert np.mean(np.abs(P[:, 2]) > 2.0) < 0.002     # the ghost layer is gone
    assert np.std(P[:, 2]) < 0.3                       # fused noise below one map's 0.3 mm
    # control: with the two reflecting views agreeing with each other only, a loose fusion keeps it
    import shutil
    shutil.rmtree(tmp_path / "cache")
    L = strict_points(dd, body, tmp_path / "cache", params={"min_views": 1, "min_baseline_deg": 0.0,
                                                           "max_violation_frac": 99, "rel_depth": 0.05,
                                                           "voxel_mm": 1.0})
    assert np.mean(np.abs(L[:, 2]) > 2.0) > 0.01


def test_profile_spline_stretch_and_closed_ends():
    """An S-shaped (ogee) edge between two flat faces of a section becomes one spline stretch whose
    ends are carried on until they cross the faces they join."""
    from foto2step import profiles
    rng = np.random.default_rng(0)
    pts, nrm = [], []
    for y in np.arange(0, 40, 0.05):                     # top face z = 50 (outward +z)
        pts.append((y, 50.0)); nrm.append((0, 1))
    for z in np.arange(0, 30, 0.05):                     # front face y = 60 (outward +y)
        pts.append((60.0, z)); nrm.append((1, 0))
    for y in np.arange(40, 60, 0.02):                    # ogee z = 40 - 10 cos(pi (60 - y) / 20)
        z = 40.0 - 10.0 * np.cos(np.pi * (60.0 - y) / 20.0)
        dz = -10.0 * np.pi / 20.0 * np.sin(np.pi * (60.0 - y) / 20.0)
        n = np.array([-dz, 1.0]) / np.hypot(dz, 1.0)
        pts.append((y, z)); nrm.append(tuple(n))
    X = np.array(pts, float)
    NX = np.array(nrm, float)
    X += rng.normal(0, 0.15, X.shape) * NX
    out = profiles.fit_profile(X, NX, 0.15, min_support=40)
    polys = [m for m, _w, _s in out if m[0] == "poly"]
    assert len(polys) == 1
    P2 = polys[0][1]
    # on the curve (away from the smoothed ends)
    mid = P2[len(P2) // 4: 3 * len(P2) // 4]
    zt = 40.0 - 10.0 * np.cos(np.pi * (60.0 - mid[:, 0]) / 20.0)
    assert np.max(np.abs(mid[:, 1] - zt)) < 0.4
    lines = [(m[1], m[2], s) for m, _w, s in out if m[0] == "line"]
    end = profiles._close_end(P2, lines)
    beg = profiles._close_end(P2[::-1], lines)
    # one end crosses the top face (z = 50), the other the front face (y = 60)
    zs = np.r_[end[:, 1], beg[:, 1]]
    ys = np.r_[end[:, 0], beg[:, 0]]
    assert zs.max() > 50.0 and ys.max() > 60.0
