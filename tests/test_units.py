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
