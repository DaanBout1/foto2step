"""Triangle mesh -> B-rep solid -> STEP (AP214), with coplanar facets merged into single planar faces.

Built directly with shared topology (no sewing), so it scales to 100k+ triangles.
Writes named parts and an assembly through OCAF/XCAF so digital-twin software sees proper part names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .log import log


def _occ():
    from OCP import BRepBuilderAPI, BRep, TopoDS, gp, ShapeUpgrade, ShapeFix, BRepCheck, TopAbs, TopExp
    return BRepBuilderAPI, BRep, TopoDS, gp, ShapeUpgrade, ShapeFix, BRepCheck, TopAbs, TopExp


def mesh_to_shape(V: np.ndarray, F: np.ndarray, merge_coplanar=True, plane_tol=0.005, ang_tol_deg=0.2):
    """Return (TopoDS_Shape, info). Solid if mesh is closed & manifold, otherwise shell.

    Coplanar triangle groups become ONE planar B-rep face (outer loop + holes) built from the same
    shared edges as their neighbours, so the result is a clean, light B-rep without sewing.
    """
    BRepBuilderAPI, BRep, TopoDS, gp, ShapeUpgrade, ShapeFix, BRepCheck, TopAbs, TopExp = _occ()
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    tri = V[F]
    ar = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    F = F[ar > 1e-8]
    verts = {}

    def vert(i):
        v = verts.get(i)
        if v is None:
            v = BRepBuilderAPI.BRepBuilderAPI_MakeVertex(gp.gp_Pnt(*map(float, V[i]))).Vertex()
            verts[i] = v
        return v

    edges = {}

    def edge(a, b):
        key = (a, b) if a < b else (b, a)
        e = edges.get(key)
        if e is None:
            e = BRepBuilderAPI.BRepBuilderAPI_MakeEdge(vert(key[0]), vert(key[1])).Edge()
            edges[key] = e
        return e

    groups = []
    if merge_coplanar and len(F) > 1:
        groups = _coplanar_groups(V, F, plane_tol, ang_tol_deg)
    in_group = np.zeros(len(F), bool)
    builder = BRep.BRep_Builder()
    shell = TopoDS.TopoDS_Shell()
    builder.MakeShell(shell)
    n_fail = 0
    n_planar = 0
    for faces, normal in groups:
        face = _planar_face(V, F[faces], normal, edge, BRepBuilderAPI, gp, ShapeFix)
        if face is None:
            continue
        builder.Add(shell, face)
        in_group[faces] = True
        n_planar += 1
    for a, b, c in F[~in_group]:
        try:
            mw = BRepBuilderAPI.BRepBuilderAPI_MakeWire(edge(a, b), edge(b, c), edge(c, a))
            if not mw.IsDone():
                n_fail += 1
                continue
            mf = BRepBuilderAPI.BRepBuilderAPI_MakeFace(mw.Wire(), True)
            if not mf.IsDone():
                n_fail += 1
                continue
            face = mf.Face()
            n_tri = np.cross(V[b] - V[a], V[c] - V[a])
            n_face = _face_normal(face)
            if n_face is not None and np.dot(n_face, n_tri) < 0:
                face = _as_face(face.Reversed())
            builder.Add(shell, face)
        except Exception:
            n_fail += 1
    closed = _is_closed(F)
    shape = shell
    kind = "shell"
    if closed:
        try:
            ms = BRepBuilderAPI.BRepBuilderAPI_MakeSolid(shell)
            solid = ms.Solid()
            sf = ShapeFix.ShapeFix_Solid(solid)
            sf.Perform()
            shape = sf.Solid()
            kind = "solid"
        except Exception as e:
            log(f"  WAARSCHUWING: solid maken mislukt ({e}); schil wordt geschreven")
    nf = _count(shape, TopAbs.TopAbs_FACE)
    valid = BRepCheck.BRepCheck_Analyzer(shape).IsValid()
    if not valid:
        try:
            fx = ShapeFix.ShapeFix_Shape(shape)
            fx.Perform()
            shape = fx.Shape()
            valid = BRepCheck.BRepCheck_Analyzer(shape).IsValid()
        except Exception:
            pass
    info = {"kind": kind, "triangles": int(len(F)), "brep_faces": int(nf), "planar_faces": n_planar,
            "failed_triangles": n_fail, "valid": bool(valid), "closed_mesh": bool(closed)}
    return shape, info


def _coplanar_groups(V, F, tol, ang_deg, min_tris=2):
    """Connected groups of (numerically) coplanar triangles: [(face_idx_array, normal)]."""
    from .features import face_adjacency
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    adj = face_adjacency(F)
    c = (n[adj[:, 0]] * n[adj[:, 1]]).sum(1)
    ok = c > np.cos(np.radians(ang_deg))
    # vertex of neighbour on plane of face
    ca = tri[adj[:, 0]].mean(1)
    dist = np.abs(((tri[adj[:, 1]] - ca[:, None, :]) * n[adj[:, 0]][:, None, :]).sum(2)).max(1)
    ok &= dist < tol
    A = coo_matrix((np.ones(ok.sum()), (adj[ok, 0], adj[ok, 1])), shape=(len(F), len(F)))
    _, comp = connected_components(A, directed=False)
    out = []
    order = np.argsort(comp, kind="stable")
    cs = comp[order]
    starts = np.r_[0, np.where(np.diff(cs))[0] + 1, len(cs)]
    for s, e in zip(starts[:-1], starts[1:]):
        if e - s < min_tris:
            continue
        faces = order[s:e]
        w = np.linalg.norm(np.cross(tri[faces, 1] - tri[faces, 0], tri[faces, 2] - tri[faces, 0]), axis=1)
        nn = (n[faces] * w[:, None]).sum(0)
        nn /= np.linalg.norm(nn)
        # global flatness check (growing with local tolerance can drift on gently curved surfaces)
        pts = tri[faces].reshape(-1, 3)
        d = pts @ nn
        if np.ptp(d) > max(4 * tol, 0.02):
            continue
        out.append((faces, nn))
    return out


def _boundary_loops(Fg):
    """Directed boundary loops of a triangle group (list of vertex index lists) or None if ambiguous."""
    e = np.vstack([Fg[:, [0, 1]], Fg[:, [1, 2]], Fg[:, [2, 0]]])
    key = {}
    for a, b in e:
        key[(a, b)] = key.get((a, b), 0) + 1
    bnd = [(a, b) for (a, b) in key if (b, a) not in key]
    if not bnd:
        return []
    nxt = {}
    for a, b in bnd:
        nxt.setdefault(a, []).append(b)
    loops = []
    used = set()
    for a0, b0 in bnd:
        if (a0, b0) in used:
            continue
        loop = [a0]
        a, b = a0, b0
        guard = 0
        while True:
            used.add((a, b))
            if b == a0:
                break
            loop.append(b)
            cands = [c for c in nxt.get(b, []) if (b, c) not in used]
            if not cands:
                return None
            a, b = b, cands[0]
            guard += 1
            if guard > len(bnd) + 5:
                return None
        loops.append(loop)
    return loops


def _planar_face(V, Fg, normal, edge, BRepBuilderAPI, gp, ShapeFix):
    loops = _boundary_loops(Fg)
    if not loops:
        return None
    # signed areas w.r.t. normal
    def sarea(lp):
        P = V[lp]
        return 0.5 * np.cross(P - P[0], np.roll(P, -1, 0) - P[0]).sum(0) @ normal
    areas = [sarea(lp) for lp in loops]
    k = int(np.argmax(areas))
    if areas[k] <= 0:
        return None
    if sum(1 for a in areas if a > 0) > 1:
        return None  # several outer loops (pinched region): keep triangles
    c = V[np.unique(Fg)].mean(0)
    pln = gp.gp_Pln(gp.gp_Pnt(*map(float, c)), gp.gp_Dir(*map(float, normal)))

    def wire(lp):
        mw = BRepBuilderAPI.BRepBuilderAPI_MakeWire()
        for i in range(len(lp)):
            mw.Add(edge(lp[i], lp[(i + 1) % len(lp)]))
        if not mw.IsDone():
            raise RuntimeError("wire")
        return mw.Wire()

    try:
        mf = BRepBuilderAPI.BRepBuilderAPI_MakeFace(pln, wire(loops[k]), True)
        for i, lp in enumerate(loops):
            if i != k:
                mf.Add(wire(lp))
        if not mf.IsDone():
            return None
        face = mf.Face()
        fx = ShapeFix.ShapeFix_Face(face)
        fx.FixOrientation()
        fx.Perform()
        face = _as_face(fx.Face())
        n_face = _face_normal(face)
        if n_face is not None and n_face @ normal < 0:
            face = _as_face(face.Reversed())
        return face
    except Exception:
        return None


def _as_face(shape):
    from OCP import TopoDS as _T
    for fn in (getattr(getattr(_T, "TopoDS", None), "Face", None), getattr(getattr(_T, "TopoDS", None), "Face_s", None)):
        if fn is not None:
            return fn(shape)
    raise RuntimeError("TopoDS.Face cast niet gevonden")


def _face_normal(face):
    from OCP.BRep import BRep_Tool
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.TopAbs import TopAbs_REVERSED
    try:
        surf = BRep_Tool.Surface_s(face)
        pl = surf.Pln() if hasattr(surf, "Pln") else None
        if pl is None:
            from OCP.Geom import Geom_Plane
            pl = Geom_Plane.DownCast(surf).Pln() if hasattr(Geom_Plane, "DownCast") else None
        d = pl.Axis().Direction()
        n = np.array([d.X(), d.Y(), d.Z()])
        if face.Orientation() == TopAbs_REVERSED:
            n = -n
        return n
    except Exception:
        try:
            props = GeomLProp_SLProps(BRep_Tool.Surface_s(face), 0.0, 0.0, 1, 1e-6)
            d = props.Normal()
            n = np.array([d.X(), d.Y(), d.Z()])
            if face.Orientation() == TopAbs_REVERSED:
                n = -n
            return n
        except Exception:
            return None


def _is_closed(F):
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(cnt == 2))


def _count(shape, kind):
    from OCP.TopExp import TopExp_Explorer
    ex = TopExp_Explorer(shape, kind)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def write_step(parts: list, path: Path, assembly_name="machine", schema="AP214"):
    """parts: list of (name, TopoDS_Shape, rgb tuple or None). Writes one STEP (single part or assembly)."""
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
    from OCP.TDataStd import TDataStd_Name
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.Interface import Interface_Static
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.IFSelect import IFSelect_RetDone

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    if len(parts) == 1:
        name, shp, rgb = parts[0]
        lab = st.AddShape(shp, False)
        TDataStd_Name.Set_s(lab, TCollection_ExtendedString(name))
        if rgb:
            ct.SetColor(lab, Quantity_Color(*rgb, Quantity_TOC_RGB), XCAFDoc_ColorType.XCAFDoc_ColorGen)
    else:
        asm = st.NewShape()
        TDataStd_Name.Set_s(asm, TCollection_ExtendedString(assembly_name))
        from OCP.TopLoc import TopLoc_Location
        for name, shp, rgb in parts:
            plab = st.AddShape(shp, False)
            TDataStd_Name.Set_s(plab, TCollection_ExtendedString(name))
            if rgb:
                ct.SetColor(plab, Quantity_Color(*rgb, Quantity_TOC_RGB), XCAFDoc_ColorType.XCAFDoc_ColorGen)
            comp = st.AddComponent(asm, plab, TopLoc_Location())
            TDataStd_Name.Set_s(comp, TCollection_ExtendedString(name))
        st.UpdateAssemblies()
    Interface_Static.SetCVal_s("write.step.schema", {"AP203": "AP203", "AP214": "AP214IS", "AP242": "AP242DIS"}.get(schema, "AP214IS"))
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.product.name", assembly_name)
    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    w.SetColorMode(True)
    with _quiet():
        ok = w.Transfer(doc, STEPControl_AsIs)
    if not ok:
        raise RuntimeError("STEP transfer mislukt")
    with _quiet():
        status = w.Write(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP schrijven mislukt: {path}")
    return path


class _quiet:
    """Silence OCC's console banner (writes to fd 1)."""

    def __enter__(self):
        import os
        import sys
        sys.stdout.flush()
        self.fd = os.dup(1)
        self.dn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.dn, 1)

    def __exit__(self, *a):
        import os
        os.dup2(self.fd, 1)
        os.close(self.fd)
        os.close(self.dn)


def read_step_check(path: Path) -> dict:
    """Re-read a STEP file and return basic properties (used by the gauntlet)."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_SHELL
    r = STEPControl_Reader()
    with _quiet():
        ok = r.ReadFile(str(path))
    if ok != IFSelect_RetDone:
        raise RuntimeError(f"STEP niet leesbaar: {path}")
    r.TransferRoots()
    shp = r.OneShape()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shp, props)
    vol = props.Mass()
    sp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shp, sp)
    bb = Bnd_Box()
    BRepBndLib.Add_s(shp, bb)
    cmin, cmax = bb.CornerMin(), bb.CornerMax()
    x0, y0, z0, x1, y1, z1 = cmin.X(), cmin.Y(), cmin.Z(), cmax.X(), cmax.Y(), cmax.Z()
    return {"volume_mm3": float(vol), "area_mm2": float(sp.Mass()), "bbox": [x0, y0, z0, x1, y1, z1],
            "solids": _count(shp, TopAbs_SOLID), "shells": _count(shp, TopAbs_SHELL), "faces": _count(shp, TopAbs_FACE)}
