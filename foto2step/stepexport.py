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


def mesh_to_shape(V: np.ndarray, F: np.ndarray, merge_coplanar=True, lin_tol=0.01, ang_tol_deg=0.5):
    """Return (TopoDS_Shape, info). Solid if mesh is closed & manifold, otherwise shell."""
    BRepBuilderAPI, BRep, TopoDS, gp, ShapeUpgrade, ShapeFix, BRepCheck, TopAbs, TopExp = _occ()
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    # drop degenerate triangles
    tri = V[F]
    ar = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    F = F[ar > 1e-8]
    verts = [BRepBuilderAPI.BRepBuilderAPI_MakeVertex(gp.gp_Pnt(*map(float, p))).Vertex() for p in V]
    edges = {}

    def edge(a, b):
        key = (a, b) if a < b else (b, a)
        e = edges.get(key)
        if e is None:
            e = BRepBuilderAPI.BRepBuilderAPI_MakeEdge(verts[key[0]], verts[key[1]]).Edge()
            edges[key] = e
        return e

    builder = BRep.BRep_Builder()
    shell = TopoDS.TopoDS_Shell()
    builder.MakeShell(shell)
    n_fail = 0
    for a, b, c in F:
        try:
            mw = BRepBuilderAPI.BRepBuilderAPI_MakeWire(edge(a, b), edge(b, c), edge(c, a))
            if not mw.IsDone():
                n_fail += 1
                continue
            w = mw.Wire()
            mf = BRepBuilderAPI.BRepBuilderAPI_MakeFace(w, True)
            if not mf.IsDone():
                n_fail += 1
                continue
            face = mf.Face()
            # make face orientation follow triangle winding
            n_tri = np.cross(V[b] - V[a], V[c] - V[a])
            n_face = _face_normal(face)
            if n_face is not None and np.dot(n_face, n_tri) < 0:
                face = TopoDS.TopoDS.Face_s(face.Reversed())
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
    n_faces_before = len(F)
    if merge_coplanar:
        try:
            u = ShapeUpgrade.ShapeUpgrade_UnifySameDomain(shape, True, True, False)
            u.SetLinearTolerance(lin_tol)
            u.SetAngularTolerance(np.radians(ang_tol_deg))
            u.Build()
            shape = u.Shape()
        except Exception as e:
            log(f"  WAARSCHUWING: vlakken samenvoegen mislukt ({e})")
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
    info = {"kind": kind, "triangles": int(n_faces_before), "brep_faces": int(nf), "failed_triangles": n_fail,
            "valid": bool(valid), "closed_mesh": bool(closed)}
    return shape, info


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


def write_step(parts: list, path: Path, assembly_name="machine"):
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
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.product.name", assembly_name)
    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    w.SetColorMode(True)
    if not w.Transfer(doc, STEPControl_AsIs):
        raise RuntimeError("STEP transfer mislukt")
    status = w.Write(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP schrijven mislukt: {path}")
    return path


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
    if r.ReadFile(str(path)) != IFSelect_RetDone:
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
