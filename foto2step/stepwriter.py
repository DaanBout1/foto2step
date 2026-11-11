"""Streaming STEP AP214 writer for triangle meshes with merged coplanar faces.

Writes the B-rep directly as text (no OCC objects in memory), so parts with hundreds of thousands
of triangles stay cheap. Coplanar triangle groups become one planar ADVANCED_FACE (outer + inner
loops); all other triangles become planar triangle faces. Edges and vertices are shared, so a
closed mesh becomes a MANIFOLD_SOLID_BREP. Produces single-part files and assemblies with named,
coloured parts (NEXT_ASSEMBLY_USAGE_OCCURRENCE), in millimetres.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _f(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    if s in ("-0.0", "-0."):
        s = "0.0"
    return s


def _p3(v) -> str:
    return f"({_f(v[0])},{_f(v[1])},{_f(v[2])})"


def _s(name: str) -> str:
    return "'" + str(name).replace("'", "''").encode("ascii", "replace").decode() + "'"


class _Out:
    def __init__(self, fh):
        self.fh = fh
        self.n = 0

    def add(self, text: str) -> int:
        self.n += 1
        self.fh.write(f"#{self.n}={text};\n")
        return self.n

    def reserve(self) -> int:
        self.n += 1
        return self.n

    def put(self, i: int, text: str):
        self.fh.write(f"#{i}={text};\n")


def _header(fh, name):
    t = time.strftime("%Y-%m-%dT%H:%M:%S")
    fh.write("ISO-10303-21;\nHEADER;\n")
    fh.write("FILE_DESCRIPTION(('foto2step model'),'2;1');\n")
    fh.write(f"FILE_NAME({_s(name)},'{t}',('foto2step'),(''),'foto2step','foto2step','');\n")
    fh.write("FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n")


def _context(o: _Out) -> int:
    lu = o.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
    au = o.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    su = o.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    unc = o.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{lu},'distance_accuracy_value','confusion accuracy')")
    return o.add(f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) "
                 f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su})) REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )")


def _axis(o: _Out, origin=(0, 0, 0), z=(0, 0, 1), x=(1, 0, 0)) -> int:
    p = o.add(f"CARTESIAN_POINT('',{_p3(origin)})")
    dz = o.add(f"DIRECTION('',{_p3(z)})")
    dx = o.add(f"DIRECTION('',{_p3(x)})")
    return o.add(f"AXIS2_PLACEMENT_3D('',#{p},#{dz},#{dx})")


def _perp(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = a - (a @ n) * n
    return x / np.linalg.norm(x)


def _brep(o: _Out, V: np.ndarray, F: np.ndarray, merge_coplanar=True):
    """Write topology + geometry. Returns (body_item_id, is_solid, n_faces, n_planar)."""
    from .stepexport import _boundary_loops, _coplanar_groups, _is_closed
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    closed_full = _is_closed(F) if len(F) else False
    tri = V[F]
    ar = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    F = F[ar > 1e-10]  # zero-area slivers have no plane; the gap they leave has zero area
    pts = {}
    vtx = {}

    def point(i):
        p = pts.get(i)
        if p is None:
            p = o.add(f"CARTESIAN_POINT('',{_p3(V[i])})")
            pts[i] = p
        return p

    def vertex(i):
        v = vtx.get(i)
        if v is None:
            v = o.add(f"VERTEX_POINT('',#{point(i)})")
            vtx[i] = v
        return v

    edges = {}

    def edge(a, b):
        """Return oriented edge id for traversal a->b."""
        key = (a, b) if a < b else (b, a)
        e = edges.get(key)
        if e is None:
            d = V[key[1]] - V[key[0]]
            L = float(np.linalg.norm(d))
            d = d / L if L > 0 else np.array([1.0, 0, 0])
            dr = o.add(f"DIRECTION('',{_p3(d)})")
            vec = o.add(f"VECTOR('',#{dr},{_f(max(L, 1e-9))})")
            ln = o.add(f"LINE('',#{point(key[0])},#{vec})")
            e = o.add(f"EDGE_CURVE('',#{vertex(key[0])},#{vertex(key[1])},#{ln},.T.)")
            edges[key] = e
        return o.add(f"ORIENTED_EDGE('',*,*,#{e},{'.T.' if a < b else '.F.'})")

    def loop(lp):
        oe = [edge(lp[i], lp[(i + 1) % len(lp)]) for i in range(len(lp))]
        return o.add("EDGE_LOOP('',(" + ",".join(f"#{x}" for x in oe) + "))")

    def plane(origin_idx, n):
        dz = o.add(f"DIRECTION('',{_p3(n)})")
        dx = o.add(f"DIRECTION('',{_p3(_perp(n))})")
        ax = o.add(f"AXIS2_PLACEMENT_3D('',#{point(origin_idx)},#{dz},#{dx})")
        return o.add(f"PLANE('',#{ax})")

    faces = []
    in_group = np.zeros(len(F), bool)
    n_planar = 0
    if merge_coplanar and len(F) > 1:
        for gf, normal in _coplanar_groups(V, F, 0.005, 0.2):
            loops = _boundary_loops(F[gf])
            if not loops:
                continue

            def sarea(lp):
                P = V[lp]
                return 0.5 * np.cross(P - P[0], np.roll(P, -1, 0) - P[0]).sum(0) @ normal
            areas = [sarea(lp) for lp in loops]
            if sum(1 for a in areas if a > 0) != 1:
                continue  # pinched / several outer loops: keep as triangles
            k = int(np.argmax(areas))
            bounds = [o.add(f"FACE_OUTER_BOUND('',#{loop(loops[k])},.T.)")]
            for i, lp in enumerate(loops):
                if i != k:
                    bounds.append(o.add(f"FACE_BOUND('',#{loop(lp)},.T.)"))
            pl = plane(loops[k][0], normal)
            faces.append(o.add("ADVANCED_FACE('',(" + ",".join(f"#{b}" for b in bounds) + f"),#{pl},.T.)"))
            in_group[gf] = True
            n_planar += 1
    for a, b, c in F[~in_group]:
        n = np.cross(V[b] - V[a], V[c] - V[a])
        n = n / np.linalg.norm(n)
        bd = o.add(f"FACE_OUTER_BOUND('',#{loop([a, b, c])},.T.)")
        pl = plane(a, n)
        faces.append(o.add(f"ADVANCED_FACE('',(#{bd}),#{pl},.T.)"))
    closed = closed_full
    flist = ",".join(f"#{x}" for x in faces)
    if closed:
        sh = o.add(f"CLOSED_SHELL('',({flist}))")
        item = o.add(f"MANIFOLD_SOLID_BREP('',#{sh})")
    else:
        sh = o.add(f"OPEN_SHELL('',({flist}))")
        item = o.add(f"SHELL_BASED_SURFACE_MODEL('',(#{sh}))")
    return item, closed, len(faces), n_planar


def _color(o: _Out, item: int, ctx: int, rgb):
    if rgb is None:
        return
    col = o.add(f"COLOUR_RGB('',{_f(rgb[0])},{_f(rgb[1])},{_f(rgb[2])})")
    fac = o.add(f"FILL_AREA_STYLE_COLOUR('',#{col})")
    fas = o.add(f"FILL_AREA_STYLE('',(#{fac}))")
    sfa = o.add(f"SURFACE_STYLE_FILL_AREA(#{fas})")
    sss = o.add(f"SURFACE_SIDE_STYLE('',(#{sfa}))")
    ssu = o.add(f"SURFACE_STYLE_USAGE(.BOTH.,#{sss})")
    psa = o.add(f"PRESENTATION_STYLE_ASSIGNMENT((#{ssu}))")
    si = o.add(f"STYLED_ITEM('color',(#{psa}),#{item})")
    o.add(f"MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#{si}),#{ctx})")


def _product(o: _Out, app: int, name: str):
    pc = o.add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    pr = o.add(f"PRODUCT({_s(name)},{_s(name)},'',(#{pc}))")
    o.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{pr}))")
    pdf = o.add(f"PRODUCT_DEFINITION_FORMATION('','',#{pr})")
    pdc = o.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    pd = o.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdc})")
    pds = o.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    return pd, pds


def _part(o: _Out, app: int, name, V, F, rgb, merge_coplanar):
    pd, pds = _product(o, app, name)
    ctx = _context(o)
    org = _axis(o)
    item, closed, nf, npl = _brep(o, V, F, merge_coplanar)
    kind = "ADVANCED_BREP_SHAPE_REPRESENTATION" if closed else "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"
    rep = o.add(f"{kind}('',(#{org},#{item}),#{ctx})")
    o.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    _color(o, item, ctx, rgb)
    return pd, rep, org, {"kind": "solid" if closed else "shell", "brep_faces": nf, "planar_faces": npl,
                          "triangles": int(len(F)), "closed_mesh": bool(closed)}


def write_part(path: Path, name: str, V, F, rgb=None, merge_coplanar=True) -> dict:
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        _header(fh, name)
        o = _Out(fh)
        app = o.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
        o.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
        _, _, _, info = _part(o, app, name, V, F, rgb, merge_coplanar)
        fh.write("ENDSEC;\nEND-ISO-10303-21;\n")
    return info


def write_assembly(path: Path, name: str, parts: list, merge_coplanar=True) -> list:
    """parts: list of (name, V, F, rgb). All parts are placed with identity transforms (world coordinates)."""
    infos = []
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        _header(fh, name)
        o = _Out(fh)
        app = o.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
        o.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
        apd, apds = _product(o, app, name)
        actx = _context(o)
        aorg = _axis(o)
        placements = [_axis(o) for _ in parts]
        arep = o.add("SHAPE_REPRESENTATION('',(" + ",".join(f"#{x}" for x in [aorg] + placements) + f"),#{actx})")
        o.add(f"SHAPE_DEFINITION_REPRESENTATION(#{apds},#{arep})")
        for k, (pname, V, F, rgb) in enumerate(parts):
            pd, rep, org, info = _part(o, app, pname, V, F, rgb, merge_coplanar)
            nauo = o.add(f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('{k + 1}',{_s(pname)},'',#{apd},#{pd},$)")
            pds = o.add(f"PRODUCT_DEFINITION_SHAPE('Placement','Placement of an item',#{nauo})")
            idt = o.add(f"ITEM_DEFINED_TRANSFORMATION('','',#{org},#{placements[k]})")
            rr = o.add(f"( REPRESENTATION_RELATIONSHIP('','',#{rep},#{arep}) "
                       f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{idt}) SHAPE_REPRESENTATION_RELATIONSHIP() )")
            o.add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{rr},#{pds})")
            infos.append(info)
        fh.write("ENDSEC;\nEND-ISO-10303-21;\n")
    return infos
