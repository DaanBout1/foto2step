"""Optional URDF (kinematic model) for simulators: base link + one joint per moving part."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


def write_urdf(work: Path, out: Path, project: str):
    info = json.loads((work / "bodies.json").read_text())
    joints = {j["body"]: j for j in info["joints"]}
    mdir = out / "urdf" / "meshes"
    mdir.mkdir(parents=True, exist_ok=True)
    lines = ['<?xml version="1.0"?>', f'<robot name="{project}">']
    base = None
    for h, b in enumerate(info["bodies"]):
        f = work / "bodies" / f"{h:02d}_{b['name']}_final.ply"
        if not f.exists():
            continue
        m = o3d.io.read_triangle_mesh(str(f))
        j = joints.get(b["name"])
        origin = np.zeros(3)
        if j and j["type"] == "revolute":
            origin = np.array(j["point"])
        elif j:
            origin = np.asarray(m.get_center())
        V = np.asarray(m.vertices) - origin
        m.vertices = o3d.utility.Vector3dVector(V / 1000.0)  # URDF in meters
        m.compute_triangle_normals()
        o3d.io.write_triangle_mesh(str(mdir / f"{b['name']}.stl"), m)
        lines += [f'  <link name="{b["name"]}">',
                  f'    <visual><geometry><mesh filename="meshes/{b["name"]}.stl"/></geometry></visual>',
                  f'    <collision><geometry><mesh filename="meshes/{b["name"]}.stl"/></geometry></collision>',
                  "  </link>"]
        if base is None:
            base = b["name"]
            continue
        if not j or j["type"] not in ("prismatic", "revolute"):
            jt = "fixed"
            lo = hi = 0.0
            axis = [1, 0, 0]
        elif j["type"] == "prismatic":
            jt = "prismatic"
            pos = list(j["positions_mm"].values())
            lo, hi = min(pos) / 1000.0, max(pos) / 1000.0
            axis = j["axis"]
        else:
            jt = "revolute"
            ang = list(j["angles_deg"].values())
            lo, hi = np.radians(min(ang)), np.radians(max(ang))
            axis = j["axis"]
        o = origin / 1000.0
        lines += [f'  <joint name="{b["name"]}_joint" type="{jt}">',
                  f'    <parent link="{base}"/><child link="{b["name"]}"/>',
                  f'    <origin xyz="{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}" rpy="0 0 0"/>',
                  f'    <axis xyz="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}"/>',
                  f'    <limit lower="{lo:.6f}" upper="{hi:.6f}" effort="1000" velocity="1"/>' if jt != "fixed" else "",
                  "  </joint>"]
    lines.append("</robot>")
    (out / "urdf" / f"{project}.urdf").write_text("\n".join(l for l in lines if l))
