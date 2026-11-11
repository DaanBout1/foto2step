"""Static preview images (PNG) of the reconstructed parts for the report."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import open3d as o3d


def render_parts_png(meshes: list, colors: list, views=((30, -60), (25, 30)), max_tris=25000, size=(9, 5)) -> str:
    """Return a data-URI PNG with the assembly from a few viewpoints (matplotlib, headless)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    light = np.array([0.4, -0.5, 0.8])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=size, dpi=110)
    allv = np.vstack([np.asarray(m.vertices) for m in meshes if len(m.vertices)])
    lo, hi = allv.min(0), allv.max(0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2
    for vi, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), vi + 1, projection="3d")
        for m, col in zip(meshes, colors):
            if len(m.triangles) == 0:
                continue
            mm = m if len(m.triangles) <= max_tris else m.simplify_quadric_decimation(max_tris)
            V = np.asarray(mm.vertices)
            F = np.asarray(mm.triangles)
            tri = V[F]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            shade = 0.35 + 0.65 * np.abs(n @ light)
            fc = np.clip(np.outer(shade, col), 0, 1)
            pc = Poly3DCollection(tri, facecolors=fc, edgecolors="none", linewidths=0)
            ax.add_collection3d(pc)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(max(c[2] - r, lo[2] - 10), max(c[2] - r, lo[2] - 10) + 2 * r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_final_meshes(work: Path, bodies: list):
    out = []
    for h, b in enumerate(bodies):
        f = Path(work) / "bodies" / f"{h:02d}_{b['name']}_final.ply"
        out.append(o3d.io.read_triangle_mesh(str(f)) if f.exists() else o3d.geometry.TriangleMesh())
    return out
