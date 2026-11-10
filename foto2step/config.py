"""Project configuration (YAML) with defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "project": "machine",
    # Alle maten in mm.
    "markers": {
        "dictionary": "DICT_5X5_250",
        "size_mm": 80.0,          # zijde van het zwarte ArUco-vierkant ZOALS GEPRINT (meet na!)
        "sizes_mm": {},           # per-id afwijkende maat, bv {200: 40.0}
        "scale_bars": [],         # [{a: id, b: id, distance_mm: 1000.0}] gemeten afstand tussen marker-middens
        "check_distances": [],    # zelfde vorm; ALLEEN ter controle (niet gebruikt voor schaal)
        "world": {"origin_id": None, "x_axis_id": None},  # origin-marker ligt op de vloer
        "min_views": 3,
    },
    "frames": {
        "video_fps": 2.0,         # frames per seconde uit video
        "blur_reject_quantile": 0.25,  # gooi de vaagste 25% weg
        "max_side": 0,            # 0 = originele resolutie behouden (aanbevolen voor nauwkeurigheid)
        "min_motion_px": 30.0,    # sla frames over die nauwelijks bewegen
    },
    "sfm": {
        "backend": "auto",        # auto | pycolmap | colmap_cli
        "colmap_bin": "colmap",
        "camera_model": "OPENCV", # OPENCV = k1,k2,p1,p2 (telefoon)
        "single_camera": True,    # 1 lens, vaste zoom!
        "matcher": "exhaustive",  # exhaustive | sequential (lange video's)
        "max_features": 16384,
        "use_gpu": True,
    },
    "dense": {
        "backend": "auto",        # auto | colmap | openmvs | none
        "openmvs_bin": "",        # map met DensifyPointCloud etc. (leeg = PATH)
        "resolution_level": 0,    # 0 = volle resolutie (traag maar nauwkeurig), 1 = half
        "max_resolution": 4096,
        "number_views_fuse": 3,
    },
    "mesh": {
        "voxel_mm": 1.0,          # puntenwolk-uitdunning
        "poisson_depth": 0,       # 0 = automatisch uit voxel_mm
        "poisson_max_depth": 11,  # 11 ~ 4 GB RAM, 12 ~ 12-16 GB RAM (fijner detail op grote machines)
        "density_trim_quantile": 0.0,  # 0 = alleen afstand-gebaseerd trimmen
        "crop_margin_mm": 300.0,  # marge rond markers/object
        "remove_floor": True,     # vloer-markers definieren vloervlak; vloer wordt verwijderd
        "floor_clearance_mm": 3.0,
        "floor_snap_mm": 40.0,    # open onderrand lager dan dit -> doorgetrokken tot de vloer
        "target_max_error_mm": 0.15,  # decimatie mag max zoveel afwijken
        "max_triangles": 400000,
        "fill_holes_mm": 3000.0,
        "working_triangles": 1200000,  # max. driehoeken per mesh tijdens verwerking (geheugen)  # gaten tot deze straal dichten (nodig voor solids)
    },
    "features": {
        "enable": True,
        "plane_tol_mm": 0.0,      # 0 = automatisch uit ruis
        "min_region_area_mm2": 400.0,
        "angle_deg": 12.0,
        "cylinders": True,
        "sharpen_edges": True,
    },
    "motion": {
        "move_threshold_mm": 3.0, # verplaatsing waarboven een marker 'bewogen' heet
        "min_markers_per_body": 1,
        "label_tol_mm": 0.0,      # 0 = automatisch
        "manual_bodies": [],      # [{name:, box_min:[x,y,z], box_max:[x,y,z]}] handmatige override
    },
    "export": {
        "step_schema": "AP214",
        "merge_coplanar": True,
        "urdf": True,
    },
    "states": [],                 # [{name: A, images: pad/naar/fotos_of_video}]
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    cfg["_root"] = str(path.parent.resolve())
    validate(cfg)
    return cfg


def default_config() -> dict:
    return copy.deepcopy(DEFAULTS)


def validate(cfg: dict) -> None:
    m = cfg["markers"]
    if float(m["size_mm"]) <= 0:
        raise ValueError("markers.size_mm moet > 0 zijn")
    for sb in list(m.get("scale_bars", [])) + list(m.get("check_distances", [])):
        for k in ("a", "b", "distance_mm"):
            if k not in sb:
                raise ValueError(f"scale_bars/check_distances regel mist '{k}': {sb}")
    if not cfg.get("states"):
        raise ValueError("Geen 'states' gedefinieerd. Geef minstens 1 state met 'images'.")
    names = [s["name"] for s in cfg["states"]]
    if len(set(names)) != len(names):
        raise ValueError("State-namen moeten uniek zijn")


def resolve(cfg: dict, p: str) -> Path:
    q = Path(p)
    if not q.is_absolute():
        q = Path(cfg.get("_root", ".")) / q
    return q


EXAMPLE_YAML = """\
# foto2step projectbestand  (alle maten in mm)
project: machine_01

markers:
  dictionary: DICT_5X5_250
  size_mm: 80.0            # MEET het geprinte zwarte vierkant na met een schuifmaat!
  # sizes_mm: {200: 40.0}  # kleinere markers op bewegende delen
  scale_bars:              # stijve lat met 2 markers, afstand gemeten (hart-hart)
    - {a: 0, b: 1, distance_mm: 1000.0}
  check_distances:         # extra gemeten afstanden, alleen ter CONTROLE
    - {a: 2, b: 3, distance_mm: 1500.0}
  world:
    origin_id: 0           # vloer-marker = oorsprong, Z omhoog
    x_axis_id: 1           # richting oorsprong -> deze marker = +X

states:                    # 1 state = 1 stand van de machine. Meerdere standen => bewegende delen
  - name: A
    images: fotos/stand_A  # map met foto's, of een videobestand
  - name: B
    images: fotos/stand_B

dense:
  backend: auto            # colmap (NVIDIA GPU) of openmvs (CPU)
"""
