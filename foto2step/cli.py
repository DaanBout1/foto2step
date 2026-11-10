"""Command line interface.

  python -m foto2step markers --ids 0-11 --size 100 -o markers_vloer.pdf
  python -m foto2step init mijn_machine           (maakt projectmap + config.yaml)
  python -m foto2step run mijn_machine/config.yaml
  python -m foto2step selftest                    (gauntlet op synthetische machine met bekende maten)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ids(s: str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="foto2step", description="Foto's/video -> nauwkeurige STEP-bestanden")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("markers", help="printbare ArUco-markers (PDF op ware grootte)")
    m.add_argument("--ids", default="0-11")
    m.add_argument("--size", type=float, default=100.0, help="zijde zwart vierkant in mm")
    m.add_argument("--dict", default="DICT_5X5_250")
    m.add_argument("-o", "--out", default="markers.pdf")

    i = sub.add_parser("init", help="nieuw project aanmaken")
    i.add_argument("folder")

    r = sub.add_parser("run", help="volledige pijplijn draaien (hervat waar hij gebleven was)")
    r.add_argument("config")
    r.add_argument("--work", default=None, help="werkmap (standaard <projectmap>/work)")
    r.add_argument("--out", default=None, help="uitvoermap (standaard <projectmap>/output)")
    r.add_argument("--force", action="store_true", help="alles opnieuw berekenen")
    r.add_argument("--until", default=None, help="stop na stap: frames|sfm|calibrate|dense|mesh|bodies|parts|report")
    r.add_argument("--redo", default=None, help="vanaf deze stap opnieuw (wist latere resultaten)")

    s = sub.add_parser("selftest", help="gauntlet: synthetische machine met bekende maten end-to-end narekenen")
    s.add_argument("--dir", default="selftest_run")
    s.add_argument("--quick", action="store_true", help="kleinere beelden, sneller (minder nauwkeurig)")
    s.add_argument("--openmvs-bin", default="", help="map met OpenMVS binaries")

    c = sub.add_parser("check", help="controleer installatie (COLMAP/OpenMVS/OCC)")

    a = ap.parse_args(argv)
    if a.cmd == "markers":
        from .markers import write_marker_pdf
        p = write_marker_pdf(a.out, a.dict, _ids(a.ids), a.size)
        print(f"Geschreven: {p}  (print op 100%, meet daarna het zwarte vierkant na!)")
    elif a.cmd == "init":
        from .config import EXAMPLE_YAML
        d = Path(a.folder)
        (d / "fotos" / "stand_A").mkdir(parents=True, exist_ok=True)
        (d / "fotos" / "stand_B").mkdir(parents=True, exist_ok=True)
        cfgp = d / "config.yaml"
        if not cfgp.exists():
            cfgp.write_text(EXAMPLE_YAML.replace("machine_01", d.name), encoding="utf-8")
        print(f"Project aangemaakt in {d}. Pas {cfgp} aan en zet foto's/video's in fotos/stand_A (en _B ...).")
    elif a.cmd == "run":
        from .config import load_config
        from .pipeline import Pipeline
        cfg = load_config(a.config)
        root = Path(a.config).resolve().parent
        work = Path(a.work) if a.work else root / "work"
        out = Path(a.out) if a.out else root / "output"
        if a.redo:
            _clear_from(work, a.redo, [s["name"] for s in cfg["states"]])
        Pipeline(cfg, work, out, a.force).run(a.until)
        print(f"\nKlaar. STEP-bestanden en rapport in: {out}")
    elif a.cmd == "selftest":
        from .selftest import run_selftest
        ok = run_selftest(Path(a.dir), quick=a.quick, openmvs_bin=a.openmvs_bin)
        sys.exit(0 if ok else 1)
    elif a.cmd == "check":
        from .doctor import doctor
        sys.exit(0 if doctor() else 1)


def _clear_from(work: Path, step: str, states):
    import shutil
    order = ["frames", "sfm", "calibrate", "dense", "mesh", "bodies", "parts", "report"]
    if step not in order:
        raise SystemExit(f"onbekende stap {step}")
    k = order.index(step)
    per_state = {"frames": ["images", "images_info.json"], "sfm": ["sparse", "sparse_raw", "database.db", "sfm_stats.json"],
                 "calibrate": ["calibration.json", "markers_world.json", "sparse_world"],
                 "dense": ["dense", "dense.ply"], "mesh": ["mesh_full.ply", "cloud_clean.ply", "noise.json"]}
    for st in order[k:]:
        for s in states:
            for f in per_state.get(st, []):
                p = work / s / f
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
        if st == "bodies":
            for p in (work / "bodies.json", work / "bodies"):
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
        if st == "parts" and (work / "parts.json").exists():
            (work / "parts.json").unlink()


if __name__ == "__main__":
    main()
