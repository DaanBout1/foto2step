"""HTML accuracy report (Dutch): what was measured, how good it is, and what to improve."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np


def _r(x, n=2):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return "-"


def accuracy_summary(work: Path, states: list) -> dict:
    """Honest overall accuracy estimate (mm, ~95%)."""
    ref = states[0]
    cal = json.loads((work / ref / "calibration.json").read_text())
    parts = json.loads((work / "parts.json").read_text()) if (work / "parts.json").exists() else []
    scale_err = cal["scale_error_at_extent_mm"] * 2  # 2 sigma
    checks = [abs(c["error_mm"]) for s in states
              for c in json.loads((work / s / "calibration.json").read_text()).get("checks", [])]
    surf = max([p["deviation_mm"]["p95"] for p in parts], default=0.0)
    total = float(np.sqrt(scale_err ** 2 + surf ** 2))
    if checks:
        total = max(total, max(checks))
    return {"scale_2sigma_at_extent_mm": scale_err, "surface_p95_mm": surf,
            "check_max_abs_mm": max(checks) if checks else None, "estimate_95_mm": total,
            "extent_mm": cal["marker_extent_mm"]}


def write_report(work: Path, out: Path, cfg: dict, states: list):
    work = Path(work)
    acc = accuracy_summary(work, states)
    parts = json.loads((work / "parts.json").read_text()) if (work / "parts.json").exists() else []
    bodies = json.loads((work / "bodies.json").read_text()) if (work / "bodies.json").exists() else {}
    H = []
    a = H.append
    a("<!DOCTYPE html><html lang='nl'><head><meta charset='utf-8'><title>foto2step rapport</title>")
    a("<style>body{font-family:system-ui,Segoe UI,Arial;margin:2rem;max-width:1100px;color:#222}"
      "table{border-collapse:collapse;margin:.5rem 0 1.5rem}td,th{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}"
      "th{background:#f2f2f2}.ok{color:#17803d}.warn{color:#b26b00}.bad{color:#c0262d}.big{font-size:1.4rem}</style></head><body>")
    a(f"<h1>foto2step rapport: {html.escape(cfg['project'])}</h1>")
    est = acc["estimate_95_mm"]
    cls = "ok" if est <= 0.5 else "warn" if est <= 2 else "bad"
    a(f"<p class='big'>Geschatte nauwkeurigheid (95%): <b class='{cls}'>± {_r(est)} mm</b> "
      f"over {_r(acc['extent_mm'] / 1000, 2)} m meetvolume</p>")
    a("<p>Opgebouwd uit: schaalonzekerheid over het hele meetvolume (2σ) "
      f"{_r(acc['scale_2sigma_at_extent_mm'])} mm, oppervlakte-afwijking meetpunten→STEP (95%) "
      f"{_r(acc['surface_p95_mm'])} mm"
      + (f", grootste fout op onafhankelijke controlematen {_r(acc['check_max_abs_mm'])} mm" if acc['check_max_abs_mm'] is not None
         else ". <b class='warn'>Geen controlematen opgegeven: zet 'check_distances' in de config voor een onafhankelijke toets.</b>")
      + "</p>")
    a("<p><i>Let op: dit geldt voor oppervlakken die goed op de foto's stonden. Gaten, blinde hoeken en glimmende/zwarte "
      "vlakken zijn ingevuld/geïnterpoleerd en kunnen meer afwijken.</i></p>")
    # per state
    a("<h2>Per stand</h2><table><tr><th>Stand</th><th>Beelden</th><th>Reproj. fout</th><th>Markers</th>"
      "<th>Schaal-onzekerheid</th><th>Hoekpunt-precisie</th><th>Vloervlak RMS</th><th>Ruis</th></tr>")
    for s in states:
        d = work / s
        sfm = json.loads((d / "sfm_stats.json").read_text()) if (d / "sfm_stats.json").exists() else {}
        cal = json.loads((d / "calibration.json").read_text())
        mk = json.loads((d / "markers_world.json").read_text())
        nz = json.loads((d / "noise.json").read_text())["noise_mm"] if (d / "noise.json").exists() else None
        a(f"<tr><td>{s}</td><td>{sfm.get('registered', '-')}/{sfm.get('total', '-')}</td>"
          f"<td>{_r(sfm.get('mean_reproj_px'), 3)} px</td><td>{len(mk)}</td>"
          f"<td>{_r(cal['scale']['rel_sigma'] * 1e3, 3)} mm/m</td><td>{_r(cal['scale']['sigma_corner_mm'], 3)} mm</td>"
          f"<td>{_r(cal['frame']['floor_plane_rms_mm'], 3)} mm</td><td>{_r(nz, 3)} mm</td></tr>")
    a("</table>")
    rows = []
    for s in states:
        cal = json.loads((work / s / "calibration.json").read_text())
        for c in cal.get("checks", []):
            rows.append((s, c))
        for b in cal["scale"].get("bars", []):
            pass
    if rows:
        a("<h2>Onafhankelijke controlematen</h2><table><tr><th>Stand</th><th>Markers</th><th>Gemeten</th><th>Uit foto's</th><th>Fout</th></tr>")
        for s, c in rows:
            e = abs(c["error_mm"])
            k = "ok" if e <= 0.5 else "warn" if e <= 2 else "bad"
            a(f"<tr><td>{s}</td><td>{c['a']}–{c['b']}</td><td>{_r(c['measured_mm'])}</td><td>{_r(c['photo_mm'])}</td>"
              f"<td class='{k}'>{c['error_mm']:+.2f} mm</td></tr>")
        a("</table>")
    # parts
    a("<h2>STEP-onderdelen</h2><table><tr><th>Deel</th><th>Bestand</th><th>Type</th><th>B-rep vlakken</th>"
      "<th>Afwijking mediaan / 95%</th><th>Vlakken / cilinders</th><th>Volume</th></tr>")
    for p in parts:
        a(f"<tr><td>{html.escape(p['name'])}</td><td>{html.escape(p['step'])}</td><td>{p['kind']}"
          f"{'' if p['valid'] else ' <b class=warn>(check)</b>'}</td><td>{p['readback']['faces']}</td>"
          f"<td>{_r(p['deviation_mm']['median'])} / {_r(p['deviation_mm']['p95'])} mm</td>"
          f"<td>{p['n_planes']} / {len(p['cylinders'])}</td><td>{_r(p['readback']['volume_mm3'] / 1e6, 3)} dm³</td></tr>")
    a("</table>")
    cyl = [(p["name"], c) for p in parts for c in p["cylinders"]]
    if cyl:
        a("<h2>Gevonden cilinders (assen, gaten, buizen)</h2><table><tr><th>Deel</th><th>Ø</th><th>Lengte</th><th>As-richting</th><th>RMS</th></tr>")
        for n, c in sorted(cyl, key=lambda x: -x[1]["length"])[:60]:
            a(f"<tr><td>{html.escape(n)}</td><td>{_r(c['diameter'])} mm</td><td>{_r(c['length'], 1)} mm</td>"
              f"<td>{', '.join(_r(v, 3) for v in c['axis'])}</td><td>{_r(c['rms'], 3)}</td></tr>")
        a("</table>")
    if bodies.get("joints"):
        a("<h2>Bewegingen</h2><table><tr><th>Deel</th><th>Type</th><th>As</th><th>Standen</th></tr>")
        for j in bodies["joints"]:
            if j["type"] == "prismatic":
                st = ", ".join(f"{k}: {v:+.2f} mm" for k, v in j["positions_mm"].items())
                ax = ", ".join(_r(v, 4) for v in j["axis"])
            elif j["type"] == "revolute":
                st = ", ".join(f"{k}: {v:+.2f}°" for k, v in j["angles_deg"].items())
                ax = ", ".join(_r(v, 4) for v in j["axis"]) + " door (" + ", ".join(_r(v, 1) for v in j["point"]) + ")"
            else:
                st, ax = "-", "-"
            a(f"<tr><td>{html.escape(j['body'])}</td><td>{j['type']}</td><td>{ax}</td><td>{st}</td></tr>")
        a("</table>")
    # advice
    adv = []
    cal0 = json.loads((work / states[0] / "calibration.json").read_text())
    if not cal0["scale"].get("bars"):
        adv.append("Gebruik een schaallat (2 markers op een stijve aluminium lat, hart-hart gemeten). Dat verbetert de schaal sterk.")
    if acc["check_max_abs_mm"] is None:
        adv.append("Meet 1–2 extra afstanden tussen markers na (rolmaat/schuifmaat) en zet ze onder check_distances.")
    if acc["surface_p95_mm"] > 1.0:
        adv.append("Oppervlakteruis is hoog: meer foto's van dichterbij, scan-spray of speckle-stickers op egale/glimmende delen.")
    if cal0["scale"]["sigma_corner_mm"] > 0.3:
        adv.append("Marker-hoekpunten zijn onnauwkeurig: scherpere foto's, markers vlak geplakt, mat papier, geen bewegingsonscherpte.")
    if adv:
        a("<h2>Advies voor een nauwkeuriger resultaat</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in adv) + "</ul>")
    a("</body></html>")
    (out / "rapport.html").write_text("\n".join(H), encoding="utf-8")
    (out / "nauwkeurigheid.json").write_text(json.dumps(acc, indent=1))
