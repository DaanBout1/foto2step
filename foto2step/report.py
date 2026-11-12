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
    a("<style>"
      ":root{--ink:#1d2329;--mut:#5b6670;--line:#e3e7eb;--bg:#f6f8fa;--ok:#17803d;--warn:#b26b00;--bad:#c0262d;--acc:#1f5fbf}"
      "*{box-sizing:border-box}body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;"
      "color:var(--ink);background:var(--bg);line-height:1.45}"
      ".wrap{max-width:1120px;margin:0 auto;padding:28px 24px 60px}"
      "header{background:#fff;border-bottom:1px solid var(--line)}header .wrap{padding:22px 24px}"
      "h1{font-size:1.5rem;margin:0}h2{font-size:1.1rem;margin:2rem 0 .6rem}.sub{color:var(--mut);margin-top:.2rem}"
      ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}"
      ".card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px}"
      ".card .k{color:var(--mut);font-size:.82rem}.card .v{font-size:1.45rem;font-weight:650;margin-top:2px}"
      ".panel{background:#fff;border:1px solid var(--line);border-radius:10px;padding:6px 6px;overflow-x:auto}"
      "table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{padding:.45rem .7rem;text-align:left;"
      "border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mut);font-weight:600;font-size:.8rem;"
      "text-transform:uppercase;letter-spacing:.03em}tr:last-child td{border-bottom:none}"
      ".ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}"
      ".note{color:var(--mut);font-size:.88rem}.preview{width:100%;border-radius:10px;border:1px solid var(--line);background:#fff}"
      "ul.adv li{margin:.3rem 0}</style></head><body>")
    import time as _t
    a(f"<header><div class='wrap'><h1>{html.escape(cfg['project'])}</h1>"
      f"<div class='sub'>foto2step rapport · {_t.strftime('%d-%m-%Y %H:%M')} · {len(states)} stand(en) · "
      f"{len(parts)} onderdeel/onderdelen</div></div></header><div class='wrap'>")
    est = acc["estimate_95_mm"]
    cls = "ok" if est <= 0.5 else "warn" if est <= 2 else "bad"
    chk = acc["check_max_abs_mm"]
    a("<div class='cards'>"
      f"<div class='card'><div class='k'>Geschatte nauwkeurigheid (95%)</div><div class='v {cls}'>± {_r(est)} mm</div></div>"
      f"<div class='card'><div class='k'>Meetvolume</div><div class='v'>{_r(acc['extent_mm'] / 1000, 2)} m</div></div>"
      f"<div class='card'><div class='k'>Schaal (2σ over meetvolume)</div><div class='v'>{_r(acc['scale_2sigma_at_extent_mm'])} mm</div></div>"
      f"<div class='card'><div class='k'>Oppervlak punten→STEP (95%)</div><div class='v'>{_r(acc['surface_p95_mm'])} mm</div></div>"
      f"<div class='card'><div class='k'>Controlematen (max. fout)</div><div class='v'>"
      + (f"{_r(chk)} mm" if chk is not None else "<span class='warn'>geen</span>") + "</div></div></div>")
    if chk is None:
        a("<p class='note warn'>Geen controlematen opgegeven: zet 'check_distances' in de config voor een onafhankelijke toets.</p>")
    # capture device(s): warnings first (e.g. iPhone Air -> lower accuracy)
    dev_rows, dev_warn, dev_adv = [], [], []
    for s_ in states:
        fi = work / s_ / "images_info.json"
        if not fi.exists():
            continue
        inf = json.loads(fi.read_text())
        dv = inf.get("device") or {}
        dev_warn += [w for w in dv.get("warnings", []) if w not in dev_warn]
        dev_adv += [x for x in dv.get("advice", []) if x not in dev_adv]
        for g, gi in (inf.get("groups") or {}).items():
            dev_rows.append((s_, g, gi))
    for w in dev_warn:
        a(f"<div class='card' style='border-color:#e9c46a;background:#fff8e6;margin:10px 0'><b class='warn'>Let op:</b> {html.escape(w)}</div>")
    a("<p class='note'>De nauwkeurigheid geldt voor oppervlakken die goed op de foto's stonden. Ingevulde gaten, blinde hoeken en "
      "glimmende/zwarte vlakken kunnen meer afwijken (zie kolom 'Ingevuld').</p>")
    try:
        from .preview import load_final_meshes, render_parts_png
        from .pipeline import PART_COLORS
        if bodies.get("bodies"):
            ms = load_final_meshes(work, bodies["bodies"])
            img = render_parts_png(ms, [PART_COLORS[i % len(PART_COLORS)] for i in range(len(ms))])
            a(f"<h2>Resultaat</h2><img class='preview' src='{img}' alt='voorbeeld'>")
    except Exception as e:  # preview is optional
        a(f"<p class='note'>(voorbeeldafbeelding niet beschikbaar: {html.escape(str(e))})</p>")
    if dev_rows:
        from .device import LENS_NL
        a("<h2>Opname</h2><div class='panel'><table><tr><th>Stand</th><th>Toestel</th><th>Lens</th><th>Foto's</th>"
          "<th>Formaat</th><th>Eigen kalibratie</th></tr>")
        for s_, g, gi in dev_rows:
            a(f"<tr><td>{s_}</td><td>{html.escape(str(gi.get('model') or 'onbekend'))}</td>"
              f"<td>{html.escape(LENS_NL.get(gi.get('lens_class'), gi.get('lens_class') or ''))}"
              f"{' · ' + str(int(gi['focal_35mm'])) + ' mm' if gi.get('focal_35mm') else ''}</td><td>{gi.get('count')}</td>"
              f"<td>{'ProRAW (DNG)' if gi.get('raw') else 'JPEG/HEIC'}</td><td>{html.escape(g)}</td></tr>")
        a("</table></div>")
    # per state
    a("<h2>Per stand</h2><div class='panel'><table><tr><th>Stand</th><th>Beelden</th><th>Reproj. fout</th><th>Markers</th>"
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
    a("</table></div>")
    rows = []
    for s in states:
        cal = json.loads((work / s / "calibration.json").read_text())
        for c in cal.get("checks", []):
            rows.append((s, c))
        for b in cal["scale"].get("bars", []):
            pass
    if rows:
        a("<h2>Onafhankelijke controlematen</h2><div class='panel'><table><tr><th>Stand</th><th>Markers</th><th>Gemeten</th><th>Uit foto's</th><th>Fout</th></tr>")
        for s, c in rows:
            e = abs(c["error_mm"])
            k = "ok" if e <= 0.5 else "warn" if e <= 2 else "bad"
            a(f"<tr><td>{s}</td><td>{c['a']}–{c['b']}</td><td>{_r(c['measured_mm'])}</td><td>{_r(c['photo_mm'])}</td>"
              f"<td class='{k}'>{c['error_mm']:+.2f} mm</td></tr>")
        a("</table></div>")
    # parts
    a("<h2>STEP-onderdelen</h2><div class='panel'><table><tr><th>Deel</th><th>Bestand</th><th>Type</th><th>B-rep vlakken</th>"
      "<th>Afwijking mediaan / 95%</th><th>Ingevuld (niet gezien)</th><th>Vlakken / cilinders</th><th>Volume</th></tr>")
    for p in parts:
        a(f"<tr><td>{html.escape(p['name'])}</td><td>{html.escape(p['step'])}</td><td>{p['kind']}"
          f"{'' if p.get('valid') is not False else ' <b class=warn>(check)</b>'}</td><td>{p['brep_faces']}</td>"
          f"<td>{_r(p['deviation_mm']['median'])} / {_r(p['deviation_mm']['p95'])} mm</td>"
          f"<td>{p.get('filled_fraction', 0) * 100:.1f}%</td>"
          f"<td>{p['n_planes']} / {len(p['cylinders'])}</td><td>{_r((p.get('volume_mm3') or 0) / 1e6, 3) if p.get('volume_mm3') else '-'} dm³</td></tr>")
    a("</table></div>")
    cyl = [(p["name"], c) for p in parts for c in p["cylinders"]]
    if cyl:
        a("<h2>Gevonden cilinders (assen, gaten, buizen)</h2><div class='panel'><table><tr><th>Deel</th><th>Ø</th><th>Lengte</th><th>As-richting</th><th>RMS</th></tr>")
        for n, c in sorted(cyl, key=lambda x: -x[1]["length"])[:60]:
            a(f"<tr><td>{html.escape(n)}</td><td>{_r(c['diameter'])} mm</td><td>{_r(c['length'], 1)} mm</td>"
              f"<td>{', '.join(_r(v, 3) for v in c['axis'])}</td><td>{_r(c['rms'], 3)}</td></tr>")
        a("</table></div>")
    if bodies.get("joints"):
        a("<h2>Bewegingen</h2><div class='panel'><table><tr><th>Deel</th><th>Type</th><th>As</th><th>Standen</th></tr>")
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
        a("</table></div>")
    # advice
    adv = list(dev_adv)
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
        a("<h2>Advies voor een nauwkeuriger resultaat</h2><ul class='adv'>" + "".join(f"<li>{html.escape(x)}</li>" for x in adv) + "</ul>")
    a("</div></body></html>")
    (out / "rapport.html").write_text("\n".join(H), encoding="utf-8")
    (out / "nauwkeurigheid.json").write_text(json.dumps(acc, indent=1))
