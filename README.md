# foto2step

Telefoonfoto's of video → geschaalde 3D-reconstructie → **STEP-bestanden** (solids) per onderdeel,
met de **bewegende delen als aparte STEP** en hun bewegingsassen. Bedoeld voor digital twins en simulaties.

```
foto's/video per stand ──► SfM (COLMAP) ──► ArUco-markers: schaal in mm + assenstelsel
        ──► dichte puntenwolk (COLMAP-CUDA of OpenMVS) ──► mesh
        ──► bewegende delen uit meerdere standen (markers + vrije-ruimte-redenering)
        ──► per deel: vlakken/cilinders herkennen, ruis wegprojecteren, randen aanscherpen
        ──► B-rep solid ──► STEP (AP214, mm), per deel + samenstelling, + rapport met nauwkeurigheid
```

## Wat kun je verwachten (eerlijk)

| Situatie | Realistische nauwkeurigheid (95%) |
|---|---|
| Onderdeel 5–30 cm, markers, goed fotoprotocol, 12 MP+ | 0,2 – 0,5 mm |
| Machine 1–2 m, schaallat + markers, 100+ scherpe foto's | 0,5 – 1,5 mm |
| Machine 2–4 m, alleen foto's | 1 – 3 mm |
| Glimmende, zwarte of egale vlakken zonder spray | slechter, soms gaten |
| Alleen video (4K) | ca. 1,5× slechter dan foto's |

Het **rapport** (`output/rapport.html`) meet dit per project: schaalonzekerheid, ruis, afwijking
punten→STEP en (als je ze opgeeft) onafhankelijke controlematen. Vertrouw op dat getal, niet op deze tabel.

De STEP bevat een **facetmodel** (vlakke stukjes) waarbij alle vlakke delen van de machine zijn
samengevoegd tot echte vlakken. Cilinders/gaten worden herkend en gemeld in het rapport (diameter, as),
maar zijn in de STEP gefacetteerd. Voor digital twin, botsingscontrole en simulatie is dat prima.

**Solid of schil?** Een onderdeel wordt een gesloten *solid* als het rondom gefotografeerd is. Vlakken die
nooit zichtbaar waren (onderkant van een arm, het contactvlak waar een slede op het frame ligt) blijven
standaard open; dan wordt het een *schil* (surface body), want een verzonnen dichting zou niet nauwkeurig zijn.
Wil je altijd solids (bv. voor massa/botsing), zet dan `export: force_solid: true`; het rapport toont per deel
hoeveel procent "ingevuld" is.

## Installatie (Windows, eenmalig)

1. Installeer **Miniforge**: https://github.com/conda-forge/miniforge (Windows x86_64 installer).
2. Open "Miniforge Prompt", ga naar deze map en:
   ```
   conda env create -f environment.yml
   conda activate foto2step
   python -m foto2step check
   ```
   Dit installeert ook COLMAP en OpenMVS (conda-forge). Met een NVIDIA-kaart gaat alles veel sneller.
3. Test of alles werkt (gauntlet op een synthetische machine met exact bekende maten; 1–3 uur op CPU, veel sneller met NVIDIA):
   ```
   python -m foto2step selftest
   ```
   Eindigt met `GAUNTLET GESLAAGD` en een lijst PASS/FAIL per controle.

## Draaien in Visual Studio Code

1. Installeer eenmalig de conda-omgeving (zie hierboven) en VS Code met de extensie **Python**.
2. Open de map `foto2step` in VS Code (File → Open Folder).
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → kies **foto2step** (de conda-omgeving).
4. Ga naar **Run and Debug** (`Ctrl+Shift+D`) en kies bovenin:
   * *foto2step: installatiecheck*
   * *foto2step: markers printen*
   * *foto2step: run (kies config.yaml)*: vraagt om het pad naar je `config.yaml`
   * *foto2step: selftest (gauntlet)*
5. Druk op **F5**. De voortgang verschijnt in de terminal onderin.

Unit tests: het Testing-paneel (erlenmeyer-icoon) vindt `tests/` vanzelf.

## Voorbereiden: markers en schaallat

```
python -m foto2step markers --ids 0-15 --size 120 -o vloer.pdf       # vloer rondom de machine
python -m foto2step markers --ids 100-115 --size 60 -o frame.pdf     # vast frame van de machine
python -m foto2step markers --ids 200-231 --size 40 -o delen.pdf     # bewegende delen
```

* Print op **100%** (niet "passend maken"), controleer de 100 mm-lijn, **meet het zwarte vierkant na** met een schuifmaat.
* Plak markers op iets stijfs en vlaks (dibond/alu/3 mm MDF): papier bolt op.
* **Vloer**: 8–16 markers rondom de machine, plat op de vloer. Kies er één als oorsprong (`origin_id`), één als X-richting (`x_axis_id`).
* **Schaallat**: 2 markers op een stijf alu-profiel van ~1 m. Meet hart-hart met een schuifmaat/rolmaat (`scale_bars`). Liefst 2 latten in verschillende richtingen. Dit is de belangrijkste factor voor nauwkeurigheid op grote maten.
  Met schaallatten bepalen die de schaal; de markergroottes dienen dan alleen als controle. Het programma meldt
  het als je printer de markers net te groot/klein heeft geprint.
* **Zonder markers** (bv. een willekeurige video van internet) kan het programma geen millimeters bepalen:
  dan bestaat er geen schaal. Markers zijn dus verplicht.
* **Controlemaat** (aanrader): meet 1–2 afstanden tussen andere vloermarkers en zet die onder `check_distances`. Die worden niet gebruikt voor de schaal, alleen om de fout eerlijk te meten.
* **Vast frame**: een paar markers op het machineframe.
* **Elk bewegend deel**: minstens **2 markers** (liefst 3, ver uit elkaar, op verschillende vlakken). Zo herkent het programma welk deel beweegt en hoe.

## Fotoprotocol (bepaalt 90% van het resultaat)

**Telefooninstellingen**
* Hoofdcamera, **1× zoom, nooit wisselen** van lens (iPhone: "Macrobediening" uit; Android: Pro-modus).
* **Focus en belichting vastzetten** (lang tikken → AE/AF-vergrendeling). Geen HDR/portret/nachtmodus.
* Hoogste resolutie, geen flits. JPG en HEIC (iPhone) werken allebei. Staand en liggend mag door elkaar:
  het programma draait alles terug naar de sensororiëntatie, zodat één lenskalibratie voor alle foto's geldt.
* Foto's met een ander beeldformaat (andere lens/zoom) worden overgeslagen en gemeld.

**Machine-oppervlak**
* Egale lak, glimmend metaal, zwart of transparant = slecht. Gebruik **scanspray** (bv. AESUB, verdampt vanzelf) of plak
  willekeurige stippen/speckle-stickers of afplaktape met krabbels. Hoe meer willekeurige textuur, hoe beter.
* Diffuus licht, geen bewegende schaduwen (blijf zelf uit de lamp).

**Fotograferen (per stand)**
* Loop **3 rondes** om de machine: laag (kniehoogte), midden (ooghoogte), hoog (arm omhoog / trapje), elke **10–15°** een foto.
* Daarna **close-ups** van details (tooling, bewegende delen) met 60–80% overlap, in kleine stapjes, geen sprongen.
* Elke marker moet op minstens 5 foto's scherp staan, vanuit verschillende hoeken.
* Reken op 100–250 foto's voor een machine. Rekentijd maakt niet uit, foto's die ontbreken wel.
* **Video mag ook** (ook meerdere per stand): 4K, langzaam lopen, focus vast. Het programma kiest zelf de scherpste frames (`frames.video_fps`).

**Bewegende delen: meerdere standen**
* Stand A: alle assen in een stand → volledige fotoronde.
* Stand B: beweeg de assen (elk minstens ~20 mm of 10°) → weer een volledige fotoronde. Vloermarkers en schaallat **niet** verplaatsen.
* Extra standen (C, D, ...) maken de assen nauwkeuriger en de delen completer (wat in A verborgen zat, komt in B in beeld).
* Tip: beweeg per stand liefst meerdere assen tegelijk een **verschillende** hoeveelheid; alles wat altijd samen beweegt wordt als één deel gezien.

## Gebruik

```
python -m foto2step init machine_01          # maakt map + config.yaml
# zet foto's in machine_01/fotos/stand_A, stand_B ... en pas config.yaml aan (marker-maten!)
python -m foto2step run machine_01/config.yaml
```

Het programma hervat waar het gebleven was. Opnieuw vanaf een stap: `--redo dense` (of `sfm`, `calibrate`, `mesh`, `bodies`, `parts`).

**Uitvoer** (`machine_01/output/`)
* `<project>_samenstelling.step`: alle delen als genoemde onderdelen in één assembly (mm, wereldassenstelsel: Z omhoog, oorsprong op de oorsprongmarker).
* `onderdelen/<deel>.step`: elk deel los (vast frame = `basis`, bewegende delen `deel_1`, ...).
* `bewegingen.json`: per bewegend deel type (lineair/roterend), as-richting, draaipunt, gemeten standen.
* `rapport.html`: nauwkeurigheid, controlematen, gevonden cilinders/gaten, advies.
* `urdf/`: optioneel kinematisch model voor simulators (ROS/Isaac/Gazebo).

Delen een naam geven: in `config.yaml` onder `motion: names: {200: slede, 210: arm}` (marker-id → naam).
Delen handmatig afbakenen (ook met maar 1 stand): `motion: manual_bodies: [{name: klem, box_min: [x,y,z], box_max: [x,y,z]}]`.

## Belangrijkste instellingen (`config.yaml`)

| Sleutel | Betekenis |
|---|---|
| `markers.size_mm` / `sizes_mm` | gemeten zijde zwart vierkant (per id te overschrijven) |
| `markers.scale_bars` | gemeten afstanden voor de schaal |
| `markers.check_distances` | gemeten afstanden alleen ter controle |
| `dense.backend` | `auto`, `colmap` (NVIDIA) of `openmvs` |
| `dense.resolution_level` | 0 = volle resolutie (nauwkeurigst) |
| `mesh.voxel_mm` | detailniveau puntenwolk (standaard 1,0; kleiner = fijner, trager) |
| `mesh.target_max_error_mm` | max. afwijking door vereenvoudigen van de mesh |
| `features.*` | vlak/cilinder-herkenning en randen aanscherpen |
| `mesh.max_triangles` | max. driehoeken per deel in de STEP (standaard 150.000) |
| `mesh.poisson_max_depth` | 11 (~4 GB RAM); 12 voor fijner detail op grote machines (~12-16 GB RAM) |
| `export.force_solid` | ook ongeziene openingen dichten zodat elk deel een solid is |

## Hoe het werkt (kort)

1. **Beelden**: EXIF-rotatie genormaliseerd, vage beelden eruit; uit video de scherpste frame per tijdvenster met bewegingsfilter.
2. **SfM**: COLMAP met DSP-SIFT, lensmodel OPENCV (k1,k2,p1,p2), één camera voor alle beelden.
3. **Schaal en assen**: ArUco-hoekpunten sub-pixel gedetecteerd, robuust getrianguleerd in het SfM-model.
   Schaal uit marker-afmetingen én schaallatten, gewogen naar onzekerheid, met consistentietoets.
   Vloervlak uit alle vloermarkers → Z omhoog.
4. **Dense**: COLMAP PatchMatch (CUDA) of OpenMVS, direct in mm.
5. **Bewegende delen**: markers die tussen standen samen bewegen vormen een star lichaam; de beweging geeft
   as-type en as. Elk oppervlakdeel wordt in de andere stand "teruggezet" en getoetst aan wat de camera's daar
   zagen (oppervlak / vrije ruimte / verborgen). Zo wordt elk vlak aan het juiste deel toegekend.
   Punten uit alle standen worden per deel samengevoegd (completer).
6. **Per deel**: Poisson-mesh, vlakken en cilinders robuust gefit, vertices erop geprojecteerd (ruis weg),
   snijlijnen van vlakken teruggebracht (scherpe randen), gaten gedicht, begrensde vereenvoudiging.
7. **STEP**: B-rep met gedeelde topologie, coplanaire facetten samengevoegd tot echte vlakken, solid, AP214 in mm,
   met namen en kleuren, teruggelezen en gecontroleerd.

## Gauntlet (zelftest)

`python -m foto2step selftest` rendert een synthetische machine (frame, lineaire slede, draaiarm, markers,
twee standen) als telefoonfoto's met lensvertekening, ruis en JPEG, draait de volledige pijplijn en vergelijkt
alles met de exacte maten: markerposities, controlematen, cameraposities, aantal delen, as-richting, slag, hoek,
draaipunt, segmentatie, oppervlakte-afwijking, volledigheid en STEP-geldigheid. Unit tests: `pytest -q`.

## Gauntlet-resultaat (laatste run)

Synthetische machine (frame 1,2 m + kolom, lineaire slede 150 mm, draaiarm 30°), 79–80 foto's per stand,
1600×1200 px, dichte reconstructie op **halve resolutie** (CPU-testomgeving). Alles vergeleken met exacte maten:

| Controle | Resultaat |
|---|---|
| Controlematen 2,2 m / 2,5 m (onafhankelijk) | 0,02 / 0,20 mm |
| Marker-hoekpunten 3D (95%) | 0,9 mm |
| Bewegende delen gevonden | 2 van 2, juiste type |
| Slede: as-richting / slag | 0,07° / 0,11 mm |
| Draaiarm: as-richting / hoek / draaipunt | 0,08° / 0,005° / 0,18 mm |
| Frame: oppervlak mediaan / 95% / volledigheid | 0,13 mm / 0,96 mm / 97% |
| Segmentatie (oppervlak bij juiste deel) | frame 98%, slede 97%, arm 99,7% |
| Arm: oppervlak | mediaan ~1 mm: de onderkant is in deze test nooit gefotografeerd |

Les: wat niet gefotografeerd is, kan niet nauwkeurig worden. Fotografeer bewegende delen ook van onderen.
