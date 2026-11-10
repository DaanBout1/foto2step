# foto2step: notes for development

- Language of user-facing text (logs, report, README): Dutch. Code/comments: English.
- All geometry in **mm**, world frame from floor markers (Z up). Each pipeline step is resumable; outputs live in `work/<state>/...`.
- Accuracy is the priority; runtime is not. Never trade accuracy for speed in defaults.
- Fast checks: `pytest -q tests`. Full gauntlet (ground-truth synthetic machine): `python -m foto2step selftest`.
- Geometry-only dev loop (skips SfM/dense): `python tests/downstream_harness.py <dataset> <work>`.
- Memory: Poisson depth 11 needs ~4 GB; do not run two heavy stages in parallel on small machines.
- OpenCV pixel centres are integers, COLMAP's are +0.5: marker corners get `+0.5` before `cam_from_img`.
- Rigid-body transforms `T[state]` map reference-pose (A) coordinates to that state's coordinates.
