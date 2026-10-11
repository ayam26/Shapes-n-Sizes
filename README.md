# Shapes-n-Sizes

Do road-object detectors rely on texture rather than shape — and does training them to be shape-biased make them find more real objects and invent fewer fake ones?

This research project trains an object detector on an ASUS Ascent GX10 in four arms (standard vs. shape-biased backbone × normal vs. texture-scrambled training images), trains only on clear daytime dashcam images from BDD100K, and tests on night, rain, snow, and fog, on texture-scrambled images, and on unfiltered dashcam footage with a blind audit of real, invented, and missed objects.

## Contents

- [`DATASETS.md`](DATASETS.md) — open dashcam/driving datasets, weather and corner-case test sets, raw footage, texture-vs-shape probes, and hallucination benchmarks for the GX10's VLMs (Nemotron etc.), with licenses and download notes.
- [`docs/superpowers/specs/2026-10-11-shape-biased-detector-design.md`](docs/superpowers/specs/2026-10-11-shape-biased-detector-design.md) — current research design spec.
- [`docs/superpowers/specs/2026-10-11-shapes-n-sizes-design.md`](docs/superpowers/specs/2026-10-11-shapes-n-sizes-design.md) — earlier VLM depiction-cue design (superseded).
- [`perception-lab/`](perception-lab/README.md) — local automotive-perception testbed on the ASUS Ascent GX10 (UCSC ASUS Local AI hackathon). Start with `perception-lab/scripts/start.sh`; results in [`perception-lab/artifacts/report.md`](perception-lab/artifacts/report.md).
- [`perception-lab-design.md`](perception-lab-design.md) — the hackathon build specification.
