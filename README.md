# Shapes-n-Sizes

Can vision-language models tell a real three-dimensional object from a flat picture of it — and which visual cues do they rely on?

This is a research project in two phases:

1. **Diagnose:** render controlled synthetic road scenes where a printed picture is matched to the real object from the camera's viewpoint, re-introduce one depth cue at a time (viewing angle, shadows, framing, shading consistency), and measure where open VLMs change their answer.
2. **Train:** fine-tune a small open VLM on synthetic examples with different objects, and test whether the skill generalizes to unseen objects, an unseen environment, and real photographs — or whether it learns shortcuts.

Models run locally on an ASUS Ascent GX10. Stimulus generation, the batch runner, and analysis live in this repository.

- Design spec: [`docs/superpowers/specs/2026-10-11-shapes-n-sizes-design.md`](docs/superpowers/specs/2026-10-11-shapes-n-sizes-design.md)
- Original hackathon spec (background): [`perception-lab-design.md`](perception-lab-design.md)
