# Perception Lab — 3-minute demo script

Setup before presenting (≈1 min): `./scripts/start.sh` on the laptop, open `http://127.0.0.1:8765`, then click
**Analyze** once on any scene so the model is resident (cold load ≈ 12 s). Header must read **LIVE local inference**,
`✓ digest`, `prompts frozen · lock frozen`, `offline guard on`.

## 0:00–0:20 — The task
"This is a small driving scene. In the yellow box there is either a real 3D truck, a picture of that truck printed on a
flat board, or a plain board. We change appearance — never the physical geometry — and ask a vision-language model
running locally on the ASUS GX10 a narrow question: is the marked target a physical vehicle, a flat depiction, or other?
It's an image-classification probe, not a driving system."

## 0:20–1:00 — Observer vs sensor
- Column 1: drag to orbit the observer view. Point out the yellow slot box and the wireframe sensor camera — "these helpers
  live on a separate render layer; the model never sees them."
- Pick **Truck picture on a flat board**, camera **A · ego lane**: column 2 shows the exact 512×512 input. "From the ego lane
  the picture has the truck's own perspective — we photographed the truck from this camera to make the board."
- Switch to **B · oblique**: "From here you can see it's a flat board on posts."
- Point at the ground-truth box: "Truth is shown only in these panels; it is never drawn into the image or sent in the prompt."

## 1:00–1:40 — Fresh local run (honest queue)
- Select **Physical truck**, camera **B · oblique**, press **Analyze with both methods**.
- Narrate the real status chips: capturing → uploading → queued → running → completed (≈ 3–7 s). "That is the GX10 doing
  two sequential inferences; nothing is cached — origin says **live**."
- Column 3: both methods received the byte-identical input (banner + sha256). Expect `flat_depiction` → red
  `✗ vehicle_as_depiction` (the held-out pattern; if the model answers differently live, say so — temperature 0 still
  changed 1/36 labels on our repeat run).
- Open **Exact prompt** and **Raw model response**. "The evidence text is what the model says, not verified reasoning."

## 1:40–2:20 — Method comparison and one failure
- Results → run **heldout-v1**. Gallery filter `pair = neither`, open a plain-board case: "Every plain board, every run,
  is called `flat_depiction` — the model never uses `other` for a blank sign. A real failure; we didn't tune it away."
- Filter `pair = variant-only`: one case where the structure sentence helped — "and our repeat run shows one-case changes
  happen by chance, so we don't claim an improvement. Regressions from the variant on this model: zero."

## 2:20–3:00 — Counts, latency, limits
- Metrics table, held-out row: baseline **6/18**, structure prompt **7/18**, false-vehicle on depictions **0/6**, vehicle recall
  **0/6 vs 1/6**. "35 of 36 answers were `flat_depiction` — a constant guesser gets 6/18 too. So the honest headline is: from an
  oblique view this model calls our low-poly truck a drawing."
- Results → run **xmodel-gemma4-heldout-v1**: "Same images, same frozen prompts, different local model: gemma4 gets 17/18 and recognises
  the oblique trucks. So this failure belongs to nemotron3, not to the picture — and model choice mattered more than our prompt
  trick, which actually caused 5 regressions on qwen3.6." (Exploratory, single runs.)
- Tick **High-fidelity render**, keep Physical truck + B · oblique, Analyze: "We re-rendered everything with textures, reflections and
  soft shadows. gemma4 and qwen3.6 get these trucks right; nemotron3 still says flat_depiction — now 'a computer-generated image'."
- Latency: ≈1.2 s baseline, ≈1.4 s with the longer prompt, per request, warm.
- Limits (one breath): synthetic low-poly scenes may themselves look like drawings; 18 cases per split; one headline model plus two exploratory single-run comparisons; no safety
  claims. "Everything — images, hashes, prompts, raw outputs, metrics — is exported and re-verified by a script."

If you must avoid waiting, open a stored case from the gallery and say "this is a **recorded** result from run …"; the fresh
local run in 1:00–1:40 remains the live evidence.
