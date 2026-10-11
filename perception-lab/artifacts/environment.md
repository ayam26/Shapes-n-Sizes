# Environment (recorded 2026-10-11T05:18Z, build start 05:17:24Z)

## Where things run

| Role | Machine | Notes |
|---|---|---|
| Coding agent (Claude Code), browser, 3D rendering (Three.js/WebGL) | User's MacBook, macOS 26.6.2, arm64 | Node 24.2.0, Python 3.12.4, Google Chrome |
| FastAPI backend, SQLite, artifact store, job runner | ASUS Ascent GX10 (`asus@100.104.225.90`, Tailscale) | Python 3.12.3 venv |
| VLM inference | GX10 GPU via Ollama 0.23.2 on 127.0.0.1:11434 | loopback only |

**The GX10 GPU does not render the browser scene.** Scenes are rendered by the laptop browser's WebGL; the captured PNG is uploaded to the GX10 backend, which runs inference there. The Mac reaches the GX10 backend through an SSH tunnel over Tailscale (LAN-class link); captured images therefore leave the laptop and are stored on the GX10.

## GX10 hardware/software

- OS: Ubuntu 24.04.4 LTS, aarch64, 20 cores (Arm, CPU part 0xd87)
- GPU: NVIDIA GB10, driver 580.142, CUDA 13.0; unified memory (nvidia-smi reports memory as N/A)
- System memory: 121 GiB total (shared CPU/GPU). Disk: 916 GB, 613 GB free at start.
- Docker installed, but user `asus` lacks socket permission — not used (no privileged change made).
- PyPI reachable from GX10 (for dependency install during development only).

## Models available in Ollama

| Model | Digest | Params | Quant | Ollama capabilities |
|---|---|---|---|---|
| nemotron3:33b | f6d8b7ff496c… | 33.0B | Q4_K_M | completion, **vision**, audio, tools, thinking |
| qwen3.6:35b | 07d35212591f… | 36.0B | Q4_K_M | completion, vision, tools, thinking |
| gemma4:26b | 5571076f3d70… | 25.8B | Q4_K_M | completion, vision, tools, thinking |
| nemotron-3-super:latest | 95acc78b3ffd… | 123.6B | Q4_K_M | completion, tools, thinking — **no vision**, excluded |

Selected: **nemotron3:33b** (architecture `nemotron_h_omni`, context 131072). License: NVIDIA Open Model License Agreement (last modified 2025-10-24), shown by `ollama show --license`.

## Image smoke test (verified)

Synthetic 256×256 PNG (red square on blue, sha256 `0754e836…f415a3`), `/api/chat`, `think:false`, temperature 0, seed 7, num_predict 200.

| Call | Wall time | load_duration | Prompt tokens | Output |
|---|---|---|---|---|
| 1 (cold) | 11.8 s | 11.28 s | 288 | "A red square is centered on a blue background." |
| 2 (warm) | 0.5 s | 0.11 s | 288 | identical |

Resident size when loaded: 32 GB (`ollama ps`, 100% GPU). At start, `nemotron-3-super` (~90 GB) was resident from other use; loading nemotron3:33b displaced it.

Conclusion: real local image inference on the GX10 is verified (`hardware_verified=true`).
