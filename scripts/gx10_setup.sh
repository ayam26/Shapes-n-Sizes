#!/usr/bin/env bash
# One-time GX10 environment: uv + project venv with CUDA 13.0 PyTorch (spec §5.5).
# Fallback if the wheels fail: NVIDIA PyTorch container (nvcr.io/nvidia/pytorch).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
print("device", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
PY
