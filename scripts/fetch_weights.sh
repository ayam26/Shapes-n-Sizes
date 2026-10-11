#!/usr/bin/env bash
# Backbone + AdaIN weights into $SNZ_DATA/weights, verified against README §5.1.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
DATA="${SNZ_DATA:-$HOME/snz-data}"
mkdir -p "$DATA/weights"
fetch() {  # url file
  [ -s "$DATA/weights/$2" ] && return
  curl -fL --retry 3 -o "$DATA/weights/$2.part" "$1"
  mv "$DATA/weights/$2.part" "$DATA/weights/$2"
}
fetch "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar" \
  resnet50_train_60_epochs-c8e5653e.pth.tar
fetch https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth decoder.pth
fetch https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth \
  vgg_normalised.pth
(cd "$DATA/weights" && sha256sum -c -) <<'SUMS'
c8e5653e47bbf3f6efad5da531f3acea9963acb93362b6530cf39b2ba4750da3  resnet50_train_60_epochs-c8e5653e.pth.tar
379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472  decoder.pth
804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a  vgg_normalised.pth
SUMS
# torchvision IN1K_V1 (cached under ~/.cache/torch; hash prefix checked by torchvision)
uv run python -c "from torchvision.models import ResNet50_Weights as W; W.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True); print('in1k_v1 ok')"
