#!/usr/bin/env bash
# Describable Textures Dataset ("for research purposes") into $SNZ_DATA/dtd.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
DATA="${SNZ_DATA:-$HOME/snz-data}"
mkdir -p "$DATA" manifests
cd "$DATA"
[ -s dtd-r1.0.1.tar.gz ] || curl -fL --retry 3 -o dtd-r1.0.1.tar.gz \
  https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
[ -d dtd/images ] || tar -xzf dtd-r1.0.1.tar.gz
sha256sum dtd-r1.0.1.tar.gz | tee -a "$REPO/manifests/downloads.sha256"
echo "styles: $(find dtd/images -name '*.jpg' | wc -l) images"
