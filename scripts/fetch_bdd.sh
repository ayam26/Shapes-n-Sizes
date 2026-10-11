#!/usr/bin/env bash
# BDD100K 100K images + 2018 detection labels into $SNZ_DATA/bdd100k.
# Read and accept the BDD100K terms (UC Berkeley: education, research, not-for-profit).
#
# Source: the official hosts (dl.cv.ethz.ch, doc.bdd100k.com, bdd-data.berkeley.edu) were
# offline on 2026-10-11 (bdd100k/bdd100k issues #353, #369). The maintainer's Berkeley
# mirror dl.yf.io still serves the images and the 2018 labels, not det_20 (its
# "det_20" zip actually holds 2,000 test images). The md5s below are of the files
# downloaded on 2026-10-11 (zip CRCs verified, 70k/10k/20k images); no official md5 for
# these packages was reachable to compare against.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
export PATH="$HOME/.local/bin:$PATH"
[ "${BDD_TERMS_ACCEPTED:-}" = "yes" ] || {
  echo "Set BDD_TERMS_ACCEPTED=yes after reading the BDD100K terms." >&2; exit 1; }
DATA="${SNZ_DATA:-$HOME/snz-data}"
BASE="${BDD_BASE:-http://dl.yf.io/bdd100k}"
mkdir -p "$DATA/bdd100k/images" "$DATA/bdd100k/labels" manifests
cd "$DATA"
for f in bdd100k_images_100k.zip bdd100k_labels.zip; do
  [ -s "$f" ] || curl -fL --retry 5 -C - -o "$f" "$BASE/$f"
done
md5sum -c - <<'SUMS'
c7a1d4db9af5a4691d5f6fee2a62e132  bdd100k_images_100k.zip
2d0165fd89b11025206e21a25ae1089d  bdd100k_labels.zip
SUMS
md5sum bdd100k_images_100k.zip bdd100k_labels.zip > "$REPO/manifests/downloads.md5"
[ -d bdd100k/images/100k/val ] || unzip -qn bdd100k_images_100k.zip -d bdd100k/images
(cd "$REPO" && uv run snz convert-labels-2018 --zip "$DATA/bdd100k_labels.zip" \
  --out "$DATA/bdd100k/labels/det_2018")
for s in train val; do echo "$s: $(ls bdd100k/images/100k/$s | wc -l) images"; done
