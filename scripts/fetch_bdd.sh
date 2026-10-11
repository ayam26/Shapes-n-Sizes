#!/usr/bin/env bash
# BDD100K 100K images + det_20 labels into $SNZ_DATA/bdd100k.
# Read and accept the BDD100K terms (education, research, not-for-profit) first:
#   https://doc.bdd100k.com/download.html  (registration portal: https://bdd-data.berkeley.edu)
# BDD_BASE defaults to the host named in the official docs. On 2026-10-11 it did not resolve
# (DNS); if it still fails, take the links from the registration portal and set BDD_BASE,
# or download the three files by hand into $SNZ_DATA and rerun (existing files are kept).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
[ "${BDD_TERMS_ACCEPTED:-}" = "yes" ] || {
  echo "Set BDD_TERMS_ACCEPTED=yes after reading the BDD100K terms." >&2; exit 1; }
DATA="${SNZ_DATA:-$HOME/snz-data}"
BASE="${BDD_BASE:-https://dl.cv.ethz.ch/bdd100k/data}"
FILES=(100k_images_train.zip 100k_images_val.zip bdd100k_det_20_labels_trainval.zip)
mkdir -p "$DATA" manifests
cd "$DATA"
for f in "${FILES[@]}"; do
  [ -s "$f" ] || curl -fL --retry 3 -C - -o "$f" "$BASE/$f"
done
md5sum "${FILES[@]}" | tee -a "$REPO/manifests/downloads.md5"
for f in "${FILES[@]}"; do unzip -qn "$f"; done
ls bdd100k/images/100k bdd100k/labels/det_20
