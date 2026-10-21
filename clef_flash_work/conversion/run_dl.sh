#!/usr/bin/env bash
# aria2c download of the 6 LFS files of Cloudflare/clef-flash at REV, sha256 against the Hub's LFS oids, then each file
# placed in the Hugging Face cache layout under K/hf (blobs/<sha256> + snapshots/<REV>/<name> -> ../../blobs/<sha256>),
# so the author's load_release_model(<snapshot dir>) and from_pretrained read it with no network.
# The file list, sizes and oids come from K/results/hub_tree_<rev8>.json (the tree API listing read just before).
# The download of an earlier port of this model (same aria2c flags) with the revision f0b7d3cd and the cache placement.
#   bash scripts/run_dl.sh            (from K; downloads into K/dl, log lines START / ARIA_EXIT / SIZE / DONE)
set -euo pipefail
K=$(cd "$(dirname "$0")/.." && pwd)
REV=f0b7d3cd65296abfe459e6c228b618f5d8b26d2f
TREE=$K/results/hub_tree_${REV:0:8}.json
D=$K/dl
C=$K/hf/hub/models--Cloudflare--clef-flash
mkdir -p "$D" "$C/blobs" "$C/snapshots/$REV"
cd "$D"
jq -r --arg rev "$REV" '.[] | select(.lfs) | "https://huggingface.co/Cloudflare/clef-flash/resolve/\($rev)/\(.path)\n  out=\(.path)"' \
  "$TREE" > urls.txt
jq -r '.[] | select(.lfs) | "\(.lfs.oid)  \(.path)"' "$TREE" > SHA256SUMS.lfs
jq -r '.[] | select(.lfs) | "\(.path) \(.size)"' "$TREE" > SIZES.lfs
[ "$(wc -l < SHA256SUMS.lfs)" -eq 6 ] || { echo "EXPECTED 6 LFS FILES"; exit 1; }
echo "START $(date +%F_%T)"
rc=0
aria2c -i urls.txt -j2 -x16 -s16 -k1M --continue=true --auto-file-renaming=false --allow-overwrite=false \
  --summary-interval=60 --console-log-level=warn --log=aria2.log --log-level=notice || rc=$?
echo "ARIA_EXIT $rc $(date +%F_%T)"
[ "$rc" -eq 0 ] || exit 1
if ls ./*.aria2 >/dev/null 2>&1; then echo "INCOMPLETE: control files remain"; ls ./*.aria2; exit 1; fi
while read -r name size; do
  got=$(stat -f %z "$name")
  echo "SIZE $name $got (tree $size)"
  [ "$got" = "$size" ] || { echo "SIZE_MISMATCH $name"; exit 1; }
done < SIZES.lfs
shasum -a 256 -c SHA256SUMS.lfs || { echo "SHA_MISMATCH $(date +%F_%T)"; exit 1; }
while read -r oid name; do
  if [ -e "$C/blobs/$oid" ]; then
    echo "blob $oid exists ($name): the downloaded copy is removed"
    rm "$D/$name"
  else
    mv "$D/$name" "$C/blobs/$oid"
  fi
  [ -L "$C/snapshots/$REV/$name" ] || ln -s "../../blobs/$oid" "$C/snapshots/$REV/$name"
done < SHA256SUMS.lfs
ls -lL "$C/snapshots/$REV/"
echo "DONE $(date +%F_%T)"
