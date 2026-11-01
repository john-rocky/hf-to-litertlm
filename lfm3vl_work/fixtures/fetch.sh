#!/usr/bin/env bash
# Rebuild lfm3vl_work/fixtures/img/ (the 12 test pictures of the LFM3-VL-2B tools; not in git) and verify every hash
# against fixtures.json.
# usage: bash fetch.sh <python with Pillow 13.0.0> [write]
#   1. the four NASA originals (public domain; pinned URLs, sha256 checked)
#   2. "write" mode: draw the synthetic images, make the 1024-px sources and the G^2 PNGs, write fixtures.json
#   3. check: recompute every sha256 (file bytes and decoded RGB) against fixtures.json
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="${1:?python path}"
mode="${2:-check}"
mkdir -p "$here/img/orig"
while read -r name sha url; do
  out="$here/img/orig/$name"
  if [ ! -f "$out" ] || [ "$(shasum -a 256 "$out" | cut -d' ' -f1)" != "$sha" ]; then
    curl -sS --fail --max-time 600 -o "$out.part" "$url"
    mv "$out.part" "$out"
  fi
  got="$(shasum -a 256 "$out" | cut -d' ' -f1)"
  [ "$got" = "$sha" ] || { echo "sha256 mismatch: $name $got"; exit 1; }
  echo "ok $name"
done <<'LIST'
nasa_S67-34267.jpg 7e748be3b43b91a25652329f4026b1c19c96ae60decc2ea99cd71a538754db21 https://images-assets.nasa.gov/image/S67-34267/S67-34267~orig.jpg
nasa_KSC-20170424-PH_SWW01_0007.jpg 497419455401d41fd0042b2af1d07944d4013a8362d549f6787f4cc6e66110a2 https://images-assets.nasa.gov/image/KSC-20170424-PH_SWW01_0007/KSC-20170424-PH_SWW01_0007~orig.JPG
nasa_KSC-08pd3809.jpg 4b187db43aaea1720286e4fc35989e020a170510d3d11fa3d21dfefb07fabfd4 https://images-assets.nasa.gov/image/KSC-08pd3809/KSC-08pd3809~orig.jpg
nasa_PIA12503.jpg 7b3a301daab2254f311913d173e10f5ff2c6a10a589f8d36918f8f3ecc6cb008 https://images-assets.nasa.gov/image/PIA12503/PIA12503~orig.jpg
LIST
if [ "$mode" = "write" ]; then
  "$py" -I "$here/make_fixtures.py" write
fi
"$py" -I "$here/make_fixtures.py" check
