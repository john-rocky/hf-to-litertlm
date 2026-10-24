#!/usr/bin/env bash
# Rebuild fixtures/img/ for the 18 public rows of fixtures.json and check every image's bytes against it.
# usage: bash fetch_public.sh <python with Pillow 12.2.0 + numpy>
# 1. downloads the four NASA originals (pinned URLs, sha256 checked), 2. draws syn_a / syn_b (make_synth.py: the
# pixels depend on the Pillow version), 3. checks the file sha256 of every image the rows name.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="${1:?python path}"
mkdir -p "$here/img"
while read -r name sha url; do
  out="$here/img/$name"
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
"$py" -I "$here/make_synth.py" "$here/img"
"$py" -I - "$here" <<'PY'
import hashlib, json, sys
from pathlib import Path
here = Path(sys.argv[1])
rows = json.loads((here / "fixtures.json").read_text())["rows"]
bad = 0
for img, sha in sorted({(r["image"], r["sha256"]) for r in rows}):
    got = hashlib.sha256((here / img).read_bytes()).hexdigest()
    bad += got != sha
    print("ok" if got == sha else "MISMATCH", img)
sys.exit(1 if bad else 0)
PY
