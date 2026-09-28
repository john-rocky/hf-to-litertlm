#!/bin/bash
# Download Mapika/decider-2b-vision at a pinned revision into out/src/decider-2b-vision/.
# Serial single-stream plain HTTP (no Xet, no parallel ranges), resumed with `curl -C -`
# in a loop whose condition is the byte count; before each file waits until the shared
# line is quiet. Every file is verified: LFS files by the sha256 the Hub publishes,
# the others by their git blob id (`git hash-object`). Writes out/src/DOWNLOAD_OK on success.
#
#   bash scripts/download_source.sh            (run from decider2bv_work/)
set -u
REPO=Mapika/decider-2b-vision
REV=863e290863655f1d6b69324d77d09ac972d21609
DST=out/src/decider-2b-vision
API=out/src/api_revision_863e290.json
IF=${IF:-en1}
IDLE_BPS=$((400*1024))
mkdir -p "$DST"
[ -s "$API" ] || curl -sS --fail --max-time 60 -o "$API" \
  "https://huggingface.co/api/models/$REPO/revision/$REV?blobs=true" || { echo "API FETCH FAILED"; exit 1; }

if_rate() {
  local a b
  a=$(netstat -ib -I "$IF" | awk 'NR==2 {print $7+$10}')
  sleep 8
  b=$(netstat -ib -I "$IF" | awk 'NR==2 {print $7+$10}')
  echo $(( (b - a) / 8 ))
}
wait_for_idle() {
  local n=0 r
  while true; do
    r=$(if_rate)
    if [ "$r" -lt "$IDLE_BPS" ]; then echo "$(date +%T) line free (${r} B/s)"; return; fi
    n=$((n+1)); echo "$(date +%T) line busy ($((r/1024)) KB/s), waiting"
    if [ "$n" -ge 20 ]; then echo "$(date +%T) line still busy after 20 checks, proceeding anyway"; return; fi
    sleep 60
  done
}

python3 - "$API" > out/src/filelist.tsv <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for s in d["siblings"]:
    lfs = s.get("lfs") or {}
    print("\t".join([s["rfilename"], str(s["size"]), s["blobId"], lfs.get("sha256", "")]))
PY

fail=0
while IFS=$'\t' read -r fn size blob sha; do
  out="$DST/$fn"; mkdir -p "$(dirname "$out")"
  url="https://huggingface.co/$REPO/resolve/$REV/$fn"
  if [ -f "$out" ] && [ "$(stat -f %z "$out")" = "$size" ]; then echo "exists $fn"; else
    [ "$size" -gt 50000000 ] && wait_for_idle
    t0=$(date +%s); attempt=0
    [ "$size" -eq 0 ] && : > "$out.part"
    while [ "$size" -gt 0 ]; do
      cur=0; [ -f "$out.part" ] && cur=$(stat -f %z "$out.part")
      [ "$cur" -ge "$size" ] && break
      curl -L -sS --fail -C - --speed-limit 30720 --speed-time 180 -o "$out.part" "$url" < /dev/null && \
        [ "$(stat -f %z "$out.part")" -ge "$size" ] && break
      attempt=$((attempt+1))
      echo "$(date +%T) $fn attempt $attempt ended at $( [ -f "$out.part" ] && stat -f %z "$out.part" || echo 0) bytes; resuming"
      [ "$attempt" -ge 60 ] && { echo "GIVING UP $fn"; break; }
      sleep 10
    done
    got=$( [ -f "$out.part" ] && stat -f %z "$out.part" || echo 0)
    [ "$got" = "$size" ] || { echo "SIZE MISMATCH $fn $got != $size"; fail=1; continue; }
    mv "$out.part" "$out"
    [ "$size" -gt 50000000 ] && echo "$(date +%T) $fn done in $(( $(date +%s) - t0 )) s ($(( size / ( $(date +%s) - t0 + 1 ) / 1024 )) KB/s)"
  fi
  if [ -n "$sha" ]; then
    got=$(shasum -a 256 "$out" | awk '{print $1}')
    [ -n "$got" ] && [ "$got" = "$sha" ] && echo "sha256 OK $fn" || { echo "SHA256 MISMATCH $fn $got != $sha"; fail=1; }
  else
    got=$(git hash-object "$out")
    [ -n "$got" ] && [ "$got" = "$blob" ] && echo "blob OK $fn" || { echo "BLOB MISMATCH $fn $got != $blob"; fail=1; }
  fi
done < out/src/filelist.tsv

if [ "$fail" = 0 ]; then echo "$REV" > out/src/DOWNLOAD_OK; echo "$(date +%T) DOWNLOAD_OK"; else echo "$(date +%T) DOWNLOAD_FAILED"; exit 1; fi
