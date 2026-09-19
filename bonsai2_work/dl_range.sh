#!/usr/bin/env bash
# dl_range.sh <url> <out> [N]  — parallel byte-range download (HF throttles ~0.35 MB/s per connection).
set -u
URL="$1"; OUT="$2"; N=${3:-8}
SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
echo "size $SIZE start $(date +%T) parts=$N -> $OUT"
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  ( for try in 1 2 3 4 5 6; do curl -sL -r $s-$e -o "$OUT.part_$i" "$URL" && [ "$(stat -f%z "$OUT.part_$i")" -eq $((e-s+1)) ] && break; echo "part $i retry $try"; sleep 10; done ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo "$OUT.part_$i"; done) > "$OUT" && rm -f "$OUT".part_*
echo "done $(date +%T) bytes $(stat -f%z "$OUT") expect $SIZE"; [ "$(stat -f%z "$OUT")" -eq "$SIZE" ] && touch "$OUT.DONE"
