#!/usr/bin/env bash
# Parallel-range download of prism-ml/Ternary-Bonsai-2-27B-mlx-2bit model.safetensors (8.6 GB) into bonsai2_work/src.
# Per-connection HF throttle ~0.35 MB/s on this link; N ranges scale (memory hf-download-disable-xet). sha256 checked at the end.
set -u
cd "$(dirname "$0")/src"
URL="https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/resolve/main/model.safetensors"
OUT=model.safetensors; N=${N:-16}
SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
echo "size $SIZE start $(date +%T) parts=$N"
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  ( for try in 1 2 3 4 5 6; do curl -sL -r $s-$e -o part_$i "$URL" && [ "$(stat -f%z part_$i)" -eq $((e-s+1)) ] && break; echo "part $i retry $try $(date +%T)"; sleep 10; done ) &
done
wait
echo "parts done $(date +%T)"
cat $(for i in $(seq 0 $((N-1))); do echo part_$i; done) > "$OUT" && rm -f part_*
GOT=$(shasum -a 256 "$OUT" | awk '{print $1}')
echo "done $(date +%T) bytes $(stat -f%z $OUT) expect $SIZE sha256 $GOT"
[ "$(stat -f%z $OUT)" -eq "$SIZE" ] && echo "$GOT" > "$OUT.sha256" && touch DONE
