#!/usr/bin/env bash
# Download the two ~2 GB weight files of Fun-ASR-Nano-2512 and verify sha256.
#   out/hf_vllm/model.safetensors  <- FunAudioLLM/Fun-ASR-Nano-2512-vllm (port / export source)
#   out/hf_official/model.pt       <- FunAudioLLM/Fun-ASR-Nano-2512      (funasr oracle)
# MODE=serial (default): one curl stream per file, resumed with -C - in a loop on the byte count
#   (a clean early close exits 0, so --retry does not help).
# MODE=ranges: N concurrent byte ranges per file (for the per-object CDN throttle), then concat.
# Usage: MODE=serial|ranges N=24 ./dl_weights.sh [vllm|official|both]
set -u
cd "$(dirname "$0")"
MODE=${MODE:-serial}; N=${N:-24}; WHICH=${1:-both}

fetch() { # url size expect_sha dest
  local URL=$1 SIZE=$2 EXPECT=$3 DEST=$4
  mkdir -p "$(dirname "$DEST")"
  if [ -f "$DEST" ] && [ "$(stat -f%z "$DEST")" -eq "$SIZE" ] && [ -f "$DEST.sha256_ok" ]; then
    echo "SKIP $DEST (already verified)"; return 0; fi
  echo "start $(date +%T) mode=$MODE dest=$DEST size=$SIZE"
  if [ "$MODE" = serial ]; then
    local PART="$DEST.part"
    while :; do
      local have=0; [ -f "$PART" ] && have=$(stat -f%z "$PART")
      [ "$have" -ge "$SIZE" ] && break
      curl -sL -C - -o "$PART" "$URL"
      echo "  $(date +%T) bytes $(stat -f%z "$PART" 2>/dev/null || echo 0)/$SIZE"
      sleep 2
    done
    mv "$PART" "$DEST"
  else
    local DIR="$DEST.parts"; mkdir -p "$DIR"
    local CH=$(( (SIZE + N - 1) / N ))
    for i in $(seq 0 $((N-1))); do
      local s=$((i*CH)) e=$((i*CH+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
      ( for try in 1 2 3 4 5 6 7 8; do
          curl -sL -r $s-$e -o "$DIR/part_$i" "$URL" && [ "$(stat -f%z "$DIR/part_$i")" -eq $((e-s+1)) ] && break
          echo "  part $i retry $try"; sleep 5; done ) &
    done
    wait
    echo "  parts done $(date +%T)"; ls -l "$DIR"/part_* | awk '{s+=$5} END {print "  total bytes", s}'
    cat $(for i in $(seq 0 $((N-1))); do echo "$DIR/part_$i"; done) > "$DEST" && rm -rf "$DIR"
  fi
  local GOT; GOT=$(shasum -a 256 "$DEST" | awk '{print $1}')
  echo "sha256 $GOT (expect $EXPECT) size $(stat -f%z "$DEST")"
  if [ -n "$EXPECT" ] && [ "$GOT" = "$EXPECT" ]; then touch "$DEST.sha256_ok"; echo "DONE $DEST $(date +%T)"
  else echo "SHA MISMATCH $DEST"; return 1; fi
}

if [ "$WHICH" = vllm ] || [ "$WHICH" = both ]; then
  fetch "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-vllm/resolve/a4362c943d48951f98ca2a62181cc028970270c5/model.safetensors" \
    1970899072 96dfbec48282dd24d3334369a01e9e909f321ee39a1b0003c528c5379f68c1a6 out/hf_vllm/model.safetensors
fi
if [ "$WHICH" = official ] || [ "$WHICH" = both ]; then
  fetch "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/272c57b82523ada6fd87095e955f8e29100979ab/model.pt" \
    1971149431 55ae0d2fee369f0f11cce0795f6927934ad17cf11b278a7e56a51272074160bb out/hf_official/model.pt
fi
