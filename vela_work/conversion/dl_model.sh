#!/usr/bin/env bash
# Download vllm-sr/Vela-1.0-Encoder-307M-Embedding @ 290318fdefda0ef569745fed362881914fac650a into hf/ (plain HTTP, Xet off;
# the 1.23 GB safetensors by 24 parallel byte ranges = the per-connection throttle here, memory hf-download-disable-xet).
set -u
export HF_HUB_DISABLE_XET=1
cd "$(dirname "$0")/../hf"
REV=290318fdefda0ef569745fed362881914fac650a
BASE="https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M-Embedding/resolve/$REV"
for f in README.md AMD_RUNTIME.md config.json config_sentence_transformers.json modules.json sentence_bert_config.json special_tokens_map.json tokenizer_config.json tokenizer.json modernbert_sdpa_layout.py 1_Pooling/config.json evaluation/known13.json evaluation/2d-grid.md; do
  mkdir -p "$(dirname "$f")"; curl -sL -o "$f" "$BASE/$f" || echo "small file $f failed"
done
URL="$BASE/model.safetensors"; SIZE=1227771776; N=24; EXPECT=0ad845c4b4d09c6cf45b5a3233cfde4dfa936a939cc8e48f7fdb6532569aa537
CH=$(( (SIZE + N - 1) / N ))
echo "start $(date +%T) parts=$N chunk=$CH"
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  ( for try in 1 2 3 4 5; do curl -sL -r $s-$e -o part_$i "$URL" && [ "$(stat -f%z part_$i)" -eq $((e-s+1)) ] && break; echo "part $i retry $try"; sleep 5; done ) &
done
wait
echo "parts done $(date +%T)"; ls -l part_* | awk '{s+=$5} END {print "total bytes", s}'
cat $(for i in $(seq 0 $((N-1))); do echo part_$i; done) > model.safetensors && rm -f part_*
GOT=$(shasum -a 256 model.safetensors | awk '{print $1}')
echo "sha256 $GOT (expect $EXPECT)"; [ "$GOT" = "$EXPECT" ] && echo "$REV" > REVISION && shasum -a 256 *.json tokenizer.json README.md 1_Pooling/config.json > SHA256SUMS && echo "SRC_DONE $(date +%T)" || echo "SHA MISMATCH"
