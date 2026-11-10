#!/usr/bin/env bash
# convert_gpt_oss_120b.sh OUTDIR [STEP...] - openai/gpt-oss-120b (pinned b5c939de8f754692c1647ca79fbf85e8c1e70f8a,
# Apache-2.0) -> OUTDIR/gpt_oss_120b_mixed_int4_ekv4096.litertlm, the file of litert-community/gpt-oss-120b: the decoder
# written one tensor at a time with the LiteRT Tensor API (no float copy of the model), each layer's 128 experts as one
# `moe` custom op (int4 blockwise 32, gelu_tanh, no clamps: the op of LiteRT-LM 0.19.1 has neither the clamp nor
# x * sigmoid(1.702 x)), attention and lm_head int8 per channel, KV cache 4,096, the checkpoint's tokenizer.json, the
# source's chat template with four edits (template/), fp32 activations declared.
# LAYERS=N writes the first N layers only (+ final norm + lm_head; the file is then named *_l<N>_*): a test of the
# pack, writer and bundle steps on ~6 GB of reads instead of 65 GB (fetch still downloads the whole snapshot). It is
# not a smaller model.
# Disk for the 120B: the snapshot 65.3 GB, the pack 77.2 GB, the TFLite 77.3 GB (removed after the bundle step), the
# .litertlm 77.3 GB; SNAPSHOT=<dir> uses an existing snapshot of the pinned revision.
# Memory: the pack's peak footprint was 15.0 GB, the writer's 0.22 GB, the bundle step's small (a streamed copy).
# Steps (in order; a finished step leaves OUTDIR/.done/<step> and is skipped on a re-run; name steps to run only those):
#   fetch    hf download of the pinned revision's top-level files into OUTDIR/snapshot (the repo's original/ and metal/
#            hold the weights twice more, 65 GB each, and are left out; skipped with SNAPSHOT=<dir>)
#   pack     tools/pack_gptoss.py: the checkpoint read one tensor at a time (large sequential preads); experts in the
#            op's layout, MXFP4 -> int4 blockwise (an fp32 scale per block of 32 by least squared error, floored at
#            2^-14), the experts' biases as one more input block, attention and lm_head int8 per channel, the op options
#            torch runs with 4 threads (OMP_NUM_THREADS=4), as it did for the shipped file: the float32 RoPE tables
#            depend on torch's thread count (with 12 threads, 300 cos rows and 32 sin rows of 4,096 differ in the
#            last bit)
#   writer   a LiteRT checkout at dcb657e8 (the LiteRT that LiteRT-LM 0.19.1 pins) in OUTDIR/LiteRT (or LITERT=<dir>),
#            tensor_api/overlay.sh (the blockwise-scales patch + the writer linked as tensor/examples/gptoss), then
#            bazelisk build -c opt //tensor/examples/gptoss:gptoss_writer
#   write    the writer: prefill_128, prefill_1024 and decode signatures, KV cache 4,096, --moe_fusion_guard (the residual
#            add after each moe op written as (x + moe) + moe * 0, exact, so a GPU delegate cannot fuse it into the op)
#   bundle   tools/make_bundle.py --stream: the LlmMetadata (template/gptoss_litertlm.pbtext), the tokenizer section =
#            the snapshot's tokenizer.json bytes (sha256 checked), the TFLite, fp32 activations declared; then
#            tools/bundle_sections.py prints every section's sha256 (the header's uuid and creation time are new on every
#            build, so compare sections, not the whole file)
# Python: tools/requirements.txt (Python 3.12). Bazel: bazelisk (LiteRT pins its Bazel version); BAZEL_OUTPUT_BASE=<dir>
# puts Bazel's output base there.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: convert_gpt_oss_120b.sh OUTDIR [STEP...]}
shift
mkdir -p "$OUT/.done"
OUT=$(cd "$OUT" && pwd)
REPO=openai/gpt-oss-120b
REV=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
LITERT_REV=dcb657e8a537b2dc944e53e14c7eb43bcd9ededa
PY=${PY:-python3}
LAYERS=${LAYERS:-36}
SNAP=${SNAPSHOT:-$OUT/snapshot}
LITERT=${LITERT:-$OUT/LiteRT}
if [ "$LAYERS" = 36 ]; then NAME=gpt_oss_120b_mixed_int4_ekv4096; else NAME=gpt_oss_120b_l${LAYERS}_mixed_int4_ekv4096; fi
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(fetch pack writer write bundle)
log() { echo "[$(date '+%F %T')] $*"; }
run() {  # step
  local s=$1
  [ -f "$OUT/.done/$s" ] && { log "skip $s (done $(cat "$OUT/.done/$s"))"; return 0; }
  log "step $s"
  "step_$s"
  date '+%F %T' > "$OUT/.done/$s"
}
step_fetch() {
  if [ -n "${SNAPSHOT:-}" ]; then log "SNAPSHOT=$SNAPSHOT"; return 0; fi
  hf download "$REPO" --revision "$REV" --local-dir "$SNAP" --exclude 'original/*' --exclude 'metal/*'
}
step_pack() {
  OMP_NUM_THREADS=4 "$PY" "$HERE/tools/pack_gptoss.py" "$SNAP" "$OUT/pack" --layers "$LAYERS" --rope-max 4096 --experts xb4 \
    --xb4-rule mse --act gelu_tanh --fc int8 --lm-head int8
}
step_writer() {
  if [ ! -d "$LITERT/.git" ]; then
    git clone https://github.com/google-ai-edge/LiteRT.git "$LITERT"
  fi
  git -C "$LITERT" checkout -q "$LITERT_REV"
  bash "$HERE/tensor_api/overlay.sh" "$LITERT"
  (cd "$LITERT" && bazelisk ${BAZEL_OUTPUT_BASE:+--output_base="$BAZEL_OUTPUT_BASE"} --batch build -c opt \
    //tensor/examples/gptoss:gptoss_writer)
}
step_write() {
  "$LITERT/bazel-bin/tensor/examples/gptoss/gptoss_writer" --weights_dir="$OUT/pack" --out="$OUT/model.tflite" \
    --prefill=128,1024 --cache=4096 --moe_fusion_guard
}
step_bundle() {
  "$PY" "$HERE/tools/make_bundle.py" --stream --tflite "$OUT/model.tflite" --tokenizer "$SNAP/tokenizer.json" \
    --out "$OUT/$NAME.litertlm"
  "$PY" "$HERE/tools/bundle_sections.py" "$OUT/$NAME.litertlm"
  rm -f "$OUT/model.tflite"
}
for s in "${STEPS[@]}"; do run "$s"; done
log "done: ${STEPS[*]}$(for s in "${STEPS[@]}"; do [ "$s" = bundle ] && printf ' -> %s' "$OUT/$NAME.litertlm"; done)"
