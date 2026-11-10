#!/usr/bin/env bash
# reproduce.sh WORK_DIR [KV] [BLOCK]: Qwen/Qwen3.5-35B-A3B (revision 59d61f3c) -> the CPU file
# qwen35_35b_a3b_mixed_int4_cpu_ekv<KV>.litertlm (KV 4096 and BLOCK 32 = the published file). Each step's command is
# in README.md "Steps". SNAP=<snapshot dir> skips the download. Disk: the checkpoint 67 GB plus ~35 GB at the peak (the
# export's model.tflite is deleted after step 4 and the int8 file after step 5; the bundle takes the grafted model's
# blocks over on macOS, --consume-tflite). Memory: ~34 GB peak.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mkdir -p "$1" && cd "$1" && pwd); KV=${2:-4096}; BLOCK=${3:-32}
REPO=Qwen/Qwen3.5-35B-A3B; REV=59d61f3ce65a6d9863b86d2e96597125219dc754
PBTEXT_SHA=2bdb4bcbd0f61abd4bddeb3c32e76d172b9049281eff9b692aa4b4b965b44229
OUT=$W/qwen35_35b_a3b_mixed_int4_cpu_ekv${KV}.litertlm

# 1. LiteRT-LM's own Qwen3.5 metadata at v0.19.1
mkdir -p "$HERE/template"
curl -fsSL -o "$HERE/template/LlmMetadataProto.pbtext" \
  https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/v0.19.1/models/qwen3_5/LlmMetadataProto.pbtext
echo "$PBTEXT_SHA  $HERE/template/LlmMetadataProto.pbtext" | shasum -a 256 -c -

# 2. the checkpoint at the pinned revision
if [ -z "${SNAP:-}" ]; then
  SNAP=$(HF_HUB_DISABLE_XET=1 python3 -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', revision='$REV', cache_dir='$W/hf_cache'))")
fi

# 3. export: fp32, the experts as 4-element placeholders in each layer's moe op, prefill 128 and 1024, KV $KV
PYTHONHASHSEED=0 Q35MOE_EXPERTS=graft python3 "$HERE/export_q35.py" "$SNAP" "$W/export" plain 128,1024 "$KV"

# 4. int8 per channel for attention q / k / v / o and lm_head; every other weight stays fp32, the moe ops untouched
python3 "$HERE/quant_tflite.py" "$W/export/model.tflite" "$W/q8.tflite" "$HERE/recipes/fc8_attn_lmhead.json"
rm -f "$W/export/model.tflite"   # executor_metadata.pb stays for step 6

# 5. the experts: int4 blockwise-$BLOCK from the bf16 checkpoint, one projection of one layer at a time
python3 "$HERE/graft_moe.py" --tflite "$W/q8.tflite" --ckpt "$SNAP" --out "$W/model.tflite" --bits 4 --block "$BLOCK"
rm -f "$W/q8.tflite"

# 6. the bundle: the official metadata, the checkpoint's tokenizer.json bytes, export_hf's executor metadata, fp32
# (--consume-tflite frees the model's blocks as the bundle takes them: macOS F_PUNCHHOLE only; elsewhere the step needs
# the model's size again on disk)
EXTRA=(); [ "$KV" -lt 4096 ] && EXTRA+=(--max-tokens "$KV"); [ "$(uname)" = Darwin ] && EXTRA+=(--consume-tflite)
python3 "$HERE/make_bundle.py" --tflite "$W/model.tflite" --out "$OUT" --tokenizer "$SNAP/tokenizer.json" \
  --executor-metadata "$W/export/executor_metadata.pb" ${EXTRA[@]+"${EXTRA[@]}"}
shasum -a 256 "$OUT"
