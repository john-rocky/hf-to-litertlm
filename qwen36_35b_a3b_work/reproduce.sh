#!/usr/bin/env bash
# reproduce.sh WORK_DIR [FORM]: Qwen/Qwen3.6-35B-A3B (revision 345d467b) -> FORM cpu (default): the CPU file
# qwen36_35b_a3b_mixed_int8_cpu_ekv4096.litertlm; FORM gpu: the mixed int8 GPU file
# qwen36_35b_a3b_mixed_int8_gpu_ekv4096.litertlm (export_hf's GPU composites); FORM gpu_int8: the int8 GPU file
# qwen36_35b_a3b_int8_gpu_ekv4096.litertlm (the same export, every linear layer of at least 524,289 weights int8; it
# declares fp16 activations).
# Each step's command is in README.md "Steps". SNAP=<snapshot dir> skips the download.
# Disk: the checkpoint 72 GB plus ~52 GB at the peak (the export's model.tflite, ~10 GB, is deleted after step 5; the
# bundle takes the grafted model's blocks over on macOS, --consume-tflite). Memory: ~45 GB peak (step 5).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mkdir -p "$1" && cd "$1" && pwd); FORM=${2:-cpu}
case $FORM in
  cpu) VARIANT=plain; NAME=qwen36_35b_a3b_mixed_int8_cpu_ekv4096.litertlm ;;
  gpu) VARIANT=gpu; NAME=qwen36_35b_a3b_mixed_int8_gpu_ekv4096.litertlm ;;
  gpu_int8) VARIANT=gpu; NAME=qwen36_35b_a3b_int8_gpu_ekv4096.litertlm ;;
  *) echo "FORM is cpu, gpu or gpu_int8, not $FORM" >&2; exit 2 ;;
esac
REPO=Qwen/Qwen3.6-35B-A3B; REV=345d467b117d1fd5eda9b879a5988de9ae5f47a7
PBTEXT_SHA=2bdb4bcbd0f61abd4bddeb3c32e76d172b9049281eff9b692aa4b4b965b44229
OUT=$W/$NAME

# 1. LiteRT-LM's own Qwen3.5 metadata at v0.19.1 (Qwen3.6-35B-A3B has the same model type, qwen3_5_moe)
mkdir -p "$HERE/template"
curl -fsSL -o "$HERE/template/LlmMetadataProto.pbtext" \
  https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/v0.19.1/models/qwen3_5/LlmMetadataProto.pbtext
echo "$PBTEXT_SHA  $HERE/template/LlmMetadataProto.pbtext" | shasum -a 256 -c -

# 2. the checkpoint at the pinned revision
if [ -z "${SNAP:-}" ]; then
  SNAP=$(HF_HUB_DISABLE_XET=1 python3 -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', revision='$REV', cache_dir='$W/hf_cache'))")
fi

# 3. export: fp32, the experts as 4-element placeholders in each layer's moe op, prefill 128 and 1024, KV 4096
# (the gpu variant also loads the q / k norm weights unpermuted: export_q35.py install_norm_perm_once)
PYTHONHASHSEED=0 Q35MOE_EXPERTS=graft python3 "$HERE/export_q35.py" "$SNAP" "$W/export" "$VARIANT" 128,1024 4096

# 4. gpu_int8 only: every FULLY_CONNECTED of at least 524,289 weights int8 per channel (the router, 256 x 2048, and the
# shared expert's gate, 1 x 2048, stay fp32; the token table is not a FULLY_CONNECTED), the moe ops left alone
SRC=$W/export/model.tflite
if [ "$FORM" = gpu_int8 ]; then
  python3 "$HERE/quant_tflite.py" "$W/export/model.tflite" "$W/q8.tflite" "$HERE/recipes/fc8_gpu_int8.json"
  rm -f "$W/export/model.tflite"
  SRC=$W/q8.tflite
fi

# 5. the experts: int8 per channel from the bf16 checkpoint, one projection of one layer at a time, into the float
# export itself (cpu, gpu: no other weight is quantized) or into step 4's model (gpu_int8)
python3 "$HERE/graft_moe.py" --tflite "$SRC" --ckpt "$SNAP" --out "$W/model.tflite" --bits 8 --block 0
rm -f "$W/export/model.tflite" "$W/q8.tflite"   # executor_metadata.pb stays for step 6

# 6. the bundle: the official metadata, the checkpoint's tokenizer.json bytes, export_hf's executor metadata, the
# activation type it declares (fp32; fp16 for gpu_int8)
# (--consume-tflite frees the model's blocks as the bundle takes them: macOS F_PUNCHHOLE only; elsewhere the step needs
# the model's size again on disk)
EXTRA=(); [ "$(uname)" = Darwin ] && EXTRA+=(--consume-tflite)
ACT=fp32; [ "$FORM" = gpu_int8 ] && ACT=fp16
python3 "$HERE/make_bundle.py" --tflite "$W/model.tflite" --out "$OUT" --tokenizer "$SNAP/tokenizer.json" \
  --executor-metadata "$W/export/executor_metadata.pb" --act "$ACT" ${EXTRA[@]+"${EXTRA[@]}"}
shasum -a 256 "$OUT"
