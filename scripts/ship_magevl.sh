#!/bin/bash
# Full microsoft/Mage-VL (4.7B) -> fast_vlm .litertlm pipeline.
#
# Vision = Mage-VL's Qwen2-VL-style ViT (24L/1024, grid_thw dynamic-res) made
# static-448: temporal_patch_size=1 means the patch-embed IS a stride-16 Conv2d
# (no Conv3d fold), 3-D rope (4:6:6 t:h:w, interleaved) precomputed from raster
# positions, single image = one cu_seqlens chunk = plain full attention, and the
# 2x2 merge done GPU-safe with strided slices in the adapter (NO GATHER_ND —
# same mobile-GPU constraint as qwen2-vl-2b).
# Decoder = the inner text model is a STOCK Qwen3-4B (AutoModel.from_config,
# plain 1-D positions — no M-RoPE compromise at all), re-hosted as a standalone
# Qwen3ForCausalLM (strict 1:1 state dict, untied lm_head), int4 blockwise-128.
# Cache 2048: fp32 KV for 36L/kv8 stays ~0.6 GB -> ~1.5 GiB peak on a phone.
#
# Env: litert-torch 0.9.2 + ai-edge-quantizer + transformers(>=5.7) +
# torch==2.12.1 + torchvision==0.27.1 + pillow + safetensors, python <= 3.13
# (3.14 breaks torchao's import). Override PY= to point at it.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/clipconv/bin/python}"
SRC=src_models/mage-vl
LLM=src_models/mage-vl-llm
DECO=out/magevl-decoder
VISO=out/magevl-vision
GREP='Loading weights|it/s|Redirects|register_constant|FlashAttention|torch_dtype.*deprecated|KernelPreference|ScaleCalculation|FutureWarning|copyreg|InitializeLog|XNNPACK|arithmetic ops|Fetching|Quantization Param|Applying Transform|Estimated count'

echo "### 0. download microsoft/Mage-VL (weights + processor; codec/video assets skipped)"
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/Mage-VL', local_dir='$SRC', ignore_patterns=['neural_codec/*','assets/*','examples/*','streammind_gate.safetensors','*.mp4','*.jpg','*.png'])" 2>&1 | grep -vE "$GREP" | tail -2
# transformers' remote-code hash walker requires every relatively-imported .py
# to EXIST (even lazily-imported ones) — streammind_gate.py is tiny, fetch it.
[ -f "$SRC/streammind_gate.py" ] || curl -sL -o "$SRC/streammind_gate.py" https://huggingface.co/microsoft/Mage-VL/raw/main/streammind_gate.py

echo "### 0.5 vendor the remote modeling code (needed as a local package for export)"
mkdir -p magevl_work/vendor && touch magevl_work/vendor/__init__.py
for f in configuration_mage_vl.py modeling_mage_vl.py config.json; do
  cp "$SRC/$f" magevl_work/vendor/$f
done

echo "### 1. extract the stock Qwen3-4B decoder (strict 1:1, untied lm_head)"
$PY magevl_work/prep_magevl_decoder.py "$SRC" "$LLM" 2>&1 | grep -vE "$GREP" | tail -3

echo "### 2. export decoder (int4 blockwise-128 OCTAV + externalize + single_token_embedder)"
CACHE=2048 PREFILL=128,512,1024 RECIPE=BOCTAV4_128 \
  $PY scripts/export_internvl_decoder.py "$LLM" "$DECO" 2>&1 | grep -vE "$GREP" | tail -2

echo "### 3. SP tokenizer"
OMP_NUM_THREADS=1 $PY - <<PYEOF 2>&1 | grep -vE "$GREP" | tail -1
from transformers import AutoTokenizer
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tok_spm
tok = AutoTokenizer.from_pretrained("$LLM", use_fast=False)
vf = getattr(tok, "vocab_file", None)
if vf and not str(vf).endswith((".model",".spiece",".spm")): tok.vocab_file = None
open("$DECO/tokenizer.spiece","wb").write(tok_spm.convert(tok))
print("SP tokenizer written")
PYEOF

echo "### 4. vision encoder+adapter (static-448, raster order, NO GATHER_ND, int8)"
IMG=448 MODEL="$SRC" $PY magevl_work/convert_magevl_vision.py "$VISO" 2>&1 | grep -vE "$GREP" | tail -3

echo "### 5. build fast_vlm bundle (ChatML, <|vision_start|><image_soft_token><|vision_end|>)"
DEC="$DECO" VIS="$VISO" TOK=sp IMAGE_SIZE=448 CACHE=2048 \
  IMG_RENDER='<|vision_start|><image_soft_token><|vision_end|>' \
  OUT_NAME=Mage-VL.litertlm \
  $PY scripts/build_internvl_bundle.py 2>&1 | tail -1
mkdir -p out/magevl-bundle && mv out/internvl-bundle/Mage-VL.litertlm out/magevl-bundle/
ls -la out/magevl-bundle/Mage-VL.litertlm | awk '{print "BUNDLE:", $NF, $5/1e9" GB"}'
echo "### DONE"
