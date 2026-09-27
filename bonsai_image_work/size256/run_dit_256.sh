#!/bin/zsh
# Whole 256x256 DiT leg, in order: wait for the transformer weights, export fp32,
# quantize (same recipe as 512: quantize_dit.py), fix zero block scales, verify vs
# torch with real ids, torch reference image, all-tflite image with the same noise.
set -e
WT=$(cd "$(dirname "$0")/.." && pwd)          # bonsai_image_work/, wherever this checkout lives
WORK=${WORK:-~/models/bonsai-image-256}
PY=${PY:-python}                               # an env with litert-torch, ai-edge-quantizer, ai-edge-litert, diffusers>=0.37
SNAPDIR=~/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked
PROMPT="a small bonsai tree in a blue ceramic pot on a wooden table, soft daylight"
export SIZE=256 WORK
cd $WORK
until ls $SNAPDIR/snapshots/*/transformer/*.safetensors >/dev/null 2>&1 && ! ls $SNAPDIR/blobs/*.incomplete >/dev/null 2>&1; do sleep 15; done
echo "[$(date +%H:%M:%S)] transformer weights present"
$PY $WT/size256/export_dit_256.py > logs/export_dit_256.log 2>&1
grep -E "^size|^params|^ref|convert OK|export OK" logs/export_dit_256.log
echo "[$(date +%H:%M:%S)] quantize"
$PY $WT/quantize_dit.py > logs/quantize_dit_256.log 2>&1
grep -E "quantize OK|GiB" logs/quantize_dit_256.log
$PY $WT/size256/fix_scales.py dit_int4b32.tflite dit_256_int4b32.tflite 2>&1 | tail -2
rm -f dit_int4b32.tflite
echo "[$(date +%H:%M:%S)] verify"
Q=$WORK/dit_256_int4b32.tflite $PY $WT/size256/verify_dit_256.py > logs/verify_dit_256.log 2>&1
grep -v -E "^\s*$|INFO|WARNING" logs/verify_dit_256.log
echo "[$(date +%H:%M:%S)] torch reference 256"
$PY $WT/size256/reference_torch.py --hub-dir $WORK/hub512 --size 256 --prompt "$PROMPT" --seed 0 \
   --out $WORK/ref_torch_256_seed0.png --noise-out $WORK/noise_256_seed0.bin > logs/ref_torch_256.log 2>&1
grep -E "prompt embeds|pipeline|sigmas|generate.py|torch DiT|generated" logs/ref_torch_256.log
echo "[$(date +%H:%M:%S)] tflite 256 (same noise)"
BONSAI_INIT_LATENTS=$WORK/noise_256_seed0.bin $PY $WT/size256/generate.py --model-dir $WORK/hub512 --size 256 \
   --dit $WORK/dit_256_int4b32.tflite --vae $WORK/vae_dec_256_fp32.tflite --prompt "$PROMPT" --seed 0 \
   --out $WORK/gen_256_seed0.png --threads 8 > logs/gen_256_seed0.log 2>&1
grep -E "encoded|loaded|^step|decoded|saved" logs/gen_256_seed0.log
$PY $WT/size256/compare.py $WORK/ref_torch_256_seed0.png $WORK/gen_256_seed0.png
echo "[$(date +%H:%M:%S)] DONE"
