#!/usr/bin/env bash
# reproduce.sh WORK_DIR [cpu|gpu]: moonshotai/Kimi-Linear-48B-A3B-Instruct (revision e1df551a) -> the CPU file
# kimi_linear_48b_a3b_mixed_int8_cpu_ekv4096.litertlm (cpu, the default) or the GPU file
# kimi_linear_48b_a3b_mixed_int8_gpu_ekv4096.litertlm (gpu: the other fully connected ops with a constant weight int8 per
# channel too, all but KDA's 16 x 16 chunk masks; steps 3b and 3c). Each step is in README.md "Steps". SNAP=<snapshot
# dir> skips the download (98 GB).
# Disk besides the checkpoint: the export 8 GB (gpu: and its quantized copy, 3 GB), the grafted model (cpu 55 GB, gpu 50 GB)
# and, while the graft runs, its spill file of the finished experts (47 GB; deleted at the end); the bundle takes the
# grafted model's blocks over on macOS (--consume-tflite). Memory: ~27 GB peak (the export).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mkdir -p "$1" && cd "$1" && pwd)
FORM=${2:-cpu}
case $FORM in cpu|gpu) ;; *) echo "the second argument is cpu or gpu" >&2; exit 2 ;; esac
REPO=moonshotai/Kimi-Linear-48B-A3B-Instruct; REV=e1df551a447157d4658b573f9a695d57658590e9
OUT=$W/kimi_linear_48b_a3b_mixed_int8_${FORM}_ekv4096.litertlm

# the checkpoint at the pinned revision
if [ -z "${SNAP:-}" ]; then
  SNAP=$(HF_HUB_DISABLE_XET=1 python3 -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', revision='$REV', cache_dir='$W/hf_cache'))")
fi

# 1. the tokenizer: the source's tiktoken tokenizer as tokenizer.json (runs the source's tokenization_kimi.py)
mkdir -p "$HERE/tok"
cp "$SNAP/tokenization_kimi.py" "$SNAP/tiktoken.model" "$SNAP/tokenizer_config.json" "$HERE/tok/"
python3 "$HERE/tools/tok_convert.py" "$HERE/tokjson"
echo "475d3a67952bcc7ece90c0288165ff4193797b3b247b1869b9abbea0946c6100  $HERE/tokjson/tokenizer.json" | shasum -a 256 -c -

# 2. the chat template: four rewrites for LiteRT-LM's template engine, each found exactly once
mkdir -p "$HERE/template"
python3 - "$SNAP/chat_template.jinja" "$HERE/template/chat_template.lite.jinja" <<'EOF'
import sys
s = open(sys.argv[1]).read()
for old, new in [("msg.get('content')", "msg['content']"), ("message.get('name')", "message['name']"),
                 ("message.get('tool_calls')", "message['tool_calls']"),
                 ("tools | tojson(separators=(',', ':'))", "tools | tojson")]:
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
open(sys.argv[2], "w").write(s)
EOF
echo "8aebb7a4414511393c00c45c36a861905c8dca45bc2f688dfd5a93afb0b635a0  $HERE/template/chat_template.lite.jinja" | shasum -a 256 -c -

# 3. export: fp32, the routed experts as 4-element placeholders in each layer's moe op, prefill 128 and 1024, KV 4096
KIMI_EXPERTS=graft python3 "$HERE/tools/kimi_export.py" "$SNAP" "$W/plain.tflite" --prefill 128,1024 --cache 4096
SRC=$W/plain.tflite

if [ "$FORM" = gpu ]; then
  # 3b. the recipe (form f6): every fully connected op with a constant weight int8 per channel but KDA's 16 x 16 chunk
  # masks, chosen one op at a time
  python3 "$HERE/tools/make_recipe.py" "$W/plain.tflite" "$W/recipe_gpu.json" --form f6
  # 3c. litert_quantizer with that recipe (the moe op is left as it is)
  python3 "$HERE/tools/quant_tflite.py" "$W/plain.tflite" "$W/q.tflite" "$W/recipe_gpu.json"
  rm -f "$W/plain.tflite"
  SRC=$W/q.tflite
fi

# 4. the routed experts: int8 per channel from the bf16 checkpoint, one projection of one layer at a time
python3 "$HERE/tools/graft_kimi.py" --tflite "$SRC" --ckpt "$SNAP" --out "$W/model.tflite" \
  --bits 8 --block 0 --spill "$W/spill.bin"
rm -f "$SRC"

# 5. the bundle (--consume-tflite frees the model's blocks as the bundle takes them: macOS F_PUNCHHOLE only; elsewhere
# the step needs the model's size again on disk)
EXTRA=(); [ "$(uname)" = Darwin ] && EXTRA+=(--consume-tflite)
python3 "$HERE/tools/kimi_bundle.py" --tflite "$W/model.tflite" --cut "$SNAP" --out "$OUT" --cache 4096 \
  ${EXTRA[@]+"${EXTRA[@]}"}
shasum -a 256 "$OUT"
