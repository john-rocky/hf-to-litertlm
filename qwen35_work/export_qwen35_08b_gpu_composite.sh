#!/usr/bin/env bash
# export_qwen35_08b_gpu_composite.sh WORK - rebuild Qwen3.5-0.8B_int8_gpu_composite_v0190.litertlm of litert-community/Qwen3.5-0.8B in
# WORK (created if missing; a finished step is not repeated) and compare its sections with the published file. Output: WORK/model.litertlm.
# export_qwen35_08b_gpu_composite.sh --check FILE [PYTHON] - step 7 alone, on any file. PYTHON (default python3) needs litert-lm-builder.
#   1 litert-torch at 65aafa4, upstream's Qwen3.5 graph (the last commit whose gated_delta_update has the 6 inputs litert-lm 0.19.0 accepts), with
#     litert_lm_builder.py of fe5c2bd (packaging code only, the pinned litert-lm-builder 0.19.0 has no TfLiteModelType)
#   2 a Python 3.12 venv from qwen35_08b_gpu_composite.lock.txt, plus that litert-torch
#   3 Qwen/Qwen3.5-0.8B at 2fc06364
#   4 LiteRT-LM v0.19.0's models/qwen3_5/LlmMetadataProto.pbtext, sha256 checked (the export packages with it; step 6 replaces it)
#   5 export_hf with PYTHONHASHSEED=0
#   6 re-pack with litert-lm-nightly 0.20.0.dev20261103 (through uvx): the LlmMetadata becomes qwen35_08b_gpu_composite.pbtext (the chat template
#     and stop tokens of Qwen3.5-0.8B_int8.litertlm, model type qwen3, pad_token <|endoftext|>), the tokenizer section the repository's own
#     tokenizer.json (an 8-byte little-endian length, then zlib level 6); the TFLite and executor sections stay as exported
#   7 the sha256 of every section against the published file. The file's own sha256 changes with every pack (a uuid and a timestamp in the header).
# Every command is printed as `### CMD: ...` before it runs. Exit 0 = every section equal, 1 = a step failed or a section differs. Refuses to overwrite
# WORK/model.litertlm. A differing TFLite section: tools/tfl_compare/ (tfl_diff, tfl_swapcheck, tfl_struct) says whether it is the same computation.
# env: PYTHON (default python3.12)
set -euo pipefail
Q=$(cd "$(dirname "$0")" && pwd)
LITERT_TORCH_COMMIT=65aafa499fb696587c7e9abfc9ee15c5a87eda84
BUILDER_COMMIT=fe5c2bd87e53b0c99b1dfef9a55ca68b7022a096
BUILDER_FILE=litert_torch/generative/export_hf/core/litert_lm_builder.py
MODEL_REVISION=2fc06364715b967f1860aea9cf38778875588b17
OFFICIAL_URL=https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/771b06e1a2c7f70a399a718599109facf37e38fd/models/qwen3_5/LlmMetadataProto.pbtext
OFFICIAL_SHA256=2bdb4bcbd0f61abd4bddeb3c32e76d172b9049281eff9b692aa4b4b965b44229
PACK_CLI=(uvx --python 3.12 --from litert-lm-nightly==0.20.0.dev20261103 litert-lm)
run() { echo "### CMD: $*"; "$@"; }

check_sections() { # check_sections FILE PYTHON
  run "$2" -I "$Q/bundle_sections.py" "$1.sections.json" "$1"
  "$2" -I - "$1.sections.json" "$1" <<'EOF'
import json, sys
PUBLISHED = {
    "LlmMetadataProto": (746, "600a70d13d9e128587019f89631f0775886d0ab2e303e49e245edd88dfc59933"),
    "ExecutorMetadataProto": (3057, "6ac6d910c67970fbcdb65d862cbbb70b16c059cf7263bfc1e41029dd830ae02e"),
    "HF_Tokenizer_Zlib": (3614174, "f7099e031c8ef292b5b5d49c36b75dfb39c81f215ccdaf3fd87d5071ab6fe684"),
    "TFLiteModel": (884603648, "d53fbd699b6eb8b22f83346fa7020729ab1d38c0b6013e55df8643a2b10ffe3c"),
}
data = json.load(open(sys.argv[1], encoding="utf-8"))
sections = next(iter(data.values()))["sections"]
differ = len(sections) != len(PUBLISHED)
for s in sections:
    size, sha = s["bytes"], s["sha256"]
    want = PUBLISHED.get(s["type"])
    same = want == (size, sha) and "prefer_activation_type" not in s
    differ = differ or not same
    print(f"{s['type']} {size} B {sha} {'SAME' if same else 'DIFF'}")
print(("DIFF: a section differs from" if differ else "SAME: every section equals") + " the published Qwen3.5-0.8B_int8_gpu_composite_v0190.litertlm: " + sys.argv[2])
sys.exit(1 if differ else 0)
EOF
}

if [ "${1:-}" = --check ]; then
  [ $# -ge 2 ] && [ -s "$2" ] || { echo "usage: $0 --check FILE [PYTHON]"; exit 2; }
  check_sections "$2" "${3:-python3}"
  exit 0
fi
[ $# -eq 1 ] || { echo "usage: $0 WORK | $0 --check FILE [PYTHON]"; exit 2; }
mkdir -p "$1"
WORK=$(cd "$1" && pwd)
[ -e "$WORK/model.litertlm" ] && { echo "refusing to overwrite $WORK/model.litertlm"; exit 2; }
export PYTHONHASHSEED=0
step() { if [ -e "$WORK/.step$1" ]; then echo "### step $1 already done"; return 1; fi; }
finished() { touch "$WORK/.step$1"; }
V=$WORK/venv

if step 1; then
  rm -rf "$WORK/litert-torch"
  run git clone -q --filter=blob:none https://github.com/google-ai-edge/litert-torch "$WORK/litert-torch"
  run git -C "$WORK/litert-torch" checkout -q --detach "$LITERT_TORCH_COMMIT"
  run git -C "$WORK/litert-torch" checkout -q "$BUILDER_COMMIT" -- "$BUILDER_FILE"
  finished 1
fi
if step 2; then
  rm -rf "$V"
  run "${PYTHON:-python3.12}" -m venv "$V"
  run "$V/bin/pip" install -q -r "$Q/qwen35_08b_gpu_composite.lock.txt"
  run "$V/bin/pip" install -q --no-deps -e "$WORK/litert-torch"
  finished 2
fi
if step 3; then
  run env HF_HUB_DISABLE_XET=1 "$V/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', revision='$MODEL_REVISION', local_dir='$WORK/model')"
  finished 3
fi
if step 4; then
  run curl -fsSL -o "$WORK/official.pbtext" "$OFFICIAL_URL"
  [ "$(shasum -a 256 "$WORK/official.pbtext" | cut -d' ' -f1)" = "$OFFICIAL_SHA256" ] || { echo "official.pbtext: not the expected sha256 $OFFICIAL_SHA256"; exit 1; }
  finished 4
fi
if step 5; then
  rm -rf "$WORK/raw"; mkdir -p "$WORK/raw"
  (cd "$WORK/raw" && run env HF_HUB_OFFLINE=1 "$V/bin/python" -c "from litert_torch.cli import main; main()" export_hf "$WORK/model" --output_dir="$WORK/raw" \
    --quantization_recipe=dynamic_wi8_afp32 --apply_gpu_composites=True --externalize_weights=False --cache_length=4096 \
    '--prefill_lengths=[1024,512,256,128,64,32,16,8,4,2,1]' --litert_lm_llm_metadata_override="$WORK/official.pbtext")
  finished 5
fi
U=$WORK/unpacked
rm -rf "$U"
run "${PACK_CLI[@]}" unpack "$WORK/raw/model.litertlm" --output-dir "$U" < /dev/null
run cp "$Q/qwen35_08b_gpu_composite.pbtext" "$U/LlmMetadataProto.pbtext"
run "$V/bin/python" -I -c "import struct, sys, zlib; d = open(sys.argv[1], 'rb').read(); open(sys.argv[2], 'wb').write(struct.pack('<Q', len(d)) + zlib.compress(d, 6))" \
  "$WORK/model/tokenizer.json" "$U/Section2_HF_Tokenizer_Zlib.zlib"
run "${PACK_CLI[@]}" pack "$U" --output "$WORK/model.litertlm" < /dev/null
rm -rf "$U"
shasum -a 256 "$WORK/model.litertlm"
check_sections "$WORK/model.litertlm" "$V/bin/python"
