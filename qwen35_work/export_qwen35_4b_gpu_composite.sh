#!/usr/bin/env bash
# export_qwen35_4b_gpu_composite.sh WORK - rebuild Qwen3.5-4B_int8_gpu_composite_v0190.litertlm and
# Qwen3.5-4B_mixed_int4_gpu_composite_v0190.litertlm of litert-community/Qwen3.5-4B in WORK (created if missing; a finished
# step is not repeated) and compare their sections with the published files. Output: WORK/<the two file names>.
# export_qwen35_4b_gpu_composite.sh --check FILE [PYTHON] - step 9 alone, on any file. PYTHON (default python3) needs litert-lm-builder.
#   1 litert-torch at 65aafa4, upstream's Qwen3.5 graph (the last commit whose gated_delta_update has the 6 inputs litert-lm 0.19.0
#     accepts), with litert_lm_builder.py of fe5c2bd (packaging code only) - as export_qwen35_08b_gpu_composite.sh
#   2 a Python 3.12 venv from qwen35_08b_gpu_composite.lock.txt, plus that litert-torch; a second one from qwen35_rt0192.lock.txt
#     (litert-lm 0.19.2: the CLI, litert-lm-builder and litert-lm-peek of step 8)
#   3 Qwen/Qwen3.5-4B at 851bf6e8
#   4 LiteRT-LM v0.19.0's models/qwen3_5/LlmMetadataProto.pbtext, sha256 checked (the export packages with it; step 7 replaces it)
#   5 export_hf through export_hf_lowmem.py with PYTHONHASHSEED=0 and OMP_NUM_THREADS=4 (torch's float32 results can depend on its
#     thread count; the published files were exported with 4): the assign / share / filecat changes, the lightweight conversion
#     with RESOURCE_CONSTANT_NUMEL_THRESHOLD=10000000, the quantization over an mmap of the float model with
#     qwen35_gpu_composite_int8.json (int8 everywhere, the 48 in_proj_a / in_proj_b projections float), and
#     --keep_temporary_files=True, which keeps the float model for step 6. Prefill signatures 1024, 256, 64, 16, 4 and 1, 4,096-token cache.
#   6 the float model quantized again with qwen35_gpu_composite_mixed_int4.json (blockwise-32 int4 fully connected weights, int8
#     lm_head and token table, in_proj_a / in_proj_b float) and the 2^-14 int4 scale floor (requant_tflite.py), and put into a copy
#     of step 5's bundle in place of its TFLite section
#   7 per file, a re-pack with litert-lm-nightly 0.20.0.dev20261103 (through uvx): the LlmMetadata becomes
#     qwen35_4b_gpu_composite.pbtext (the LlmMetadata of the published Qwen3.5-4B_int8.litertlm, plus pad_token <|endoftext|> and
#     min_runtime_version 0.19.0), the tokenizer section the repository's own tokenizer.json (an 8-byte little-endian length, then
#     zlib level 6); then prefer_activation_type fp32 (scripts/set_activation_type.py with the same packer: litert-lm 0.19.2's
#     pack moves the weights out of the graph by default, and the script's read-back would refuse the new section)
#   8 per file, the weights out of the graph into a TFLiteWeights section: tools/externalize/externalize_repo.sh --no-hub with
#     litert-lm 0.19.2 (checked section by section and tensor by tensor, smoke on the CPU)
#   9 the sha256 of every section against the published files. A file's own sha256 changes with every pack (a uuid and a timestamp
#     in the header).
# Every command is printed as `### CMD: ...` before it runs. Exit 0 = every section of both files equal, 1 = a step failed or a
# section differs. Refuses to overwrite an output file. A differing TFLite section: tools/tfl_compare/ (tfl_diff, tfl_swapcheck,
# tfl_struct) says whether it is the same computation.
# Needs python3.12, uvx, git, curl, perl and about 60 GB of disk in WORK; the export peaks at about 25 GB of memory.
# env: PYTHON (default python3.12)
set -euo pipefail
Q=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$Q/.." && pwd)
LITERT_TORCH_COMMIT=65aafa499fb696587c7e9abfc9ee15c5a87eda84
BUILDER_COMMIT=fe5c2bd87e53b0c99b1dfef9a55ca68b7022a096
BUILDER_FILE=litert_torch/generative/export_hf/core/litert_lm_builder.py
MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
OFFICIAL_URL=https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/771b06e1a2c7f70a399a718599109facf37e38fd/models/qwen3_5/LlmMetadataProto.pbtext
OFFICIAL_SHA256=2bdb4bcbd0f61abd4bddeb3c32e76d172b9049281eff9b692aa4b4b965b44229
PACK_CLI=(uvx --python 3.12 --from litert-lm-nightly==0.20.0.dev20261103 litert-lm)
F8=Qwen3.5-4B_int8_gpu_composite_v0190.litertlm
F4=Qwen3.5-4B_mixed_int4_gpu_composite_v0190.litertlm
run() { echo "### CMD: $*"; "$@"; }

check_sections() { # check_sections FILE PYTHON
  run "$2" -I "$Q/bundle_sections.py" "$1.sections.json" "$1"
  "$2" -I - "$1.sections.json" "$1" <<'EOF'
import json, os, sys
PUBLISHED = {
    "Qwen3.5-4B_int8_gpu_composite_v0190.litertlm": {
        "LlmMetadataProto": (4458, "75ad5a722a74096b354925158165aafe4f566cc037ba0bfc54a32db836cf8b9e"),
        "ExecutorMetadataProto": (4101, "0346374de412454ac5931dbea7a2551af53720599040d0364274d6e122284e4a"),
        "HF_Tokenizer_Zlib": (3614174, "f7099e031c8ef292b5b5d49c36b75dfb39c81f215ccdaf3fd87d5071ab6fe684"),
        "TFLiteModel": (147221464, "b7261c4e05ae53105f83aee31a27a2d4dcc95d2d826b1a086af11942feee2330"),
        "TFLiteWeights": (4216586240, "6e1ecb4567c594388d7374c55a8d12a68a17175e643c19f22b3a9dd6510ab4a0"),
    },
    "Qwen3.5-4B_mixed_int4_gpu_composite_v0190.litertlm": {
        "LlmMetadataProto": (4458, "75ad5a722a74096b354925158165aafe4f566cc037ba0bfc54a32db836cf8b9e"),
        "ExecutorMetadataProto": (4101, "0346374de412454ac5931dbea7a2551af53720599040d0364274d6e122284e4a"),
        "HF_Tokenizer_Zlib": (3614174, "f7099e031c8ef292b5b5d49c36b75dfb39c81f215ccdaf3fd87d5071ab6fe684"),
        "TFLiteModel": (275161096, "264774bc686797faa07aceda82bc70c48f2d1245dcfd153cdad0786590f69830"),
        "TFLiteWeights": (2434007040, "43d59ac9d5bf647af543958586d1666ac861620d16ea762ab6adaa23b5f2cc0f"),
    },
}
data = json.load(open(sys.argv[1], encoding="utf-8"))
sections = next(iter(data.values()))["sections"]
name = os.path.basename(sys.argv[2])
want = PUBLISHED.get(name)
if want is None:
    print(f"DIFF: no published sections for the name {name} (one of {sorted(PUBLISHED)})")
    sys.exit(1)
differ = len(sections) != len(want)
for s in sections:
    size, sha = s["bytes"], s["sha256"]
    same = want.get(s["type"]) == (size, sha)
    differ = differ or not same
    print(f"{s['type']} {size} B {sha} {'SAME' if same else 'DIFF'}")
print(("DIFF: a section differs from" if differ else "SAME: every section equals") + f" the published {name}: " + sys.argv[2])
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
for f in "$F8" "$F4"; do [ -e "$WORK/$f" ] && { echo "refusing to overwrite $WORK/$f"; exit 2; }; done
export PYTHONHASHSEED=0
step() { if [ -e "$WORK/.step$1" ]; then echo "### step $1 already done"; return 1; fi; }
finished() { touch "$WORK/.step$1"; }
V=$WORK/venv RT=$WORK/rt

if step 1; then
  rm -rf "$WORK/litert-torch"
  run git clone -q --filter=blob:none https://github.com/google-ai-edge/litert-torch "$WORK/litert-torch"
  run git -C "$WORK/litert-torch" checkout -q --detach "$LITERT_TORCH_COMMIT"
  run git -C "$WORK/litert-torch" checkout -q "$BUILDER_COMMIT" -- "$BUILDER_FILE"
  finished 1
fi
if step 2; then
  rm -rf "$V" "$RT"
  run "${PYTHON:-python3.12}" -m venv "$V"
  run "$V/bin/pip" install -q -r "$Q/qwen35_08b_gpu_composite.lock.txt"
  run "$V/bin/pip" install -q --no-deps -e "$WORK/litert-torch"
  run "${PYTHON:-python3.12}" -m venv "$RT"
  run "$RT/bin/pip" install -q -r "$Q/qwen35_rt0192.lock.txt"
  finished 2
fi
if step 3; then
  run env HF_HUB_DISABLE_XET=1 "$V/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', revision='$MODEL_REVISION', local_dir='$WORK/model')"
  finished 3
fi
if step 4; then
  run curl -fsSL -o "$WORK/official.pbtext" "$OFFICIAL_URL"
  [ "$(shasum -a 256 "$WORK/official.pbtext" | cut -d' ' -f1)" = "$OFFICIAL_SHA256" ] || { echo "official.pbtext: not the expected sha256 $OFFICIAL_SHA256"; exit 1; }
  finished 4
fi
if step 5; then
  rm -rf "$WORK/raw" "$WORK/filecat"; mkdir -p "$WORK/raw"
  (cd "$WORK/raw" && run env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 RESOURCE_CONSTANT_NUMEL_THRESHOLD=10000000 \
    Q35_FIX=assign,share,filecat Q35_FILECAT_DIR="$WORK/filecat" Q35_QUANT_RECIPE="$Q/qwen35_gpu_composite_int8.json" \
    Q35_EVENT_LOG="$WORK/export_events.jsonl" \
    "$V/bin/python" "$Q/export_hf_lowmem.py" export_hf "$WORK/model" --output_dir="$WORK/raw" \
    --quantization_recipe="$Q/qwen35_gpu_composite_int8.json" --apply_gpu_composites=True --externalize_weights=False --cache_length=4096 \
    '--prefill_lengths=[1024,256,64,16,4,1]' --litert_lm_llm_metadata_override="$WORK/official.pbtext" \
    --experimental_lightweight_conversion=True --keep_temporary_files=True)
  [ -s "$WORK/raw/model.tflite" ] && [ -s "$WORK/raw/model.litertlm" ] || { echo "step 5 left no float model.tflite / model.litertlm"; exit 1; }
  rm -rf "$WORK/filecat"
  finished 5
fi
if step 6; then
  rm -f "$WORK/model_int4.tflite" "$WORK/raw_int4.litertlm"; rm -rf "$WORK/swap"
  run "$V/bin/python" -I "$Q/requant_tflite.py" "$WORK/raw/model.tflite" "$Q/qwen35_gpu_composite_mixed_int4.json" "$WORK/model_int4.tflite" --int4-floor
  run "${PACK_CLI[@]}" unpack "$WORK/raw/model.litertlm" --output-dir "$WORK/swap" < /dev/null
  T=$(ls "$WORK"/swap/*TFLiteModel*.tflite)
  [ "$(echo "$T" | wc -l | tr -d ' ')" = 1 ] || { echo "expected one TFLiteModel .tflite in the unpacked bundle: $T"; exit 1; }
  run cp "$WORK/model_int4.tflite" "$T"
  run "${PACK_CLI[@]}" pack "$WORK/swap" --output "$WORK/raw_int4.litertlm" < /dev/null
  [ -s "$WORK/raw_int4.litertlm" ] || { echo "step 6: pack wrote no file"; exit 1; }
  rm -rf "$WORK/swap"
  finished 6
fi
finish() { # finish RAW NAME
  local raw=$1 name=$2 d=$WORK/finish_${2%.litertlm}
  rm -rf "$d"; mkdir -p "$d"
  run "${PACK_CLI[@]}" unpack "$raw" --output-dir "$d/unpacked" < /dev/null
  run cp "$Q/qwen35_4b_gpu_composite.pbtext" "$d/unpacked/LlmMetadataProto.pbtext"
  run "$V/bin/python" -I -c "import struct, sys, zlib; d = open(sys.argv[1], 'rb').read(); open(sys.argv[2], 'wb').write(struct.pack('<Q', len(d)) + zlib.compress(d, 6))" \
    "$WORK/model/tokenizer.json" "$d/unpacked/Section2_HF_Tokenizer_Zlib.zlib"
  run "${PACK_CLI[@]}" pack "$d/unpacked" --output "$d/step1.litertlm" < /dev/null
  printf '#!/bin/sh\nexec %s "$@"\n' "${PACK_CLI[*]}" > "$d/packer.sh"; chmod +x "$d/packer.sh"
  run env PATH="$RT/bin:$PATH" "$RT/bin/python" "$R/scripts/set_activation_type.py" "$d/step1.litertlm" "$d/step2.litertlm" --type fp32 --litert-lm "$d/packer.sh"
  run env XR_PY="$V/bin/python" XR_LITERT_LM="$RT/bin/litert-lm" XR_BUILDER="$RT/bin/litert-lm-builder" XR_PEEK="$RT/bin/litert-lm-peek" \
    XR_WORK_ROOT="$d/xr_work" XR_STAGING_ROOT="$d/xr_staging" bash "$R/tools/externalize/externalize_repo.sh" litert-community/Qwen3.5-4B "$name" \
    --source "$d/step2.litertlm" --no-hub
  run mv "$d/xr_staging/litert-community__Qwen3.5-4B/$name" "$WORK/$name"
  rm -rf "$d/unpacked" "$d/step1.litertlm" "$d/step2.litertlm"
  shasum -a 256 "$WORK/$name"
}
if step 7; then
  run env XR_PY="$V/bin/python" bash "$R/tools/externalize/fetch_tools.sh"
  finish "$WORK/raw/model.litertlm" "$F8"
  finish "$WORK/raw_int4.litertlm" "$F4"
  finished 7
fi
rc=0
check_sections "$WORK/$F8" "$V/bin/python" || rc=1
check_sections "$WORK/$F4" "$V/bin/python" || rc=1
exit $rc
