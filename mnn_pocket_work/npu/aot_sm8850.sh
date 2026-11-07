#!/bin/bash
# aot_sm8850.sh OUTDIR - litert-community/MNN-Pocket-0.3B's SM8850 file, MNN-Pocket-0.3B_qualcomm_sm8850_ekv2048.litertlm:
# the NPU file (MNN-Pocket-0.3B_qualcomm_npu_ekv2048.litertlm, key mnn-pocket-0.3b-npu of scripts/reproduce_llm.sh)
# compiled ahead of time for the HTP of the Qualcomm SM8850 (Snapdragon 8 Elite Gen 5) in a container of the image of
# lfm3audio_work/npu/Dockerfile (tag npu-aot-qc:2.2.0-full: ai-edge-litert 2.2.0 and its Qualcomm compiler plugin with
# QAIRT 2.47.0.260601; build it as that file's header says). Needs docker, git and curl.
#   IN=<file>     the NPU file (default: downloaded into OUTDIR from the repository at revision d95e1d8d); its sha256
#                 must be the published file's, e8af457e...
#   REF=<file>    compare the result with this file by aot_compare.py, in the same container; exit 1 unless the two
#                 are the same bytes or the same computation
#   TREE=<dir>    a litert-torch checkout at 6d4c622 (default: cloned into OUTDIR/litert-torch); only the three files of
#                 its LiteRT-LM NPU compiler are used, checked by sha256
#   IMAGE=<tag>   the image (default npu-aot-qc:2.2.0-full); its QAIRT must be 2.47.0 build 260601114230
#   CONTAINER=<n> the name of the container the script creates (default aot_mnn_pocket_<pid>); removed at the end
# Steps: the image's QAIRT version; the input and its sha256; the compiler files; the three pure-Python packages the
# compiler imports and the image lacks (litert-lm-builder 0.17.0, absl-py 2.5.0, tomli 2.4.1), installed in the container
# from their platform-independent wheels on PyPI; the compile (aot_stage4.py); the result copied out with its sha256; with
# REF, the comparison (aot_compare.py). Every command and its output: OUTDIR/recipe.log.
set -euo pipefail
OUT=${1:?usage: aot_sm8850.sh OUTDIR}
N=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
LOG=$OUT/recipe.log; : > "$LOG"
exec > >(tee -a "$LOG") 2>&1
IMAGE=${IMAGE:-npu-aot-qc:2.2.0-full}
NAME=MNN-Pocket-0.3B_qualcomm_sm8850_ekv2048.litertlm
NPU_FILE=MNN-Pocket-0.3B_qualcomm_npu_ekv2048.litertlm
NPU_SHA=e8af457e3d2840a1c5b35f66e7e8bf17d1c88212608f2ec0c374ef01966060a0
NPU_URL=https://huggingface.co/litert-community/MNN-Pocket-0.3B/resolve/d95e1d8df7218f3ef64fe8cb2965a82721cf08d9/$NPU_FILE
LT_URL=https://github.com/google-ai-edge/litert-torch.git; LT_REV=6d4c622c9a3aade4d411a1e859c6bbe571d38ee0
CD=litert_torch/generative/export_hf/experimental/litert_lm_npu_compiler
COMPILER_FILES="__init__.py litert_lm_npu_compiler.py litert_lm_npu_compiler_configs.py"
COMPILER_SUMS="cbc703f53235f2cd50cb9a7191e3a899790d9fc4d6caaae7ae869b6339d14433  __init__.py
f394267a6c68aaa8059037c6766d0de61c149e478934432099a4e433db8c83d1  litert_lm_npu_compiler.py
68b1433c081cf1f99b97321bc6ad3e3085edff88f768f08dce0c3a6c74732fd8  litert_lm_npu_compiler_configs.py"
run() { echo "+ $*"; "$@"; }
sha() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-64; }
size() { wc -c < "$1" | tr -d ' '; }
echo "=== aot_sm8850.sh $(date '+%F %T') OUTDIR $OUT"

# 1. the image and its QAIRT
run docker image inspect "$IMAGE" --format '{{.Id}} {{.Created}}'
Q=$(docker run --rm --platform linux/amd64 "$IMAGE" sh -c \
  'grep -E "^(version|build_id):" /usr/local/lib/python3.12/site-packages/ai_edge_litert_sdk_qualcomm/data/sdk.yaml' | tr '\n' ' ')
echo "QAIRT in the image: $Q"
case "$Q" in *"version: 2.47.0"*"build_id: 260601114230"*) ;; *) echo "STOP: the image's QAIRT is not 2.47.0.260601114230"; exit 1 ;; esac

# 2. the input
IN=${IN:-$OUT/$NPU_FILE}
if [ ! -f "$IN" ]; then run curl -fL --retry 3 -o "$IN.part" "$NPU_URL"; mv "$IN.part" "$IN"; fi
got=$(sha "$IN"); echo "input $IN $(size "$IN") B, sha256 $got"
[ "$got" = "$NPU_SHA" ] || { echo "STOP: the input is not the published NPU file ($NPU_SHA)"; exit 1; }

# 3. the compiler files
TREE=${TREE:-$OUT/litert-torch}
if [ ! -d "$TREE/$CD" ]; then
  rm -rf "$TREE"
  run git clone -q --filter=blob:none "$LT_URL" "$TREE"
  run git -C "$TREE" checkout -q "$LT_REV"
fi
( cd "$TREE/$CD" && for f in $COMPILER_FILES; do echo "$(sha "$f")  $f"; done ) > "$OUT/compiler.sha256"
cat "$OUT/compiler.sha256"
[ "$(cat "$OUT/compiler.sha256")" = "$COMPILER_SUMS" ] || { echo "STOP: the compiler files are not litert-torch 6d4c622's"; exit 1; }

# 4. the container: the compiler files, the scripts, the packages, the input (its sha256 read back inside)
C=${CONTAINER:-aot_mnn_pocket_$$}
trap 'docker rm -f "$C" > /dev/null 2>&1 || true' EXIT
run docker create --platform linux/amd64 --name "$C" "$IMAGE" sleep infinity > /dev/null
run docker start "$C" > /dev/null
docker exec "$C" mkdir -p "/w/tree/$CD" /w/pylib /w/out
for f in $COMPILER_FILES; do docker cp "$TREE/$CD/$f" "$C:/w/tree/$CD/$f"; done
docker cp "$N/aot_stage4.py" "$C:/w/aot_stage4.py"
docker cp "$N/aot_compare.py" "$C:/w/aot_compare.py"
run docker exec "$C" pip install --quiet --disable-pip-version-check --root-user-action=ignore --no-deps --no-compile \
  --only-binary=:all: --platform any --python-version 3.12 --implementation py --target /w/pylib \
  litert-lm-builder==0.17.0 absl-py==2.5.0 tomli==2.4.1
docker exec "$C" sh -c 'cd /w/pylib && find . -name "*.py" | LC_ALL=C sort | xargs sha256sum' > "$OUT/pylib.sha256"
echo "pure-Python packages: $(grep -c . "$OUT/pylib.sha256") .py files (OUTDIR/pylib.sha256)"
run docker cp "$IN" "$C:/w/in.litertlm"
cin=$(docker exec "$C" sha256sum /w/in.litertlm | cut -c1-64)
[ "$cin" = "$NPU_SHA" ] || { echo "STOP: the input in the container is $cin"; exit 1; }

# 5. the compile
t0=$(date +%s)
run docker exec "$C" sh -c 'cd /w && PYTHONPATH=/w/pylib python -u aot_stage4.py /w/in.litertlm /w/out /w/tree'
echo "stage 4: $(( $(date +%s) - t0 )) s; container memory peak $(docker exec "$C" sh -c 'cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo -') B"
run docker cp "$C:/w/out/model_qualcomm_SM8850.litertlm" "$OUT/$NAME"
for f in $(docker exec "$C" sh -c 'cd /w/out && ls apply_plugin_*.log'); do docker cp "$C:/w/out/$f" "$OUT/$f"; done
echo "output $OUT/$NAME $(size "$OUT/$NAME") B, sha256 $(sha "$OUT/$NAME")"

# 6. the comparison
if [ -n "${REF:-}" ]; then
  echo "reference $REF $(size "$REF") B, sha256 $(sha "$REF")"
  run docker cp "$REF" "$C:/w/ref.litertlm"
  rc=0
  run docker exec "$C" sh -c 'cd /w && PYTHONPATH=/w/pylib python -u aot_compare.py /w/out/model_qualcomm_SM8850.litertlm /w/ref.litertlm /w/out/compare.json' || rc=$?
  docker cp "$C:/w/out/compare.json" "$OUT/compare.json"
  [ "$rc" = 0 ] || { echo "the rebuilt file is not the same computation as $REF (OUTDIR/compare.json)"; exit 1; }
fi
echo "=== done $(date '+%F %T')"
