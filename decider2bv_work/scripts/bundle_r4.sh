#!/bin/zsh
# Round 4: build, ExecutorMetadata, unpack, content readout and inspection of ONE weight-form bundle.
#   scripts/bundle_r4.sh <variant>        (from decider2bv_work/; variant = fp16 | dyn8 | v7c)
# Layout: out/bundle_r4/<v>/decider-2b-vision_<v>{_noexec,}.litertlm, unpack in out/bundle_r4/<v>/unpack, fresh XNNPACK
# weight cache out/xnn_cache/<v>_bundle_r4.xnnpack_cache (so the unpacked bytes are repacked, not read back through the
# variant readout's cache). After the inspection PASSes, the regenerable intermediates of THIS round (noexec copy,
# unpack dir, the bundle-readout weight cache) are deleted and logged to logs/r4_deletions.log (disk cap).
set -u
V=$1
cd "$(dirname $0)/.."
PY=out/venv-readout/bin/python
D=out/bundle_r4/$V
N=decider-2b-vision_$V
L=logs/bundle_r4_$V.log
mkdir -p $D out/tmp logs
{
echo "# $(date '+%F %T %Z') bundle_r4.sh $V"
$PY -B scripts/build_bundle_g256.py --name $N --out-dir $D --decoder out/weights_r4/$V/decoder.tflite \
    --embedder out/weights_r4/$V/embedder.tflite --vision-encoder out/fp16_r3/vision_encoder_fp16.tflite \
    --vision-adapter out/fp16_r3/vision_adapter_fp16.tflite || exit 1
TMPDIR=out/tmp $PY -B ../scripts/add_executor_metadata.py $D/${N}_noexec.litertlm $D/$N.litertlm \
    --litert-lm out/venv-readout/bin/litert-lm --python $PY || exit 1
out/venv-readout/bin/litert-lm unpack $D/$N.litertlm --output-dir $D/unpack < /dev/null || exit 1
echo "# $(date '+%F %T %Z') content readout"
$PY -B -u scripts/graph_readout.py --decoder $D/unpack/Section4_TFLiteModel_tf_lite_prefill_decode.tflite \
    --embedder $D/unpack/Section3_TFLiteModel_tf_lite_embedder.tflite \
    --vision $V=$D/unpack/Section5_TFLiteModel_tf_lite_vision_encoder.tflite:$D/unpack/Section6_TFLiteModel_tf_lite_vision_adapter.tflite \
    --no-hf-arm --gate-arm $V --xnn-cache out/xnn_cache/${V}_bundle_r4.xnnpack_cache --out results/bundle_readout_r4_$V.json \
    > logs/graph_readout_bundle_r4_$V.log 2>&1
tail -1 logs/graph_readout_bundle_r4_$V.log
$PY -B scripts/inspect_bundle_r4.py --variant $V || exit 1
echo "# $(date '+%F %T %Z') done"
} > $L 2>&1
grep -E "BUNDLE_BUILT|^OK:|GRAPH_READOUT|BUNDLE_INSPECT|Error|error|Traceback" $L | head -20
if grep -q "BUNDLE_INSPECT PASS" $L; then
  for p in $D/${N}_noexec.litertlm $D/unpack out/xnn_cache/${V}_bundle_r4.xnnpack_cache; do
    [ -e $p ] || continue
    echo "$(date '+%F %T %Z') delete $(du -sk $p | cut -f1) KiB $p (round-4 regenerable intermediate, after BUNDLE_INSPECT PASS $V)" >> logs/r4_deletions.log
    rm -rf $p
  done
fi
