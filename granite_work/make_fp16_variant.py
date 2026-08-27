#!/usr/bin/env python3
"""Build an fp16 (weight-only float-casting) variant of a .litertlm bundle.

Casts FULLY_CONNECTED + EMBEDDING_LOOKUP weights to fp16 (ai-edge-quantizer
FLOAT_CASTING); convs and the SSM scan stay fp32 — the same scope as the wi8fc
int8 recipe, so the two variants differ only in weight precision. Roughly 1.9x
smaller than fp32 and greedy-faithful to the fp32 export in our gates.

Note for phones: the CPU runtime unpacks fp16 weights back to fp32 in RAM, so
peak memory is close to the fp32 model's — treat fp16 as the desktop/quality
variant and ship int8 for mobile.

Usage: python make_fp16_variant.py in_fp32.litertlm out_fp16.litertlm [--drop-start-token]
Needs the `litert-lm` CLI and ai-edge-quantizer installed.
"""
import glob, os, shutil, subprocess, sys, tempfile

LITERT_LM = shutil.which("litert-lm") or sys.exit("litert-lm CLI not on PATH (pip install litert-lm)")
src, dst = sys.argv[1], sys.argv[2]
drop_bos = "--drop-start-token" in sys.argv[3:]

work = tempfile.mkdtemp(prefix="fp16var_")
unpack = os.path.join(work, "u")
subprocess.run([LITERT_LM, "unpack", src, "--output-dir", unpack], check=True)

if drop_bos:
    meta = os.path.join(unpack, "LlmMetadataProto.pbtext")
    s = open(meta).read()
    assert s.startswith("start_token {"), "expected a leading start_token block"
    open(meta, "w").write(s.split("}\n", 1)[1])

# Locate the TFLite section by pattern, not by index: the section number depends
# on how many sections precede it, so a float export straight out of the
# exporter (no ExecutorMetadata yet) names it Section2 while a bundle that
# already carries ExecutorMetadata names it Section3.
matches = glob.glob(os.path.join(unpack, "Section*_TFLiteModel_*.tflite"))
if len(matches) != 1:
    sys.exit(f"expected exactly one TFLiteModel section, found {matches}")
tfl = matches[0]
from ai_edge_quantizer import quantizer, recipe_manager, qtyping

rm = recipe_manager.RecipeManager()
for op in (qtyping.TFLOperationName.FULLY_CONNECTED,
           qtyping.TFLOperationName.EMBEDDING_LOOKUP):
    rm.add_quantization_config(
        regex=".*",
        operation_name=op,
        algorithm_key="float_casting",
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT,
        ),
    )
qt = quantizer.Quantizer(tfl, rm.get_quantization_recipe())
if qt.need_calibration:
    raise SystemExit("float casting unexpectedly needs calibration")
res = qt.quantize()
fp16_tfl = os.path.join(work, "model_fp16.tflite")
res.export_model(fp16_tfl, overwrite=True)
os.replace(fp16_tfl, tfl)

if os.path.exists(dst):
    os.remove(dst)  # `litert-lm pack` silently no-ops when the output exists
subprocess.run([LITERT_LM, "pack", unpack, "--output", dst], check=True)
shutil.rmtree(work)
print("OK", dst, os.path.getsize(dst))
