"""Round 3: fp16 (and one int8-adapter) variants of the four G = 256 fp32 tflites, ai-edge-quantizer 0.8.0
(out/venv-export), one part per call so each quantization is its own process.

  decoder / embedder : decider_work's V6 `wfp16` recipe, imported unchanged from
                       decider_work/scripts/quantize_decider_ab.py::build_recipe('wfp16') =
                       add_weight_only_config(regex '.*', FULLY_CONNECTED and EMBEDDING_LOOKUP, 16 bits, CHANNELWISE,
                       FLOAT_CASTING) -> weight-only, compute FLOAT, explicit_dequantize True.
  vision enc / adp   : the qwen35vl precedent (qwen35vl_work/vision_quant_ab.py::quant_fp16): JSON recipe
                       float_casting, 16-bit FLOAT weights, compute FLOAT, on FULLY_CONNECTED + CONV_2D only
                       (qwen35vl_work/FINDINGS.md: the recipe_manager dynamic-config route silently no-ops for float
                       casting; the JSON route rejects BATCH_MATMUL, which is weightless here anyway).
  adapter int8       : ai_edge_quantizer.recipe.dynamic_wi8_afp32() (the qwen35vl / Qwen3.5-2B-VL ship adapter).

Every output gets a flatbuffer weight-dtype census (scripts/weight_dtypes.py) in its record; TMPDIR is expected to
point under out/ so nothing is written outside decider2bv_work.

    TMPDIR=out/tmp_quant PYTHONDONTWRITEBYTECODE=1 out/venv-export/bin/python -B -u scripts/quantize_fp16_r3.py <part>
    part = decoder | embedder | vision_encoder | vision_adapter_fp16 | vision_adapter_int8
"""
import copy
import hashlib
import importlib.metadata as md
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DECIDER_SCRIPTS = os.path.join(ROOT, '../decider_work/scripts')
OUT = os.path.join(ROOT, 'out/fp16_r3')

PARTS = {
    'decoder': ('out/decoder_g256_fp32/model.tflite', 'decoder_fp16.tflite', 'wfp16'),
    'embedder': ('out/decoder_g256_fp32/embedder.tflite', 'embedder_fp16.tflite', 'wfp16'),
    'vision_encoder': ('out/vision_g256/vision_encoder.tflite', 'vision_encoder_fp16.tflite', 'json_float_casting_fc_conv'),
    'vision_adapter_fp16': ('out/vision_g256/vision_adapter.tflite', 'vision_adapter_fp16.tflite', 'json_float_casting_fc_conv'),
    'vision_adapter_int8': ('out/vision_g256/vision_adapter.tflite', 'vision_adapter_int8.tflite', 'dynamic_wi8_afp32'),
}


def sha256_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def recipe_for(kind):
    if kind == 'wfp16':
        sys.path.insert(0, DECIDER_SCRIPTS)
        import quantize_decider_ab                                   # read-only import (python -B, no .pyc written)
        return quantize_decider_ab.build_recipe('wfp16'), dict(
            source='decider_work/scripts/quantize_decider_ab.py::build_recipe("wfp16")',
            source_sha256=sha256_file(os.path.join(DECIDER_SCRIPTS, 'quantize_decider_ab.py')))
    import ai_edge_quantizer.recipe as r
    if kind == 'dynamic_wi8_afp32':
        return r.dynamic_wi8_afp32(), dict(source='ai_edge_quantizer.recipe.dynamic_wi8_afp32()')
    assert kind == 'json_float_casting_fc_conv'
    base = copy.deepcopy(r.dynamic_wi8_afp32()[0])
    base['algorithm_key'] = 'float_casting'
    base['op_config']['weight_tensor_config'] = {'num_bits': 16, 'symmetric': True, 'granularity': 'TENSORWISE', 'dtype': 'FLOAT'}
    base['op_config']['compute_precision'] = 'FLOAT'
    base['op_config']['explicit_dequantize'] = False
    recipes = []
    for op in ('FULLY_CONNECTED', 'CONV_2D'):
        rr = copy.deepcopy(base)
        rr['operation'] = op
        recipes.append(rr)
    return recipes, dict(source='qwen35vl_work/vision_quant_ab.py::quant_fp16 (copied: FULLY_CONNECTED + CONV_2D)',
                         source_sha256=sha256_file(os.path.join(ROOT, '../qwen35vl_work/vision_quant_ab.py')))


def main():
    part = sys.argv[1]
    src_rel, dst_name, kind = PARTS[part]
    os.makedirs(OUT, exist_ok=True)
    src, dst = os.path.join(ROOT, src_rel), os.path.join(OUT, dst_name)
    assert not os.path.exists(dst), f'refusing to overwrite {dst}'
    t0 = time.monotonic()
    rec = dict(status='RUNNING', part=part, kind=kind, source=src_rel, output=os.path.relpath(dst, ROOT),
               tmpdir=os.environ.get('TMPDIR'), versions={p: md.version(p) for p in ('ai-edge-quantizer', 'ai-edge-litert')})
    log = os.path.join(OUT, f'{part}.quant.json')
    try:
        rec['source_sha256'] = sha256_file(src)
        rec['source_bytes'] = os.path.getsize(src)
        recipe, prov = recipe_for(kind)
        rec['recipe'] = json.loads(json.dumps(recipe, default=str))
        rec['recipe_provenance'] = prov
        from ai_edge_quantizer import quantizer
        q = quantizer.Quantizer(src, recipe)
        rec['need_calibration'] = bool(q.need_calibration)
        assert not q.need_calibration
        t1 = time.monotonic()
        res = q.quantize()
        rec['quantize_seconds_contended'] = time.monotonic() - t1
        res.export_model(dst)
        rec['bytes'] = os.path.getsize(dst)
        rec['sha256'] = sha256_file(dst)
        rec['ratio_vs_source'] = rec['bytes'] / rec['source_bytes']
        from weight_dtypes import census
        rec['census'] = census(dst)
        rec['source_census'] = census(src)
        rec['status'] = 'DONE'
    except BaseException as e:  # noqa: BLE001
        rec['status'] = 'ERROR'
        rec['error_type'] = type(e).__name__
        rec['traceback'] = traceback.format_exc()
        print(rec['traceback'], flush=True)
    rec['wall_seconds_contended'] = time.monotonic() - t0
    with open(log, 'w') as f:
        json.dump(rec, f, indent=1)
    print('QUANT', part, rec['status'], rec.get('bytes'), rec.get('census', {}).get('weights_by_op'),
          f"{rec['wall_seconds_contended']:.0f}s", flush=True)
    if rec['status'] != 'DONE':
        sys.exit(1)


if __name__ == '__main__':
    main()
