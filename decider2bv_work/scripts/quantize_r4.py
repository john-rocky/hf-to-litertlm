"""Round 4: three weight forms of the RELU-free G = 256 decoder + embedder (out/decoder_g256_r4_fp32), ai-edge-quantizer
0.8.0 (out/venv-export), one part per process. The vision tower is not re-quantized: every variant uses round 3's
fp16 encoder + fp16 adapter (out/fp16_r3/vision_{encoder,adapter}_fp16.tflite).

  fp16 : decider_work V6 `wfp16` = weight-only FLOAT_CASTING 16-bit CHANNELWISE on FULLY_CONNECTED and
         EMBEDDING_LOOKUP, float compute, explicit DEQUANTIZE (round 3's recipe) - decoder and embedder.
  dyn8 : the house `wi8fc` = dynamic int8 (activations quantized on the fly) CHANNELWISE on FULLY_CONNECTED + int8
         EMBEDDING_LOOKUP CHANNELWISE (decider_work V1 / the qwen35vl ship recipe) - decoder and embedder.
  v7c  : decider_work/gpu_run `fp16fc_i8emb_dynhead` = every FULLY_CONNECTED weight-only fp16 (FLOAT_CASTING), then
         the lm_head scope `^decode_logits_output;$` DYNAMIC int8 CHANNELWISE (the later regex wins in
         recipe_manager), EMBEDDING_LOOKUP dynamic int8 CHANNELWISE. The scope regex was checked against THIS
         export's flatbuffer before use (scripts/lm_head_scope.py -> results/lm_head_scope_r4.json: exactly one
         match, the decode signature's [248320, 2048] vocabulary FC whose output is the `logits` signature output).
         Decoder and embedder (on the embedder only the EMBEDDING_LOOKUP rule can match).

Recipes are imported read-only (python -B) from decider_work/scripts/quantize_decider_ab.py (wfp16, wi8fc) and
decider_work/gpu_run/scripts/quantize_decider_ab.py (fp16fc_i8emb_dynhead); both files' sha256 are recorded. Every
output gets the flatbuffer weight-dtype census (scripts/weight_dtypes.py) and the op histogram.

    TMPDIR=out/tmp_quant PYTHONDONTWRITEBYTECODE=1 out/venv-export/bin/python -B -u scripts/quantize_r4.py <variant> <part>
    variant = fp16 | dyn8 | v7c ; part = decoder | embedder
"""
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SRC = {'decoder': 'out/decoder_g256_r4_fp32/model.tflite', 'embedder': 'out/decoder_g256_r4_fp32/embedder.tflite'}
RECIPES = {'fp16': ('decider_work/scripts/quantize_decider_ab.py', 'wfp16'),
           'dyn8': ('decider_work/scripts/quantize_decider_ab.py', 'wi8fc'),
           'v7c': ('decider_work/gpu_run/scripts/quantize_decider_ab.py', 'fp16fc_i8emb_dynhead')}


def sha256_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_recipe(variant):
    rel, kind = RECIPES[variant]
    path = os.path.join(ROOT, '..', rel)
    spec = importlib.util.spec_from_file_location(f'qab_{variant}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                  # python -B: no .pyc written next to the source
    return mod.build_recipe(kind), dict(source=f'{rel}::build_recipe({kind!r})', source_sha256=sha256_file(path))


def main():
    variant, part = sys.argv[1], sys.argv[2]
    src = os.path.join(ROOT, SRC[part])
    out_dir = os.path.join(ROOT, 'out/weights_r4', variant)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f'{part}.tflite')
    assert not os.path.exists(dst), f'refusing to overwrite {dst}'
    t0 = time.monotonic()
    rec = dict(status='RUNNING', variant=variant, part=part, source=SRC[part], output=os.path.relpath(dst, ROOT),
               tmpdir=os.environ.get('TMPDIR'), versions={p: md.version(p) for p in ('ai-edge-quantizer', 'ai-edge-litert')})
    log = os.path.join(out_dir, f'{part}.quant.json')
    try:
        rec['source_sha256'] = sha256_file(src)
        rec['source_bytes'] = os.path.getsize(src)
        recipe, prov = load_recipe(variant)
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
        from tflite_scan import scan
        rec['census'] = census(dst)
        s = scan(dst, with_sha=False)
        rec['op_histogram'] = s['op_histogram']
        rec['operator_count'] = s['operator_count']
        rec['forbidden'] = s['forbidden']
        rec['status'] = 'DONE'
    except BaseException as e:  # noqa: BLE001
        rec['status'] = 'ERROR'
        rec['error_type'] = type(e).__name__
        rec['traceback'] = traceback.format_exc()
        print(rec['traceback'], flush=True)
    rec['wall_seconds_contended'] = time.monotonic() - t0
    with open(log, 'w') as f:
        json.dump(rec, f, indent=1)
    print('QUANT', variant, part, rec['status'], rec.get('bytes'), rec.get('census', {}).get('weights_by_op'),
          f"{rec['wall_seconds_contended']:.0f}s", flush=True)
    if rec['status'] != 'DONE':
        sys.exit(1)


if __name__ == '__main__':
    main()
