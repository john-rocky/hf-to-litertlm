#!/usr/bin/env python3
"""fp32 export of Qwen/Qwen3-ASR-1.7B-hf with litert-torch export_hf, task automatic_speech_recognition, the audio
encoder attention in transformers' 104-token windows (encoder_window.py, process-local; the litert-torch tree is not
edited). = confucius4_r2t2_work/export_bundle.py + export_bundle_r2.py in one file: the recipe of
litert-community/Qwen3-ASR-0.6B (prefill 128, external embedder + audio encoder sections) with a 30 s window and a
1024-token KV cache. Quantization happens per section afterwards (quant_sections.py).

  PYTHONPATH=out/pyoverlay_builder017 ~/venvs/ltmain0918/bin/python export_fp32.py --out out/export/q17_f32_30s

litert-lm-builder 0.17.0 on PYTHONPATH: litert-torch 0918 writes LlmMetadata.pad_token, which the venv's builder
0.16.1 lacks.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import encoder_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.MODEL_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input_sec", type=float, default=30.0)
    ap.add_argument("--cache_length", type=int, default=1024)
    ap.add_argument("--prefill_lengths", default="128")
    ap.add_argument("--quantization_recipe", default="none")
    args = ap.parse_args()
    encoder_window.install()
    import litert_torch
    import litert_lm_builder
    from litert_torch.generative.export_hf import export as export_lib
    kwargs = dict(task="automatic_speech_recognition", input_sec=args.input_sec, cache_length=args.cache_length,
                  prefill_lengths=[int(x) for x in args.prefill_lengths.split(",")],
                  quantization_recipe=args.quantization_recipe, bundle_litert_lm=True)
    print("litert_torch", litert_torch.__file__, "| litert_lm_builder", litert_lm_builder.__file__, flush=True)
    print("EXPORT_ARGS", json.dumps({"model": args.model, "output_dir": args.out, **kwargs}), flush=True)
    t0 = time.time()
    export_lib.export(model=args.model, output_dir=args.out, **kwargs)
    path = os.path.join(args.out, "model.litertlm")
    print("EXPORT_DONE", path, os.path.getsize(path), "bytes", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
