#!/usr/bin/env python3
"""Export a Qwen3-ASR (-hf layout) checkpoint to a .litertlm bundle with litert-torch export_hf, task
automatic_speech_recognition, the recipe of litert-community/Qwen3-ASR-0.6B qwen3_asr_0.6b_5s_i8.litertlm as read back
from that bundle (out/census_official06b.json): 5 s audio window (audio encoder input [1, 128, 500] -> 73 embeddings),
prefill 128, KV cache 512, external embedder + audio encoder sections, dynamic int8 (dynamic_wi8_afp32: int8
per-channel weights, fp32 activations) on all three tflite models. Everything not listed is the export_hf default.

  ~/venvs/ltmain0918/bin/python export_bundle.py --model Qwen/Qwen3-ASR-0.6B-hf --out out/export/ours06b
  ~/venvs/ltmain0918/bin/python export_bundle.py --model out/hf_layout --out out/export/c4_i8
"""
import argparse
import json
import os
import time

import litert_torch
from litert_torch.generative.export_hf import export as export_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input_sec", type=float, default=5.0)
    ap.add_argument("--cache_length", type=int, default=512)
    ap.add_argument("--prefill_lengths", default="128")
    ap.add_argument("--quantization_recipe", default="dynamic_wi8_afp32")
    args = ap.parse_args()
    kwargs = dict(task="automatic_speech_recognition", input_sec=args.input_sec, cache_length=args.cache_length,
                  prefill_lengths=[int(x) for x in args.prefill_lengths.split(",")],
                  quantization_recipe=args.quantization_recipe, bundle_litert_lm=True)
    print("litert_torch", litert_torch.__file__, flush=True)
    print("EXPORT_ARGS", json.dumps({"model": args.model, "output_dir": args.out, **kwargs}), flush=True)
    t0 = time.time()
    export_lib.export(model=args.model, output_dir=args.out, **kwargs)
    path = os.path.join(args.out, "model.litertlm")
    print("EXPORT_DONE", path, os.path.getsize(path), "bytes", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
