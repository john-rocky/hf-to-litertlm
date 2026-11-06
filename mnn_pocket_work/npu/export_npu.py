#!/usr/bin/env python3
"""export_npu.py STEP ... - the three stages of litert-torch's experimental npu_export for MNN-Pocket-0.3B on the
Qualcomm HTP of the Snapdragon 8 Elite Gen 5 (SM8850). No AOT stage: LiteRT-LM compiles the graphs for the HTP each time
an engine is created on the phone. Run by convert_mnn_pocket_0_3b_npu.sh, with its litert-torch tree (commit 6d4c622 +
litert_torch_6d4c622_split_cache_mask.diff) first on PYTHONPATH and PYTHONHASHSEED=0.

  export_npu.py export    MODEL_DIR PBTEXT OUT_DIR
      stage 1 -> OUT_DIR/model.litertlm: the float export with the KV cache split in two (npu_export requires it), int8
      weights with one scale per output channel, one prefill signature of 128 tokens, a 2,048-token KV cache, the
      source's tokenizer.json bytes and the LlmMetadata pbtext
  export_npu.py calibrate MODEL_DIR STAGE1 CALIB_DIR OUT_DIR
      stage 2 -> OUT_DIR/: every prompt of CALIB_DIR/ALL.jsonl (already rendered with the chat template, so no second
      formatting pass) run through stage 1's graphs, 64 decode steps after each; the min / max of every tensor
  export_npu.py quantize  MODEL_DIR STAGE1 BOUNDS_DIR OUT
      stage 3 -> OUT: static range quantization of stage 1's graphs (int16 activations, int8 weights) on the ranges in
      BOUNDS_DIR
"""
import os
import sys
import time

import litert_torch
from litert_torch.generative.export_hf.experimental.npu_export import stages
from litert_torch.generative.export_hf.experimental.npu_export.config_manager import build_pipeline_config


def config(model_dir):
  return build_pipeline_config(
      model_id=model_dir,
      target_vendor="qualcomm",
      target_soc="SM8850",
      weight_quantization_recipe="dynamic_wi8_afp32",
      prefill_lengths=[128],
      cache_length=2048,
      max_calibration_decode_steps=64,
  )


def main(argv):
  step, rest = argv[1], argv[2:]
  print("litert_torch from", os.path.dirname(litert_torch.__file__), flush=True)
  t0 = time.time()
  if step == "export":
    model_dir, pbtext, out_dir = rest
    cfg = config(model_dir)
    cfg.print_summary()
    os.makedirs(out_dir, exist_ok=True)
    stages.npu_export(cfg=cfg, output_dir=out_dir,
                      tokenizer_path_override=os.path.join(model_dir, "tokenizer.json"),
                      litert_lm_llm_metadata_override=pbtext)
  elif step == "calibrate":
    model_dir, stage1, calib_dir, out_dir = rest
    cfg = config(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    stages.npu_calibrate(cfg=cfg, input_litertlm=stage1, calibration_dataset_dir=calib_dir,
                         calibration_result_save_dir=out_dir, enable_formatting=False)
  elif step == "quantize":
    model_dir, stage1, bounds_dir, out = rest
    cfg = config(model_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    stages.npu_quantize(cfg=cfg, input_litertlm=stage1, output_litertlm=out, calibration_dir=bounds_dir)
  else:
    sys.exit(f"unknown step {step!r} (export, calibrate or quantize)")
  print(f"### {step} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
  main(sys.argv)
