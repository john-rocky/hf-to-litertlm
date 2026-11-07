#!/usr/bin/env python3
"""stage3_only.py EXPORTED.litertlm BOUNDS_DIR OUT.litertlm - stage 3 (static-range quantize) of litert-torch's
experimental npu_export alone, from stage 1's file and a stage-2 bounds directory, with the configuration of the
gemma-3-270m NPU files: the gemma3_270m_qualcomm_sm8850.json preset that ships in the tree, 32 calibration decode steps
and a 896-token KV cache. Run it with litert-torch on PYTHONPATH (tree and environment: docs/android-npu.md section 4)."""
import os
import sys
import time

from litert_torch.generative.export_hf.experimental.npu_export import stages
from litert_torch.generative.export_hf.experimental.npu_export.config_manager import NpuPipelineConfig

if len(sys.argv) != 4:
  sys.exit(__doc__)
exported, bounds, out = sys.argv[1:4]
preset = os.path.join(os.path.dirname(os.path.abspath(stages.__file__)), "configs", "gemma3_270m_qualcomm_sm8850.json")
cfg = NpuPipelineConfig.load_json(preset)
cfg.max_calibration_decode_steps = 32
cfg.cache_length = 896
print("litert_torch from", os.path.dirname(os.path.abspath(stages.__file__)), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
t = time.time()
stages.npu_quantize(cfg=cfg, input_litertlm=exported, output_litertlm=out, calibration_dir=bounds)
print(f"stage 3 done in {time.time() - t:.1f} s -> {out} ({os.path.getsize(out)} B)", flush=True)
