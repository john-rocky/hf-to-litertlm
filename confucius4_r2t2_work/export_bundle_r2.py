#!/usr/bin/env python3
"""Export recipe with the original attention windows: export_bundle.py (the recipe read back from
litert-community/Qwen3-ASR-0.6B) with the audio
encoder attention windows equal to transformers 5.14.1 (encoder_window.py: 104-token block-diagonal windows instead
of litert-torch's 13-token chunks). Everything else is unchanged: task automatic_speech_recognition, prefill 128,
external embedder + audio encoder sections, builder 0.17.0 overlay (litert-torch 0918 writes LlmMetadata.pad_token).

  PYTHONPATH=out/pyoverlay_builder017 ~/venvs/ltmain0918/bin/python export_bundle_r2.py --model out/hf_layout \
      --out out/export/c4r2_f32_5s --quantization_recipe none
  PYTHONPATH=out/pyoverlay_builder017 ~/venvs/ltmain0918/bin/python export_bundle_r2.py --model out/hf_layout \
      --out out/export/c4r2_f32_30s --quantization_recipe none --input_sec 30 --cache_length 1024
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encoder_window  # noqa: E402
import export_bundle  # noqa: E402

if __name__ == "__main__":
    encoder_window.install()
    export_bundle.main()
    sys.exit(0)
