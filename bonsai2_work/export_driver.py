#!/usr/bin/env python3
"""Float export of a (Bonsai 2 / Qwen3.5) checkpoint to model.tflite with the Qwen3.5 hybrid patch
(PYTHONPATH=qwen35_work/litert-torch-qwen35) plus the Hadamard activation transform when the checkpoint
carries hadamard.json + hadamard_signs.safetensors. Stops at the tflite (no fp32 .litertlm packaging: the
27B float tflite is ~108 GB and the int4 build packages its own), keeps the work dir.

  PYTHONPATH=qwen35_work/litert-torch-qwen35 QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 \
    ~/venvs/ltconv040dev/bin/python3 bonsai2_work/export_driver.py <ckpt_dir> <out_dir> [extra export_hf flags]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hadamard_export  # noqa: E402

ckpt, out = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
os.makedirs(out, exist_ok=True)

from litert_torch.generative.export_hf.core import export_lib  # noqa: E402
from litert_torch.generative.export_hf.core import litert_lm_builder  # noqa: E402

_orig_load = export_lib.load_model


def load_model_with_hadamard(*a, **k):
  arts = _orig_load(*a, **k)
  n = hadamard_export.install_hadamard(arts.model, ckpt)
  print("hadamard modules:", n)
  return arts


export_lib.load_model = load_model_with_hadamard


def no_package(*a, **k):
  print("package_model skipped (tflite kept in work dir)")
  return a[-1] if a else None


litert_lm_builder.package_model = no_package

from litert_torch.cli import main  # noqa: E402

sys.argv = ["litert-torch", "export_hf", "--model", ckpt, "--output_dir", out,
            "--prefill_lengths", os.environ.get("QWEN35_PREFILL_LADDER", "1024,256,64,16,4,1"),
            "--cache_length", os.environ.get("CACHE_LENGTH", "4096"),
            "--quantization_recipe", ""] + extra
print("argv:", sys.argv)
raise SystemExit(main())
