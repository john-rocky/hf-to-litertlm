#!/usr/bin/env python3
"""Convert LiquidAI LFM2.5 models to .litertlm (with the ShortConv prefill fix).

  python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_lfm25_12b [--fp]

Default exports the int8 file (litert-torch's export-time dynamic-int8 recipe,
convs included — safe at export time, unlike post-hoc conv int8). Pass --fp to
export unquantized instead, e.g. to post-hoc quantize int4 with
../minicpm_work/quantize_litertlm.py (recipe wi4b32_wi8 --algo octav; that
recipe touches only linears + embedding, convs stay float).

Multi-length prefill signatures (1..1024) are exported so the runtime can pick
tight chunks. Requires litert-torch >= 0.9.1 and litert-lm >= 0.14 to run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfm_short_conv_patch import apply_patch

apply_patch()

model = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2.5-1.2B-Instruct"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_lfm25"
fp = "--fp" in sys.argv

metadata = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "lfm25_LlmMetaProto.pbtext")
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4099",
    "--bundle_litert_lm", "True",
    "--use_jinja_template", "True",
    "--litert_lm_llm_metadata_override", metadata,
]
if fp:
    argv += ["--quantization_recipe", ""]

from litert_torch.cli import main

sys.argv = argv
sys.exit(main())
