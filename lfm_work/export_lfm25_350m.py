#!/usr/bin/env python3
"""LiquidAI/LFM2.5-350M export: 11-signature prefill ladder (1024 ... 1) + a 4096-token KV cache.

    python lfm_work/export_lfm25_350m.py OUTDIR [--fp]

Environment: litert-torch==0.9.4, litert-converter==0.4.0, ai-edge-quantizer==0.9.0, transformers==5.14.1 (torch 2.13.0).
transformers 5.15 breaks the lfm2 export (Lfm2ShortConv has no L_cache), so pin 5.14.x.

Same flags as lfm_work/convert_lfm25_230m.py except the cache: 4096, not 4099 (the macOS WebGPU prefill_1024 x
cache 4099 pair fails the shader compile, see the LFM2.5-230M entry of REPRODUCE.md).
--fp = no quantization: the base of the two post-hoc recipes in convert_lfm25_350m.sh. Without --fp the exporter's
default recipe applies (dynamic_wi8_afp32 = export-time int8 including the ShortConv convolutions; not what ships).
litert-torch 0.9.4 writes the ExecutorMetadata section itself (the hybrid's state-buffer binding; no retrofit needed).
The vendor chat template embedded here does not parse on litert-lm's minijinja; repack_official_lfm2.py replaces the
whole LlmMetadata afterwards.
"""
import sys

outdir = sys.argv[1] if len(sys.argv) > 1 else "out_350m"
fp = "--fp" in sys.argv

argv = [
    "litert-torch", "export_hf",
    "--model", "LiquidAI/LFM2.5-350M",
    "--output_dir", outdir,
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4096",
    "--bundle_litert_lm", "True",
    "--use_jinja_template", "True",
]
if fp:
    argv += ["--quantization_recipe", ""]

import importlib.metadata as md

# Print the stack first: the export depends on the exact transformers minor version.
print("versions:", ", ".join(
    f"{p} {md.version(p)}" for p in ("litert-torch", "litert-converter", "ai-edge-quantizer",
                                       "litert-lm-builder", "transformers", "torch")), flush=True)

from litert_torch.cli import main

sys.argv = argv
sys.exit(main())
