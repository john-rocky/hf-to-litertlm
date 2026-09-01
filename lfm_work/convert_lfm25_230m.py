#!/usr/bin/env python3
"""LFM2.5-230M export, family shape (mirrors lfm_work/convert_lfm25.py flags:
11-signature prefill ladder + cache 4099, jinja template embedded verbatim).

    ~/venvs/ltconv040dev/bin/python convert_230m.py OUTDIR [--fp]

The raw export's embedded template does NOT run on litert-lm (minijinja:
`{% generation %}` statements + map .get method) — every bundle must go
through fix_230m_template.py afterwards, and 0.9.3 bundling omits the
ExecutorMetadata section (lfm_work/add_executor_metadata.py retrofits).
"""
import sys

outdir = sys.argv[1] if len(sys.argv) > 1 else "out_230m"
fp = "--fp" in sys.argv

argv = [
    "litert-torch", "export_hf",
    "--model", "LiquidAI/LFM2.5-230M",
    "--output_dir", outdir,
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4099",
    "--bundle_litert_lm", "True",
    "--use_jinja_template", "True",
]
if fp:
    argv += ["--quantization_recipe", ""]

from litert_torch.cli import main

sys.argv = argv
sys.exit(main())
