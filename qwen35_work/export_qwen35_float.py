#!/usr/bin/env python3
"""export_qwen35_float.py HF_ID OUTDIR - the float export of convert_qwen35_hybrid.py's step 1 (litert-torch export_hf, no
quantization; OUTDIR/model.litertlm) with two settings that script fixes:
  --cache_length QWEN35_CACHE_LENGTH (default 32768; convert_qwen35_hybrid.py writes 4096) - a model whose replies run to tens of
    thousands of tokens (Qwen3.5-4B-3D-Coder: REPRODUCE.md) needs a KV cache that holds the prompt and the reply;
  --prefill_lengths QWEN35_PREFILL_LADDER (default 1024,256,64,16,4,1 = the Qwen3.5-4B ladder).
Everything else as in convert_qwen35_hybrid.py step 1 (--quantization_recipe ""). Run it as a script (not imported) with the pinned
litert-torch checkout (115a13607c73 + qwen35_hybrid_litert_torch.patch, REPRODUCE.md) first on PYTHONPATH and PYTHONHASHSEED=0 in
the environment, without `python -I` (it would drop the seed: export_hf writes the stop tokens in the iteration order of a Python
set), HF_HUB_CACHE = a cache that holds the pinned snapshot (fetch_pinned.py) and HF_HUB_OFFLINE=1. Prints the litert_torch path it
imported, so the log shows the pinned checkout was used."""
import os
import sys

model, outdir = sys.argv[1], sys.argv[2]
assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0 (and without python -I)"
import litert_torch  # noqa: E402

print("litert_torch from", os.path.dirname(litert_torch.__file__), flush=True)
assert "litert-torch-qwen35" in litert_torch.__file__, "PYTHONPATH must put the pinned litert-torch checkout first"
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--prefill_lengths", os.environ.get("QWEN35_PREFILL_LADDER", "1024,256,64,16,4,1"),
    "--cache_length", os.environ.get("QWEN35_CACHE_LENGTH", "32768"),
    "--quantization_recipe", "",
]
print("argv", argv, flush=True)
from litert_torch.cli import main  # noqa: E402

sys.argv = argv
rc = main()
print("DONE (float export):", os.path.join(outdir, "model.litertlm"), "rc", rc, flush=True)
sys.exit(rc or 0)
