"""Export the re-hosted North-Micro-Vision decoder (Cohere2ForCausalLM checkpoint written
by prep_northmv_decoder.py) for the fast_vlm bundle = scripts/export_internvl_decoder.py
with the Compass rope patch active (northmv_rope_patch.py: Llama-style half-split rope,
which is what the checkpoint's verbatim Compass weights expect, and which lowers to the
standard GPU-safe rope graph -- stock Cohere2's interleaved rope emits a 5-D CONCATENATION
and a BROADCAST_TO that the LiteRT GPU delegate rejects).

    CACHE=4096 PREFILL=128,512,1024 RECIPE=dynamic_wi8_afp32 \
      .venv-vl093/bin/python northmv_work/export_northmv_decoder.py <llm_dir> <out_dir>
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from northmv_rope_patch import patch_cohere2_rope  # noqa: E402

patch_cohere2_rope()
print("[northmv] Cohere2 rope patched to Compass Llama-style layout")

sys.argv = [os.path.join(HERE, "..", "scripts", "export_internvl_decoder.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
