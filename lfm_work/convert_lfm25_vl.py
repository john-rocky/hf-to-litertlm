#!/usr/bin/env python3
"""Convert LiquidAI/LFM2.5-VL-3B to .litertlm (litert-torch >= 0.9.3).

  python convert_lfm25_vl3b.py [hf_id_or_path] [out_dir] [--fp] [--keep-softmax-composite]

Default = export-time dynamic int8 (text + vision, dynamic_wi8_afp32).
--fp = unquantized TEXT export for post-hoc int4-b32 via
  minicpm_work/quantize_litertlm.py (vision stays dynamic int8:
  quantizing the tower post-hoc is not supported there and int8 is the
  shipped default recipe anyway).

VL specifics vs lfm_work/convert_lfm25_26b.py (2.6B text ship):
- --task image_text_to_text: exports vision_encoder + vision_adapter tflites
  and bundles them as VISION_ENCODER / VISION_ADAPTER sections
  (externalize_embedder + single_token_embedder are auto-set by this task).
- metadata pbtext sets llm_model_type { lfm2 {} } so litert-lm's factory
  picks Lfm2DataProcessor (proto defaults == runtime defaults == this model:
  512x512, patch 16, max 1024 patches, pooling 2, mean/std 0.5).
- template renders bare "<image>" per image part; the runtime splits on it
  and inserts boi/eoi + preprocessed pixels itself (lfm2_data_processor.cc).
- template is append-only (no thinking-strip branches); VL-3B is a
  non-thinking model (vendor generation prompt does not open <think>).

After export, retrofit executor metadata (0.9.3 does not write the section):
  lfm_work/add_executor_metadata.py <out>.litertlm <out>_fix.litertlm
"""
import os
import sys
from importlib.metadata import version

lt_ver = tuple(int(x) for x in version("litert-torch").split(".")[:3])
assert lt_ver >= (0, 9, 3), "needs litert-torch >= 0.9.3 (lfm2_vl export path)"

if "--keep-softmax-composite" not in sys.argv:
    # Strip the odml.softmax composite marker: released litert-converter 0.3.x
    # does not lower it and the GPU delegate rejects the composite. Math unchanged.
    from types import SimpleNamespace
    from litert_torch.generative.export_hf.core import attention

    class _PassthroughBuilder:
        def __init__(self, *args, **kwargs):
            pass

        def mark_inputs(self, *xs):
            return xs[0] if len(xs) == 1 else xs

        def mark_outputs(self, *xs):
            return xs[0] if len(xs) == 1 else xs

    attention.composite = SimpleNamespace(
        StableHLOCompositeBuilder=_PassthroughBuilder
    )

model = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "LiquidAI/LFM2.5-VL-3B"
outdir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "out_lfm25_vl3b"
fp = "--fp" in sys.argv

# 450M is the old 64k vocab (stops 7/2, verified in its generation_config);
# 3B is the 128k one (124900/124895). Any other size: verify the vocab first.
if "450" in model.lower():
    pbtext = "lfm25_vl450m_LlmMetaProto.pbtext"
elif "3b" in model.lower().replace("-", ""):
    pbtext = "lfm25_vl3b_LlmMetaProto.pbtext"
else:
    raise SystemExit(f"unknown vocab for {model}: check stop token ids first")
metadata = os.path.join(os.path.dirname(os.path.abspath(__file__)), pbtext)
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--task", "image_text_to_text",
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4099",
    "--bundle_litert_lm", "True",
    "--use_jinja_template", "True",
    "--litert_lm_llm_metadata_override", metadata,
]
if fp:
    argv += ["--quantization_recipe", ""]

# Forward any other --key value pairs verbatim (e.g.
# --vision_encoder_quantization_recipe "") for probe runs.
_consumed = {"--fp", "--keep-softmax-composite"}
argv += [a for a in sys.argv[3:] if a not in _consumed]

from litert_torch.cli import main

sys.argv = argv
sys.exit(main())
