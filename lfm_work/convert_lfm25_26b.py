#!/usr/bin/env python3
"""Convert LiquidAI/LFM2.5-2.6B (thinking flagship) to .litertlm.

  python convert_lfm25_26b.py [hf_id] [out_dir] [--fp] [--keep-softmax-composite]

Default = export-time dynamic int8 (convs included — safe at export time).
Full int4 pipeline (the published int4 file):

  python convert_lfm25_26b.py LiquidAI/LFM2.5-2.6B out_fp --fp
  python ../minicpm_work/quantize_litertlm.py apply out_fp/model.litertlm int4.litertlm \
      --recipe wi4b32_wi8 --algo octav
  python ../bonsai_work/fix_zero_block_scales.py int4.litertlm int4_zfix.litertlm
  python add_executor_metadata.py int4_zfix.litertlm LFM2.5-2.6B_int4.litertlm

Order matters: quantize -> zero-scale fix -> executor metadata. The zero-scale
step is NOT ternary-specific here — this dense-trained checkpoint carries ~746k
all-zero 32-blocks across 26 tensors, so OCTAV emits zero scales that XNNPACK
refuses at load ("unsupported scale value (0.000000) ... for INT4 tensor"; at
the engine level all you see is "Failed to invoke the compiled model").
The int8 file needs only add_executor_metadata.py after export.

Differences vs the 1.2B family script (convert_lfm25.py):

- metadata pbtext regenerated for the 2.6B tokenizer. Stop ids are
  per-tokenizer, not per-family: <|im_end|> is id 7 in the 1.2B and 124900
  here — reusing the sibling pbtext verbatim ships broken stops.
- the generation prompt pre-fills "<think>" exactly as the vendor template
  does. This is load-bearing for quantized builds: with a bare assistant
  prompt, think-block emission is the model's choice, and int4 loses that
  discipline first (unscaffolded plain-text deliberation instead of answers
  from turn 2). The pre-fill also routes the thought channel cleanly.
- history assistant turns render content verbatim — no position-dependent
  think-strip branches. The vendor template keys thinking-keep on the last
  user index, which changes between turns and violates the runtime's
  incremental conversation rendering (each render must string-extend the
  previous one; violation = turn-3 hard fail or silent mid-stream rewind).

Requires litert-torch >= 0.9.2 (upstream ShortConv prefill-pad fix) and
litert-lm >= 0.14 to run (>= 0.15 needs the executor-metadata step above).
"""
import os
import sys
from importlib.metadata import version

lt_ver = tuple(int(x) for x in version("litert-torch").split(".")[:3])
assert lt_ver >= (0, 9, 2), "needs litert-torch >= 0.9.2 (upstream ShortConv pad fix)"

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

model = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "LiquidAI/LFM2.5-2.6B"
outdir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "out_lfm25_26b"
fp = "--fp" in sys.argv

metadata = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "lfm25_26b_LlmMetaProto.pbtext")
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
