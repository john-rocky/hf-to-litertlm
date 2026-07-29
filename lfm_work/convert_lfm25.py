#!/usr/bin/env python3
"""Convert LiquidAI LFM2.5 models to .litertlm.

  python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_lfm25_12b [--fp] [--keep-softmax-composite]

Default exports the int8 file (litert-torch's export-time dynamic-int8 recipe,
convs included — safe at export time, unlike post-hoc conv int8). Pass --fp to
export unquantized instead, e.g. to post-hoc quantize int4 with
../minicpm_work/quantize_litertlm.py (recipe wi4b32_wi8 --algo octav; that
recipe touches only linears + embedding, convs stay float).

ShortConv prefill-pad fix: litert-torch 0.9.2 fixed this upstream (valid-token
masking + one-hot matmul state select — no GATHER_ND, no int64). On older
litert-torch (== 0.9.1) the local patch in lfm_short_conv_patch.py is applied
automatically; on >= 0.9.2 it is skipped.

GPU note (litert-torch >= 0.9.2): the exporter wraps attention softmax in a
`odml.softmax` StableHLO composite. The GPU delegate in the current released
runtime (litert-lm 0.14) does not accept that composite, so GPU engine creation
fails; CPU is unaffected. By default this script strips the composite marker
(math is unchanged — plain softmax stays in the graph), which makes the export
fully GPU-delegable today (Mac WebGPU: ~4750 tok/s prefill, ~195 tok/s decode
on the 1.2B int8). Pass --keep-softmax-composite to keep the marker for newer
runtimes that fuse it.

Multi-length prefill signatures (1..1024) are exported so the runtime can pick
tight chunks. Requires litert-torch >= 0.9.1 and litert-lm >= 0.14 to run.
"""
import os
import sys
from importlib.metadata import version

lt_ver = tuple(int(x) for x in version("litert-torch").split(".")[:3])

if lt_ver < (0, 9, 2):
    # Upstream litert-torch gained the ShortConv prefill-pad fix in 0.9.2.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lfm_short_conv_patch import apply_patch

    apply_patch()

if lt_ver >= (0, 9, 2) and "--keep-softmax-composite" not in sys.argv:
    # Strip the odml.softmax composite marker (see GPU note above). Rebinds the
    # name only inside the attention module; other composites (rms_norm) stay.
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
