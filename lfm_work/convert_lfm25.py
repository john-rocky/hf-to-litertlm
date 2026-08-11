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
`odml.softmax` StableHLO composite. litert-converter < 0.3.1 leaves that
composite in the graph and the GPU delegate rejects it, so GPU engine creation
fails; CPU is unaffected. This script strips the composite marker on those old
converters (math is unchanged — plain softmax stays in the graph), which makes
the export fully GPU-delegable (Mac WebGPU: ~4750 tok/s prefill, ~195 tok/s
decode on the 1.2B int8).

litert-converter >= 0.3.1 lowers the composite itself, to a fused SOFTMAX
builtin, so the strip is skipped automatically. Both paths converge on the same
graph: exporting with 0.3.1 and keeping the marker gives an op histogram
identical to the stripped export, marker count 0, Mac GPU gate PASS and 8/8 on
the quality gate for both backends. litert-torch 0.9.3 pins
`litert-converter==0.3.*`, so a plain install already gets you the lowering
converter — no flags. Pass --keep-softmax-composite to force the marker through
on an older converter (an old runtime will then refuse the GPU delegate).

Multi-length prefill signatures (1..1024) are exported so the runtime can pick
tight chunks. Requires litert-torch >= 0.9.1 and litert-lm >= 0.14 to run.
"""
import os
import sys
from importlib.metadata import version


def _ver(pkg):
    """(major, minor, patch) of an installed package; dev/rc suffixes dropped."""
    parts = version(pkg).split(".")[:3]
    return tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in parts)


def _converter_lowers_softmax():
    """True if this litert-converter turns odml.softmax into a fused builtin.

    Verified on 0.3.1 (released 2026-08-10) and on the 0.4.0.dev20260806
    nightly, which produce identical op histograms. The 0.4.0.dev line was
    renumbered to 0.3.1.dev when the release branch was cut, so an *older*
    0.4.0.dev nightly is not covered by that check on the numeric tuple alone
    — hence the date floor for dev builds.
    """
    raw = version("litert-converter")
    if _ver("litert-converter") < (0, 3, 1):
        return False
    if ".dev" in raw:
        try:
            return int(raw.split(".dev")[1]) >= 20260806
        except ValueError:
            return False
    return True


lt_ver = _ver("litert-torch")

if lt_ver < (0, 9, 2):
    # Upstream litert-torch gained the ShortConv prefill-pad fix in 0.9.2.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lfm_short_conv_patch import apply_patch

    apply_patch()

_strip_softmax = (lt_ver >= (0, 9, 2)
                  and not _converter_lowers_softmax()
                  and "--keep-softmax-composite" not in sys.argv)

if _strip_softmax:
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
