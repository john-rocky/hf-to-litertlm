#!/usr/bin/env python3
"""LiquidAI/LFM2-2.6B-Exp export: 11-signature prefill ladder (1024 ... 1) + a 4096-token KV cache, from a pinned commit.

    HF_HUB_CACHE=<cache> python lfm_work/export_lfm2_26b_exp.py OUTDIR [--fp] [--prefill L1,L2,...]

Environment: litert-torch==0.9.4, litert-converter==0.4.0, ai-edge-quantizer==0.9.0, transformers==5.14.1 (torch 2.13.0).
transformers 5.15 breaks the lfm2 export (Lfm2ShortConv has no L_cache), so pin 5.14.x.

The checkpoint is LiquidAI/LFM2-2.6B-Exp at commit 2251c865342649ffb4a9f57d07ab2428bb552cf5, the revision the published files
were converted from. `litert-torch export_hf` takes no revision, so fetch that commit first into a separate cache
(`python lfm_work/fetch_pinned.py LiquidAI/LFM2-2.6B-Exp 2251c865342649ffb4a9f57d07ab2428bb552cf5 <cache>`) and point
HF_HUB_CACHE at it: this script runs offline (HF_HUB_OFFLINE=1) and stops unless the cache's refs/main names that commit.
convert_lfm2_26b_exp.sh does both.

Same flags as lfm_work/export_lfm2_26b.py (LFM2-2.6B, the checkpoint this one was trained from; the same 30-layer
config) and lfm_work/export_lfm25_350m.py (and convert_lfm25_230m.py with cache 4096, not 4099: the macOS WebGPU
prefill_1024 x cache 4099 pair fails the shader compile, see the LFM2.5-230M entry of REPRODUCE.md).
--fp = no quantization: the base of the two post-hoc recipes in convert_lfm2_26b_exp.sh. Without --fp the exporter's default
recipe applies (dynamic_wi8_afp32 = export-time int8 including the ShortConv convolutions; not what ships).
--prefill overrides the prefill signature ladder, e.g. 1024,256,64,16,4,1 (six signatures).
litert-torch 0.9.4 writes the ExecutorMetadata section itself (the hybrid's binding of 22 ShortConv states and 8 key + 8 value
caches; no retrofit needed). The vendor chat template embedded here does not render on litert-lm 0.18.0 (its
`message.get("content")`: "Failed to apply template: unknown method: map has no method named get"); repack_official_lfm2.py
replaces the whole LlmMetadata afterwards.
"""
import os
import sys

REPO = "LiquidAI/LFM2-2.6B-Exp"
PINNED = "2251c865342649ffb4a9f57d07ab2428bb552cf5"

if "HF_HUB_CACHE" not in os.environ:
    sys.exit("set HF_HUB_CACHE to the cache fetch_pinned.py filled (see the docstring)")
os.environ["HF_HUB_OFFLINE"] = "1"  # before any hub import: resolve refs/main locally
ref = os.path.join(os.environ["HF_HUB_CACHE"], "models--" + REPO.replace("/", "--"), "refs", "main")
if not os.path.exists(ref) or open(ref).read().strip() != PINNED:
    sys.exit(f"{ref} does not name {PINNED}: run fetch_pinned.py first")

outdir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "out_lfm2_26b_exp"
fp = "--fp" in sys.argv
prefill = sys.argv[sys.argv.index("--prefill") + 1] if "--prefill" in sys.argv else "1024,512,256,128,64,32,16,8,4,2,1"

argv = [
    "litert-torch", "export_hf",
    "--model", REPO,
    "--output_dir", outdir,
    "--prefill_lengths", prefill,
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
print(f"HF_HUB_CACHE={os.environ['HF_HUB_CACHE']} HF_HUB_OFFLINE=1 revision={PINNED} prefill_lengths={prefill}", flush=True)

from litert_torch.cli import main

sys.argv = argv
sys.exit(main())
