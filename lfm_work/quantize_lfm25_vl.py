#!/usr/bin/env python3
"""Post-process an fp LFM2.5-VL .litertlm: int4-b32 text + zero-scale fix +
ExecutorMetadata, preserving ALL sections (vision encoder/adapter, embedder).

Why not quantize_litertlm.py / fix_zero_block_scales.py directly: both rebuild
with litert-lm-builder passing a SINGLE prefill_decode tflite, which would drop
the vision sections of a VLM bundle. This script goes through `litert-lm
unpack/pack` instead, so every section survives; only the prefill_decode tflite
(the one with kv_cache_* inputs) is rewritten in place.

Pipeline (same order as the 2.6B ship): quantize -> zero-scale fix -> executor
metadata -> pack.

Usage:
  quantize_vl.py <in_fp.litertlm> <out.litertlm> [--recipe wi4b32_wi8] [--algo octav]
  quantize_vl.py <in_int8.litertlm> <out.litertlm> --recipe none   # metadata only

Run with .venv-vl093 python (ai-edge-quantizer + ai-edge-litert present).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "minicpm_work"))
sys.path.insert(0, os.path.join(REPO, "bonsai_work"))
sys.path.insert(0, os.path.join(REPO, "lfm_work"))

from quantize_litertlm import build_recipe  # noqa: E402
from fix_zero_block_scales import patch_tflite  # noqa: E402
from add_executor_metadata import build_pbtext, read_state_buffers  # noqa: E402


def find_prefill_decode(py, unpack_dir):
    tflites = [f for f in os.listdir(unpack_dir) if f.endswith(".tflite")]
    with_state = []
    for f in tflites:
        try:
            if read_state_buffers(py, os.path.join(unpack_dir, f)):
                with_state.append(f)
        except subprocess.CalledProcessError:
            pass
    if len(with_state) != 1:
        raise SystemExit(f"expected exactly one tflite with kv_cache_* inputs, "
                         f"got {with_state} out of {tflites}")
    return os.path.join(unpack_dir, with_state[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--recipe", default="wi4b32_wi8",
                    choices=["wi4b32_wi8", "wi8", "wi8fc", "none"])
    ap.add_argument("--algo", default="octav", choices=["minmax", "octav"])
    ap.add_argument("--litert-lm",
                    default=os.environ.get("LITERT_LM", "litert-lm"))
    args = ap.parse_args()

    td = tempfile.mkdtemp(prefix="qvl_")
    unpack = os.path.join(td, "unpack")
    subprocess.run([args.litert_lm, "unpack", args.src, "--output-dir", unpack],
                   check=True, capture_output=True, text=True)
    toml_path = os.path.join(unpack, "model.toml")
    toml = open(toml_path).read()

    pd_path = find_prefill_decode(sys.executable, unpack)
    print("prefill_decode:", os.path.basename(pd_path),
          round(os.path.getsize(pd_path) / 1e6), "MB")

    if args.recipe != "none":
        from ai_edge_quantizer import quantizer
        qt = quantizer.Quantizer(pd_path, build_recipe(args.recipe, args.algo))
        if qt.need_calibration:
            raise SystemExit("recipe unexpectedly needs calibration")
        result = qt.quantize()
        qpath = os.path.join(td, "model_quant.tflite")
        result.export_model(qpath)
        print("quantized:", round(os.path.getsize(qpath) / 1e6), "MB")
        shutil.move(qpath, pd_path)
        n = patch_tflite(pd_path)
        print("zero-scale fix:", n)

        # The VLM embedder is externalized into its own tflite, so the text
        # recipe never reaches it; an fp-text export leaves it float32 (1GB at
        # 128k vocab). Quantize it dynamic int8, matching the export-time int8
        # variant (official 2.6B int4 layout: embed int8).
        emb = [f for f in os.listdir(unpack) if f.endswith("_embedder.tflite")]
        if emb:
            emb_path = os.path.join(unpack, emb[0])
            before = os.path.getsize(emb_path)
            qt = quantizer.Quantizer(emb_path, build_recipe("wi8"))
            result = qt.quantize()
            qpath = os.path.join(td, "embedder_quant.tflite")
            result.export_model(qpath)
            print(f"embedder: {round(before / 1e6)} -> "
                  f"{round(os.path.getsize(qpath) / 1e6)} MB")
            shutil.move(qpath, emb_path)

    if "ExecutorMetadata" not in toml:
        buffers = read_state_buffers(sys.executable, pd_path)
        if not buffers:
            raise SystemExit("no kv_cache_* state inputs found")
        with open(os.path.join(unpack, "ExecutorMetadataProto.pbtext"), "w") as f:
            f.write(build_pbtext(buffers))
        marker = 'data_path = "LlmMetadataProto.pbtext"\n'
        if marker not in toml:
            raise SystemExit("model.toml has no LlmMetadata section to anchor on")
        toml = toml.replace(marker, marker +
                            '\n[[section]]\nsection_type = "ExecutorMetadata"\n'
                            'data_path = "ExecutorMetadataProto.pbtext"\n')
        open(toml_path, "w").write(toml)

    if os.path.exists(args.dst):
        os.remove(args.dst)
    subprocess.run([args.litert_lm, "pack", toml_path, "--output",
                    os.path.abspath(args.dst)], check=True)
    print("wrote", args.dst, round(os.path.getsize(args.dst) / 1e6), "MB")
    shutil.rmtree(td)


if __name__ == "__main__":
    main()
