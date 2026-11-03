#!/usr/bin/env python3
"""text_export.py TEXT_DIR OUT_DIR [--prefill 1024,256,64,16] [--cache 4096] [--quant none]
Exports MiniCPM-V-4's text decoder (a dir from make_text_dir.py, class from v4_text_lm.py) with litert-torch
export_hf, task text_generation, externalized embedder (the VLM contract: the decoder takes `embeddings`; the EMBEDDER
section maps token ids; the vision features land at the image tokens). Pattern: muse_glimmer_work/muse_text_export.py.
Run with PYTHONPATH=<the TF schema shim> (export_hf 0.10.0 imports a TensorFlow schema module without declaring
TensorFlow; muse_glimmer_work/shim stands in for it). --quant defaults to "none" (export_hf's unset recipe is dynamic_wi8_afp32)."""
import argparse
import importlib.util
import os
import sys
import time


def load_wrapper():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4_text_lm.py")
    spec = importlib.util.spec_from_file_location("v4_text_lm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--prefill", default="1024,256,64,16")
    ap.add_argument("--cache", type=int, default=4096)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--single-token-embedder", action="store_true")
    a = ap.parse_args()
    # Import export_hf first: importing it puts a new `transformers` module object into sys.modules (seen in
    # trap 1), so a class registered before the import is lost.
    from litert_torch.generative.export_hf import export as export_lib
    load_wrapper().register()
    t0 = time.time()
    export_lib.export(
        model=os.path.abspath(a.text_dir),
        output_dir=os.path.abspath(a.out_dir),
        task="text_generation",
        keep_temporary_files=True,
        prefill_lengths=[int(x) for x in a.prefill.split(",")],
        cache_length=a.cache,
        quantization_recipe=a.quant,
        externalize_embedder=True,
        single_token_embedder=a.single_token_embedder,
        bundle_litert_lm=False,
    )
    print(f"EXPORT_DONE {time.time() - t0:.1f} s -> {a.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
