#!/usr/bin/env python3
"""muse_text_export.py TEXT_DIR OUT_DIR [--prefill 128,...] [--cache 2048] [--quant none] [--v2]
                       [--bundle] [--lightweight]

Exports the Muse-Glimmer text decoder (a directory from muse_textdir.py) with litert-torch
export_hf, text_generation task, externalized embedder (the VLM contract: the decoder takes
embeddings, the EMBEDDER section maps token ids, the vision adapter's output lands at the image
tokens). The wrapper class is registered with AutoModelForCausalLM first (muse_text_lm.py).
Run with PYTHONPATH=shim (export_hf 0.10.0 imports a TensorFlow schema module; shim/ stands in for it). --quant
defaults to "none": export_hf's default for an unset recipe is dynamic_wi8_afp32, not float.
"""
import argparse
import importlib.util
import os
import sys
import time


def load_wrapper():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muse_text_lm.py")
    spec = importlib.util.spec_from_file_location("muse_text_lm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def readonly_clamp_patch():
    """litert-torch's backend/inline_consts.py `_clamp_inf_values` clamps every float constant IN PLACE while the MLIR
    module is built; on an mmapped F32 safetensors weight that write dirties every page (the whole checkpoint turns into
    anonymous memory). clamp(finfo.min, finfo.max) changes only +-inf, so this check reads in chunks and calls the
    original only when a tensor holds +-inf: the same constants, the weights stay file pages."""
    import atexit
    import torch
    from litert_torch.backend import inline_consts as ic
    orig = ic._clamp_inf_values
    stats = {"float_tensors": 0, "clamped": 0}

    def readonly(tensor):
        if not torch.is_floating_point(tensor):
            return tensor
        stats["float_tensors"] += 1
        flat = tensor.reshape(-1)
        step = 1 << 26
        for i in range(0, flat.numel(), step):
            if torch.isinf(flat[i:i + step]).any():
                stats["clamped"] += 1
                return orig(tensor)
        return tensor

    ic._clamp_inf_values = readonly
    atexit.register(lambda: print(f"readonly clamp check: {stats}", flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--prefill", default="128")
    ap.add_argument("--cache", type=int, default=2048)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--bundle", action="store_true")
    ap.add_argument("--lightweight", action="store_true")
    ap.add_argument("--readonly-clamp", action="store_true", help="the read-only +-inf check (F32 mmapped source)")
    a = ap.parse_args()
    if a.lightweight:
        # bytes, not elements: constants >= 1e8 B stay lazy through the lowering
        os.environ.setdefault("RESOURCE_CONSTANT_NUMEL_THRESHOLD", "100000000")
        print("RESOURCE_CONSTANT_NUMEL_THRESHOLD =", os.environ["RESOURCE_CONSTANT_NUMEL_THRESHOLD"], flush=True)

    # Import export_hf first: importing it puts a new `transformers` module object into
    # sys.modules (measured 2026-10-29), so a class registered before the import is lost.
    from litert_torch.generative.export_hf import export as export_lib
    load_wrapper().register()
    if a.readonly_clamp:
        readonly_clamp_patch()

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
        single_token_embedder=True,
        # No auto_model_override: export_hf 0.10.0's V2 path calls auto_model_cls.from_config(), which only
        # the Auto classes have; the class is found through AutoModelForCausalLM.register().
        bundle_litert_lm=a.bundle,
        use_v2=a.v2,
        delete_in_memory_params=a.v2,
        experimental_lightweight_conversion=a.lightweight,
    )
    print(f"EXPORT_DONE {time.time() - t0:.1f} s -> {a.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
