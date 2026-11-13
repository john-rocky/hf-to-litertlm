#!/usr/bin/env python3
"""export_hf_lowmem.py <export_hf arguments...> - litert-torch's `export_hf` for upstream's Qwen3.5 graph (litert-torch
65aafa4) with less memory, for the GPU-composite exports of the 4B and larger Qwen3.5 checkpoints. Takes the arguments of
`litert-torch export_hf ...`; add `--experimental_lightweight_conversion=True` and set RESOURCE_CONSTANT_NUMEL_THRESHOLD
(digits only: litert-torch reads it with str.isdigit(), so "1e7" is ignored with a warning and the 1 MiB default is used).

None of the changes below changes a value or the graph; the 0.8B export through this file and through the plain CLI are
the same computation (tools/tfl_compare/tfl_struct.py over all 1,968 subgraphs, see REPRODUCE.md). What they change:
  * clamp: backend/inline_consts._clamp_inf_values clamps every float constant in place (writing every page of every
    weight); here it calls the original only for a tensor that holds +-inf (for finite tensors the clamp is the identity).
  * assign (Q35_FIX): Qwen3_5StaticForCausalLM.from_hf_model without the random initialization of the fresh static model
    and with load_state_dict(assign=True): the static model holds the HF model's tensors instead of a copy. Every parameter
    of the static model must come from the HF state dict, else RuntimeError.
  * share (Q35_FIX): the prefill and decode modules use one wrapped, patched static model (65aafa4 builds one per module).
  * filecat (Q35_FIX): while apply_qwen3_5_model_patches fuses projections, every float torch.cat of at least
    Q35_FILECAT_MIN_BYTES (default 64e6) writes into a MAP_SHARED mapping of an unlinked file in Q35_FILECAT_DIR; pages
    backed by a file are not in the process footprint.
  * quant (Q35_QUANT_RECIPE = a recipe name of litert_quantizer.recipe or a .json recipe path): export_hf's quantize step
    becomes Quantizer(<mmap of the float TFLite>, recipe).quantize(serialize_to_path=...), so the float model is not read
    into memory; with Q35_INT4_FLOOR=1 every blockwise symmetric int4 scale below 2^-14 is raised to 2^-14 (XNNPACK refuses
    a zero scale). Without Q35_QUANT_RECIPE the --quantization_recipe argument runs as in export_hf.
One JSON line per event in Q35_EVENT_LOG (default ./export_hf_lowmem_events.jsonl): fixes, assign, share, filecat, quant.
"""
import json
import os
import sys
import time

_LOG = open(os.environ.get("Q35_EVENT_LOG", "export_hf_lowmem_events.jsonl"), "a", buffering=1)
_T0 = time.time()


def _emit(rec):
    rec = dict(rec, t=round(time.time() - _T0, 1))
    _LOG.write(json.dumps(rec) + "\n")
    print("export_hf_lowmem:", json.dumps(rec), flush=True)


def _fixes(fixes):
    import copy as _copy
    import math
    import tempfile
    import torch
    from torch import nn
    from litert_torch.generative.export_hf.model_ext.qwen3_5 import exportable_module as em
    from litert_torch.generative.export_hf.model_ext.qwen3_5 import modeling_qwen3_5_static as ms
    from litert_torch.generative.export_hf.model_ext.qwen3_5 import patch as qp
    _emit({"ev": "fixes", "fixes": sorted(fixes)})

    class _SkipInit:
        """nn.Linear / nn.Embedding / conv reset_parameters as no-ops: the parameter is allocated (untouched pages are
        not in the footprint) and replaced right after, so its values are never used."""
        def __enter__(self):
            self.saved = [(c, c.reset_parameters) for c in (nn.Linear, nn.Embedding, nn.modules.conv._ConvNd)]
            for c, _ in self.saved:
                c.reset_parameters = lambda self_: None

        def __exit__(self, *a):
            for c, f in self.saved:
                c.reset_parameters = f

    if "assign" in fixes:
        def from_hf_model(cls, hf_model):
            config = _copy.deepcopy(ms._unwrap_config(hf_model.config))
            with _SkipInit():
                static_model = cls(config)
            names = dict(static_model.named_parameters())
            res = static_model.load_state_dict(hf_model.state_dict(), strict=False, assign=True)
            miss = [k for k in res.missing_keys if k in names]
            _emit({"ev": "assign", "params": len(names), "missing_params": miss[:20],
                   "unexpected": list(res.unexpected_keys)[:20], "n_unexpected": len(res.unexpected_keys)})
            if miss:
                raise RuntimeError(f"export_hf_lowmem assign: parameters not in the HF state dict: {miss[:20]}")
            # buffers the static model computes itself (e.g. the rotary inv_freq) keep their __init__ values, as before
            return static_model
        ms.Qwen3_5StaticForCausalLM.from_hf_model = classmethod(from_hf_model)

    if "share" in fixes:
        orig_init = em.Qwen3_5ExportableMixin.__init__
        cache = {}

        def mixin_init(self, model, export_config, source_model_artifacts=None):
            if not isinstance(model, em.Qwen3_5StaticModelHFWrapper):
                key = id(model)
                if key not in cache:
                    w = em.Qwen3_5StaticModelHFWrapper(model)
                    em.qwen3_5_patch.apply_qwen3_5_model_patches(w.static_model, export_config)
                    cache[key] = (w, model)
                    _emit({"ev": "share_built"})
                else:
                    _emit({"ev": "share_reused"})
                model = cache[key][0]
            orig_init(self, model, export_config, source_model_artifacts)
        em.Qwen3_5ExportableMixin.__init__ = mixin_init

    if "filecat" in fixes:
        d = os.environ.get("Q35_FILECAT_DIR") or tempfile.gettempdir()
        os.makedirs(d, exist_ok=True)
        thresh = int(float(os.environ.get("Q35_FILECAT_MIN_BYTES", "64e6")))
        orig_cat = torch.cat
        stat = {"n": 0, "bytes": 0}

        def file_cat(tensors, dim=0, *args, **kw):
            ts = list(tensors)
            if (not args and not kw and ts and all(isinstance(t, torch.Tensor) for t in ts)
                    and ts[0].dtype.is_floating_point and ts[0].device.type == "cpu"):
                shape = list(ts[0].shape)
                dd = dim % len(shape)
                shape[dd] = sum(t.shape[dd] for t in ts)
                nbytes = math.prod(shape) * ts[0].element_size()
                if nbytes >= thresh:
                    fd, path = tempfile.mkstemp(dir=d, prefix="cat_", suffix=".bin")
                    os.ftruncate(fd, nbytes)
                    os.close(fd)
                    buf = torch.from_file(path, shared=True, size=math.prod(shape), dtype=ts[0].dtype).view(shape)
                    os.unlink(path)  # the mapping keeps the inode; the space is freed when it is unmapped
                    with torch.no_grad():
                        orig_cat(ts, dim=dd, out=buf)
                    stat["n"] += 1
                    stat["bytes"] += nbytes
                    return buf
            return orig_cat(tensors, dim, *args, **kw)

        orig_apply = qp.apply_qwen3_5_model_patches

        def apply(*a, **k):
            torch.cat = file_cat
            try:
                with _SkipInit():
                    return orig_apply(*a, **k)
            finally:
                torch.cat = orig_cat
                _emit({"ev": "filecat", "tensors": stat["n"], "gb": round(stat["bytes"] / 1e9, 3)})
        qp.apply_qwen3_5_model_patches = apply
        em.qwen3_5_patch.apply_qwen3_5_model_patches = apply


def _quant_override(recipe_ref):
    """export_hf's quantize step -> the quantizer over an mmap of the float model, serializing to a path."""
    import mmap
    import numpy as np
    from litert_torch.generative.export_hf.core import export_lib as el
    from litert_torch.generative.export_hf.core import quantization as qz

    def quantize_model(model_path, quantization_recipe):
        from litert_quantizer import quantizer as quantizer_lib
        from litert_quantizer import recipe as recipe_lib
        from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
        if recipe_ref.endswith(".json"):
            recipe = json.load(open(recipe_ref))
        elif recipe_ref in getattr(qz, "_LOCAL_QUANTIZATION_RECIPES", {}):
            recipe = qz._LOCAL_QUANTIZATION_RECIPES[recipe_ref]()
        else:
            recipe = recipe_lib.__dict__[recipe_ref]()
        orig = u.tensor_zp_scale_from_min_max
        floor = np.float32(2.0 ** -14)
        st = {"calls": 0, "floored": 0, "scales": 0}
        do_floor = os.environ.get("Q35_INT4_FLOOR") == "1"

        def floored(min_value, max_value, num_bits, symmetric, granularity, *a, **k):
            zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, *a, **k)
            if do_floor and "BLOCKWISE" in str(granularity) and num_bits == 4:
                if not symmetric:
                    raise ValueError("export_hf_lowmem: asymmetric blockwise int4 is not covered by the floor")
                st["calls"] += 1
                low = scale < floor
                st["scales"] += int(np.size(scale))
                n = int(np.count_nonzero(low))
                if n:
                    st["floored"] += n
                    scale = np.where(low, floor, scale).astype(np.asarray(scale).dtype)
            return zp, scale
        u.tensor_zp_scale_from_min_max = floored
        qpath = model_path.removesuffix(".tflite").removesuffix("_quantized") + "_quantized.tflite"
        _emit({"ev": "quant_start", "recipe": recipe_ref, "cli_recipe_unused": quantization_recipe,
               "float_bytes": os.path.getsize(model_path), "int4_floor": do_floor})
        with open(model_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            qt = quantizer_lib.Quantizer(mm, recipe)
            qt.quantize(serialize_to_path=qpath)
        finally:
            u.tensor_zp_scale_from_min_max = orig
        _emit({"ev": "quant_done", "bytes": os.path.getsize(qpath), "blockwise_int4_calls": st["calls"],
               "block_scales": st["scales"], "scales_floored": st["floored"]})
        return qpath
    qz.quantize_model = quantize_model
    el.quantize_model = quantize_model
    _emit({"ev": "quant_override", "recipe": recipe_ref})


def main():
    import torch
    import litert_torch.backend.inline_consts as ic
    orig = ic._clamp_inf_values

    def read_only(t):
        if isinstance(t, torch.Tensor) and t.is_floating_point() and bool(torch.isinf(t).any()):
            _emit({"ev": "clamp_inf", "shape": list(t.shape)})
            return orig(t)
        return t
    ic._clamp_inf_values = read_only
    fixes = set(x for x in os.environ.get("Q35_FIX", "").split(",") if x)
    if fixes:
        _fixes(fixes)
    if os.environ.get("Q35_QUANT_RECIPE"):
        _quant_override(os.environ["Q35_QUANT_RECIPE"])
    _emit({"ev": "start", "argv": sys.argv[1:], "threshold": os.environ.get("RESOURCE_CONSTANT_NUMEL_THRESHOLD"),
           "hashseed": os.environ.get("PYTHONHASHSEED")})
    from litert_torch.cli import main as cli_main
    sys.argv = ["litert_torch"] + sys.argv[1:]
    try:
        cli_main()
    finally:
        _emit({"ev": "end"})


if __name__ == "__main__":
    main()
