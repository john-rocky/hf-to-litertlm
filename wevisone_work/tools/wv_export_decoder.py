#!/usr/bin/env python3
"""Round 2: fp32 export of WeVisOne-VL-2B's wide-input fast_vlm decoder + EMBEDDER (litert-torch 0.10.0 export_hf).

  EMBEDDER       : token_ids [1, 1] -> embeddings [1, 1, 8196]   (lookup [1, 1, 2048] + 6148 zeros, one PAD)
  PREFILL_DECODE : embeddings [1, T, 8196] + input_pos + mask [1, 1, T, 2048] + 56 KV states (-> logits on decode)

The rail is the stock `export_hf.export.export(...)` on the derived text checkpoint (K/out/text_ckpt, Qwen3ForCausalLM,
tools/text_ckpt.sh) with tools/wv_patch.py installed in this process only (wide inputs, deepstack adds after layers 0..2,
the flagged derived rotary on model.model.rotary_emb; nothing on disk changes). use_rope_composite=False keeps the
rotary on rotary_emb (the composite reads position_ids itself). Before the export the patched rotary
is checked against the plain piecewise formula on every position of the cache for both flag values (and, for the
image positions of the 512 grid, against ref/mrope_check.json's 512 rows = HF get_rope_index).

    ~/venvs/lt0100dev/bin/python -I tools/wv_export_decoder.py [--grid 16] [--out out/r2/dec_fp32]
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parents[1]
CKPT = K / "out" / "text_ckpt"
PREFILL = [1024, 256, 64, 16, 4, 1]
CACHE = 2048


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WVP = _load("wv_patch")


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def precheck(grid, inv_freq):
    """The patched cos / sin vs the plain formula's positions through HF-style rotary math, every cache position."""
    out = {}
    m = WVP.channel_masks(inv_freq.shape[0])
    for f in (0, 1):
        # flag 0 on every cache position; flag 1 only where the template puts image tokens (P .. P + N - 1)
        lo, hi = (0, CACHE) if f == 0 else (grid.P, grid.P + grid.N)
        p = torch.arange(lo, hi, dtype=torch.int32)[None]
        flag = torch.full((1, hi - lo, 1), float(f))
        c, s = WVP.flagged_cos_sin(p, flag, inv_freq, grid)
        ref = torch.tensor([WVP.reference_thw(int(x), f, grid) for x in range(lo, hi)], dtype=torch.float32)
        own = torch.tensor([[mm[j] for mm in m].index(1.0) for j in range(inv_freq.shape[0])])
        freqs = ref[:, own] * inv_freq.float()[None, :]                                   # per-frequency channel
        emb = torch.cat((freqs, freqs), dim=-1)[None]
        out[f"flag{f}"] = {"cos_max_abs": float((c - emb.cos()).abs().max()),
                           "sin_max_abs": float((s - emb.sin()).abs().max()),
                           "bit_equal": bool(torch.equal(c, emb.cos()) and torch.equal(s, emb.sin()))}
    # image slots of one 512 row against the HF positions r1 recorded (get_rope_index), P .. P + N - 1
    mr_path = K / "ref" / "mrope_check.json"
    if mr_path.exists():
        rows512 = [r for r in json.loads(mr_path.read_text())["rows"] if r["side"] == 512]
        out["r1_mrope_512_rows_all_equal"] = all(r["equal"] == r["positions"] for r in rows512)
    t, h, w = WVP.flagged_thw(torch.arange(grid.P, grid.P + grid.N, dtype=torch.float32),
                              torch.ones(grid.N), grid)
    q = torch.arange(grid.N)
    out["image_slots_equal"] = bool(torch.equal(t, torch.full((grid.N,), float(grid.P)))
                                    and torch.equal(h, (grid.P + q // grid.G).float())
                                    and torch.equal(w, (grid.P + q % grid.G).float()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--out", default="out/r2/dec_fp32")
    a = ap.parse_args()
    out_dir = (K / a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert not any(f.endswith(".tflite") for f in os.listdir(out_dir)), f"refusing to overwrite tflites in {out_dir}"
    rec = {"status": "RUNNING", "ckpt": str(CKPT), "prefill_lengths": PREFILL, "cache_length": CACHE,
           "grid": a.grid, "argv": sys.argv[1:], "wv_patch_sha256": sha256_file(Path(WVP.__file__))}
    t0 = time.monotonic()
    try:
        import transformers
        import litert_torch
        rec["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                           "transformers": transformers.__version__,
                           **{p: md.version(p) for p in ("litert-torch", "litert-converter", "ai-edge-litert",
                                                         "ai-edge-quantizer")}}
        grid = WVP.Grid(a.grid)
        rec["grid_repr"] = repr(grid)
        restore_cls = WVP.install_classes()
        restore_load, loaded = WVP.install_on_load(grid)
        from litert_torch.generative.export_hf.export import export
        args = dict(model=str(CKPT), output_dir=str(out_dir), quantization_recipe="", externalize_embedder=True,
                    single_token_embedder=True, cache_length=CACHE, prefill_lengths=PREFILL, bundle_litert_lm=False,
                    keep_temporary_files=True, use_jinja_template=False, use_rope_composite=False,
                    trust_remote_code=False)
        rec["export_args"] = args
        # the precheck runs right after load_model, before any task traces the model
        from litert_torch.generative.export_hf.core import export_lib
        wrapped_load = export_lib.load_model

        def load_and_check(*x, **y):
            art = wrapped_load(*x, **y)
            rec["model_class"] = loaded["model_class"]
            rec["precheck"] = precheck(grid, loaded["inv_freq"])
            print("PRECHECK", json.dumps(rec["precheck"]), flush=True)
            pc = rec["precheck"]
            assert pc["flag0"]["bit_equal"] and pc["flag1"]["bit_equal"] and pc["image_slots_equal"], pc
            (out_dir / "export_driver.json").write_text(json.dumps(rec, indent=1, default=str))
            return art
        export_lib.load_model = load_and_check
        (out_dir / "export_driver.json").write_text(json.dumps(rec, indent=1, default=str))
        export(**args)
        export_lib.load_model = wrapped_load
        restore_load()
        restore_cls()
        rec["tflites"] = {f: {"bytes": os.path.getsize(out_dir / f)} for f in sorted(os.listdir(out_dir))
                          if f.endswith(".tflite")}
        rec["status"] = "DONE"
    except BaseException as e:  # noqa: BLE001
        rec["status"] = "ERROR"
        rec["error_type"] = type(e).__name__
        rec["traceback"] = traceback.format_exc()
        print(rec["traceback"], flush=True)
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    (out_dir / "export_driver.json").write_text(json.dumps(rec, indent=1, default=str))
    print("EXPORT_DRIVER", rec["status"], f"{rec['wall_s']:.0f}s", flush=True)
    return 0 if rec["status"] == "DONE" else 1


if __name__ == "__main__":
    sys.exit(main())
