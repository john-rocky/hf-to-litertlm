#!/usr/bin/env python3
# Copied from wevisone_work/tools/wv_vision_r2.py (WeVisOne-VL-2B): the 4B widths (tools/wv_dims.py), fp16 as the
# only reduced form, the records under out/vision/, the modes that need the reference visual outputs left out.
"""The two vision sections of the WeVisOne-VL-4B file at 512 (export environment, litert-torch 0.10.0).

  VISION_ENCODER : images [1, 512, 512, 3] in [0, 1] -> [1, 1024, 4096] = merger inputs h23 | h5 | h11 | h17, raster
                   (tools/vision_taps.py VisionTaps; FP16SAFE LayerNorm pre-scale, LN_SCALES or the calibration rows)
  VISION_ADAPTER : [1, 1024, 4096] -> [1, 256, 10244] = merger(h23) | deepstack mergers 0..2 | flag 1 | 0 0 0
                   (tools/vision_taps.py WideAdapter + one CONCATENATION with a constant [1, 256, 4] = 1, 0, 0, 0)

Modes:
  export   fp32 tflites -> out/vision/{vision_encoder,vision_adapter}_fp32.tflite + op tables (vision_export.json)
  quant    fp16 (float casting) forms of both, ai-edge-quantizer 0.10.0 (vision_quant.json)

Both read out/vision512/ (tools/vision_taps.py consts): consts.npz, and the picture the export traces with
(ref_syn_a_det.npz). The published file's two vision sections are the fp16 forms:
  vision_encoder_fp16.tflite   611,814,736 B  sha256 26adbb94502caed3a3ca8d02f73269393593465b6305567b18e01818659750e4
  vision_adapter_fp16.tflite   218,245,632 B  sha256 88c674a4139bb8c9ed3e91b06baad110f6871f331f8bc1eaf4258a04b95eb793
from the fp32 forms of 1,217,361,552 B (encoder) and 436,348,484 B (adapter).

    PY_EXPORT -I tools/wv_vision.py export|quant
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parents[1]
OUT = K / "out" / "vision"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the vision tower of this directory; the op table helper of WeVisOne-VL-2B (it only reads a .tflite)
VT = _load("vision_taps", Path(__file__).resolve().parent / "vision_taps.py")
OPT = _load("op_table", K.parent / "wevisone_work" / "tools" / "op_table.py")


class FlagAdapter(torch.nn.Module):
    def __init__(self, ada):
        super().__init__()
        self.ada = ada
        tail = torch.zeros(1, VT.NTOK, VT.DIMS.WIDE - VT.DIMS.FLAG)
        tail[..., 0] = 1.0
        self.register_buffer("tail", tail, persistent=False)

    def forward(self, enc):
        return torch.cat([self.ada(enc), self.tail], dim=-1)            # [1, 256, WIDE]


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def path_of(part, form):
    return OUT / f"vision_{part}_{form}.tflite"


def mode_export():
    import litert_torch
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    enc, ada, scale = VT.build()
    fada = FlagAdapter(ada).eval()
    ref0 = np.load(VT.OUT / f"ref_{VT.CALIB_ROWS[0]}.npz")
    x0 = torch.from_numpy(ref0["image"]).float()[None] / 255.0
    with torch.no_grad():
        e0 = enc(x0)
    rec = {"litert_torch": litert_torch.__version__, "torch": torch.__version__, "fp16safe": VT.FP16SAFE,
           "ln_scales_file": VT.LN_SCALES or None, "ln_scales_gt1": {k: v for k, v in sorted(scale.items()) if v > 1}}
    t0 = time.time()
    litert_torch.convert(enc, (x0,)).export(str(path_of("encoder", "fp32")))
    rec["enc_convert_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    litert_torch.convert(fada, (e0,)).export(str(path_of("adapter", "fp32")))
    rec["ada_convert_s"] = round(time.time() - t0, 1)
    rec["encoder_sha256"] = sha256_file(path_of("encoder", "fp32"))
    rec["ops"] = {p: OPT.op_table(path_of(p, "fp32")) for p in ("encoder", "adapter")}
    (OUT / "vision_export.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("VISION_EXPORT", json.dumps({k: rec[k] for k in ("enc_convert_s", "ada_convert_s")}),
          {p: rec["ops"][p]["flagged"] for p in rec["ops"]}, flush=True)


def make_recipe(kind):
    import ai_edge_quantizer.recipe as r
    base = r.dynamic_wi8_afp32()[0]
    if kind == "int8dyn":
        return [base]
    q = copy.deepcopy(base)
    if kind == "fp16":
        q["algorithm_key"] = "float_casting"
        q["op_config"]["weight_tensor_config"] = {"num_bits": 16, "symmetric": True, "granularity": "TENSORWISE",
                                                  "dtype": "FLOAT"}
        q["op_config"]["compute_precision"] = "FLOAT"
        q["op_config"]["explicit_dequantize"] = False
        return [q]
    raise ValueError(kind)


def mode_quant():
    from ai_edge_quantizer import quantizer
    rec = {}
    for part in ("encoder", "adapter"):
        for kind in ("fp16",):
            dst = path_of(part, kind)
            if not dst.exists():
                t0 = time.time()
                qt = quantizer.Quantizer(str(path_of(part, "fp32")), make_recipe(kind))
                qt.quantize().export_model(str(dst))
                print(f"quant {part} {kind}: {dst.stat().st_size} B in {time.time() - t0:.0f} s", flush=True)
            rec[f"{part}_{kind}"] = {"bytes": dst.stat().st_size, "sha256": sha256_file(dst),
                                     "ops": OPT.op_table(dst)}
    for part in ("encoder", "adapter"):
        p = path_of(part, "fp32")
        rec[f"{part}_fp32"] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    (OUT / "vision_quant.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")


if __name__ == "__main__":
    {"export": mode_export, "quant": mode_quant}[sys.argv[1]]()
