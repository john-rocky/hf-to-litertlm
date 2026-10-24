#!/usr/bin/env python3
"""Round 2: the two vision sections of the WeVisOne-VL-2B bundle at 512 (export venv, litert-torch 0.10.0).

  VISION_ENCODER : images [1, 512, 512, 3] in [0, 1] -> [1, 1024, 4096] = merger inputs h23 | h5 | h11 | h17, raster
                   (tools/vision_taps.py VisionTaps, unchanged; FP16SAFE LayerNorm pre-scale, calibration rows of r1)
  VISION_ADAPTER : [1, 1024, 4096] -> [1, 256, 8196] = merger(h23) | deepstack mergers 0..2 | flag 1 | 0 0 0
                   (tools/vision_taps.py WideAdapter + one CONCATENATION with a constant [1, 256, 4] = 1, 0, 0, 0)

Modes:
  export   fp32 tflites -> out/r2/vision/{vision_encoder,vision_adapter}_fp32.tflite + op tables
  quant    fp16 (float casting) and int8 dynamic (dynamic_wi8_afp32) forms of both, ai-edge-quantizer 0.10.0
  parity   every fixture image (17): out/r2/png512/<stem>.png -> u8 * (1 / 255) as the runtime feeds it -> encoder ->
           adapter, each encoder x adapter form, against HF's visual outputs (out/r2/hf_vis): per-token cosine and corr
           for image_embeds and the three deepstack parts, the flag column (= 1) and the zero tail -> r2/vision_parity.json

    ~/venvs/lt0100dev/bin/python -I tools/wv_vision_r2.py export|quant|parity
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
OUT = K / "out" / "r2" / "vision"
PNG = K / "out" / "r2" / "png512"
HFVIS = K / "out" / "r2" / "hf_vis"
FORMS = ("fp32", "fp16", "int8dyn")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VT = _load("vision_taps")
OPT = _load("op_table")


class FlagAdapter(torch.nn.Module):
    def __init__(self, ada):
        super().__init__()
        self.ada = ada
        tail = torch.zeros(1, VT.NTOK, 4)
        tail[..., 0] = 1.0
        self.register_buffer("tail", tail, persistent=False)

    def forward(self, enc):
        return torch.cat([self.ada(enc), self.tail], dim=-1)            # [1, 256, 8196]


class VisionTapsBarrier(VT.VisionTaps):
    """VisionTaps with northmv's fold barrier (northmv_work/convert_northmv_vision.py `_barrier`): a clamp to the fp16
    range between each LayerNorm and the FC that follows it (norm1 -> qkv, norm2 -> fc1). Semantically neutral (the
    LN outputs are far inside +-65504); it stops the converter from folding the LN gamma into the FC weights, whose
    per-output-channel int8 range the large-gamma columns then spend (northmv: int8 encoder corr 0.91-0.97 folded vs
    0.998 unfolded). Built from a calibrated VisionTaps by swapping the class (same parameters and LN scales)."""

    def forward(self, images):
        x = (images.permute(0, 3, 1, 2) - 0.5) / 0.5
        h = self.conv(x).permute(0, 2, 3, 1).reshape(1, VT.N, VT.HID) + self.pos
        taps = {}
        for i, blk in enumerate(self.blocks):
            n1 = VT.ln(h, blk.norm1.weight, blk.norm1.bias, 1e-6, self.ln_scale.get(f"b{i}.n1", 1.0))
            h = h + self.attn(blk, torch.clamp(n1, -65504.0, 65504.0))
            n2 = VT.ln(h, blk.norm2.weight, blk.norm2.bias, 1e-6, self.ln_scale.get(f"b{i}.n2", 1.0))
            h = h + blk.fc2(torch.nn.functional.gelu(blk.fc1(torch.clamp(n2, -65504.0, 65504.0)), approximate="tanh"))
            if i in VT.TAPS:
                taps[i] = h
        return torch.cat([taps[23], taps[5], taps[11], taps[17]], dim=-1)


def mode_barrier():
    """One try (supervisor 20:3x): the barrier encoder fp32 + its int8 dynamic form, parity on the 17 images."""
    import litert_torch
    from ai_edge_quantizer import quantizer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    enc, _, _ = VT.build()
    enc.__class__ = VisionTapsBarrier
    ref0 = np.load(VT.OUT / f"ref_{VT.CALIB_ROWS[0]}.npz")
    x0 = torch.from_numpy(ref0["image"]).float()[None] / 255.0
    rec = {}
    p32, p8 = OUT / "vision_encoder_barrier_fp32.tflite", OUT / "vision_encoder_barrier_int8dyn.tflite"
    if not p32.exists():
        t0 = time.time()
        litert_torch.convert(enc.eval(), (x0,)).export(str(p32))
        rec["convert_s"] = round(time.time() - t0, 1)
    if not p8.exists():
        qt = quantizer.Quantizer(str(p32), make_recipe("int8dyn"))
        qt.quantize().export_model(str(p8))
    rec["ops"] = {"fp32": OPT.op_table(p32), "int8dyn": OPT.op_table(p8)}
    rec["bytes"] = {"fp32": p32.stat().st_size, "int8dyn": p8.stat().st_size}
    from PIL import Image
    rows = json.loads((K / "fixtures" / "fixtures.json").read_text())["rows"]
    worst = {}
    rec["images"] = {}
    for s in sorted({Path(r["image"]).stem for r in rows}):
        with Image.open(PNG / f"{s}.png") as im:
            u8 = np.asarray(im.convert("RGB"))
        ref = np.load(HFVIS / f"{s}.npz")
        x = (u8.astype(np.float32) * np.float32(1.0 / 255.0))[None]
        rec["images"][s] = {}
        for name, enc_path, ada in (("barrier_fp32+fp32", p32, "fp32"), ("barrier_int8dyn+fp16", p8, "fp16"),
                                    ("barrier_int8dyn+int8dyn", p8, "int8dyn")):
            e, _ = run(enc_path, x)
            wide, _ = run(path_of("adapter", ada), e)
            m = metrics(wide[0][:, :8192], ref["wide"])
            rec["images"][s][name] = m
            w = worst.setdefault(name, {"cos_min": 1.0, "corr_min": 1.0})
            w["cos_min"] = min(w["cos_min"], *(m[k]["cos_min"] for k in m))
            w["corr_min"] = min(w["corr_min"], *(m[k]["corr"] for k in m))
        print(s, {k: round(v["image_embeds"]["cos_min"], 6) for k, v in rec["images"][s].items()}, flush=True)
    rec["worst"] = worst
    (K / "r2" / "vision_barrier.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("VISION_BARRIER", json.dumps(worst), flush=True)


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
           "ln_scales_gt1": {k: v for k, v in sorted(scale.items()) if v > 1}}
    t0 = time.time()
    litert_torch.convert(enc, (x0,)).export(str(path_of("encoder", "fp32")))
    rec["enc_convert_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    litert_torch.convert(fada, (e0,)).export(str(path_of("adapter", "fp32")))
    rec["ada_convert_s"] = round(time.time() - t0, 1)
    r1_enc = VT.OUT / "vision_encoder.tflite"
    rec["encoder_sha256"] = sha256_file(path_of("encoder", "fp32"))
    rec["r1_encoder_sha256"] = sha256_file(r1_enc) if r1_enc.exists() else None
    rec["encoder_equals_r1"] = rec["encoder_sha256"] == rec["r1_encoder_sha256"]
    rec["ops"] = {p: OPT.op_table(path_of(p, "fp32")) for p in ("encoder", "adapter")}
    (K / "r2").mkdir(exist_ok=True)
    (K / "r2" / "vision_export.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("VISION_EXPORT", json.dumps({k: rec[k] for k in ("enc_convert_s", "ada_convert_s", "encoder_equals_r1")}),
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
        for kind in ("fp16", "int8dyn"):
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
    (K / "r2" / "vision_quant.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")


def run(path, x, it_cache={}):
    from ai_edge_litert.interpreter import Interpreter
    key = str(path)
    if key not in it_cache:
        it = Interpreter(model_path=key, num_threads=int(os.environ.get("OMP_NUM_THREADS", "1")))
        it.allocate_tensors()
        it_cache[key] = it
    it = it_cache[key]
    d = it.get_input_details()[0]
    it.set_tensor(d["index"], np.ascontiguousarray(x, dtype=d["dtype"]))
    t0 = time.perf_counter()
    it.invoke()
    dt = time.perf_counter() - t0
    return it.get_tensor(it.get_output_details()[0]["index"]).copy(), dt


def token_cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def metrics(test, ref):
    out = {}
    for j, name in enumerate(("image_embeds", "deepstack0", "deepstack1", "deepstack2")):
        a, b = test[:, j * 2048:(j + 1) * 2048], ref[:, j * 2048:(j + 1) * 2048]
        c = token_cos(a, b)
        out[name] = {"cos_min": float(c.min()), "cos_mean": float(c.mean()),
                     "corr": float(np.corrcoef(a.ravel().astype(np.float64), b.ravel().astype(np.float64))[0, 1]),
                     "max_abs": float(np.abs(a - b).max()), "ref_absmax": float(np.abs(b).max())}
    return out


def mode_parity():
    from PIL import Image
    rows = json.loads((K / "fixtures" / "fixtures.json").read_text())["rows"]
    stems = sorted({Path(r["image"]).stem for r in rows})
    res = {"input": "PNG u8 * float32(1/255) (the runtime's rescale, stb_image_preprocessor.cc)", "images": {},
           "forms": {}}
    combos = [(e, a) for e in FORMS for a in FORMS]
    worst = {f"{e}+{a}": {"cos_min": 1.0, "corr_min": 1.0} for e, a in combos}
    for s in stems:
        with Image.open(PNG / f"{s}.png") as im:
            u8 = np.asarray(im.convert("RGB"))
        ref = np.load(HFVIS / f"{s}.npz")
        assert np.array_equal(ref["image"], u8), s
        x = (u8.astype(np.float32) * np.float32(1.0 / 255.0))[None]
        enc_out = {}
        res["images"][s] = {}
        for e in FORMS:
            enc_out[e], dt_e = run(path_of("encoder", e), x)
            for a in FORMS:
                wide, dt_a = run(path_of("adapter", a), enc_out[e])
                wide = wide[0]
                m = metrics(wide[:, :8192], ref["wide"])
                m["flag_ok"] = bool(np.all(wide[:, 8192] == 1.0))
                m["tail_zero"] = bool(np.all(wide[:, 8193:] == 0.0))
                m["enc_s"], m["ada_s"] = round(dt_e, 3), round(dt_a, 3)
                res["images"][s][f"{e}+{a}"] = m
                w = worst[f"{e}+{a}"]
                w["cos_min"] = min(w["cos_min"], *(m[k]["cos_min"] for k in m if isinstance(m[k], dict)))
                w["corr_min"] = min(w["corr_min"], *(m[k]["corr"] for k in m if isinstance(m[k], dict)))
                w["flag_ok"] = w.get("flag_ok", True) and m["flag_ok"] and m["tail_zero"]
        print(s, {k: round(v["image_embeds"]["cos_min"], 6) for k, v in res["images"][s].items()}, flush=True)
    res["worst"] = worst
    res["line1_fp32"] = {"cos_min": worst["fp32+fp32"]["cos_min"], "corr_min": worst["fp32+fp32"]["corr_min"],
                         "pass_cos": worst["fp32+fp32"]["cos_min"] >= 0.9999,
                         "pass_corr": worst["fp32+fp32"]["corr_min"] >= 0.99999}
    (K / "r2" / "vision_parity.json").write_text(json.dumps(res, indent=1) + "\n")
    print("VISION_PARITY", json.dumps(res["line1_fp32"]), json.dumps(worst), flush=True)


if __name__ == "__main__":
    {"export": mode_export, "quant": mode_quant, "parity": mode_parity, "barrier": mode_barrier}[sys.argv[1]]()
