"""Round 6: checks of the d1-omni vision prefix path against the provider and the oracle (round 2's ref/records_ref.json
version 2, image records = float resize path; ref/npz/<id>.npz `pixel_values`, `spatial_shapes`,
`pixel_attention_mask`, `pos_resized`, `tower_last_hidden_state`, `prefix`).

    cd d1_omni_work; Q=~/code/standup/tools/quiet
    HF_MODULES_CACHE=cache/hf_modules_r6 venv-ref/bin/python scripts/vision_check.py --host
        -> results/vision_host_check.json  (host/d1_vision_host.py vs the provider's preprocess() on the float path,
           the oracle's NaNFlex arrays, transformers' resize_positional_embeddings, the Projector's unshuffle)
    ~/venvs/lt094dev/bin/python scripts/vision_check.py --host-bits      (the same host bits from the exporter venv)
    HF_MODULES_CACHE=cache/hf_modules_r6 $Q/quiet_wait.py -- venv-ref/bin/python scripts/vision_check.py --eager
        -> results/vision_eager_check.json + out/r6_runs/eager/<id>.npy (our prefix per record)

Sections added by later steps (same file): --lrt (LiteRT CompiledModel CPU / Metal), --e2e (the decision graph on the
prefixes), --ms (Mac timing). The provider's files are never edited: its vision.py is imported as a plain module from
the pinned snapshot (no relative import), torchvision's resize is replaced in this process by round 2's float-path
copy (scripts/ref_probs.py resize_tv024_float) exactly as the oracle v2 run did.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
K = HERE.parent
_mc = os.environ.get("HF_MODULES_CACHE", "cache/hf_modules_r6")
os.environ["HF_MODULES_CACHE"] = str(_mc if os.path.isabs(_mc) else (K / _mc).resolve())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(K / "host"))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import d1_src as S  # noqa: E402
import d1_vision_host as V  # noqa: E402

ORACLE = K / "ref/records_ref.json"
NPZ = K / "ref/npz"
RUNS = K / "out/r6_runs"
IMAGE_IDS = ("card_cats", "img_dogs_01", "img_cat_02", "img_bike_03", "img_01", "img_02", "img_03")
THREADS_ORACLE = 12


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, doc):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def bits_equal(a, b) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def diff(a, b) -> dict:
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a64 - b64)
    ref_max = float(np.abs(b64).max()) if b64.size else 0.0
    ref_rms = float(np.sqrt((b64 ** 2).mean())) if b64.size else 0.0
    return {"max_abs": float(d.max()) if d.size else 0.0, "ref_absmax": ref_max,
            "rel_max_over_absmax": float(d.max() / ref_max) if ref_max else None,
            "rel_rms": float(np.sqrt((d ** 2).mean()) / ref_rms) if ref_rms else None,
            "bit_equal": bool(a64.shape == b64.shape and np.array_equal(np.asarray(a), np.asarray(b))),
            "nonfinite": int((~np.isfinite(np.asarray(a, np.float64))).sum())}


def image_records():
    """-> [(oracle entry, fixture record)] for the 7 image records (oracle order)."""
    oracle = json.loads(ORACLE.read_text())
    assert oracle.get("version") == 2 and oracle["load"]["resize_path"]["path"] == "float", "oracle v2 expected"
    fx = {r["id"]: r for r in json.loads((K / "fixtures/requests.json").read_text())["records"]}
    out = [(e, fx[e["id"]]) for e in oracle["records"] if e["mode"] == "image"]
    assert tuple(e["id"] for e, _ in out) == IMAGE_IDS, [e["id"] for e, _ in out]
    return out, oracle


def image_path(rec) -> Path:
    p = K / rec["media"]["ref"]
    assert S.sha256_file(p) == rec["media"]["sha256"], p
    return p


def npz(rid):
    with np.load(NPZ / f"{rid}.npz") as z:
        return {k: np.asarray(z[k]) for k in ("pixel_values", "spatial_shapes", "pixel_attention_mask", "pos_resized",
                                              "tower_last_hidden_state", "prefix")}


# --------------------------------------------------------------------------- provider side (venv-ref)


def provider_vision_module():
    return S.load_module("d1_provider_vision", S.SNAP / "vision.py")


def float_resize_patch():
    """Replace torchvision's v2 resize in this process by round 2's float-path copy (as the oracle v2 run did);
    -> (restore function, list of calls (input uint8 CHW, size, output uint8, info))."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.v2.functional as tvf

    import ref_probs as RP

    orig = tvf.resize
    calls = []

    def resize(inpt, size, interpolation=tvf.InterpolationMode.BILINEAR, max_size=None, antialias=True):
        ok = (isinstance(inpt, torch.Tensor) and inpt.dtype == torch.uint8 and inpt.device.type == "cpu"
              and interpolation == tvf.InterpolationMode.BILINEAR and antialias is True and max_size is None
              and len(size) == 2)
        assert ok, (type(inpt), getattr(inpt, "dtype", None), interpolation, antialias, max_size, size)
        out, info = RP.resize_tv024_float(torch, F, inpt, list(size))
        calls.append((inpt.detach().clone(), list(size), out.detach().clone(), info))
        return out

    tvf.resize = resize

    def restore():
        tvf.resize = orig

    return restore, calls


def torch_float_stage(img_chw_u8, size):
    """The float32 interpolate output before the rounding, the same steps as resize_tv024_float."""
    import torch
    import torch.nn.functional as F

    image = img_chw_u8
    shape, numel = image.shape, image.numel()
    c, oh, ow = shape[-3:]
    image = image.reshape(-1, c, oh, ow)
    strides = image.stride()
    if image.is_contiguous(memory_format=torch.channels_last) and image.shape[0] == 1 and numel != strides[0]:
        new_strides = list(strides)
        new_strides[0] = numel
        image = image.as_strided((1, c, oh, ow), new_strides)
    image = image.to(dtype=torch.float32)
    return F.interpolate(image, size=list(size), mode="bilinear", align_corners=False, antialias=True)[0]


def host_float_stage(arr_hwc_u8, out_h, out_w):
    h, w = arr_hwc_u8.shape[:2]
    x = arr_hwc_u8.astype(np.float32)
    if out_w != w:
        x = V._resample_axis_f32(x, 1, out_w)
    if out_h != h:
        x = V._resample_axis_f32(x, 0, out_h)
    return x


def host_check(a):
    import torch
    from transformers.image_utils import load_image
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings

    import torchvision

    torch.set_num_threads(THREADS_ORACLE)
    t0 = time.time()
    vmod = provider_vision_module()
    recs, oracle = image_records()
    table = V.read_position_table(S.WEIGHTS)
    from safetensors import safe_open

    with safe_open(str(S.WEIGHTS), framework="pt") as f:
        table_t = f.get_tensor(V.POSITION_KEY)
    table_ok = bits_equal(table, table_t.numpy().reshape(16, 16, 768))
    restore, calls = float_resize_patch()
    per, resize_rows, pos_rows, unshuffle_rows, bits = [], [], [], [], {}
    try:
        for e, rec in recs:
            rid = e["id"]
            path = image_path(rec)
            img_tf = load_image(str(path))
            img_host = V.load_image(path)
            same_image = bits_equal(np.asarray(img_tf), np.asarray(img_host))
            n0 = len(calls)
            prov = vmod.preprocess(img_tf)
            mycalls = calls[n0:]
            host = V.preprocess(img_host)
            z = npz(rid)
            pv_p, ss_p, m_p = (prov["pixel_values"].numpy(), prov["spatial_shapes"].numpy(),
                               prov["pixel_attention_mask"].numpy())
            row = {"id": rid, "file": rec["media"]["ref"], "px_wh": list(img_host.size),
                   "load_image_equal_host_reader": same_image,
                   "plan_host": {k: list(v) if isinstance(v, tuple) else v for k, v in host["plan"].items()},
                   "plan_provider": {k: list(v) if isinstance(v, tuple) else v
                                     for k, v in vmod.layout(*img_tf.size).items()},
                   "crops": int(pv_p.shape[0]), "spatial_shapes": ss_p.tolist(),
                   "pixel_values_bit_equal_provider": bits_equal(host["pixel_values"], pv_p),
                   "spatial_shapes_equal_provider": bool(np.array_equal(host["spatial_shapes"], ss_p))
                   and host["spatial_shapes"].dtype == ss_p.dtype,
                   "mask_equal_provider": bits_equal(host["pixel_attention_mask"], m_p),
                   "pixel_values_bit_equal_oracle_npz": bits_equal(host["pixel_values"], z["pixel_values"]),
                   "spatial_shapes_equal_oracle_npz": bool(np.array_equal(host["spatial_shapes"], z["spatial_shapes"])),
                   "mask_equal_oracle_npz": bits_equal(host["pixel_attention_mask"], z["pixel_attention_mask"]),
                   "provider_bit_equal_oracle_npz": bits_equal(pv_p, z["pixel_values"]),
                   "prefix_length_host": V.prefix_length(host["spatial_shapes"]),
                   "prefix_length_provider": int(vmod.prefix_length(prov)), "prefix_oracle": int(e["prefix"]),
                   "resize_calls": len(mycalls)}
            row["plan_equal"] = (row["plan_host"]["grid"] == row["plan_provider"]["grid"]
                                 and row["plan_host"]["thumbnail"] == row["plan_provider"]["thumbnail"]
                                 and row["plan_host"]["tiled"] == row["plan_provider"]["tiled"])
            if not row["pixel_values_bit_equal_provider"]:
                d = np.abs(host["pixel_values"].astype(np.float64) - pv_p)
                where = np.argwhere(d > 0)
                row["mismatch"] = {"elements": int(len(where)), "max_abs": float(d.max()),
                                   "crops": sorted({int(w[0]) for w in where}),
                                   "patches_first": [list(map(int, w[:2])) for w in where[:10]]}
            # each resize call: host vs the float-path copy (uint8 bits, and the float32 stage before rounding)
            for (inp, size, out, info) in mycalls:
                arr = inp.permute(1, 2, 0).contiguous().numpy()
                host_out = V.resize_float(arr, size[0], size[1])
                rr = {"id": rid, "in_hw": list(arr.shape[:2]), "out_hw": size, "identity": info["identity"],
                      "uint8_bit_equal": bits_equal(host_out, out.permute(1, 2, 0).contiguous().numpy())}
                if not info["identity"]:
                    tf = torch_float_stage(inp, size).permute(1, 2, 0).contiguous().numpy()
                    hf = host_float_stage(arr, size[0], size[1])
                    rr["float_stage_bit_equal"] = bits_equal(hf, tf)
                    rr["float_stage_max_abs"] = float(np.abs(hf.astype(np.float64) - tf).max())
                    rr["float_stage_elements_differing"] = int((hf != tf).sum())
                    frac = np.abs(tf - np.floor(tf) - 0.5)
                    rr["float_values_within_1e-4_of_half"] = int((frac < 1e-4).sum())
                resize_rows.append(rr)
            # position table per crop: host vs transformers' resize_positional_embeddings vs the oracle's tap
            for i, (h, w) in enumerate(ss_p.tolist()):
                ours = V.positions_padded(table, (h, w))
                theirs = Siglip2VisionEmbeddings.resize_positional_embeddings(
                    table_t.reshape(16, 16, 768), torch.tensor([[h, w]]), max_length=1024)[0].numpy()
                pos_rows.append({"id": rid, "crop": i, "grid": [h, w], "bit_equal_transformers": bits_equal(ours, theirs),
                                 "bit_equal_oracle_tap": bits_equal(ours, z["pos_resized"][i]),
                                 "max_abs_transformers": float(np.abs(ours.astype(np.float64) - theirs).max())})
                # unshuffle: host vs the provider's Projector reshape / permute on the oracle's features
                feat = z["tower_last_hidden_state"][i, : h * w]
                x = torch.from_numpy(feat.copy()).reshape(1, h, w, 768)
                b, hh, ww, c = x.shape
                x = x.reshape(b, hh, ww // 2, c * 2).permute(0, 2, 1, 3)
                x = x.reshape(b, ww // 2, hh // 2, c * 4).permute(0, 2, 1, 3)
                cells_p = x.reshape(-1, c * 4).contiguous().numpy()
                cells_h = V.pixel_unshuffle(feat, (h, w))
                unshuffle_rows.append({"id": rid, "crop": i, "grid": [h, w], "cells": int(cells_h.shape[0]),
                                       "bit_equal_provider": bits_equal(cells_h, cells_p)})
            bits[rid] = {"pixel_values": sha(host["pixel_values"]), "spatial_shapes": host["spatial_shapes"].tolist(),
                         "mask": sha(host["pixel_attention_mask"]),
                         "pos": [sha(V.positions_padded(table, tuple(g))) for g in host["spatial_shapes"].tolist()]}
            per.append(row)
            print(rid, row["pixel_values_bit_equal_provider"], row["pixel_values_bit_equal_oracle_npz"],
                  row["prefix_length_host"], flush=True)
    finally:
        restore()

    # random sizes: host resize vs the float-path copy on uint8 noise
    import torch.nn.functional as F

    import ref_probs as RP

    g = np.random.default_rng(11)
    rand_rows = []
    for k in range(a.random_resizes):
        H, W = int(g.integers(8, 1300)), int(g.integers(8, 1300))
        mode = k % 4
        oh = int(g.integers(1, 49)) * 32 if mode != 1 else H
        ow = int(g.integers(1, 49)) * 32 if mode != 2 else W
        arr = g.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        out, info = RP.resize_tv024_float(torch, F, t, [oh, ow])
        ho = V.resize_float(arr, oh, ow)
        rand_rows.append({"in_hw": [H, W], "out_hw": [oh, ow],
                          "bit_equal": bits_equal(ho, out.permute(1, 2, 0).contiguous().numpy())})
    # position tables: every even grid with h * w <= 1024 and both sides <= 64, plus long thin grids
    grids = sorted({(h, w) for h in range(2, 65, 2) for w in range(2, 65, 2) if h * w <= 1024}
                   | {(2, 188), (188, 2), (2, 512), (512, 2), (4, 256), (256, 4), (8, 128), (128, 8)})
    pos_sweep_bad = []
    for (h, w) in grids:
        ours = V.positions_padded(table, (h, w))
        theirs = Siglip2VisionEmbeddings.resize_positional_embeddings(
            table_t.reshape(16, 16, 768), torch.tensor([[h, w]]), max_length=max(1024, h * w))[0].numpy()[:1024]
        if not bits_equal(ours, theirs):
            pos_sweep_bad.append([h, w])
    # layout(): the host's vs the provider's on a sweep of sizes
    sizes = {(w, h) for w in (1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 100, 216, 255, 256, 257, 384, 511, 512, 513, 640,
                              700, 768, 1000, 1023, 1024, 1025, 1280, 1500, 2048, 3000, 4000)
             for h in (1, 2, 16, 20, 32, 60, 216, 288, 320, 384, 480, 512, 600, 853, 1024, 1280, 1600, 2289, 3000)}
    sizes |= {(int(g.integers(1, 5000)), int(g.integers(1, 5000))) for _ in range(3000)}
    layout_bad = [list(s) for s in sorted(sizes) if V.layout(*s) != vmod.layout(*s)]

    summary = {
        "records": len(per), "crops": sum(r["crops"] for r in per),
        "pixel_values_bit_equal_provider": sum(r["pixel_values_bit_equal_provider"] for r in per),
        "spatial_shapes_equal_provider": sum(r["spatial_shapes_equal_provider"] for r in per),
        "mask_equal_provider": sum(r["mask_equal_provider"] for r in per),
        "pixel_values_bit_equal_oracle_npz": sum(r["pixel_values_bit_equal_oracle_npz"] for r in per),
        "spatial_shapes_equal_oracle_npz": sum(r["spatial_shapes_equal_oracle_npz"] for r in per),
        "mask_equal_oracle_npz": sum(r["mask_equal_oracle_npz"] for r in per),
        "provider_float_path_bit_equal_oracle_npz": sum(r["provider_bit_equal_oracle_npz"] for r in per),
        "prefix_length_all_equal": all(r["prefix_length_host"] == r["prefix_length_provider"] == r["prefix_oracle"]
                                       for r in per),
        "plan_all_equal": all(r["plan_equal"] for r in per),
        "load_image_all_equal": all(r["load_image_equal_host_reader"] for r in per),
        "resize_calls": len(resize_rows),
        "resize_uint8_bit_equal": sum(r["uint8_bit_equal"] for r in resize_rows),
        "resize_float_stage_bit_equal": sum(r.get("float_stage_bit_equal", True) for r in resize_rows),
        "random_resizes": len(rand_rows), "random_resizes_bit_equal": sum(r["bit_equal"] for r in rand_rows),
        "position_crops": len(pos_rows),
        "position_bit_equal_transformers": sum(r["bit_equal_transformers"] for r in pos_rows),
        "position_bit_equal_oracle_tap": sum(r["bit_equal_oracle_tap"] for r in pos_rows),
        "position_sweep_grids": len(grids), "position_sweep_not_bit_equal": pos_sweep_bad,
        "unshuffle_crops": len(unshuffle_rows),
        "unshuffle_bit_equal_provider": sum(r["bit_equal_provider"] for r in unshuffle_rows),
        "layout_sizes": len(sizes), "layout_differs": layout_bad[:20], "layout_differs_count": len(layout_bad),
        "position_table_reader_bit_equal_safetensors": table_ok,
    }
    n_c = summary["crops"]
    summary["pass"] = bool(
        summary["pixel_values_bit_equal_provider"] == summary["spatial_shapes_equal_provider"]
        == summary["mask_equal_provider"] == summary["pixel_values_bit_equal_oracle_npz"] == len(per)
        and summary["resize_uint8_bit_equal"] == summary["resize_calls"]
        and summary["random_resizes_bit_equal"] == summary["random_resizes"]
        and summary["position_bit_equal_transformers"] == summary["position_bit_equal_oracle_tap"] == n_c
        and not pos_sweep_bad and summary["unshuffle_bit_equal_provider"] == n_c and not layout_bad
        and summary["prefix_length_all_equal"] and summary["plan_all_equal"] and table_ok)
    doc = {"step": "round 6 step 1: host/d1_vision_host.py vs the provider's preprocess() (float resize path)",
           "written": now(), "summary": summary,
           "venv": {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
                    "numpy": np.__version__, "threads": torch.get_num_threads(),
                    "cpu_capability": torch.backends.cpu.get_cpu_capability()},
           "oracle": {"file": str(ORACLE.relative_to(K)), "version": oracle["version"],
                      "sha256": S.sha256_file(ORACLE), "resize_path": oracle["load"]["resize_path"]["path"]},
           "provider_vision_py_sha256": S.sha256_file(S.SNAP / "vision.py"),
           "records": per, "resize_calls": resize_rows, "random_resizes": rand_rows, "positions": pos_rows,
           "unshuffle": unshuffle_rows, "host_bits": {"venv-ref": bits}, "seconds": round(time.time() - t0, 1)}
    write_json(K / "results/vision_host_check.json", doc)
    print(json.dumps(summary, indent=1))
    return 0 if summary["pass"] else 1


def host_bits(a):
    """The host's outputs recomputed in this venv (numpy / Pillow of the exporter venv) vs the venv-ref bits."""
    path = K / "results/vision_host_check.json"
    doc = json.loads(path.read_text())
    ref = doc["host_bits"]["venv-ref"]
    recs, _ = image_records()
    table = V.read_position_table(S.WEIGHTS)
    mine, equal = {}, {}
    for e, rec in recs:
        rid = e["id"]
        host = V.preprocess(V.load_image(image_path(rec)))
        mine[rid] = {"pixel_values": sha(host["pixel_values"]), "spatial_shapes": host["spatial_shapes"].tolist(),
                     "mask": sha(host["pixel_attention_mask"]),
                     "pos": [sha(V.positions_padded(table, tuple(g))) for g in host["spatial_shapes"].tolist()]}
        equal[rid] = mine[rid] == ref[rid]
    import PIL

    key = f"python{sys.version.split()[0]}_numpy{np.__version__}_pillow{PIL.__version__}"
    doc["host_bits"][key] = mine
    doc["summary"]["host_bits_equal_across_venvs"] = {key: sum(equal.values())}
    write_json(path, doc)
    print(json.dumps({"venv": key, "records_equal": equal}, indent=1))
    return 0 if all(equal.values()) else 1


# --------------------------------------------------------------------------- eager (venv-ref)


def provider_vision(dtype=None):
    """The provider's Vision (vision.py) with the checkpoint's 201 vision tensors (transformers' rename
    vision.tower.vision_model.* -> tower.*), strict; float32 eval."""
    import torch
    from safetensors import safe_open

    vmod = provider_vision_module()
    cfg = S.config()
    vis = vmod.Vision(cfg["vision_config"], cfg["projector_hidden_size"], cfg["text_config"]["hidden_size"])
    sd = {}
    with safe_open(str(S.WEIGHTS), framework="pt") as f:
        for k in f.keys():
            if k.startswith("vision.tower.vision_model."):
                sd["tower." + k[len("vision.tower.vision_model."):]] = f.get_tensor(k)
            elif k.startswith("vision.projector."):
                sd["projector." + k[len("vision.projector."):]] = f.get_tensor(k)
    res = vis.load_state_dict(sd, strict=True)
    vis = vis.float().eval().requires_grad_(False)
    return vmod, vis, {"tensors_loaded": len(sd), "missing": list(res.missing_keys),
                       "unexpected": list(res.unexpected_keys),
                       "attn_implementation": vis.tower.config._attn_implementation}


def tower_stats(tower, x_in, mask):
    """Per layer, over the real patches: residual absmax, the layer-norm inputs' max mean square of (x - mean) and max
    |x - mean|, the attention scores' absmax (scaled), fc1's absmax (before the gelu); fp32."""
    import torch
    import torch.nn.functional as F

    real = mask[0] > 0.5
    x = tower.patch_embedding(x_in["pixels"]) + x_in["pos"]
    bias = (1.0 - mask)[:, None, None, :] * -1e4
    rows = []

    def ln_stats(t):
        t = t[0][real]
        dev = t - t.mean(-1, keepdim=True)
        return float((dev ** 2).mean(-1).max()), float(dev.abs().max())

    for i, layer in enumerate(tower.layers):
        v1, d1 = ln_stats(x)
        h = layer.layer_norm1(x)
        b, n, _ = h.shape
        q = layer.q_proj(h).view(b, n, 12, 64).transpose(1, 2)
        k = layer.k_proj(h).view(b, n, 12, 64).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(2, 3)) * layer.scale
        s_abs = float(sc[:, :, real][:, :, :, real].abs().max())
        x = x + layer.attention(h, bias)
        v2, d2 = ln_stats(x)
        f1 = layer.fc1(layer.layer_norm2(x))
        f1_abs = float(f1[0][real].abs().max())
        x = x + layer.fc2(F.gelu(f1, approximate="tanh"))
        rows.append({"layer": i, "ln1_max_var": v1, "ln1_max_absdev": d1, "scores_absmax": s_abs, "ln2_max_var": v2,
                     "ln2_max_absdev": d2, "fc1_absmax": f1_abs, "residual_absmax_out": float(x[0][real].abs().max())})
    vp, dp = ln_stats(x)
    return rows, {"post_ln_max_var": vp, "post_ln_max_absdev": dp}


def eager_check(a):
    import torch

    import vision_graph as VG

    torch.set_num_threads(THREADS_ORACLE)
    t0 = time.time()
    vmod, vis, prov_load = provider_vision()
    tower, proj, table, rep = VG.load_vision()
    recs, oracle = image_records()
    (RUNS / "eager").mkdir(parents=True, exist_ok=True)
    restore, calls = float_resize_patch()
    from transformers.image_utils import load_image

    per, stats_all, pad_rows = [], {}, []
    try:
        with torch.no_grad():
            for e, rec in recs:
                rid = e["id"]
                path = image_path(rec)
                img = load_image(str(path))
                z = npz(rid)
                # provider: the oracle's call (all crops in one batch), then each crop alone
                inputs = vmod.preprocess(img)
                inputs = {k: (v.to(torch.float32) if k == "pixel_values" else v) for k, v in inputs.items()}
                hid_b = vis.tower(**inputs).last_hidden_state.numpy()
                pre_b = vis([img]).numpy()[0]
                shapes = inputs["spatial_shapes"].tolist()
                host = V.preprocess(V.load_image(path))
                assert bits_equal(host["pixel_values"], inputs["pixel_values"].numpy())
                crops, mine_pre, single_pre = [], [], []
                for i, (h, w) in enumerate(shapes):
                    n = h * w
                    one = {k: v[i:i + 1] for k, v in inputs.items()}
                    hid_s = vis.tower(**one).last_hidden_state.numpy()
                    p_single = vis.projector(torch.from_numpy(hid_s[:, :n].copy()).reshape(1, h, w, -1))[0].numpy()
                    p_batch = vis.projector(torch.from_numpy(hid_b[i:i + 1, :n].copy()).reshape(1, h, w, -1))[0].numpy()
                    crop = {"pixels": host["pixel_values"][i], "mask": host["pixel_attention_mask"][i], "grid": (h, w)}
                    x_in = {k: torch.from_numpy(v) for k, v in V.tower_inputs(crop, table).items()}
                    feat = tower(**x_in)["features"][0].numpy()
                    cells = V.pixel_unshuffle(feat[:n], (h, w))
                    mine = proj(soft=torch.from_numpy(V.projector_input(cells)))["prefix"][0].numpy()[: cells.shape[0]]
                    # (ii) unshuffle + our projector on the provider's own features (batch call)
                    cells_b = V.pixel_unshuffle(hid_b[i, :n], (h, w))
                    ours_on_theirs = proj(soft=torch.from_numpy(V.projector_input(cells_b)))["prefix"][0].numpy()
                    ours_on_theirs = ours_on_theirs[: cells_b.shape[0]]
                    c = {"crop": i, "grid": [h, w], "patches": n, "cells": int(cells.shape[0]),
                         "provider_batch_vs_oracle_features_bit_equal": bits_equal(hid_b[i], z["tower_last_hidden_state"][i]),
                         "provider_single_vs_batch_features": diff(hid_s[0, :n], hid_b[i, :n]),
                         "provider_single_vs_batch_prefix": diff(p_single, p_batch),
                         "features_vs_provider_batch": diff(feat[:n], hid_b[i, :n]),
                         "features_vs_provider_single": diff(feat[:n], hid_s[0, :n]),
                         "projector_ours_vs_provider_on_provider_features": diff(ours_on_theirs, p_batch)}
                    # (iv) pad patches: random pixels and positions -> real features bit-equal
                    if n < V.MAX_PATCHES:
                        gpad = np.random.default_rng(100 + i)
                        y = {k: v.clone() for k, v in x_in.items()}
                        y["pixels"][0, n:] = torch.from_numpy(gpad.normal(0, 3, (V.MAX_PATCHES - n, 768)).astype(np.float32))
                        y["pos"][0, n:] = torch.from_numpy(gpad.normal(0, 3, (V.MAX_PATCHES - n, 768)).astype(np.float32))
                        feat2 = tower(**y)["features"][0].numpy()
                        pad = {"id": rid, "crop": i, "pad_rows": V.MAX_PATCHES - n,
                               "real_bit_equal": bits_equal(feat2[:n], feat[:n]),
                               "pad_rows_changed": bool(not np.array_equal(feat2[n:], feat[n:]))}
                        c["pad_invariance"] = pad
                        pad_rows.append(pad)
                    st, post = tower_stats(tower, x_in, x_in["mask"])
                    stats_all[f"{rid}/{i}"] = {"layers": st, **post}
                    crops.append(c)
                    mine_pre.append(mine)
                    single_pre.append(p_single)
                mine_pre = np.concatenate(mine_pre)
                single_pre = np.concatenate(single_pre)
                np.save(RUNS / "eager" / f"{rid}.npy", mine_pre.astype(np.float32))
                row = {"id": rid, "P": int(mine_pre.shape[0]), "crops": crops,
                       "provider_rerun_prefix_bit_equal_oracle": bits_equal(pre_b, z["prefix"]),
                       "provider_single_crop_prefix_vs_oracle": diff(single_pre, z["prefix"]),
                       "prefix_vs_oracle": diff(mine_pre, z["prefix"]),
                       "prefix_vs_provider_single_crop": diff(mine_pre, single_pre)}
                per.append(row)
                print(rid, row["P"], "prefix max|d| %.3g rel %.3g" % (row["prefix_vs_oracle"]["max_abs"],
                                                                      row["prefix_vs_oracle"]["rel_max_over_absmax"]),
                      "provider single %.3g" % row["provider_single_crop_prefix_vs_oracle"]["max_abs"], flush=True)
            # (v) float64 identity on one crop (card_cats: 936 real patches + padding), both sides float64
            rid = "card_cats"
            rec = next(r for e, r in recs if e["id"] == rid)
            img = load_image(str(image_path(rec)))
            inputs = vmod.preprocess(img)
            vis64 = vis.double()
            t64, p64 = tower.double(), proj.double()
            hid64 = vis64.tower(pixel_values=inputs["pixel_values"].double(), spatial_shapes=inputs["spatial_shapes"],
                                pixel_attention_mask=inputs["pixel_attention_mask"]).last_hidden_state[0].numpy()
            (h, w) = inputs["spatial_shapes"][0].tolist()
            n = h * w
            pre64_p = vis64.projector(torch.from_numpy(hid64[:n].copy()).reshape(1, h, w, -1))[0].numpy()
            host = V.preprocess(V.load_image(image_path(rec)))
            crop = {"pixels": host["pixel_values"][0], "mask": host["pixel_attention_mask"][0], "grid": (h, w)}
            x64 = {k: torch.from_numpy(v.astype(np.float64)) for k, v in V.tower_inputs(crop, table).items()}
            f64 = t64(**x64)["features"][0].numpy()
            cells64 = V.pixel_unshuffle(f64[:n], (h, w))
            soft64 = np.zeros((1, V.PROJ_ROWS, cells64.shape[1]), np.float64)
            soft64[0, : cells64.shape[0]] = cells64          # float64 all the way (projector_input is float32)
            pre64_m = p64(soft=torch.from_numpy(soft64))["prefix"][0].numpy()[: n // 4]
            f64_doc = {"record": rid, "grid": [h, w], "features_max_abs": float(np.abs(f64[:n] - hid64[:n]).max()),
                       "prefix_max_abs": float(np.abs(pre64_m - pre64_p).max()),
                       "dtype_ours": str(f64.dtype), "dtype_provider": str(hid64.dtype)}
            vis.float(), tower.float(), proj.float()
    finally:
        restore()

    pm = [r["prefix_vs_oracle"] for r in per]
    summary = {
        "records": len(per), "crops": sum(len(r["crops"]) for r in per),
        "prefix_max_abs_vs_oracle": max(d["max_abs"] for d in pm),
        "prefix_rel_max_over_absmax": max(d["rel_max_over_absmax"] for d in pm),
        "prefix_rel_rms": max(d["rel_rms"] for d in pm),
        "features_max_abs_vs_provider_batch": max(c["features_vs_provider_batch"]["max_abs"] for r in per
                                                  for c in r["crops"]),
        "features_rel_rms_vs_provider_batch": max(c["features_vs_provider_batch"]["rel_rms"] for r in per
                                                  for c in r["crops"]),
        "projector_ours_vs_provider_max_abs": max(c["projector_ours_vs_provider_on_provider_features"]["max_abs"]
                                                  for r in per for c in r["crops"]),
        "provider_own_single_vs_batch_prefix_max_abs": max(r["provider_single_crop_prefix_vs_oracle"]["max_abs"]
                                                           for r in per),
        "provider_own_single_vs_batch_features_max_abs": max(c["provider_single_vs_batch_features"]["max_abs"]
                                                             for r in per for c in r["crops"]),
        "provider_rerun_bit_equal_oracle": sum(r["provider_rerun_prefix_bit_equal_oracle"] for r in per),
        "provider_batch_features_bit_equal_oracle": sum(c["provider_batch_vs_oracle_features_bit_equal"]
                                                        for r in per for c in r["crops"]),
        "pad_crops": len(pad_rows), "pad_real_bit_equal": sum(p["real_bit_equal"] for p in pad_rows),
        "float64": f64_doc, "keymap_missing": 0 if rep["tensors_vision"] == 201 else None,
        "bar": {"prefix_max_abs": 1e-4, "prefix_rel": "1e-5 class", "float64": 1e-12, "stop_if_prefix_max_abs_above": 1e-3},
    }
    summary["pass_float64"] = f64_doc["features_max_abs"] <= 1e-12 and f64_doc["prefix_max_abs"] <= 1e-12
    summary["pass_pad"] = summary["pad_real_bit_equal"] == summary["pad_crops"]
    summary["pass_prefix_1e-4"] = summary["prefix_max_abs_vs_oracle"] <= 1e-4
    summary["stop_condition_1e-3"] = summary["prefix_max_abs_vs_oracle"] > 1e-3
    doc = {"step": "round 6 step 2: our VisionTower / Projector (export form) vs the provider's Vision and the oracle",
           "written": now(), "summary": summary, "provider_load": prov_load, "our_load": rep,
           "venv": {"python": sys.version.split()[0], "torch": torch.__version__, "threads": torch.get_num_threads()},
           "records": per, "tower_activation_stats": stats_all, "seconds": round(time.time() - t0, 1)}
    write_json(K / "results/vision_eager_check.json", doc)
    print(json.dumps(summary, indent=1))
    return 0


def _layers_ours(tower, x_in, attn=None):
    """Our tower layer by layer -> ([per-layer output [1, N, 768]], features); attn = an optional replacement of the
    attention arithmetic (the SDPA variant of the diagnosis)."""
    import torch.nn.functional as F

    x = tower.patch_embedding(x_in["pixels"]) + x_in["pos"]
    bias = (1.0 - x_in["mask"])[:, None, None, :] * -1e4
    outs = []
    for layer in tower.layers:
        h = layer.layer_norm1(x)
        x = x + (layer.attention(h, bias) if attn is None else attn(layer, h, x_in["mask"]))
        x = x + layer.fc2(F.gelu(layer.fc1(layer.layer_norm2(x)), approximate="tanh"))
        outs.append(x)
    return outs, tower.post_layernorm(x)


def _sdpa_attention(layer, h, mask):
    """The same projections with torch's scaled_dot_product_attention and a boolean key mask (the provider's path)."""
    import torch
    import torch.nn.functional as F

    b, n, _ = h.shape
    q = layer.q_proj(h).view(b, n, 12, 64).transpose(1, 2)
    k = layer.k_proj(h).view(b, n, 12, 64).transpose(1, 2)
    v = layer.v_proj(h).view(b, n, 12, 64).transpose(1, 2)
    keep = (mask > 0.5)[:, None, None, :].expand(b, 1, n, n)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, scale=layer.scale)
    return layer.out_proj(y.transpose(1, 2).reshape(b, n, 768))


def eager_diag(a):
    """Why our fp32 prefix differs from the oracle's: the provider in float64 = the truth; per layer (real patches)
    max |d| of the provider fp32 (= the oracle) and of ours fp32 against it; ours with SDPA attention vs the provider
    fp32; and the decision graph in torch (D1Decision, fp32, checkpoint) on our eager prefix vs the oracle prefix
    -> the `fp32_floor` section of results/vision_eager_check.json."""
    import torch
    from transformers.image_utils import load_image

    import graph_build as B
    import d1_host as H
    import vision_graph as VG
    from types import SimpleNamespace

    torch.set_num_threads(THREADS_ORACLE)
    t0 = time.time()
    vmod, vis, _ = provider_vision()
    tower, proj, table, _ = VG.load_vision()
    recs, oracle = image_records()
    restore, calls = float_resize_patch()
    out = {}
    try:
        with torch.no_grad():
            for rid in a.diag_ids.split(","):
                rec = next(r for e, r in recs if e["id"] == rid)
                img = load_image(str(image_path(rec)))
                inputs = vmod.preprocess(img)
                assert inputs["pixel_values"].shape[0] == 1, "one-crop records only"
                (h, w) = inputs["spatial_shapes"][0].tolist()
                n = h * w

                def provider_run(dtype):
                    vis.to(dtype)
                    outs = []
                    hooks = [L.register_forward_hook(lambda m, ar, o: outs.append(o.detach().clone()))
                             for L in vis.tower.encoder.layers]
                    hid = vis.tower(pixel_values=inputs["pixel_values"].to(dtype),
                                    spatial_shapes=inputs["spatial_shapes"],
                                    pixel_attention_mask=inputs["pixel_attention_mask"]).last_hidden_state
                    for hk in hooks:
                        hk.remove()
                    pre = vis.projector(hid[:, :n].reshape(1, h, w, -1))[0]
                    vis.float()
                    return [o[0, :n].double() for o in outs], hid[0, :n].double(), pre.double()

                def ours_run(dtype, attn=None):
                    tower.to(dtype), proj.to(dtype)
                    host = V.preprocess(V.load_image(image_path(rec)))
                    crop = {"pixels": host["pixel_values"][0], "mask": host["pixel_attention_mask"][0], "grid": (h, w)}
                    x_in = {k: torch.from_numpy(v).to(dtype) for k, v in V.tower_inputs(crop, table).items()}
                    outs, feat = _layers_ours(tower, x_in, attn)
                    cells = V.pixel_unshuffle(feat[0, :n].numpy(), (h, w))
                    soft = np.zeros((1, V.PROJ_ROWS, cells.shape[1]), cells.dtype)
                    soft[0, : cells.shape[0]] = cells
                    pre = proj(soft=torch.from_numpy(soft))["prefix"][0, : n // 4]
                    tower.float(), proj.float()
                    return [o[0, :n].double() for o in outs], feat[0, :n].double(), pre.double()

                truth = provider_run(torch.float64)
                prov32 = provider_run(torch.float32)
                ours32 = ours_run(torch.float32)
                ours64 = ours_run(torch.float64)
                ours_sdpa = ours_run(torch.float32, _sdpa_attention)

                def cmp(x, y):
                    lay = [float((u - v).abs().max()) for u, v in zip(x[0], y[0])]
                    return {"per_layer_residual_max_abs": lay, "features_max_abs": float((x[1] - y[1]).abs().max()),
                            "prefix_max_abs": float((x[2] - y[2]).abs().max()),
                            "prefix_rel_rms": float(((x[2] - y[2]) ** 2).mean().sqrt() / (y[2] ** 2).mean().sqrt())}

                d_po = (ours32[2] - prov32[2]).abs()
                tok, ch = divmod(int(d_po.argmax()), d_po.shape[1])
                out[rid] = {"grid": [h, w],
                            "provider_fp32_vs_float64_truth": cmp(prov32, truth),
                            "ours_fp32_vs_float64_truth": cmp(ours32, truth),
                            "ours_fp32_vs_provider_fp32": cmp(ours32, prov32),
                            "ours_float64_vs_float64_truth": cmp(ours64, truth),
                            "ours_fp32_with_sdpa_vs_provider_fp32": cmp(ours_sdpa, prov32),
                            "worst_prefix_element_ours_vs_provider": {
                                "token": tok, "channel": ch, "ours": float(ours32[2][tok, ch]),
                                "provider": float(prov32[2][tok, ch]), "truth": float(truth[2][tok, ch]),
                                "token_row_absmax": float(truth[2][tok].abs().max())}}
                o = out[rid]
                print(rid, "prefix max|d| vs float64: provider %.3g ours %.3g | ours vs provider %.3g | sdpa variant %.3g"
                      % (o["provider_fp32_vs_float64_truth"]["prefix_max_abs"], o["ours_fp32_vs_float64_truth"]["prefix_max_abs"],
                         o["ours_fp32_vs_provider_fp32"]["prefix_max_abs"],
                         o["ours_fp32_with_sdpa_vs_provider_fp32"]["prefix_max_abs"]), flush=True)
    finally:
        restore()

    # the decision graph (torch fp32, checkpoint) on the 16 image rows: our eager prefix vs the oracle's prefix
    temps = S.config()["temperatures"]
    qt = {"choice": 0, "score": 1, "noul": 2}
    rows = []
    built = {}
    for e, rec in recs:
        z = npz(e["id"])
        ours = np.load(RUNS / "eager" / f"{e['id']}.npy")
        for q in e["questions"]:
            L = next(b for b in (256, 512, 1024, 2048, 4096) if q["positions"] <= b)
            if L not in built:
                built[L] = B.build(L, "checkpoint", S.config())[0]
            res = {}
            for name, pre in (("oracle_prefix", z["prefix"]), ("eager_prefix", ours)):
                x = H.build_inputs(q["ids"], pre, L)
                oh = np.zeros((1, 3), np.float32)
                oh[0, qt[q["type"]]] = 1.0
                x["qtype_onehot"] = oh
                with torch.no_grad():
                    sc = built[L](**{k: torch.from_numpy(v) for k, v in x.items()})["scores"].numpy().reshape(-1)
                pq = H.readout(sc, q["prefix"], q["markers"], SimpleNamespace(type=q["type"], options=q["K"]),
                               bool(q["calibrate"]), temps)
                res[name] = [float(v) for v in pq]
            rows.append({"key": f"{e['id']}/{q['qid']}", "L": L, "oracle": q["probs"], **res,
                         "max_abs_dp_eager_vs_oracle": max(abs(u - v) for u, v in zip(res["eager_prefix"], q["probs"])),
                         "max_abs_dp_oracleprefix_vs_oracle": max(abs(u - v) for u, v in zip(res["oracle_prefix"], q["probs"])),
                         "max_abs_dp_eager_vs_oracleprefix": max(abs(u - v) for u, v in zip(res["eager_prefix"], res["oracle_prefix"])),
                         "argmax_equal": int(np.argmax(res["eager_prefix"])) == int(np.argmax(q["probs"]))})
    e2e = {"rows": len(rows), "argmax_equal": sum(r["argmax_equal"] for r in rows),
           "max_abs_dp_eager_vs_oracle": max(r["max_abs_dp_eager_vs_oracle"] for r in rows),
           "max_abs_dp_oracleprefix_vs_oracle": max(r["max_abs_dp_oracleprefix_vs_oracle"] for r in rows),
           "max_abs_dp_eager_vs_oracleprefix": max(r["max_abs_dp_eager_vs_oracleprefix"] for r in rows),
           "per_row": rows}
    path = K / "results/vision_eager_check.json"
    doc = json.loads(path.read_text())
    doc["fp32_floor"] = {"written": now(), "what": eager_diag.__doc__, "records": out, "e2e_torch_decision_graph": e2e,
                         "seconds": round(time.time() - t0, 1)}
    write_json(path, doc)
    print(json.dumps({k: v for k, v in e2e.items() if k != "per_row"}, indent=1))
    return 0


def truth(a):
    """The float64 truth for every image record: the provider's Vision in float64 (its position resize stays float32
    on the CPU, as transformers does it) -> out/r6_runs/truth/<id>.npy; per record the provider fp32 (= the oracle
    npz prefix) and our eager fp32 against it, and ours in float64 against it (the identity on every crop) -> the
    `float64_truth` section of results/vision_eager_check.json."""
    import torch
    from transformers.image_utils import load_image

    import vision_graph as VG

    torch.set_num_threads(THREADS_ORACLE)
    t0 = time.time()
    vmod, vis, _ = provider_vision()
    tower, proj, table, _ = VG.load_vision(dtype=torch.float64)
    vis = vis.double()
    recs, oracle = image_records()
    (RUNS / "truth").mkdir(parents=True, exist_ok=True)
    restore, calls = float_resize_patch()
    rows = {}
    try:
        with torch.no_grad():
            for e, rec in recs:
                rid = e["id"]
                img = load_image(str(image_path(rec)))
                inputs = vmod.preprocess(img)
                hid = vis.tower(pixel_values=inputs["pixel_values"].double(), spatial_shapes=inputs["spatial_shapes"],
                                pixel_attention_mask=inputs["pixel_attention_mask"]).last_hidden_state
                host = V.preprocess(V.load_image(image_path(rec)))
                pre_t, pre_o, feat_dev = [], [], []
                for i, (h, w) in enumerate(inputs["spatial_shapes"].tolist()):
                    n = h * w
                    pre_t.append(vis.projector(hid[i:i + 1, :n].reshape(1, h, w, -1))[0].numpy())
                    crop = {"pixels": host["pixel_values"][i], "mask": host["pixel_attention_mask"][i], "grid": (h, w)}
                    x64 = {k: torch.from_numpy(v.astype(np.float64)) for k, v in V.tower_inputs(crop, table).items()}
                    f = tower(**x64)["features"][0].numpy()
                    feat_dev.append(float(np.abs(f[:n] - hid[i, :n].numpy()).max()))
                    cells = V.pixel_unshuffle(f[:n], (h, w))
                    soft = np.zeros((1, V.PROJ_ROWS, cells.shape[1]), np.float64)
                    soft[0, : cells.shape[0]] = cells
                    pre_o.append(proj(soft=torch.from_numpy(soft))["prefix"][0].numpy()[: cells.shape[0]])
                pre_t, pre_o = np.concatenate(pre_t), np.concatenate(pre_o)
                np.save(RUNS / "truth" / f"{rid}.npy", pre_t)
                z = npz(rid)
                eager = np.load(RUNS / "eager" / f"{rid}.npy")
                rows[rid] = {"P": int(pre_t.shape[0]), "crops": len(feat_dev),
                             "provider_fp32_vs_truth": diff(z["prefix"], pre_t),
                             "eager_fp32_vs_truth": diff(eager, pre_t),
                             "ours_float64_vs_truth_features_max_abs": max(feat_dev),
                             "ours_float64_vs_truth_prefix": diff(pre_o, pre_t)}
                r = rows[rid]
                r["eager_over_provider_distance"] = (r["eager_fp32_vs_truth"]["max_abs"]
                                                     / r["provider_fp32_vs_truth"]["max_abs"])
                print(rid, "provider->truth %.3g  eager->truth %.3g  ratio %.2f  ours64 feat %.2g prefix %.2g" % (
                    r["provider_fp32_vs_truth"]["max_abs"], r["eager_fp32_vs_truth"]["max_abs"],
                    r["eager_over_provider_distance"], r["ours_float64_vs_truth_features_max_abs"],
                    r["ours_float64_vs_truth_prefix"]["max_abs"]), flush=True)
    finally:
        restore()
    path = K / "results/vision_eager_check.json"
    doc = json.loads(path.read_text())
    summ = {"records": len(rows),
            "ours_float64_features_max_abs": max(r["ours_float64_vs_truth_features_max_abs"] for r in rows.values()),
            "ours_float64_prefix_max_abs": max(r["ours_float64_vs_truth_prefix"]["max_abs"] for r in rows.values()),
            "eager_over_provider_distance_max": max(r["eager_over_provider_distance"] for r in rows.values()),
            "eager_within_2x_provider_distance": sum(r["eager_over_provider_distance"] <= 2.0 for r in rows.values())}
    summ["identity_pass_1e-10"] = summ["ours_float64_prefix_max_abs"] <= 1e-10 and summ["ours_float64_features_max_abs"] <= 1e-10
    doc["float64_truth"] = {"written": now(), "what": truth.__doc__, "summary": summ, "records": rows,
                            "seconds": round(time.time() - t0, 1)}
    doc["bar_decision"] = {
        "from": "the review of 2026-10-08 15:3x",
        "pass_fail": "e2e = FACTS §7 on the 16 image rows against the oracle (argmax 100 % + max |dp| <= 0.02 + "
                     "mean |dp| <= 0.002)",
        "module_identity": "float64 identity <= 1e-10 (features and prefix)",
        "prefix_distances_recorded": ["vs the oracle (provider fp32)", "vs the float64 truth", "vs our eager fp32"],
        "litert_fp32_bar": "distance to the float64 truth within 2x the provider fp32's (per record, max |d|) + e2e PASS",
        "rel_rms_vs_eager": "recorded, not a bar",
        "replaces": "prefix max |d| <= 1e-4 and the stop condition 1e-3 (both measured unattainable for any fp32 "
                    "implementation that is not the provider's own SDPA kernel order: fp32_floor section)"}
    doc["summary"]["float64_identity_all_records"] = summ
    write_json(path, doc)
    print(json.dumps(summ, indent=1))
    return 0 if summ["identity_pass_1e-10"] else 1


# --------------------------------------------------------------------------- LiteRT (exporter venv)

PAD_CHECK = (("card_cats", 0), ("img_dogs_01", 0), ("img_03", 6))
BAR_E2E = {"max_abs_dp": 0.02, "mean_abs_dp": 0.002, "near_tie_gap": 0.02}


def lrt_tag(backend, precision, form):
    return f"{backend}_{'fp32' if backend == 'cpu' else precision}_{form}"


def lrt_result_path(tag):
    return K / f"results/vision_parity_{tag}.json"


def lrt_child(a):
    """One run: the tower and projector files of `form` on one backend, every crop of the 7 image records through
    the host -> out/r6_runs/<tag>/<id>.npy, prefix / feature distances, pad invariance, delegation."""
    import importlib.metadata as md
    import resource

    import litert_run as R

    tag = lrt_tag(a.backend, a.precision, a.form)
    rdir = RUNS / tag
    rdir.mkdir(parents=True, exist_ok=True)
    tower_path = K / f"out/d1omni_vision_tower_{a.form}.tflite"
    proj_path = K / f"out/d1omni_projector_{a.form}.tflite"
    k = len(list(K.glob(f"logs/r6_{tag}_run*.compile_tower.log")))
    logs = {n: K / f"logs/r6_{tag}_run{k}.{n}.log" for n in ("compile_tower", "compile_projector", "runs")}
    info = {"tag": tag, "started": now(), "pid": os.getpid(), "backend": a.backend, "precision": a.precision,
            "form": a.form, "threads": 8 if a.backend == "cpu" else None,
            "files": {"tower": {"file": str(tower_path.relative_to(K)), "bytes": tower_path.stat().st_size},
                      "projector": {"file": str(proj_path.relative_to(K)), "bytes": proj_path.stat().st_size}},
            "ai_edge_litert": md.version("ai-edge-litert")}
    accel = "cpu" if a.backend == "cpu" else "gpu"
    status, err = "OK", None
    per, pads = [], []
    t0 = time.time()
    try:
        with R.capture_fd2(logs["compile_tower"]):
            info["logger"] = R.runtime_log_verbose()
            t = time.time()
            tower = V.LiteRTGraph(tower_path, accel, a.precision, threads=8)
            info["compile_seconds_tower"] = round(time.time() - t, 2)
        with R.capture_fd2(logs["compile_projector"]):
            t = time.time()
            proj = V.LiteRTGraph(proj_path, accel, a.precision, threads=8)
            info["compile_seconds_projector"] = round(time.time() - t, 2)
        info["is_fully_accelerated"] = {"tower": tower.fully_accelerated, "projector": proj.fully_accelerated}
        table = V.read_position_table(S.WEIGHTS)
        recs, _ = image_records()
        with R.capture_fd2(logs["runs"]):
            for e, rec in recs:
                rid = e["id"]
                z = npz(rid)
                crops, plan = V.crops_of(V.load_image(image_path(rec)))
                pres, crow = [], []
                for i, c in enumerate(crops):
                    crop = V.to_patches(c)
                    h, w = crop["grid"]
                    n = h * w
                    x = V.tower_inputs(crop, table)
                    feat = tower(**x)[0]
                    cells = V.pixel_unshuffle(feat[:n], (h, w))
                    pre = proj(soft=V.projector_input(cells))[0][: cells.shape[0]]
                    pres.append(pre)
                    crow.append({"crop": i, "grid": [h, w], "features_vs_oracle": diff(feat[:n],
                                                                                     z["tower_last_hidden_state"][i, :n]),
                                 "nonfinite_features_all_rows": int((~np.isfinite(feat)).sum())})
                    if (rid, i) in PAD_CHECK and n < V.MAX_PATCHES:
                        g = np.random.default_rng(500 + i)
                        y = {kk: v.copy() for kk, v in x.items()}
                        y["pixels"][0, n:] = g.normal(0, 3, (V.MAX_PATCHES - n, 768)).astype(np.float32)
                        y["pos"][0, n:] = g.normal(0, 3, (V.MAX_PATCHES - n, 768)).astype(np.float32)
                        feat2 = tower(**y)[0]
                        pads.append({"id": rid, "crop": i, "pad_rows": V.MAX_PATCHES - n,
                                     "real_bit_equal": bits_equal(feat2[:n], feat[:n]),
                                     "real_max_abs": float(np.abs(feat2[:n].astype(np.float64) - feat[:n]).max()),
                                     "pad_rows_changed": bool(not np.array_equal(feat2[n:], feat[n:]))})
                prefix = np.concatenate(pres).astype(np.float32)
                np.save(rdir / f"{rid}.npy", prefix)
                per.append({"id": rid, "P": int(prefix.shape[0]), "crops": crow})
                print(rid, prefix.shape, flush=True)
        tower.close()
        proj.close()
    except BaseException as ex:  # recorded, then the json is written
        import traceback
        status, err = "FAIL", f"{type(ex).__name__}: {ex}"
        traceback.print_exc()
    info["seconds_wall"] = round(time.time() - t0, 1)
    info["ru_maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    info["status"], info["error"] = status, err
    info["delegation"] = {n: R.delegation_from_log(p) for n, p in logs.items() if p.exists()}
    raw = {"info": info, "records": per, "pad_invariance": pads}
    write_json(rdir / "run.json", raw)
    doc = lrt_score(tag)
    print(json.dumps({"tag": tag, "status": status, "error": err, "summary": doc.get("summary")}, indent=1, default=str))
    return 0 if status == "OK" else 1


def lrt_score(tag):
    """The run's prefixes against the oracle (provider fp32), the float64 truth and our eager fp32 -> the
    `prefix` part of results/vision_parity_<tag>.json (the e2e part is kept when present)."""
    rdir = RUNS / tag
    raw = json.loads((rdir / "run.json").read_text())
    path = lrt_result_path(tag)
    old = json.loads(path.read_text()) if path.exists() else {}
    rows = []
    recs, oracle = image_records()
    for e, _ in recs:
        rid = e["id"]
        f = rdir / f"{rid}.npy"
        if not f.exists():
            continue
        got = np.load(f)
        z = npz(rid)
        tru = np.load(RUNS / "truth" / f"{rid}.npy")
        eag = np.load(RUNS / "eager" / f"{rid}.npy")
        d_o, d_t, d_e = diff(got, z["prefix"]), diff(got, tru), diff(got, eag)
        d_pt = diff(z["prefix"], tru)
        rr = next((r for r in raw["records"] if r["id"] == rid), {})
        rows.append({"id": rid, "P": int(got.shape[0]), "vs_oracle": d_o, "vs_float64_truth": d_t, "vs_eager": d_e,
                     "provider_fp32_vs_truth_max_abs": d_pt["max_abs"],
                     "provider_fp32_vs_truth_rel_rms": d_pt["rel_rms"],
                     "truth_distance_over_provider": d_t["max_abs"] / d_pt["max_abs"],
                     "within_2x_provider_distance": d_t["max_abs"] <= 2.0 * d_pt["max_abs"],
                     "truth_rel_rms_over_provider": d_t["rel_rms"] / d_pt["rel_rms"],
                     "within_2x_provider_rel_rms": d_t["rel_rms"] <= 2.0 * d_pt["rel_rms"],
                     "worst_token": int(np.abs(got.astype(np.float64) - tru).max(1).argmax()),
                     "second_worst_token_max_abs": float(np.sort(np.abs(got.astype(np.float64) - tru).max(1))[-2])
                     if got.shape[0] > 1 else None,
                     "features_vs_oracle_max_abs": max((c["features_vs_oracle"]["max_abs"] for c in rr.get("crops", [])),
                                                       default=None),
                     "nonfinite": d_o["nonfinite"] + sum(c.get("nonfinite_features_all_rows", 0)
                                                         for c in rr.get("crops", []))})
    info = raw["info"]
    dl = info.get("delegation", {})
    rep = {n: d.get("replacing") for n, d in dl.items()}
    summ = {"status": info["status"], "error": info["error"], "records": len(rows),
            "prefix_vs_oracle_max_abs": max((r["vs_oracle"]["max_abs"] for r in rows), default=None),
            "prefix_vs_oracle_rel_rms_max": max((r["vs_oracle"]["rel_rms"] for r in rows), default=None),
            "prefix_vs_truth_max_abs": max((r["vs_float64_truth"]["max_abs"] for r in rows), default=None),
            "prefix_vs_eager_max_abs": max((r["vs_eager"]["max_abs"] for r in rows), default=None),
            "prefix_vs_eager_rel_rms_max": max((r["vs_eager"]["rel_rms"] for r in rows), default=None),
            "truth_distance_over_provider_max": max((r["truth_distance_over_provider"] for r in rows), default=None),
            "within_2x_provider_distance": sum(r["within_2x_provider_distance"] for r in rows),
            "truth_rel_rms_over_provider_max": max((r["truth_rel_rms_over_provider"] for r in rows), default=None),
            "within_2x_provider_rel_rms": sum(r["within_2x_provider_rel_rms"] for r in rows),
            "features_vs_oracle_max_abs": max((r["features_vs_oracle_max_abs"] or 0 for r in rows), default=None),
            "nonfinite": sum(r["nonfinite"] for r in rows),
            "pad_invariance": [{k: p[k] for k in ("id", "crop", "real_bit_equal", "real_max_abs")}
                               for p in raw["pad_invariance"]],
            "replacing": rep, "is_fully_accelerated": info.get("is_fully_accelerated"),
            "compile_seconds": {"tower": info.get("compile_seconds_tower"),
                                "projector": info.get("compile_seconds_projector")}}
    summ["prefix_max_element_2x_truth_distance"] = bool(rows and len(rows) == 7
                                                        and summ["within_2x_provider_distance"] == 7
                                                        and summ["nonfinite"] == 0)
    # 15:4x: the auxiliary prefix bar = the rel_rms distance to the float64 truth within 2x the provider
    # fp32's (per record); the max-element ratio is kept as a value; PASS / FAIL = e2e
    summ["prefix_bar_rel_rms_2x"] = bool(rows and len(rows) == 7 and summ["within_2x_provider_rel_rms"] == 7
                                         and summ["nonfinite"] == 0)
    if old.get("e2e"):
        summ["e2e"] = {k: v["summary"] for k, v in old["e2e"].items()}
    doc = {"step": "round 6 step 4: the vision graphs on the Mac (CompiledModel) against the oracle / float64 truth / "
                   "eager", "written": now(), "tag": tag, "summary": summ, "e2e": old.get("e2e", {}),
           "bar": {"pass_fail": "e2e (FACTS §7 on the 16 image rows, oracle v2): argmax 100 % outside near-tie rows "
                                "+ max |dp| <= 0.02 + mean |dp| <= 0.002 + no non-finite value",
                   "prefix_auxiliary": "rel_rms distance to the float64 truth within 2x the provider fp32's, per "
                                       "record, no non-finite value (15:4x; the fp32 files); the "
                                       "max-element ratio (15:3x proposal) is kept as a value",
                   "e2e": BAR_E2E},
           "run": info, "records": rows, "pad_invariance": raw["pad_invariance"],
           "history": old.get("history", []) + [{"scored": now(), "status": info["status"]}]}
    write_json(path, doc)
    return doc


def lrt_parent(a):
    """GPU: the run in a child process; a child that dies without its record gets a failure record here."""
    tag = lrt_tag(a.backend, a.precision, a.form)
    rdir = RUNS / tag
    rdir.mkdir(parents=True, exist_ok=True)
    errlog = K / f"logs/r6_{tag}.child_stdio.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--lrt", "--backend", a.backend, "--precision", a.precision,
           "--form", a.form, "--child"]
    import subprocess
    import signal

    t0 = time.time()
    with open(errlog, "a") as fo:
        fo.write(f"--- {now()} {' '.join(cmd)}\n")
        fo.flush()
        child = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=str(K))
        _, st, ru = os.wait4(child.pid, 0)
    rc = os.waitstatus_to_exitcode(st)
    sig = signal.Signals(-rc).name if rc < 0 else None
    runj = rdir / "run.json"
    recorded = runj.exists() and json.loads(runj.read_text())["info"].get("pid") == child.pid
    if not recorded:
        tails = {}
        for p in sorted(K.glob(f"logs/r6_{tag}_run*.log"), key=lambda q: q.stat().st_mtime)[-3:]:
            tails[p.name] = p.read_text(errors="replace").splitlines()[-40:]
        raw = {"info": {"tag": tag, "started": now(), "pid": child.pid, "status": "GPU_FAIL", "returncode": rc,
                        "signal": sig, "error": f"child died rc={rc} signal={sig}", "seconds_wall": round(time.time() - t0, 1),
                        "child_ru_maxrss_bytes": int(ru.ru_maxrss), "runtime_log_tails": tails,
                        "child_stdio_tail": errlog.read_text(errors="replace").splitlines()[-40:]},
               "records": [], "pad_invariance": []}
        write_json(runj, raw)
        lrt_score(tag)
    else:
        raw = json.loads(runj.read_text())
        raw["info"]["child_ru_maxrss_bytes"] = int(ru.ru_maxrss)
        write_json(runj, raw)
        lrt_score(tag)
    print(f"gpu child pid={child.pid} rc={rc} signal={sig} recorded_by_child={recorded} "
          f"seconds={time.time() - t0:.1f} -> {lrt_result_path(tag).relative_to(K)}")
    return rc


def e2e(a):
    """The decision graph (CompiledModel CPU, 8 threads) on the 16 image rows with this tag's vision prefix and with
    the oracle's prefix -> the `e2e` part of results/vision_parity_<tag>.json, one entry per text form."""
    from types import SimpleNamespace

    import d1_host as H
    import litert_run as R

    tag = a.tag
    path = lrt_result_path(tag)
    doc = json.loads(path.read_text())
    recs, oracle = image_records()
    temps = S.config()["temperatures"]
    qt = {"choice": 0, "score": 1, "noul": 2}
    rdir = RUNS / tag
    rows, by_L = [], {}
    for e, _ in recs:
        for q in e["questions"]:
            L = next(b for b in (256, 512, 1024, 2048, 4096) if q["positions"] <= b)
            by_L.setdefault(L, []).append((e, q))
    out = doc.get("e2e", {})
    for tform in a.text_forms.split(","):
        t0 = time.time()
        rows = []
        runs = {}
        for L, items in sorted(by_L.items()):
            cm, desc = R.open_compiled(K / f"out/d1omni_decide_L{L}_{tform}.tflite", "cpu", threads=8)
            run = R.Runner(cm, next(iter(cm.get_signature_list())))
            for e, q in items:
                rid = e["id"]
                res = {}
                for name, pre in (("vision_graph", np.load(rdir / f"{rid}.npy")), ("oracle_prefix", npz(rid)["prefix"])):
                    x = H.build_inputs(q["ids"], pre, L)
                    oh = np.zeros((1, 3), np.float32)
                    oh[0, qt[q["type"]]] = 1.0
                    x["qtype_onehot"] = oh
                    sc = run(x)
                    pq = H.readout(sc, q["prefix"], q["markers"], SimpleNamespace(type=q["type"], options=q["K"]),
                                   bool(q["calibrate"]), temps)
                    res[name] = {"probs": [float(v) for v in pq],
                                 "logits": [float(sc[q["prefix"] + m]) for m in q["markers"][: q["K"]]],
                                 "finite": bool(np.isfinite(sc[: q["positions"]]).all())}
                po = q["probs"]
                srt = sorted(po, reverse=True)
                gap = srt[0] - srt[1] if len(srt) > 1 else 1.0
                pv, pr = res["vision_graph"]["probs"], res["oracle_prefix"]["probs"]
                rows.append({"key": f"{rid}/{q['qid']}", "L": L, "type": q["type"], "oracle": po, "vision_graph": pv,
                             "oracle_prefix": pr, "near_tie": gap <= BAR_E2E["near_tie_gap"],
                             "argmax_equal": int(np.argmax(pv)) == int(np.argmax(po)),
                             "max_abs_dp_vs_oracle": max(abs(u - v) for u, v in zip(pv, po)),
                             "dp_vs_oracle": [abs(u - v) for u, v in zip(pv, po)],
                             "max_abs_dp_oracleprefix_vs_oracle": max(abs(u - v) for u, v in zip(pr, po)),
                             "max_abs_dp_vision_vs_oracleprefix": max(abs(u - v) for u, v in zip(pv, pr)),
                             "max_abs_dlogit_vision_vs_oracleprefix": max(abs(u - v) for u, v in zip(
                                 res["vision_graph"]["logits"], res["oracle_prefix"]["logits"])),
                             "finite": res["vision_graph"]["finite"]})
            run.close()
            runs[str(L)] = {"file": f"out/d1omni_decide_L{L}_{tform}.tflite", "rows": len(items), "options": desc}
        dps = [d for r in rows for d in r["dp_vs_oracle"]]
        main = [r for r in rows if not r["near_tie"]]
        st = {"rows": len(rows), "near_tie_rows": len(rows) - len(main),
              "argmax_equal_outside_near_tie": sum(r["argmax_equal"] for r in main), "rows_outside_near_tie": len(main),
              "argmax_equal_all": sum(r["argmax_equal"] for r in rows),
              "max_abs_dp_vs_oracle": max(dps), "mean_abs_dp_vs_oracle": float(np.mean(dps)),
              "max_abs_dp_oracleprefix_vs_oracle": max(r["max_abs_dp_oracleprefix_vs_oracle"] for r in rows),
              "max_abs_dp_vision_vs_oracleprefix": max(r["max_abs_dp_vision_vs_oracleprefix"] for r in rows),
              "max_abs_dlogit_vision_vs_oracleprefix": max(r["max_abs_dlogit_vision_vs_oracleprefix"] for r in rows),
              "nonfinite_rows": sum(not r["finite"] for r in rows)}
        st["bar_pass"] = bool(st["argmax_equal_outside_near_tie"] == st["rows_outside_near_tie"]
                              and st["max_abs_dp_vs_oracle"] <= BAR_E2E["max_abs_dp"]
                              and st["mean_abs_dp_vs_oracle"] <= BAR_E2E["mean_abs_dp"] and st["nonfinite_rows"] == 0)
        out[f"text_{tform}_cpu"] = {"written": now(), "summary": st, "text_graphs": runs, "per_row": rows,
                                    "seconds": round(time.time() - t0, 1)}
        print(tag, tform, json.dumps(st), flush=True)
    doc["e2e"] = out
    doc["summary"]["e2e"] = {k: v["summary"] for k, v in out.items()}
    write_json(path, doc)
    return 0


def half_ab(a):
    """Torch fp32 with every FULLY_CONNECTED weight of the graphs rounded to fp16 and back (patch_embedding, the 72
    layer linears, the projector's 2; biases and layer norms unchanged) = what the FC-only fp16 file stores, on the
    host's inputs -> out/r6_runs/torch_fc16/<id>.npy and the `fc16_weights_torch` section of
    results/vision_eager_check.json (distances to the oracle / float64 truth / eager fp32)."""
    import torch

    import vision_graph as VG

    torch.set_num_threads(8)
    t0 = time.time()
    tower, proj, table, _ = VG.load_vision()
    n_w = 0
    for m in list(tower.modules()) + list(proj.modules()):
        if isinstance(m, torch.nn.Linear):
            m.weight.copy_(m.weight.half().float())
            n_w += 1
    recs, _ = image_records()
    (RUNS / "torch_fc16").mkdir(parents=True, exist_ok=True)
    rows = {}
    with torch.no_grad():
        for e, rec in recs:
            rid = e["id"]
            crops, _ = V.crops_of(V.load_image(image_path(rec)))
            pres = []
            for c in crops:
                crop = V.to_patches(c)
                h, w = crop["grid"]
                x = {k: torch.from_numpy(v) for k, v in V.tower_inputs(crop, table).items()}
                feat = tower(**x)["features"][0].numpy()
                cells = V.pixel_unshuffle(feat[: h * w], (h, w))
                pres.append(proj(soft=torch.from_numpy(V.projector_input(cells)))["prefix"][0].numpy()[: cells.shape[0]])
            pre = np.concatenate(pres)
            np.save(RUNS / "torch_fc16" / f"{rid}.npy", pre)
            z = npz(rid)
            tru = np.load(RUNS / "truth" / f"{rid}.npy")
            rows[rid] = {"vs_oracle": diff(pre, z["prefix"]), "vs_float64_truth": diff(pre, tru),
                         "vs_eager": diff(pre, np.load(RUNS / "eager" / f"{rid}.npy"))}
            for tag in ("cpu_fp32_fp16", "gpu_fp32_fp16"):
                f = RUNS / tag / f"{rid}.npy"
                if f.exists():
                    rows[rid][f"litert_{tag}_vs_this"] = diff(np.load(f), pre)
            print(rid, "fc16 torch vs truth %.3g" % rows[rid]["vs_float64_truth"]["max_abs"], flush=True)
    path = K / "results/vision_eager_check.json"
    doc = json.loads(path.read_text())
    doc["fc16_weights_torch"] = {"written": now(), "what": half_ab.__doc__, "linear_weights_rounded": n_w,
                                 "records": rows, "seconds": round(time.time() - t0, 1)}
    write_json(path, doc)
    return 0


# --------------------------------------------------------------------------- Mac ms (exporter venv, quiet_hold)

WARMUP, REPS = 5, 20
PEER_CPU_MAX, IDLE_MIN, GATE_WAIT_S, GATE_POLL_S = 120.0, 50.0, 300, 10
MS_IMAGE = "img_dogs_01"          # the 384 px picture of round 4's workload (e): 384 x 216 -> 224 x 384, 84 tokens
MS_TILED = "img_03"               # 768 x 1024 -> 6 tiles + thumbnail = 7 crops, 1,770 tokens
R4_TEXT = {  # round 4's medians (results/timing_mac_r4_*.json `sets.<form>.result.workloads.<w>.ms_write_run_read`)
    ("gpu", "fp16", 256): ("results/timing_mac_r4_gpu_fp32_L256.json", "fp16", "e_L256"),
    ("gpu", "fp32", 256): ("results/timing_mac_r4_gpu_fp32_L256.json", "fp32", "e_L256"),
    ("cpu", "fp16", 256): ("results/timing_mac_r4_cpu_L256.json", "fp16", "e_L256"),
    ("cpu", "fp32", 256): ("results/timing_mac_r4_cpu_L256.json", "fp32", "e_L256"),
    ("gpu", "fp16", 2048): ("results/timing_mac_r4_gpu_fp32_L2048.json", "fp16", "g"),
    ("gpu", "fp32", 2048): ("results/timing_mac_r4_gpu_fp32_L2048.json", "fp32", "g"),
    ("cpu", "fp16", 2048): ("results/timing_mac_r4_cpu_L2048.json", "fp16", "g"),
    ("cpu", "fp32", 2048): ("results/timing_mac_r4_cpu_L2048.json", "fp32", "g"),
}


def _contention():
    import subprocess

    out = subprocess.run(["top", "-l", "2", "-s", "1", "-o", "cpu", "-n", "12", "-stats", "pid,ppid,cpu,command"],
                         capture_output=True, text=True).stdout.splitlines()
    cpu_lines = [ln for ln in out if ln.startswith("CPU usage")]
    idle = None
    if cpu_lines:
        for part in cpu_lines[-1].split(","):
            if "idle" in part:
                idle = float(part.strip().split("%")[0])
    heads = [i for i, ln in enumerate(out) if ln.strip().startswith("PID")]
    procs = []
    for ln in (out[heads[-1] + 1:] if heads else []):
        f = ln.split(None, 3)
        if len(f) < 4:
            continue
        try:
            procs.append({"pid": int(f[0]), "ppid": int(f[1]), "cpu": float(f[2]), "command": f[3].strip()})
        except ValueError:
            continue
    mine = {os.getpid(), os.getppid()}
    peers = [q for q in procs if q["cpu"] > PEER_CPU_MAX and q["pid"] not in mine and q["ppid"] not in mine]
    return {"idle_pct": idle, "peers_above_120": peers, "top": procs[:6],
            "ok": idle is not None and idle >= IDLE_MIN and not peers}


def _load_avg():
    import subprocess

    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return [float(x) for x in out.replace("{", " ").replace("}", " ").split()[:3]]


def _gate():
    t0 = time.time()
    polls = []
    while True:
        c = _contention()
        polls.append({"t": round(time.time() - t0, 1), "load": _load_avg(), "idle_pct": c["idle_pct"],
                      "peers_above_120": c["peers_above_120"]})
        if c["ok"]:
            return {"ok": True, "waited_s": round(time.time() - t0, 1), "polls": polls[-4:], "contention": c}
        if time.time() - t0 > GATE_WAIT_S:
            return {"ok": False, "waited_s": round(time.time() - t0, 1), "polls": polls, "contention": c}
        time.sleep(GATE_POLL_S)


def _stats(v):
    v = sorted(v)
    return {"median": float(np.median(v)), "min": float(v[0]), "max": float(v[-1]), "n": len(v)}


def _timed(graph, feeds):
    """One call as the host makes it: write every input, run, read the output back -> (ms total, ms run only)."""
    t = time.perf_counter()
    for n, v in feeds.items():
        graph.inputs[n].write(v)
    tr = time.perf_counter()
    graph.model.run_by_name(graph.signature, graph.inputs, graph.outputs)
    run_ms = (time.perf_counter() - tr) * 1000.0
    size = int(np.prod(graph.output_shape))
    out = np.asarray(graph.outputs[graph.output_name].read(size, np.float32), np.float32)
    total = (time.perf_counter() - t) * 1000.0
    return total, run_ms, out


def ms_child(a):
    """Fresh process: compile the tower and projector of one form on one backend, then 5 warm-up + 20 timed calls of
    each on img_dogs_01's crop (the static graph runs all 1024 / 256 rows whatever the crop) -> json on stdout."""
    import litert_run as R

    accel = "cpu" if a.backend == "cpu" else "gpu"
    res = {"backend": a.backend, "precision": a.precision, "form": a.form, "pid": os.getpid(), "started": now()}
    try:
        t = time.time()
        tower = V.LiteRTGraph(K / f"out/d1omni_vision_tower_{a.form}.tflite", accel, a.precision, threads=8)
        res["compile_seconds_tower"] = round(time.time() - t, 3)
        t = time.time()
        proj = V.LiteRTGraph(K / f"out/d1omni_projector_{a.form}.tflite", accel, a.precision, threads=8)
        res["compile_seconds_projector"] = round(time.time() - t, 3)
        res["memory_after_compile"] = R.memory()
        table = V.read_position_table(S.WEIGHTS)
        recs, _ = image_records()
        rec = next(r for e, r in recs if e["id"] == MS_IMAGE)
        crops, _ = V.crops_of(V.load_image(image_path(rec)))
        crop = V.to_patches(crops[0])
        h, w = crop["grid"]
        x = {k: np.ascontiguousarray(v, np.float32) for k, v in V.tower_inputs(crop, table).items()}
        ref = np.load(RUNS / lrt_tag(a.backend, a.precision, a.form) / f"{MS_IMAGE}.npy") if (
            RUNS / lrt_tag(a.backend, a.precision, a.form) / f"{MS_IMAGE}.npy").exists() else None
        for name, graph, feeds in (("tower", tower, x), ("projector", proj, None)):
            if name == "projector":
                feat = tower(**x)[0]
                feeds = {"soft": np.ascontiguousarray(V.projector_input(V.pixel_unshuffle(feat[: h * w], (h, w))))}
            warm, timed, runs = [], [], []
            out = None
            for i in range(WARMUP + REPS):
                tot, run_ms, out = _timed(graph, feeds)
                (warm if i < WARMUP else timed).append(tot)
                if i >= WARMUP:
                    runs.append(run_ms)
            res[name] = {"ms_write_run_read": _stats(timed), "ms_run_only": _stats(runs), "warmup_ms": warm,
                         "timed_ms": timed, "finite": bool(np.isfinite(out).all())}
        cells = feeds["soft"][0, : (h // 2) * (w // 2)]
        pre = proj(soft=feeds["soft"])[0][: cells.shape[0]]
        if ref is not None:
            res["timed_graph_prefix_vs_parity_run_max_abs"] = float(np.abs(pre.astype(np.float64) - ref).max())
        res["memory_end"] = R.memory()
        tower.close()
        proj.close()
        res["status"] = "OK"
    except BaseException as ex:
        import traceback
        res["status"], res["error"] = "FAIL", f"{type(ex).__name__}: {ex}"
        res["traceback"] = traceback.format_exc()[-3000:]
    print("MS_RESULT " + json.dumps(res, default=str), flush=True)
    return 0 if res["status"] == "OK" else 1


def host_ms():
    """The reference host's steps (numpy + Pillow, this process, one thread of Python) on img_dogs_01 and img_03:
    2 warm-up + 20 timed runs each."""
    recs, _ = image_records()
    table = V.read_position_table(S.WEIGHTS)
    out = {}
    for rid in (MS_IMAGE, MS_TILED):
        rec = next(r for e, r in recs if e["id"] == rid)
        path = image_path(rec)
        feats = {}
        steps = {"load_image": [], "layout_and_resize": [], "patches": [], "positions": [], "unshuffle_and_pad": []}
        for i in range(2 + REPS):
            t = time.perf_counter()
            img = V.load_image(path)
            t1 = time.perf_counter()
            crops, plan = V.crops_of(img)
            t2 = time.perf_counter()
            cs = [V.to_patches(c) for c in crops]
            t3 = time.perf_counter()
            ins = [V.tower_inputs(c, table) for c in cs]
            t4 = time.perf_counter()
            if not feats:
                g = np.random.default_rng(0)
                feats = {j: g.standard_normal((1024, 768)).astype(np.float32) for j in range(len(cs))}
            soft = [V.projector_input(V.pixel_unshuffle(feats[j][: c["grid"][0] * c["grid"][1]], c["grid"]))
                    for j, c in enumerate(cs)]
            t5 = time.perf_counter()
            if i >= 2:
                for k, (u, v) in zip(steps, ((t, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5))):
                    steps[k].append((v - u) * 1000.0)
        tot = [sum(x) for x in zip(*steps.values())]
        out[rid] = {"crops": len(cs), "plan": {k: list(v) if isinstance(v, tuple) else v for k, v in plan.items()},
                    "ms": {k: _stats(v) for k, v in steps.items()}, "ms_total": _stats(tot),
                    "note": "numpy reference port (Python loops over output rows for the resize); the position "
                            "table of a 32 x 32 tile can be cached once per grid"}
    return out


def ms_window(a):
    """quiet_hold window: the contention gate, then one fresh child per form, then the host steps."""
    import subprocess

    label = "cpu" if a.backend == "cpu" else f"gpu_{a.precision}"
    path = K / f"results/timing_mac_r6_{label}.json"
    n = 2
    while path.exists():
        path = K / f"results/timing_mac_r6_{label}_take{n}.json"
        n += 1
    try:
        lock = Path(os.path.expanduser("~/code/coreai/_GPU_LOCK")).read_text().strip()
    except OSError as ex:
        lock = f"unreadable: {ex}"
    doc = {"what": f"Mac ms of the vision graphs, backend {label}, forms {a.forms}, one quiet window",
           "protocol": f"per form a fresh child process: compile tower + projector, {WARMUP} warm-up then {REPS} timed "
                       "calls of each; 1 call = write every input + run + read the output back (ms_write_run_read); "
                       "the static tower runs 1024 rows and the projector 256 rows whatever the crop",
           "window_lock_line_at_start": lock, "started": now(), "sets": {}}
    rc = 0
    for form in a.forms.split(","):
        gate = _gate()
        st = {"gate": gate}
        if not gate["ok"]:
            st["status"] = "discarded: contention for the whole wait"
            doc["sets"][form] = st
            rc = 3
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--ms-child", "--backend", a.backend, "--precision",
               a.precision, "--form", form]
        pr = subprocess.run(cmd, capture_output=True, text=True, cwd=str(K))
        line = [ln for ln in pr.stdout.splitlines() if ln.startswith("MS_RESULT ")]
        res = json.loads(line[-1][len("MS_RESULT "):]) if line else {"status": "FAIL", "returncode": pr.returncode,
                                                                       "stderr_tail": pr.stderr[-2000:]}
        st["result"] = res
        st["status"] = "measured" if res.get("status") == "OK" else f"failed: {res.get('error')}"
        st["load_after"] = _load_avg()
        doc["sets"][form] = st
    if a.backend == "cpu":
        doc["host"] = {"gate": _gate(), "steps": host_ms()}
    doc["finished"] = now()
    write_json(path, doc)
    summ = {f: {"status": s["status"], "tower_ms": (s.get("result") or {}).get("tower", {}).get("ms_write_run_read"),
                "projector_ms": (s.get("result") or {}).get("projector", {}).get("ms_write_run_read")}
            for f, s in doc["sets"].items()}
    print(json.dumps({"out": str(path.relative_to(K)), "sets": summ,
                      "host": {k: v["ms_total"] for k, v in (doc.get("host", {}).get("steps") or {}).items()}},
                     indent=1, default=str))
    return rc


def ms_aggregate(a):
    """results/timing_mac_r6.json: per backend x form the tower / projector medians (the latest take of each window),
    the host steps, round 4's text row (img_dogs_01/count at L256 = (e_L256); img_03 at L2048 = (g), the same graph
    call), and the totals for one question on img_dogs_01 (1 crop) and on img_03 (7 crops)."""
    wins = {}
    for p in sorted(K.glob("results/timing_mac_r6_*.json")):
        if p.name == "timing_mac_r6.json":
            continue
        d = json.loads(p.read_text())
        label = p.stem[len("timing_mac_r6_"):].split("_take")[0]
        wins[label] = (p.name, d)          # sorted: a later take replaces an earlier one
    host = None
    for name, d in wins.values():
        if d.get("host"):
            host = {"file": name, **d["host"]["steps"]}
    rows = []
    for label, (name, d) in wins.items():
        backend = "cpu" if label == "cpu" else "gpu"
        for form, st in d["sets"].items():
            r = st.get("result") or {}
            if st.get("status") != "measured":
                rows.append({"window": name, "backend": label, "form": form, "status": st.get("status")})
                continue
            tw, pj = r["tower"]["ms_write_run_read"]["median"], r["projector"]["ms_write_run_read"]["median"]
            row = {"window": name, "backend": label, "form": form, "status": "measured", "tower_ms": tw,
                   "projector_ms": pj, "tower_min_max": [r["tower"]["ms_write_run_read"]["min"],
                                                         r["tower"]["ms_write_run_read"]["max"]],
                   "compile_s": [r.get("compile_seconds_tower"), r.get("compile_seconds_projector")],
                   "memory_after_compile": r.get("memory_after_compile")}
            tform = "fp16"   # the decision graph's ship form (round 3 / 4)
            for L, key in ((256, "text_L256_e_ms"), (2048, "text_L2048_g_ms")):
                f, fm, wk = R4_TEXT[(backend, tform, L)]
                td = json.loads((K / f).read_text())
                row[key] = td["sets"][fm]["result"]["workloads"][wk]["ms_write_run_read"]["median"]
                row[key + "_source"] = f"{f} sets.{fm} {wk}"
            if host:
                h1, h7 = host[MS_IMAGE]["ms_total"]["median"], host[MS_TILED]["ms_total"]["median"]
                row["host_ms_img_dogs_01"], row["host_ms_img_03"] = h1, h7
                row["total_one_question_384px_ms"] = h1 + tw + pj + row["text_L256_e_ms"]
                row["total_one_question_img_03_7crops_ms"] = h7 + 7 * (tw + pj) + row["text_L2048_g_ms"]
                row["graphs_only_384px_ms"] = tw + pj + row["text_L256_e_ms"]
                row["graphs_only_img_03_ms"] = 7 * (tw + pj) + row["text_L2048_g_ms"]
            rows.append(row)
    doc = {"what": ms_aggregate.__doc__, "written": now(), "windows": {k: v[0] for k, v in wins.items()},
           "host": host, "rows": rows}
    write_json(K / "results/timing_mac_r6.json", doc)
    for r in rows:
        print(json.dumps({k: r.get(k) for k in ("backend", "form", "status", "tower_ms", "projector_ms",
                                                 "host_ms_img_dogs_01", "text_L256_e_ms", "total_one_question_384px_ms",
                                                 "total_one_question_img_03_7crops_ms")}))
    return 0


def contract_vision(a):
    """Add the `vision` section to results/contract_draft.json (other keys untouched), from this round's files."""
    import litert_run as R

    path = K / "results/contract_draft.json"
    doc = json.loads(path.read_text())
    sig = {g: json.loads((K / f"results/signature_vision_{g}.json").read_text())["flatbuffer_signatures"][0]
           for g in ("tower", "projector")}
    quant = json.loads((K / "results/quant_vision.json").read_text())["forms"]
    ops = {g: json.loads((K / f"results/opscan_vision_{g}_fp32.json").read_text()) for g in ("tower", "projector")}
    host = json.loads((K / "results/vision_host_check.json").read_text())["summary"]
    eag = json.loads((K / "results/vision_eager_check.json").read_text())
    par = {}
    for f in sorted(K.glob("results/vision_parity_*.json")):
        d = json.loads(f.read_text())
        sm = d["summary"]
        par[d["tag"]] = {"status": sm["status"], "prefix_vs_oracle_max_abs": sm["prefix_vs_oracle_max_abs"],
                         "prefix_vs_truth_max_abs": sm["prefix_vs_truth_max_abs"],
                         "truth_distance_over_provider_max": sm["truth_distance_over_provider_max"],
                         "within_2x_provider_distance": f"{sm['within_2x_provider_distance']}/7",
                         "replacing": {k: [(x["delegated"], x["total"], x["delegate"]) for x in v]
                                       for k, v in (sm.get("replacing") or {}).items() if v},
                         "e2e_text_fp16": (sm.get("e2e") or {}).get("text_fp16_cpu")}
    files = {}
    for g, stem in (("tower", "d1omni_vision_tower"), ("projector", "d1omni_projector")):
        files[g] = {"fp32": {"file": f"out/{stem}_fp32.tflite", "bytes": ops[g]["bytes"], "sha256": ops[g]["sha256"]},
                    "fp16": {"file": quant[g]["output"], "bytes": quant[g]["bytes"], "sha256": quant[g]["sha256"]}}
    io = {g: {"signature": sig[g]["key"],
              "inputs": sorted([{"name": i["name"], "dtype": i["dtype"], "shape": i["shape"],
                                 "tensor_index": i["tensor_index"]} for i in sig[g]["inputs"]],
                               key=lambda i: i["tensor_index"]),
              "outputs": [{"name": o["name"], "dtype": o["dtype"], "shape": o["shape"]} for o in sig[g]["outputs"]]}
          for g in sig}
    vision = {
        "written": now(), "round": 6,
        "what": "image -> prefix rows for the decision graph: host preprocessing (numpy + Pillow), the vision tower "
                "graph per crop, the host unshuffle, the projector graph per crop",
        "graphs": {g: {**io[g], "files": files[g]} for g in io},
        "inputs": {
            "pixels": "float32 [1, 1024, 768]: the crop's 16 x 16 patches in raster order, each (row, column, "
                      "channel) = 768 values of (x - 127.5) / 127.5; zero rows after the ph * pw real patches",
            "pos": "float32 [1, 1024, 768]: the 16 x 16 position table resized to (ph, pw) (bilinear antialias, "
                   "align_corners False, float32), rows in raster order; rows past ph * pw = row 0 (any value: masked)",
            "mask": "float32 [1, 1024]: 1.0 for the ph * pw real patches, else 0",
            "soft": "float32 [1, 256, 3072]: the crop's (ph/2)(pw/2) unshuffled cells, zero rows after them"},
        "outputs": {"features": "float32 [1, 1024, 768]: post-layernorm hidden states; the first ph * pw rows are "
                                "the crop's", "prefix": "float32 [1, 256, 1024]: the first (ph/2)(pw/2) rows are the "
                                                         "crop's prefix rows"},
        "host_steps": [
            "load: Pillow open, EXIF orientation applied (ImageOps.exif_transpose), RGB (= transformers load_image)",
            "layout(width, height) (the provider's vision.layout, Python round half to even): grid (cols, rows), "
            "thumbnail (h, w), tiled = max(16, round32(h)) * max(16, round32(w)) > 524,288",
            "crops: tiled -> resize(picture, [rows * 512, cols * 512]) cut into 512 x 512 tiles row-major, then the "
            "thumbnail = resize(picture, thumbnail); not tiled -> the thumbnail only",
            "resize = the float path: uint8 -> float32 -> bilinear antialias align_corners False (width, then height; "
            "an unchanged side skipped; an unchanged size returns the picture) -> round half to even -> clamp 0..255 "
            "-> uint8 (torchvision 0.24's resize_image for a uint8 CPU tensor; NOT torchvision 0.29's arm64 uint8 "
            "kernel, which moves 17 % of card_cats' pixels by one level)",
            "patches: float32 (x - 127.5) / 127.5, 16 x 16 patches raster order, (row, column, channel), zero rows to "
            "1024, mask, grid (ph, pw) = crop size / 16",
            "pos: the checkpoint's vision.tower.vision_model.embeddings.position_embedding.weight [256, 768] as "
            "[16, 16, 768], resized per crop grid (float32; PyTorch's arm64 kernel fuses the multiply-add: emulate "
            "with float64 sums, host/d1_vision_host.py _resample_axis_f32); a 32 x 32 tile grid can be cached",
            "tower(pixels, pos, mask) -> features; keep the first ph * pw rows",
            "unshuffle: (ph, pw, 768) -> (ph/2, pw/2, 3072); channel j * 1536 + k * 768 + c of cell (r, q) = input "
            "(2r + j, 2q + k, c); cells raster order",
            "projector(soft) -> keep the first (ph/2)(pw/2) rows",
            "prefix = the crops' rows concatenated (tiles row-major, thumbnail last), P = sum (ph/2)(pw/2)",
            "decision graph: prefix at positions 0 .. P-1 of `prefix`, media = 1 there, keep_right = 0 at P - 1, "
            "image mode = no temperature, noul default yes / no, max_len min(896, 16384 - P)"],
        "P_formula": "sum over crops of (ph/2)(pw/2); a 512 x 512 tile = 256 rows; 384 x 384 = 144; 640 x 480 "
                     "(card_cats) = 234; 768 x 1024 (img_03) = 6 x 256 + 234 = 1,770",
        "limits": "a crop holds <= 1024 patches (<= 256 prefix rows); a picture is 1 crop or 2..10 tiles + thumbnail "
                  "(<= 2,816 rows)",
        "precision": "GPU = fp32 precision (GpuOptions(enforce_f32=True)) only: Metal default precision (fp16 "
                     "activations) gives features max |d| 23 vs the oracle on every crop (the residual stream reaches "
                     "848 from layer 5 on: layer-norm squares overflow fp16)",
        "verified_round_6": {
            "host_vs_provider_preprocess": {k: host[k] for k in ("records", "crops", "pixel_values_bit_equal_provider",
                                                                  "resize_uint8_bit_equal", "position_bit_equal_transformers",
                                                                  "unshuffle_bit_equal_provider", "pass")},
            "module_float64_identity": eag["summary"].get("float64_identity_all_records"),
            "fp32_floor": "the provider fp32 prefix itself is up to 0.127 (img_02) from its float64 value; our "
                          "export-form module with the provider's SDPA kernel is bit-equal per layer "
                          "(results/vision_eager_check.json fp32_floor)",
            "litert": par},
        "card_note_seed": "vision prefix: fp32 rounding moves it by up to 7e-2 from the provider's (the same function "
                          "in float64; massive activations amplify the rounding), and the decisions stay the same "
                          "(argmax 16/16, max |dp| 1.2e-5)",
    }
    replaced = "vision" in doc
    doc["vision"] = vision
    R.dump_json(path, doc, overwrite=True)
    print(json.dumps({"written": str(path.relative_to(K)), "replaced_existing_vision": replaced,
                      "keys": sorted(doc)}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--host-bits", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--eager-diag", action="store_true")
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--half-ab", action="store_true")
    ap.add_argument("--diag-ids", default="img_02,card_cats,img_01")
    ap.add_argument("--random-resizes", type=int, default=60)
    ap.add_argument("--lrt", action="store_true")
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--precision", choices=("fp32", "default"), default="fp32")
    ap.add_argument("--form", default="fp32")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--rescore", metavar="TAG")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--text-forms", default="fp16")
    ap.add_argument("--ms", action="store_true")
    ap.add_argument("--ms-child", action="store_true")
    ap.add_argument("--ms-aggregate", action="store_true")
    ap.add_argument("--forms", default="fp32,fp16")
    ap.add_argument("--contract-vision", action="store_true")
    a = ap.parse_args()
    if a.host:
        return host_check(a)
    if a.host_bits:
        return host_bits(a)
    if a.eager:
        return eager_check(a)
    if a.eager_diag:
        return eager_diag(a)
    if a.truth:
        return truth(a)
    if a.half_ab:
        return half_ab(a)
    if a.lrt:
        if a.backend == "gpu" and not a.child:
            return lrt_parent(a)
        return lrt_child(a)
    if a.rescore:
        print(json.dumps(lrt_score(a.rescore)["summary"], indent=1, default=str))
        return 0
    if a.e2e:
        return e2e(a)
    if a.ms_child:
        return ms_child(a)
    if a.ms:
        return ms_window(a)
    if a.ms_aggregate:
        return ms_aggregate(a)
    if a.contract_vision:
        return contract_vision(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
