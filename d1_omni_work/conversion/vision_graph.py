"""Round 6: the d1-omni vision prefix graphs (design 5), pure torch (no transformers: imports in the
reference venv and in the exporter venv), the checkpoint key map, and the LiteRT build.

    VisionTower(pixels f32 [1, 1024, 768], pos f32 [1, 1024, 768], mask f32 [1, 1024]) -> {"features": f32 [1, 1024, 768]}
    Projector(soft f32 [1, 256, 3072]) -> {"prefix": f32 [1, 256, 1024]}

VisionTower = the provider's Siglip2VisionModel (vision.py: Siglip2VisionConfig(**vision_config), sdpa, hidden 768,
intermediate 3072, 12 layers, 12 heads, patch 16, layer_norm_eps 1e-6, gelu_pytorch_tanh, vision_use_head false)
written out, with two changes for the export:
  - the position table is an input: the host resizes the 16 x 16 table to the crop's patch grid (bilinear antialias,
    host/d1_vision_host.py `resize_positions`), so the graph holds no RESIZE_BILINEAR / SELECT_V2;
    embeddings = patch_embedding(pixels) + pos (Siglip2VisionEmbeddings.forward without the resize);
  - attention: q / k / v [1, 12, N, 64] -> scores = rank-4 matmul * 64^-0.5 + (1 - mask)[:, None, None, :] * -1e4
    -> softmax in float32 -> matmul -> out_proj. transformers runs SDPA with the bidirectional padding mask (keys only);
    a masked key gets weight exactly 0 in both (exp(-1e4 + s) underflows to 0 in fp32), so real patches never see
    padding.
  Each encoder layer: x + out_proj(attn(layer_norm1(x))), then x + fc2(gelu_tanh(fc1(layer_norm2(x)))); the output is
  post_layernorm at every position (= last_hidden_state; the real patches are the first h * w rows).
Projector = the provider's Projector without its pixel unshuffle (the host does it, `pixel_unshuffle`, the same channel
order): linear_1 [2048, 3072] -> F.gelu (exact erf) -> linear_2 [1024, 2048]. The host writes the (h/2)(w/2) cells into
the first rows of `soft` and keeps the same rows of `prefix`.

Checkpoint (model.safetensors, fp32): 201 vision tensors = 197 under `vision.tower.vision_model.` + 4 under
`vision.projector.`. KEY_MAP: 200 go into the two modules, `embeddings.position_embedding.weight` [256, 768] goes to
the host (the table it resizes; [16, 16, 768] row-major = transformers' reshape).

    cd d1_omni_work
    ~/venvs/lt094dev/bin/python scripts/vision_graph.py --env                 # results/export_env.json key round_6
    ~/venvs/lt094dev/bin/python scripts/vision_graph.py --keymap              # results/vision_keymap.json
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/vision_graph.py --write tower      (and projector)
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/vision_graph.py --half tower       (and projector)
--write: litert_torch.signature(<sig>, module, sample_kwargs).convert().export(out/d1omni_<graph>_fp32.tflite), the
static scan (litert_run.scan) -> results/opscan_vision_<graph>_fp32.json with the acceptance checks (RESIZE_* /
SELECT_V2 / GATHER_ND / CUSTOM / INT64 = 0, rank <= 4), the signature -> results/signature_vision_<graph>.json, and a
CPU smoke run of the new file on the sample vs the eager module. --half: the FC-only fp16 form (ai-edge-quantizer
float_casting 16-bit FLOAT on FULLY_CONNECTED, round 3's recipe) -> out/d1omni_<graph>_fp16.tflite +
results/quant_vision.json. Never overwrites a file.

Round 12 (additions; every call above behaves as before, the default tower is unchanged):
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/vision_graph.py --range
        -> results/vision_f16safe_range.json (the 25 LayerNorm sites' (x - mean)^2 maxima on the 13 crops, eager fp32)
           + results/vision_f16safe_k_table.json (the per-site k)
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/vision_graph.py --write tower --variant f16safe
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/vision_graph.py --half tower --variant f16safe
  --variant f16safe (tower only): every LayerNorm site (layer_norm1 / layer_norm2 x 12 + post_layernorm) whose squared
    deviation can pass fp16's 65,504 takes its input x 2^-k and eps x 4^-k (d1_graph_f16safe.ScaledLayerNorm, the round
    8 form, the same function in fp32), k per site = the smallest k >= 0 with max (x - mean)^2 x 4^-k <= 65,504 / 4
    (margin >= 4x) over the 13 crops (every row of the 1,024, real and pad); sites within the margin keep the
    original module. Files: out/d1omni_vision_tower_f16safe_{fp32,fp16}.tflite,
    results/vision_tower_{opscan,signature}_f16safe*.json, results/vision_quant_f16safe.json.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import re
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))
import d1_src as S  # noqa: E402  (stdlib only; sets dont_write_bytecode)

NEG = -1e4
N_PATCHES, R_TOKENS = 1024, 256
HID, INTER, LAYERS, HEADS, EPS = 768, 3072, 12, 12, 1e-6
PATCH_IN = 3 * 16 * 16
PROJ_HIDDEN, D_TEXT = 2048, 1024
CKPT_TOWER = "vision.tower.vision_model."
CKPT_PROJ = "vision.projector."
POS_KEY = CKPT_TOWER + "embeddings.position_embedding.weight"
GRAPHS = {"tower": ("vision_tower", "d1omni_vision_tower"), "projector": ("projector", "d1omni_projector")}


# --------------------------------------------------------------------------- modules


class TowerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(HID, eps=EPS)
        self.q_proj = nn.Linear(HID, HID)
        self.k_proj = nn.Linear(HID, HID)
        self.v_proj = nn.Linear(HID, HID)
        self.out_proj = nn.Linear(HID, HID)
        self.layer_norm2 = nn.LayerNorm(HID, eps=EPS)
        self.fc1 = nn.Linear(HID, INTER)
        self.fc2 = nn.Linear(INTER, HID)
        self.head_dim = HID // HEADS
        self.scale = self.head_dim ** -0.5

    def attention(self, h, bias):
        b, n, _ = h.shape
        q = self.q_proj(h).view(b, n, HEADS, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, n, HEADS, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, n, HEADS, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale + bias
        # fp32 softmax (float64 only in the float64 identity check)
        p = torch.softmax(scores, dim=-1, dtype=torch.float64 if scores.dtype == torch.float64 else torch.float32)
        p = p.to(q.dtype)
        y = torch.matmul(p, v).transpose(1, 2).reshape(b, n, HID)
        return self.out_proj(y)

    def forward(self, x, bias):
        x = x + self.attention(self.layer_norm1(x), bias)
        return x + self.fc2(F.gelu(self.fc1(self.layer_norm2(x)), approximate="tanh"))


class VisionTower(nn.Module):
    """pixels [1, N, 768] + pos [1, N, 768] + mask [1, N] -> {"features": [1, N, 768]}."""

    def __init__(self, variant: str | None = None):
        super().__init__()
        assert variant in (None, "f16safe"), variant
        self.patch_embedding = nn.Linear(PATCH_IN, HID)
        self.layers = nn.ModuleList([TowerLayer() for _ in range(LAYERS)])
        self.post_layernorm = nn.LayerNorm(HID, eps=EPS)
        self.variant = variant
        self.f16safe_report = apply_f16safe(self, load_k_table()) if variant == "f16safe" else None

    def forward(self, pixels, pos, mask):
        x = self.patch_embedding(pixels) + pos
        bias = (1.0 - mask)[:, None, None, :] * NEG
        for layer in self.layers:
            x = layer(x, bias)
        return {"features": self.post_layernorm(x)}


class Projector(nn.Module):
    """soft [1, R, 3072] (unshuffled on the host) -> {"prefix": [1, R, 1024]}."""

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(4 * HID, PROJ_HIDDEN)
        self.linear_2 = nn.Linear(PROJ_HIDDEN, D_TEXT)

    def forward(self, soft):
        return {"prefix": self.linear_2(F.gelu(self.linear_1(soft)))}


# --------------------------------------------------------------------------- round 12: the fp16-safe LayerNorm variant

FP16_MAX = 65504.0
K_MARGIN = 4.0
RANGE_FILE = K / "results/vision_f16safe_range.json"
K_TABLE_FILE = K / "results/vision_f16safe_k_table.json"


def norm_sites(tower) -> dict:
    """site name -> (parent module, attribute): layer_norm1 / layer_norm2 of the 12 layers + post_layernorm = 25."""
    out = {}
    for i, layer in enumerate(tower.layers):
        out[f"L{i:02d}.layer_norm1"] = (layer, "layer_norm1")
        out[f"L{i:02d}.layer_norm2"] = (layer, "layer_norm2")
    out["post_layernorm"] = (tower, "post_layernorm")
    return out


def k_for(max_sq: float) -> int:
    k = 0
    while max_sq * 4.0 ** -k > FP16_MAX / K_MARGIN:
        k += 1
    return k


def load_k_table(path=K_TABLE_FILE) -> dict:
    return {name: int(v["k"]) for name, v in json.loads(Path(path).read_text())["sites"].items()}


def apply_f16safe(tower, k_table: dict) -> dict:
    """Swap every LayerNorm with k > 0 for d1_graph_f16safe.ScaledLayerNorm (before the weights are loaded: the
    parameter keys are unchanged). Every site must be in the table and the reverse."""
    import d1_graph_f16safe as F16

    sites = norm_sites(tower)
    assert set(sites) == set(k_table), sorted(set(sites) ^ set(k_table))
    scaled = {}
    for name, (parent, attr) in sites.items():
        old = getattr(parent, attr)
        k = int(k_table[name])
        assert type(old) is nn.LayerNorm and old.elementwise_affine and len(old.normalized_shape) == 1, name
        if k == 0:
            continue
        new = F16.ScaledLayerNorm(old.normalized_shape[0], old.eps, k)
        new.load_state_dict(old.state_dict())
        setattr(parent, attr, new)
        scaled[name] = k
    return {"sites": len(sites), "scaled": len(scaled), "k_by_site": scaled,
            "unchanged": sorted(n for n in sites if n not in scaled)}


def range_probe() -> int:
    """The 25 LayerNorm sites' inputs on the 13 crops of the 7 image records (host inputs, eager fp32 tower with the
    checkpoint): per site the max of the element (x - mean)^2, the row variance max / min, max |x - mean|, max |x|,
    over the real patch rows and over all 1,024 rows -> results/vision_f16safe_range.json; then the k table."""
    import hashlib

    sys.path.insert(0, str(K / "host"))
    import d1_vision_host as V
    import litert_run as R
    import vision_check as VC

    for p in (RANGE_FILE, K_TABLE_FILE):
        assert not p.exists(), f"refusing to overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    tower, _, table, rep = load_vision()
    sites = norm_sites(tower)
    stats = {n: {"real": {"sq_dev_max": 0.0, "var_max": 0.0, "var_min": float("inf"), "dev_absmax": 0.0, "x_absmax": 0.0},
                 "all": {"sq_dev_max": 0.0, "var_max": 0.0, "var_min": float("inf"), "dev_absmax": 0.0, "x_absmax": 0.0},
                 "worst": None} for n in sites}
    cur = {}

    def hook(name):
        def f(mod, args):
            x = args[0].detach().double()[0]                     # [1024, 768]
            mean = x.mean(-1, keepdim=True)
            dev = x - mean
            sq = dev * dev
            var = sq.mean(-1)
            n = cur["n"]
            for part, sl in (("real", slice(0, n)), ("all", slice(0, x.shape[0]))):
                s = stats[name][part]
                m = float(sq[sl].max())
                if part == "real" and m > s["sq_dev_max"]:
                    stats[name]["worst"] = {"crop": cur["key"], "row": int(sq[sl].max(-1).values.argmax())}
                s["sq_dev_max"] = max(s["sq_dev_max"], m)
                s["var_max"] = max(s["var_max"], float(var[sl].max()))
                s["var_min"] = min(s["var_min"], float(var[sl].min()))
                s["dev_absmax"] = max(s["dev_absmax"], float(dev[sl].abs().max()))
                s["x_absmax"] = max(s["x_absmax"], float(x[sl].abs().max()))
        return f

    handles = [getattr(p, a).register_forward_pre_hook(hook(n)) for n, (p, a) in sites.items()]
    recs, _ = VC.image_records()
    crops_seen = []
    with torch.no_grad():
        for e, rec in recs:
            crops, _ = V.crops_of(V.load_image(VC.image_path(rec)))
            for i, c in enumerate(crops):
                crop = V.to_patches(c)
                h, w = crop["grid"]
                cur.update(n=h * w, key=f"{e['id']}/c{i}")
                x = V.tower_inputs(crop, table)
                out = tower(**{k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()})["features"]
                assert bool(torch.isfinite(out).all())
                crops_seen.append({"crop": cur["key"], "grid": [h, w], "patches": h * w})
    for hd in handles:
        hd.remove()
    out = {}
    for n, (p, a) in sites.items():
        s = stats[n]
        out[n] = {"dim": HID, "eps": EPS, **{f"{part}_{k}": v for part in ("real", "all") for k, v in s[part].items()},
                  "worst_real": s["worst"], "overflows_fp16_before": s["all"]["sq_dev_max"] > FP16_MAX}
    doc = {"step": "round 12 step 2: the vision tower's LayerNorm input range on the 13 crops (eager fp32, checkpoint, "
                   "host inputs; float64 statistics of the fp32 activations)",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "crops": crops_seen, "load": rep,
           "torch": torch.__version__, "python": sys.version.split()[0], "seconds": round(time.time() - t0, 1),
           "fp16_max": FP16_MAX, "sites": out,
           "summary": {"sites": len(out), "sites_over_fp16_max": sorted(n for n, v in out.items() if v["overflows_fp16_before"]),
                       "sq_dev_max": max(v["all_sq_dev_max"] for v in out.values()),
                       "var_max": max(v["all_var_max"] for v in out.values()),
                       "var_min_real": min(v["real_var_min"] for v in out.values())}}
    R.dump_json(RANGE_FILE, doc)
    kt, hist = {}, {}
    for n, v in out.items():
        mx = v["all_sq_dev_max"]
        k = k_for(mx)
        hist[k] = hist.get(k, 0) + 1
        kt[n] = {"kind": "layernorm", "dim": HID, "eps": EPS, "k": k, "scale_s": 2.0 ** -k, "eps_k": EPS * 4.0 ** -k,
                 "sq_dev_max": mx, "sq_dev_max_real": v["real_sq_dev_max"], "overflows_fp16_before": mx > FP16_MAX,
                 "sq_dev_max_after_k": mx * 4.0 ** -k, "margin_after_k": FP16_MAX / (mx * 4.0 ** -k),
                 "variance_max": v["all_var_max"], "variance_min_real": v["real_var_min"],
                 "variance_min_real_after_k": v["real_var_min"] * 4.0 ** -k,
                 "variance_min_after_k_below_fp16_min_normal": v["real_var_min"] * 4.0 ** -k < 2.0 ** -14,
                 # observation only (not the rule): the row's sum of squared deviations an fp16 accumulator would hold
                 "sum_sq_dev_max_after_k": v["all_var_max"] * HID * 4.0 ** -k,
                 "sum_after_k_over_fp16_max": v["all_var_max"] * HID * 4.0 ** -k > FP16_MAX}
    mins = min(kt.items(), key=lambda kv: kv[1]["margin_after_k"])
    kdoc = {"step": "round 12 step 2: the per-site k of the vision tower's fp16-safe LayerNorm (input x 2^-k, eps x 4^-k)",
            "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": {"file": str(RANGE_FILE.relative_to(K)), "sha256": hashlib.sha256(RANGE_FILE.read_bytes()).hexdigest()},
            "rule": "k = the smallest k >= 0 with max (x - mean)^2 (the element; the 13 crops, all 1,024 rows) x 4^-k "
                    "<= 65,504 / 4 (margin >= 4x); a site within the margin keeps k = 0 and its original module",
            "summary": {"sites": len(kt), "sites_k_gt_0": sum(v["k"] > 0 for v in kt.values()),
                        "k_histogram": {str(k): n for k, n in sorted(hist.items())},
                        "sites_overflowing_before": sorted(n for n, v in kt.items() if v["overflows_fp16_before"]),
                        "min_margin_after_k": mins[1]["margin_after_k"], "min_margin_site": mins[0],
                        "all_margins_ge_4_after_k": all(v["margin_after_k"] >= K_MARGIN for v in kt.values()),
                        "sites_variance_min_after_k_below_fp16_min_normal": sorted(
                            n for n, v in kt.items() if v["variance_min_after_k_below_fp16_min_normal"]),
                        "sites_sum_after_k_over_fp16_max_observation": sorted(
                            n for n, v in kt.items() if v["sum_after_k_over_fp16_max"])},
            "sites": kt}
    R.dump_json(K_TABLE_FILE, kdoc)
    print(json.dumps({"range": doc["summary"], "k": kdoc["summary"], "seconds": doc["seconds"]}, indent=1))
    return 0 if kdoc["summary"]["all_margins_ge_4_after_k"] else 1


# --------------------------------------------------------------------------- key map and weights


def key_map() -> dict:
    """checkpoint key -> (part, our key): part = tower | projector | host."""
    out = {}
    out[CKPT_TOWER + "embeddings.patch_embedding.weight"] = ("tower", "patch_embedding.weight")
    out[CKPT_TOWER + "embeddings.patch_embedding.bias"] = ("tower", "patch_embedding.bias")
    out[POS_KEY] = ("host", "position_table")
    for i in range(LAYERS):
        src = f"{CKPT_TOWER}encoder.layers.{i}."
        dst = f"layers.{i}."
        for a, b in (("layer_norm1", "layer_norm1"), ("layer_norm2", "layer_norm2"), ("self_attn.q_proj", "q_proj"),
                     ("self_attn.k_proj", "k_proj"), ("self_attn.v_proj", "v_proj"),
                     ("self_attn.out_proj", "out_proj"), ("mlp.fc1", "fc1"), ("mlp.fc2", "fc2")):
            for p in ("weight", "bias"):
                out[f"{src}{a}.{p}"] = ("tower", f"{dst}{b}.{p}")
    for p in ("weight", "bias"):
        out[CKPT_TOWER + f"post_layernorm.{p}"] = ("tower", f"post_layernorm.{p}")
        for n in ("linear_1", "linear_2"):
            out[CKPT_PROJ + f"{n}.{p}"] = ("projector", f"{n}.{p}")
    return out


def load_vision(path=None, dtype=torch.float32, variant=None):
    """-> (VisionTower, Projector, position table [16, 16, 768] float32 numpy, report). Strict: every module parameter
    set from the checkpoint, every vision key of the checkpoint used, shapes equal. variant (round 12): the tower's."""
    from safetensors import safe_open

    path = Path(path or S.WEIGHTS)
    km = key_map()
    tower, proj = VisionTower(variant), Projector()
    want = {"tower": tower.state_dict(), "projector": proj.state_dict()}
    parts = {"tower": {}, "projector": {}}
    table = None
    with safe_open(str(path), framework="pt") as f:
        present = {k for k in f.keys() if k.startswith("vision.")}
        assert present == set(km), (sorted(present - set(km))[:5], sorted(set(km) - present)[:5])
        for ck, (part, mk) in km.items():
            t = f.get_tensor(ck)
            assert t.dtype == torch.float32, (ck, t.dtype)
            if part == "host":
                assert list(t.shape) == [256, HID], t.shape
                table = t.numpy().reshape(16, 16, HID).astype(np.float32).copy()
                continue
            assert list(t.shape) == list(want[part][mk].shape), (ck, t.shape, want[part][mk].shape)
            parts[part][mk] = t
    tower.load_state_dict(parts["tower"], strict=True)
    proj.load_state_dict(parts["projector"], strict=True)
    tower = tower.to(dtype).eval().requires_grad_(False)
    proj = proj.to(dtype).eval().requires_grad_(False)
    rep = {"file": str(path), "tensors_vision": len(km), "tower_params": len(parts["tower"]),
           "projector_params": len(parts["projector"]), "host_tensors": 1,
           "tower_param_count": int(sum(p.numel() for p in tower.parameters())),
           "projector_param_count": int(sum(p.numel() for p in proj.parameters()))}
    return tower, proj, table, rep


def keymap_record() -> dict:
    """results/vision_keymap.json: the 201 keys, their shapes and targets, read with safe_open (no tensor data read
    beyond the shapes, then one strict load)."""
    from safetensors import safe_open

    km = key_map()
    tower, proj = VisionTower(), Projector()
    sds = {"tower": tower.state_dict(), "projector": proj.state_dict()}
    rows, bad, extra_in_file = [], [], []
    with safe_open(str(S.WEIGHTS), framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith("vision.")]
        for k in keys:
            if k not in km:
                extra_in_file.append(k)
                continue
            sl = f.get_slice(k)
            shape, dtype = list(sl.get_shape()), sl.get_dtype()
            part, mk = km[k]
            want = [256, HID] if part == "host" else list(sds[part][mk].shape)
            ok = shape == want
            if not ok:
                bad.append(k)
            rows.append({"checkpoint": k, "shape": shape, "dtype": dtype, "part": part, "ours": mk, "shape_ok": ok})
    missing_in_file = sorted(set(km) - set(keys))
    module_keys = {(p, k) for p, sd in sds.items() for k in sd}
    mapped = {(p, k) for p, k in km.values() if p != "host"}
    _, _, table, rep = load_vision()
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "file": str(S.WEIGHTS), "revision": S.REV,
           "checkpoint_vision_tensors": len(keys), "mapped": len(rows), "by_part": {
               p: sum(r["part"] == p for r in rows) for p in ("tower", "projector", "host")},
           "missing_in_checkpoint": missing_in_file, "unmapped_checkpoint_keys": extra_in_file,
           "module_params_without_source": sorted(f"{p}:{k}" for p, k in module_keys - mapped),
           "shape_mismatch": bad, "strict_load": rep, "position_table_shape": list(table.shape),
           "rename_note": "transformers 5.19 renames vision.tower.vision_model.* -> vision.tower.* at load; this map "
                          "goes from the file's names straight to our modules", "rows": rows}
    doc["pass"] = (not missing_in_file and not extra_in_file and not bad and not doc["module_params_without_source"]
                   and len(rows) == 201 and doc["by_part"] == {"tower": 196, "projector": 4, "host": 1})
    return doc


# --------------------------------------------------------------------------- env


def _version(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def env_append(key="round_6"):
    import subprocess

    path = K / "results/export_env.json"
    doc = json.loads(path.read_text())
    if key in doc:
        print(f"{key} already in {path.name}; not rewritten")
        return 0
    ref_py = K / "venv-ref/bin/python"
    ref = subprocess.run([str(ref_py), "-I", "-c", "import json, importlib.metadata as m, sys, torch\n"
                          "print(json.dumps({'python': sys.version.split()[0], 'packages': {p: m.version(p) for p in "
                          "('torch', 'transformers', 'torchvision', 'numpy', 'pillow', 'safetensors')}, "
                          "'torch_cpu_capability': torch.backends.cpu.get_cpu_capability()}))"],
                         capture_output=True, text=True, check=True)
    rec = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "step": "round 6 step 0 (vision prefix graph)",
           "exporter_runner": {"venv": "~/venvs/lt094dev", "python": sys.version.split()[0],
                               "executable": sys.executable, "platform": platform.platform(),
                               "packages": {p: _version(p) for p in (
                                   "litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "torch",
                                   "numpy", "pillow", "flatbuffers", "safetensors", "transformers")},
                               "torchvision": _version("torchvision"),
                               "torch_cpu_capability": torch.backends.cpu.get_cpu_capability()},
           "reference": {"venv": "K/venv-ref", **json.loads(ref.stdout)},
           "hf_modules_cache": "cache/hf_modules_r6",
           "weights": {"path": str(S.WEIGHTS), "resolved": str(S.WEIGHTS.resolve()), "bytes": S.WEIGHTS.stat().st_size,
                       "read": "safe_open, the 201 vision.* tensors only"},
           "note": "lt094dev and venv-ref are run only; nothing installed by this round. lt094dev has no torchvision: "
                   "the provider's preprocess() / Vision run in venv-ref, the graphs and CompiledModel in lt094dev."}
    doc[key] = rec
    import litert_run as R

    R.dump_json(path, doc, overwrite=True)
    print(json.dumps(rec, indent=1))
    return 0


# --------------------------------------------------------------------------- build


def sample_inputs(kind: str) -> dict:
    """Seed-fixed inputs of the graph's shapes: a crop of 24 x 24 = 576 real patches (img_01's grid); the graph has no
    data-dependent branch, so the values only feed the trace and the smoke run."""
    g = torch.Generator().manual_seed(7)
    if kind == "tower":
        mask = torch.zeros(1, N_PATCHES)
        mask[0, :576] = 1.0
        pixels = torch.rand(1, N_PATCHES, PATCH_IN, generator=g) * 2 - 1
        pixels[0, 576:] = 0.0
        return {"pixels": pixels, "pos": torch.randn(1, N_PATCHES, HID, generator=g) * 0.5, "mask": mask}
    soft = torch.randn(1, R_TOKENS, 4 * HID, generator=g)
    soft[0, 144:] = 0.0
    return {"soft": soft}


def out_path(kind, form, variant=None):
    return K / (f"out/{GRAPHS[kind][1]}_{variant}_{form}.tflite" if variant else f"out/{GRAPHS[kind][1]}_{form}.tflite")


def checks_of(sc, kind):
    hist = sc["op_histogram"]
    sig = sc["signatures"]
    if kind == "tower":
        want_in = sorted([("mask", [1, N_PATCHES], "FLOAT32"), ("pixels", [1, N_PATCHES, PATCH_IN], "FLOAT32"),
                          ("pos", [1, N_PATCHES, HID], "FLOAT32")])
        want_out = [("features", [1, N_PATCHES, HID], "FLOAT32")]
    else:
        want_in = [("soft", [1, R_TOKENS, 4 * HID], "FLOAT32")]
        want_out = [("prefix", [1, R_TOKENS, D_TEXT], "FLOAT32")]
    return {
        "one_signature_named": len(sig) == 1 and sig[0]["key"] == GRAPHS[kind][0],
        "signature_io": len(sig) == 1 and sorted((i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]) == want_in
        and [(o["name"], o["shape"], o["dtype"]) for o in sig[0]["outputs"]] == want_out,
        "RESIZE_0": sum(v for k, v in hist.items() if k.startswith("RESIZE")) == 0,
        "SELECT_V2_0": hist.get("SELECT_V2", 0) == 0 and hist.get("SELECT", 0) == 0,
        "GATHER_ND_0": hist.get("GATHER_ND", 0) == 0 and hist.get("GATHER", 0) == 0,
        "CUSTOM_0": sc["custom_op_count"] == 0,
        "INT64_0": sc["int64_tensor_count"] == 0,
        "BROADCAST_TO_0": hist.get("BROADCAST_TO", 0) == 0,
        "max_tensor_rank_le_4": sc["max_tensor_rank"] <= 4,
    }


def write_tflite(kind: str, variant=None) -> int:
    import litert_run as R

    sig, stem = GRAPHS[kind]
    path = out_path(kind, "fp32", variant)
    if variant:   # round 12: the variant's own files; round 6's are never touched
        assert kind == "tower", "the variant is the tower's"
        res = K / f"results/vision_{kind}_opscan_{variant}_fp32.json"
        sig_res = K / f"results/vision_{kind}_signature_{variant}.json"
    else:
        res = K / f"results/opscan_vision_{kind}_fp32.json"
        sig_res = K / f"results/signature_vision_{kind}.json"
    for p in (path, res, sig_res):
        assert not p.exists(), f"refusing to overwrite {p}"
    torch.set_num_threads(8)
    t0 = time.time()
    tower, proj, table, rep = load_vision(variant=variant)
    module = tower if kind == "tower" else proj
    sample = sample_inputs(kind)
    with torch.no_grad():
        ref = module(**sample)
    outname = next(iter(ref))
    extra = {}
    if variant:
        with torch.no_grad():   # the variant's function vs the default tower on the trace sample (fp32 eager)
            base, _, _, _ = load_vision()
            ref0 = base(**sample)[outname].numpy()
            del base
        got0 = ref[outname].numpy()
        extra = {"variant": variant, "f16safe_report": tower.f16safe_report,
                 "eager_vs_default_module": {"max_abs_all_rows": float(np.abs(got0.astype(np.float64) - ref0).max()),
                                             "bit_equal_all_rows": bool(np.array_equal(got0, ref0))}}
    rec = {"graph": type(module).__name__, "signature": sig, "file": str(path.relative_to(K)), "load": rep, **extra,
           "sample_kwargs": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                             for k, v in sample.items()},
           "convert_call": f"litert_torch.signature('{sig}', module, sample_kwargs=...).convert().export(path)",
           "versions": {p: _version(p) for p in ("torch", "litert-torch", "litert-converter", "ai-edge-litert")},
           "python": sys.version.split()[0], "load_seconds": round(time.time() - t0, 1)}
    try:
        import litert_torch

        t1 = time.perf_counter()
        edge = litert_torch.signature(sig, module, sample_kwargs=sample).convert()
        rec["convert_seconds"] = round(time.perf_counter() - t1, 1)
        t2 = time.perf_counter()
        edge.export(str(path))
        rec["write_seconds"] = round(time.perf_counter() - t2, 1)
        del edge
    except BaseException:
        (K / (f"logs/r12_write_{kind}_{variant}.traceback.txt" if variant else f"logs/r6_write_{kind}.traceback.txt")
         ).write_text(traceback.format_exc())
        raise
    rec["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    sc = R.scan(path)
    checks = checks_of(sc, kind)
    # CPU smoke run of the new file vs the eager module on the sample
    cm, desc = R.open_compiled(path, "cpu", threads=8)
    sname = next(iter(cm.get_signature_list()))
    ins = cm.get_input_tensor_details(sname)
    outs = cm.get_output_tensor_details(sname)
    ib = {n: cm.create_input_buffer_by_name(sname, n) for n in ins}
    ob = {n: cm.create_output_buffer_by_name(sname, n) for n in outs}
    for n, v in sample.items():
        ib[n].write(np.ascontiguousarray(v.numpy(), np.float32))
    cm.run_by_name(sname, ib, ob)
    o_shape = [int(x) for x in outs[outname]["shape"]]
    got = np.asarray(ob[outname].read(int(np.prod(o_shape)), np.float32)).reshape(o_shape)
    want = ref[outname].numpy()
    real = 576 if kind == "tower" else 144
    smoke = {"rows_compared": real, "max_abs_real_vs_eager": float(np.abs(got[0, :real].astype(np.float64)
                                                                           - want[0, :real]).max()),
             "max_abs_eager_real": float(np.abs(want[0, :real]).max()), "finite": bool(np.isfinite(got).all())}
    for b in list(ib.values()) + list(ob.values()):
        b.destroy()
    doc = {"step": (f"round 12: {kind} variant {variant} fp32 export" if variant else f"round 6 step 3: {kind} fp32 export"),
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "export": rec, "checks": checks, "checks_pass": all(checks.values()), "smoke_cpu_vs_eager": smoke,
           "bytes": sc["bytes"], "sha256": sc["sha256"], "scan": sc, "seconds_total": round(time.time() - t0, 1)}
    R.dump_json(res, doc)
    sig_doc = {"step": f"round {12 if variant else 6} step 3: signature of {path.name}", "signature": sig,
               "flatbuffer_signatures": sc["signatures"],
               "compiled_model_inputs": {n: {k: str(v) for k, v in d.items()} for n, d in ins.items()},
               "compiled_model_outputs": {n: {k: str(v) for k, v in d.items()} for n, d in outs.items()}}
    R.dump_json(sig_res, sig_doc)
    print(json.dumps({k: doc[k] for k in ("bytes", "sha256", "checks", "checks_pass", "smoke_cpu_vs_eager",
                                          "seconds_total")} | {"op_histogram": sc["op_histogram"],
                                                                "convert_seconds": rec["convert_seconds"]},
                     indent=1), flush=True)
    return 0 if doc["checks_pass"] and smoke["finite"] else 1


def half_form(kind: str, variant=None) -> int:
    """The FC-only fp16 form: round 3's recipe (scripts/quant_forms.py recipe_for('fp16'))."""
    import litert_run as R
    from ai_edge_quantizer import qtyping, quantizer, recipe_manager

    src, dst = out_path(kind, "fp32", variant), out_path(kind, "fp16", variant)
    res = K / (f"results/vision_quant_{variant}.json" if variant else "results/quant_vision.json")
    doc = json.loads(res.read_text()) if res.exists() else {
        "step": f"round 12: FC-only fp16 forms of the {variant} variant" if variant else "round 6 step 3: FC-only fp16 forms",
        "forms": {}}
    if kind in doc["forms"]:
        print(f"{kind}: already in {res.name}")
        return 0
    assert not dst.exists(), f"refusing to overwrite {dst}"
    OP = qtyping.TFLOperationName
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=OP.FULLY_CONNECTED, algorithm_key="float_casting",
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT))
    recipe = rm.get_quantization_recipe()
    sc_src = R.scan(src)
    n_fc = sc_src["fully_connected_count"]
    t0 = time.time()
    qt = quantizer.Quantizer(str(src), recipe)
    assert not (qt.need_calibration or rm.need_calibration()), "recipe needs calibration"
    result = qt.quantize()
    result.export_model(str(dst))
    secs = round(time.time() - t0, 1)
    sc = R.scan(dst)
    checks = {
        f"fc_count_{n_fc}": sc["fully_connected_count"] == n_fc,
        "fc_weight_all_FLOAT16": sc["fc_weight_source_dtype"] == {"FLOAT16": n_fc},
        f"dequantize_{n_fc}_FLOAT16_to_FLOAT32": sc["dequantize_in_out"] == {"FLOAT16->FLOAT32": n_fc},
        "op_histogram_unchanged_except_DEQUANTIZE": {k: v for k, v in sc["op_histogram"].items() if k != "DEQUANTIZE"}
        == {k: v for k, v in sc_src["op_histogram"].items() if k != "DEQUANTIZE"},
        "signature_unchanged": sc["signatures"] == sc_src["signatures"],
        **{k: v for k, v in checks_of(sc, kind).items() if k != "signature_io"},
    }
    doc["forms"][kind] = {"input": {"file": str(src.relative_to(K)), "bytes": src.stat().st_size,
                                    "sha256": sc_src["sha256"], "fully_connected": n_fc,
                                    "constant_bytes_by_dtype": sc_src["constant_bytes_by_dtype"]},
                          "output": str(dst.relative_to(K)), "recipe": recipe, "seconds": secs,
                          "bytes": sc["bytes"], "sha256": sc["sha256"], "operator_count": sc["operator_count"],
                          "op_histogram": sc["op_histogram"], "tensor_dtype_histogram": sc["tensor_dtype_histogram"],
                          "constant_bytes_by_dtype": sc["constant_bytes_by_dtype"],
                          "fc_weight_source_dtype": sc["fc_weight_source_dtype"], "fc_bias_dtype": sc["fc_bias_dtype"],
                          "dequantize_in_out": sc["dequantize_in_out"], "checks": checks,
                          "checks_pass": all(checks.values())}
    doc["versions"] = {p: _version(p) for p in ("ai-edge-quantizer", "ai-edge-litert")}
    doc["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    R.dump_json(res, doc, overwrite=True)
    print(json.dumps({k: doc["forms"][kind][k] for k in ("bytes", "op_histogram", "checks", "checks_pass", "seconds")},
                     indent=1))
    return 0 if all(checks.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--keymap", action="store_true")
    ap.add_argument("--write", choices=sorted(GRAPHS))
    ap.add_argument("--half", choices=sorted(GRAPHS))
    ap.add_argument("--variant", choices=("f16safe",), help="round 12: --write / --half tower of the variant")
    ap.add_argument("--range", action="store_true", help="round 12: results/vision_f16safe_range.json + k table")
    a = ap.parse_args()
    if a.range:
        return range_probe()
    if a.variant and (a.write or a.half):
        return write_tflite(a.write, a.variant) if a.write else half_form(a.half, a.variant)
    if a.env:
        return env_append()
    if a.keymap:
        import litert_run as R

        doc = keymap_record()
        R.dump_json(K / "results/vision_keymap.json", doc)
        print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))
        return 0 if doc["pass"] else 1
    if a.write:
        return write_tflite(a.write)
    if a.half:
        return half_form(a.half)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
