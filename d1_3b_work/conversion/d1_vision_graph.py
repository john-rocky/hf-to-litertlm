"""Round 3: the picture graphs of the d1-3B LiteRT conversion, the tiny random VL model, and the weight loaders.

    VisionTower(pixels f32 [1, N, 768], pos f32 [1, N, H], mask f32 [1, N]) -> {"features": f32 [1, N, H]}   N = 1024
    Projector(soft f32 [1, R, 4H]) -> {"mm": f32 [1, R, d]}                                               R = 256

VisionTower = transformers 5.14.1's Siglip2VisionModel modules (the checkpoint's own: patch_embedding Linear, the
encoder layers, post_layernorm; `vision_use_head` is false on d1-3B), with two changes for the export:
  - the position table is an input: the host resizes the 16 x 16 table to the tile's patch grid (bilinear antialias,
    host/d1_vision.py `resize_positions`) and pads it to N rows, so the graph holds no RESIZE_BILINEAR / SELECT_V2;
    embeddings = patch_embedding(pixels) + pos (Siglip2VisionEmbeddings.forward without the resize);
  - attention (`TowerAttention`, swapped onto the Siglip2Attention instances; same projections and parameter names):
    q/k/v [1, heads, N, head_dim] -> scores = rank-4 BMM * scale + (1 - mask)[:, None, None, :] * -1e4 -> softmax in
    float32 -> BMM -> out_proj. transformers runs SDPA / eager with the processor's mask (padding keys masked); a
    masked key gets weight exactly 0 in both, so the real patches do not see the padding.
The encoder layers (layer norms, residuals, the gelu-tanh MLP) and post_layernorm run transformers' own forward.
features = post_layernorm output at every position (`last_hidden_state`); the real patches are the first h * w rows.

Projector = Lfm2VlMultiModalProjector without its pixel_unshuffle (the host does it, host/d1_vision.py
`pixel_unshuffle`, the same channel order): linear_1 -> gelu (erf) -> linear_2; projector_use_layernorm is false on
d1-3B (the layer norm would run first if it were set). The host writes the (h/2)(w/2) unshuffled cells into the
first rows of `soft` and keeps the same rows of `mm`.

Tiny VL model (round 3): config.json with
  text  = round 2's tiny text_config (hidden 64, 6 layers, 4 / 2 heads; d1_prefill_graph.TINY_OVERRIDES) but the
          checkpoint's vocab_size (128000): the processor writes real token ids (<image> = 124907, ...), so the table
          needs those rows; the text layers are round 2's tiny weights bit for bit, the table's first 256 rows are
          round 2's table and the other rows N(0, 1);
  vision = siglip2_vision_model hidden 32, intermediate 64, 2 layers, 2 heads (everything else the checkpoint's:
          patch 16, num_patches 256, gelu_pytorch_tanh, layer_norm_eps 1e-6, vision_use_head false);
  projector_hidden_size 64 (projector_use_layernorm false, downsample_factor 2, image_token_id 124907, gelu, bias).
Weights (seed 0, sorted names): layer-norm weights 1 + 0.1 N(0,1), every bias 0.1 N(0,1), the position table
N(0,1), other matrices N(0, 1/fan_in). Saved under the checkpoint's names (vision under
`model.vision_tower.vision_model.`, as in LiquidAI/d1-3B's model.safetensors), so the tiny file and the real file go
through the same loader and the same rename to transformers 5.14.1's module names (`vision_model.` dropped; round 1).

    $EXPORT scripts/d1_vision_graph.py --make-tiny
    $EXPORT scripts/d1_vision_graph.py --check-header           # 437 + 4 keys vs Hub
    $EXPORT scripts/d1_vision_graph.py --source tiny --tag tiny --write tower
    $EXPORT scripts/d1_vision_graph.py --source tiny --tag tiny --write projector
--write: litert_torch.convert(module, sample_kwargs=...) -> exports/{tag}_vision_tower_fp32.tflite (or
{tag}_projector_fp32.tflite), then the static scan (tflite_scan.py) -> results/{tag}_vision_tower_tflite.json; never
overwrites. --source = tiny | a snapshot directory holding LiquidAI/d1-3B's model.safetensors (bfloat16 -> float32).

Round 6d (the real weights): `Out` names what a step writes. Without --rtag everything is named as in round 3. With
--rtag the exports keep the {tag} prefix (exports/real_vision_tower_fp32.tflite) while results take {rtag}
(results/realv_vision_tower_fp32_tflite.json), the per-run cache goes to cache/{rtag}/ and the logs to
logs/{log_prefix}{rtag}_*; --root (default K) moves all of it, so the driver's rerun writes a second copy under
cache/realv/run/ and its bytes can be compared with the first.

    $EXPORT scripts/d1_vision_graph.py \
        --source <snapshot> --tag real --rtag realv --log-prefix r6d_ --write tower
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_prefill_graph as G  # noqa: E402
from d1_common import HF_SMALL, K, provider, sha256_file  # noqa: E402

NEG = -1e4
N_PATCHES = 1024
R_TOKENS = 256
LM_PREFIX = G.PREFIX                                   # model.language_model.
CKPT_VISION = "model.vision_tower.vision_model."      # the checkpoint's names
MODULE_VISION = "model.vision_tower."                 # transformers 5.14.1's module names
PROJ = "model.multi_modal_projector."
VL_DIR = K / "cache/vision"
TINY_SEED = 0
TINY_VL_WEIGHTS = VL_DIR / f"tiny_vl_seed{TINY_SEED}.safetensors"
TINY_VISION = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2}
TINY_PROJECTOR_HIDDEN = 64
GRAPHS = {"tower": "vision_tower", "projector": "projector"}


class Out:
    """Where a step writes. Round 3 (rtag None): exports/{tag}_*, results/{tag}_*, cache/vision/, logs/{tag}_*.
    Round 6d (rtag given): exports/{tag}_*, results/{rtag}_*, cache/{rtag}/, logs/{log_prefix}{rtag}_*. `root`
    (K-relative or absolute, default K) holds all four folders."""

    def __init__(self, tag: str, rtag: str | None = None, root: str | None = None, log_prefix: str = ""):
        self.tag, self.rtag, self.legacy, self.log_prefix = tag, rtag or tag, rtag is None, log_prefix
        r = Path(root) if root else K
        self.root = r if r.is_absolute() else K / r

    def rname(self, stem: str) -> str:
        """An export stem ({tag}_...) -> the name its results and logs take ({rtag}_...)."""
        assert stem == self.tag or stem.startswith(self.tag + "_"), (stem, self.tag)
        return self.rtag + stem[len(self.tag):]

    def export(self, stem: str) -> Path:
        return self.root / "exports" / f"{stem}.tflite"

    def result(self, name: str) -> Path:
        return self.root / "results" / name

    def cache(self, name: str) -> Path:
        return (self.root / "cache/vision" if self.legacy else self.root / "cache" / self.rtag) / name

    def log(self, name: str) -> Path:
        return self.root / "logs" / f"{self.log_prefix}{name}"

    def rel(self, p: Path) -> str:
        return str(p.relative_to(K)) if p.is_relative_to(K) else str(p)

    def mkdirs(self) -> "Out":
        for p in (self.export("x").parent, self.result("x").parent, self.cache("x").parent, self.log("x").parent):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict:
        return {"tag": self.tag, "rtag": self.rtag, "root": self.rel(self.root) if self.root != K else ".",
                "log_prefix": self.log_prefix, "legacy_round3_names": self.legacy}

    @staticmethod
    def add_args(ap, tag_default: str | None = "tiny") -> None:
        ap.add_argument("--tag", default=tag_default, help="export prefix (exports/{tag}_*)")
        ap.add_argument("--rtag", default=None, help="results / cache prefix (round 6d: realv); none = round 3 names")
        ap.add_argument("--root", default=None, help="output root, K-relative or absolute (default K)")
        ap.add_argument("--log-prefix", default="", help="prefix of the log files (round 6d: r6d_)")

    @classmethod
    def from_args(cls, a, tag: str | None = None) -> "Out":
        return cls(tag or a.tag, a.rtag, a.root, a.log_prefix).mkdirs()


# --------------------------------------------------------------------------- #
# config, model, weights
# --------------------------------------------------------------------------- #


def config_dict() -> dict:
    return json.loads((HF_SMALL / "config.json").read_text())


def vl_config(tiny: bool):
    """The Lfm2VlConfig the provider's class builds from config.json; tiny = the round-3 tiny sizes."""
    from transformers import Lfm2VlConfig

    d = config_dict()
    if tiny:
        d["vision_config"] = {**d["vision_config"], **TINY_VISION}
        d["projector_hidden_size"] = TINY_PROJECTOR_HIDDEN
    cfg = Lfm2VlConfig.from_dict(d)
    if tiny:
        text = G.tiny_config()
        text.vocab_size = d["text_config"]["vocab_size"]
        cfg.text_config = text
    return cfg


def build_vl(cfg) -> nn.Module:
    """The provider's D1Model (Lfm2VlForConditionalGeneration with the hybrid language model), float32, eval."""
    return provider("modeling_d1").D1Model(cfg).float().eval().requires_grad_(False)


def to_module_name(name: str) -> str:
    return MODULE_VISION + name[len(CKPT_VISION):] if name.startswith(CKPT_VISION) else name


def to_ckpt_name(name: str) -> str:
    return CKPT_VISION + name[len(MODULE_VISION):] if name.startswith(MODULE_VISION) else name


def make_tiny() -> dict:
    from safetensors.torch import load_file, save_file

    cfg = vl_config(tiny=True)
    model = build_vl(cfg)
    r2 = load_file(str(G.TINY_WEIGHTS))
    g = torch.Generator().manual_seed(TINY_SEED)
    out = {}
    for name, t in sorted(model.state_dict().items()):
        if name == "lm_head.weight":           # tied to the table; not in the checkpoint
            continue
        if name.startswith(LM_PREFIX):
            sub = name[len(LM_PREFIX):]
            if sub == "embed_tokens.weight":
                z = torch.randn(t.shape, generator=g, dtype=torch.float32)
                z[: r2[sub].shape[0]] = r2[sub]
                out[name] = z
            else:
                assert r2[sub].shape == t.shape, (sub, r2[sub].shape, t.shape)
                out[name] = r2[sub].clone()
            continue
        z = torch.randn(t.shape, generator=g, dtype=torch.float32)
        if name.endswith(".bias"):
            v = 0.1 * z
        elif name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            v = 1.0 + 0.1 * z
        elif name.endswith("position_embedding.weight"):
            v = z
        else:
            v = z / t[0].numel() ** 0.5
        out[to_ckpt_name(name)] = v
    VL_DIR.mkdir(parents=True, exist_ok=True)
    assert not TINY_VL_WEIGHTS.exists(), f"refusing to overwrite {TINY_VL_WEIGHTS}"
    meta = {"what": "d1-3B LiteRT round 3 tiny random VL model (checkpoint names)", "seed": str(TINY_SEED),
            "vision_overrides": json.dumps(TINY_VISION), "projector_hidden_size": str(TINY_PROJECTOR_HIDDEN),
            "text": "round 2 tiny (d1_prefill_graph.TINY_OVERRIDES) with vocab_size 128000; text layers = "
                    f"{G.TINY_WEIGHTS.name} sha256 {sha256_file(G.TINY_WEIGHTS)}"}
    save_file(out, str(TINY_VL_WEIGHTS), metadata=meta)
    groups = {"language_model": sum(k.startswith(LM_PREFIX) for k in out), "vision": sum(k.startswith(CKPT_VISION) for k in out),
              "projector": sum(k.startswith(PROJ) for k in out)}
    return {"file": str(TINY_VL_WEIGHTS.relative_to(K)), "sha256": sha256_file(TINY_VL_WEIGHTS), "tensors": len(out),
            "groups": groups, "params": int(sum(t.numel() for t in out.values())),
            "vision_params": int(sum(t.numel() for k, t in out.items() if not k.startswith(LM_PREFIX)))}


def _weights_file(source: str) -> Path:
    return TINY_VL_WEIGHTS if source == "tiny" else Path(source) / "model.safetensors"


def load_vl(source: str = "tiny") -> tuple[nn.Module, dict]:
    """The whole VL model (provider's D1Model) with the file's weights, lm_head tied to the table. Tiny only in round
    3 (the real text weights are 5 GB of bfloat16)."""
    from safetensors import safe_open

    cfg = vl_config(tiny=source == "tiny")
    model = build_vl(cfg)
    path = _weights_file(source)
    want = model.state_dict()
    sd = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            sd[to_module_name(k)] = f.get_tensor(k).to(torch.float32)
    missing, extra = sorted(set(want) - set(sd) - {"lm_head.weight"}), sorted(set(sd) - set(want))
    assert not missing and not extra, (missing[:5], extra[:5])
    model.load_state_dict(sd, strict=False)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    info = {"source": source, "file": str(path), "sha256": sha256_file(path) if source == "tiny" else None,
            "tensors": len(sd)}
    return model, info


def vision_key_map(cfg, device: str = "cpu") -> tuple[nn.Module, nn.Module, dict]:
    """(Siglip2VisionModel, Lfm2VlMultiModalProjector) built from the config, and checkpoint name -> (part, name)."""
    from transformers import Siglip2VisionModel
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector

    with torch.device(device):
        tower = Siglip2VisionModel(cfg.vision_config)
        proj = Lfm2VlMultiModalProjector(cfg)
    names = {CKPT_VISION + k: ("tower", k) for k in tower.state_dict()}
    names.update({PROJ + k: ("projector", k) for k in proj.state_dict()})
    return tower, proj, names


def load_vision(source: str = "tiny") -> tuple[nn.Module, nn.Module, object, dict]:
    """Only the 437 + 4 vision / projector tensors (tiny file or <snapshot>/model.safetensors), float32, strict."""
    from safetensors import safe_open

    cfg = vl_config(tiny=source == "tiny")
    tower, proj, names = vision_key_map(cfg)
    path = _weights_file(source)
    parts = {"tower": {}, "projector": {}}
    with safe_open(str(path), framework="pt") as f:
        present = {k for k in f.keys() if k.startswith(CKPT_VISION) or k.startswith(PROJ)}
        assert present == set(names), (sorted(present - set(names))[:5], sorted(set(names) - present)[:5])
        for ck, (part, mk) in names.items():
            parts[part][mk] = f.get_tensor(ck).to(torch.float32)
    tower.load_state_dict(parts["tower"], strict=True)
    proj.load_state_dict(parts["projector"], strict=True)
    tower, proj = tower.float().eval().requires_grad_(False), proj.float().eval().requires_grad_(False)
    info = {"source": source, "file": str(path), "tensors": len(names), "vision_hidden": cfg.vision_config.hidden_size,
            "layers": cfg.vision_config.num_hidden_layers, "heads": cfg.vision_config.num_attention_heads,
            "text_hidden": cfg.text_config.hidden_size, "projector_hidden": cfg.projector_hidden_size,
            "attn_implementation": getattr(tower.config, "_attn_implementation", None)}
    return tower, proj, cfg, info


def check_header() -> dict:
    """Without weights: the real tower and projector on the meta device vs the Hub header (names after the rename,
    shapes)."""
    hdr = json.loads(G.HEADER.read_text())["header"]
    tower, proj, names = vision_key_map(vl_config(tiny=False), device="meta")
    sds = {"tower": tower.state_dict(), "projector": proj.state_dict()}
    in_hdr = {k for k in hdr if k.startswith(CKPT_VISION) or k.startswith(PROJ)}
    shape_bad = [k for k, (part, mk) in names.items() if k in hdr and list(sds[part][mk].shape) != hdr[k]["shape"]]
    matched = sorted(set(names) & in_hdr)
    return {"model_tensors": len(names), "header_vision_projector_tensors": len(in_hdr), "matched": len(matched),
            "shape_matched": len(matched) - len(shape_bad),
            "vision_in_header": sum(k.startswith(CKPT_VISION) for k in in_hdr),
            "projector_in_header": sum(k.startswith(PROJ) for k in in_hdr),
            "rename": f"{CKPT_VISION} -> {MODULE_VISION} (transformers 5.14.1 Siglip2VisionModel has no vision_model level)",
            "missing_in_header": sorted(set(names) - in_hdr), "extra_in_header": sorted(in_hdr - set(names)),
            "shape_mismatch": shape_bad, "dtypes": sorted({hdr[k]["dtype"] for k in in_hdr}),
            "pass": not shape_bad and set(names) == in_hdr}


# --------------------------------------------------------------------------- #
# the export forms
# --------------------------------------------------------------------------- #


def _tower_attention_class():
    from transformers.models.siglip2.modeling_siglip2 import Siglip2Attention

    class TowerAttention(Siglip2Attention):
        """Siglip2Attention with the export-form forward; `attention_mask` = the additive bias [1, 1, 1, N]."""

        def forward(self, hidden_states, attention_mask=None, **kwargs):
            b, n, _ = hidden_states.shape
            h, hd = self.num_heads, self.head_dim
            q = self.q_proj(hidden_states).view(b, n, h, hd).transpose(1, 2)
            k = self.k_proj(hidden_states).view(b, n, h, hd).transpose(1, 2)
            v = self.v_proj(hidden_states).view(b, n, h, hd).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(2, 3)) * self.scale + attention_mask
            p = torch.softmax(scores, dim=-1, dtype=torch.float32)
            y = torch.matmul(p, v).transpose(1, 2).reshape(b, n, h * hd)
            return self.out_proj(y), None

    return TowerAttention


class VisionTower(nn.Module):
    """pixels [1, N, 768] + pos [1, N, H] + mask [1, N] -> {"features": [1, N, H]}."""

    def __init__(self, tower: nn.Module):
        super().__init__()
        cls = _tower_attention_class()
        for layer in tower.encoder.layers:
            layer.self_attn.__class__ = cls
        self.tower = tower

    def forward(self, pixels, pos, mask):
        x = self.tower.embeddings.patch_embedding(pixels) + pos
        bias = (1.0 - mask)[:, None, None, :] * NEG
        for layer in self.tower.encoder.layers:
            x = layer(x, bias)
        return {"features": self.tower.post_layernorm(x)}


class Projector(nn.Module):
    """soft [1, R, 4H] (unshuffled on the host) -> {"mm": [1, R, d]}."""

    def __init__(self, proj: nn.Module):
        super().__init__()
        self.proj = proj

    def forward(self, soft):
        x = self.proj.layer_norm(soft) if self.proj.use_layer_norm else soft
        return {"mm": self.proj.linear_2(self.proj.act(self.proj.linear_1(x)))}


def graphs(source: str = "tiny") -> tuple[VisionTower, Projector, dict]:
    tower, proj, cfg, info = load_vision(source)
    return VisionTower(tower).eval(), Projector(proj).eval(), info


def sample_inputs(kind: str, info: dict) -> dict:
    """Seed-fixed inputs of the graph's shapes (a tile of 14 x 22 = 308 real patches; the graph has no data-dependent
    branch, so the values only feed the torch sanity run)."""
    g = torch.Generator().manual_seed(7)
    hid = info["vision_hidden"]
    if kind == "tower":
        mask = torch.zeros(1, N_PATCHES)
        mask[0, :308] = 1.0
        return {"pixels": torch.rand(1, N_PATCHES, 768, generator=g) * 2 - 1,
                "pos": torch.randn(1, N_PATCHES, hid, generator=g), "mask": mask}
    return {"soft": torch.randn(1, R_TOKENS, 4 * hid, generator=g)}


def write_tflite(kind: str, source: str, o: Out) -> int:
    from tflite_scan import scan

    stem = f"{o.tag}_{GRAPHS[kind]}_fp32"
    rn = o.rname(stem)
    path = o.export(stem)
    out = o.result(f"{rn}_tflite.json")
    scan_out = o.result(f"{rn}_opscan.json")
    for p in (path, out, scan_out):
        assert not p.exists(), f"never overwrite {p}"
    torch.set_num_threads(4)
    t0 = time.time()
    tower, proj, info = graphs(source)
    module = tower if kind == "tower" else proj
    sample = sample_inputs(kind, info)
    with torch.no_grad():
        ref = module(**sample)
    record = {"graph": type(module).__name__, "source": info, "file": o.rel(path), "out": o.as_dict(),
              "params": int(sum(p.numel() for p in module.parameters())), "status": "RUNNING",
              "sample_kwargs": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                                for k, v in sample.items()},
              "convert_call": "litert_torch.convert(module, sample_kwargs=...).export(path)",
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch",
                                                                      "litert-converter", "ai-edge-litert")},
              "load_seconds": round(time.time() - t0, 1)}
    t1 = time.perf_counter()
    try:
        import litert_torch

        lrt = litert_torch.convert(module, sample_kwargs=sample)
        record["convert_seconds"] = round(time.perf_counter() - t1, 1)
        t2 = time.perf_counter()
        lrt.export(str(path))
        record["write_seconds"] = round(time.perf_counter() - t2, 1)
        s = scan(path)
        scan_out.write_text(json.dumps(s, indent=1) + "\n")
        hid, d = info["vision_hidden"], info["text_hidden"]
        if kind == "tower":
            want_in = sorted([("pixels", [1, N_PATCHES, 768], "FLOAT32"), ("pos", [1, N_PATCHES, hid], "FLOAT32"),
                              ("mask", [1, N_PATCHES], "FLOAT32")])
            want_out = [("features", [1, N_PATCHES, hid], "FLOAT32")]
        else:
            want_in = [("soft", [1, R_TOKENS, 4 * hid], "FLOAT32")]
            want_out = [("mm", [1, R_TOKENS, d], "FLOAT32")]
        sig = s["signatures"]
        sig_ok = (len(sig) == 1 and sorted((i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]) == want_in
                  and [(t["name"], t["shape"], t["dtype"]) for t in sig[0]["outputs"]] == want_out)
        hist = s["op_histogram"]
        stops = {"custom_op_count": s["custom_op_count"], "rank_gt4_tensor_count": s["rank_gt4_tensor_count"]}
        watched = {op: hist.get(op, 0) for op in ("RESIZE_BILINEAR", "SELECT_V2", "GATHER_ND", "BROADCAST_TO", "CAST",
                                                   "MAXIMUM", "MEAN", "GELU", "SOFTMAX", "BATCH_MATMUL",
                                                   "FULLY_CONNECTED", "SQUARED_DIFFERENCE", "RSQRT", "TANH", "POW")}
        record.update(
            status="CUSTOM_OP_STOP" if s["custom_op_count"] else ("RANK5_STOP" if s["rank_gt4_tensor_count"] else "EXPORTED"),
            bytes=s["bytes"], sha256=s["sha256"], signatures=sig, signature_matches_contract=sig_ok, stops=stops,
            operator_count=s["operator_count"], op_histogram=hist, watched_ops=watched,
            int64_tensor_count=s["int64_tensor_count"], forbidden_counts=s["forbidden_counts"],
            tensor_rank_histogram=s["tensor_rank_histogram"], tensor_dtype_histogram=s["tensor_dtype_histogram"],
            batch_matmul_shape_groups=s["batch_matmul_shape_groups"], fully_connected_count=s["fully_connected_count"],
            pad_count=s["pad_count"], stablehlo_ops=s["stablehlo_ops"], opscan=o.rel(scan_out),
            torch_sample_finite=bool(all(torch.isfinite(v).all() for v in ref.values())))
    except BaseException:
        o.log(f"{rn}_write.traceback.txt").write_text(traceback.format_exc())
        record.update(status="FAIL", seconds=round(time.perf_counter() - t1, 1))
        o.result(f"{rn}_tflite_attempt.json").write_text(json.dumps(record, indent=1) + "\n")
        raise
    record["peak_rss_bytes_getrusage"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["seconds_wall"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("op_histogram", "batch_matmul_shape_groups",
                                                                    "source", "signatures", "tensor_rank_histogram",
                                                                    "tensor_dtype_histogram")}, indent=1))
    return 0 if record["status"] == "EXPORTED" and sig_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-tiny", action="store_true", help="write cache/vision/tiny_vl_seed0.safetensors")
    ap.add_argument("--check-header", action="store_true", help="real vision + projector key map vs the Hub header")
    ap.add_argument("--write", choices=sorted(GRAPHS), help="convert one graph to exports/{tag}_<graph>_fp32.tflite")
    ap.add_argument("--source", default="tiny")
    Out.add_args(ap)
    a = ap.parse_args()
    if a.make_tiny:
        print(json.dumps(make_tiny()))
    if a.check_header:
        r = check_header()
        (K / "results/vision_header_check.json").write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps({k: v for k, v in r.items() if k not in ("missing_in_header", "extra_in_header")}))
        return 0 if r["pass"] else 1
    if a.write:
        return write_tflite(a.write, a.source, Out.from_args(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
