#!/usr/bin/env python3
"""export_qwen3.py MODEL_DIR OUTDIR - the float32 export of a Qwen3 dense checkpoint (Qwen/Qwen3-32B) with the released
litert-torch 0.10.0 (an environment made from mnn_pocket_work/requirements.txt), for the quantized file (int8 by
default) made afterwards by make_int4_qwen3.py from the float TFLite:
  * `litert-torch export_hf` with the GPU-composite flags (enable_gpu_dynamic_prefill / enable_gpu_dynamic_cache,
    apply_gpu_composites, fuse_gate_up, fuse_qkv, use_qkv_norm_rope_composite, use_swiglu_composite, use_sdpa_composite,
    use_bool_mask), prefill 1024, cache 4096 (Q32_CACHE overrides), --quantization_recipe=none (float; leaving the flag
    out means int8 weights), lightweight conversion, --keep_temporary_files, --bundle_litert_lm;
    Q32_PLAIN=1 = none of the GPU-composite flags;
  * LlmMetadata = LiteRT-LM v0.19.1's models/qwen3/LlmMetadataProto.pbtext (qwen3_LlmMetadataProto_v0.19.1.pbtext next to
    this script, byte for byte; Q32_PBTEXT overrides the path); tokenizer = MODEL_DIR/tokenizer.json;
  * the packager runs but its builder writes no bytes (the 64-layer float .litertlm would be ~131 GB); the sections it
    would write are recorded. Q32_PACKAGE_BYTES=1 lets it write OUTDIR/model.litertlm;
  * RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000 (default here);
  * a read-only +-inf check replaces litert-torch's in-place clamp of every float constant (that clamp writes into every
    page of a memory-mapped F32 checkpoint and turns it into anonymous memory);
  * the fused q||k||v and gate||up weights are served without copies (below; Q32_PREFUSED = prefuse_qwen3.py's file).
MODEL_DIR is a local directory (the F32 copy of qwen35_work/upcast_fp32.py: transformers then hands the export memory-
mapped float32 tensors instead of converting every bf16 weight in RAM). The caller sets HF_HUB_OFFLINE=1 and puts a
stub `tensorflow` package on PYTHONPATH (litert-torch 0.10.0 imports tensorflow.lite.python.schema_py_generated without
depending on TensorFlow; the stub re-exports ai_edge_litert's schema module: convert_qwen3_32b.sh's `shim` step).
Records: OUTDIR/export_artifacts.json (artifacts, the sections the bundle would have), OUTDIR/llm_metadata.pbtext (the
LlmMetadata as packed), OUTDIR/export_flags.json (argv, versions, env)."""
import atexit
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
model_dir, outdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.environ.setdefault("RESOURCE_CONSTANT_NUMEL_THRESHOLD", "100000000")
PBTEXT = os.environ.get("Q32_PBTEXT", os.path.join(HERE, "qwen3_LlmMetadataProto_v0.19.1.pbtext"))
PLAIN = os.environ.get("Q32_PLAIN") == "1"
GPU_FLAGS = [
    "--enable_gpu_dynamic_prefill=True", "--enable_gpu_dynamic_cache=True", "--apply_gpu_composites=True",
    "--fuse_gate_up=True", "--fuse_qkv=True", "--use_qkv_norm_rope_composite=True", "--use_swiglu_composite=True",
    "--use_sdpa_composite=True", "--use_bool_mask=True",
]
argv = [
    "litert-torch", "export_hf",
    f"--model={model_dir}",
    f"--output_dir={outdir}",
    "--prefill_lengths=1024",
    f"--cache_length={os.environ.get('Q32_CACHE', '4096')}",
    "--quantization_recipe=none",  # float (without the flag the export quantizes the weights to int8)
    "--experimental_lightweight_conversion=True",
    "--keep_temporary_files=True",
    "--bundle_litert_lm=True",
    f"--litert_lm_llm_metadata_override={PBTEXT}",
    f"--tokenizer_path_override={os.path.join(model_dir, 'tokenizer.json')}",
] + ([] if PLAIN else GPU_FLAGS)

import litert_lm_builder as _lb  # noqa: E402
from litert_torch.generative.export_hf.core import litert_lm_builder as _llb  # noqa: E402

_orig_package = _llb.package_model
_REC = {}


def _record_build(self, stream):
    _REC["sections"] = [{"data_type": str(getattr(x, "data_type", None)),
                         "metadata": [[str(m.key), str(m.value)] for m in getattr(x, "metadata", [])]}
                        for x in self._sections]
    print("Q32: builder.build skipped (no bundle bytes);", len(_REC["sections"]), "sections recorded", flush=True)


def _package(source_model_artifacts, export_config, exported_model_artifacts):
    keep = os.environ.get("Q32_PACKAGE_BYTES") == "1"
    orig_build = _lb.LitertLmFileBuilder.build

    def _build_and_record(self, stream):
        _REC["sections"] = [{"data_type": str(getattr(x, "data_type", None)),
                             "metadata": [[str(m.key), str(m.value)] for m in getattr(x, "metadata", [])]}
                            for x in self._sections]
        return orig_build(self, stream)

    _lb.LitertLmFileBuilder.build = _build_and_record if keep else _record_build
    try:
        res = _orig_package(source_model_artifacts, export_config, exported_model_artifacts)
    finally:
        _lb.LitertLmFileBuilder.build = orig_build
    path = res.litert_lm_model_path
    if not keep and path and os.path.exists(path) and os.path.getsize(path) == 0:
        os.remove(path)
        res = dataclasses.replace(res, litert_lm_model_path=None)
    from google.protobuf import text_format
    from litert_lm_builder.runtime.proto import llm_metadata_pb2
    md = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(open(PBTEXT, encoding="utf-8").read(), md)
    open(os.path.join(export_config.output_dir, "llm_metadata.pbtext"), "w", encoding="utf-8").write(
        text_format.MessageToString(md, as_utf8=True))
    art = {f.name: getattr(res, f.name) for f in dataclasses.fields(res)}
    json.dump({"artifacts": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v))
                             for k, v in art.items()},
               "llm_metadata_override": PBTEXT, "package_bytes": keep, "sections": _REC.get("sections")},
              open(os.path.join(export_config.output_dir, "export_artifacts.json"), "w"), indent=1)
    return res


_llb.package_model = _package

# inline_consts._clamp_inf_values runs `tensor.clamp_(finfo.min, finfo.max)` IN PLACE on every float constant =
# copy-on-write of the whole mmapped F32 source (a 4-layer F32 export of a 27B: 0.9 -> 16 GB). clamp(min, max)
# changes only +-inf, so the replacement checks for +-inf read-only in chunks of 2^26 elements and calls the original only
# when one is there: the same constants, the weights stay file pages.
import torch  # noqa: E402
from litert_torch.backend import inline_consts as _ic  # noqa: E402

_orig_clamp_inf = _ic._clamp_inf_values
_CLAMP_STATS = {"float_tensors": 0, "clamped": 0}


def _clamp_inf_values_readonly(tensor):
    if not torch.is_floating_point(tensor):
        return tensor
    _CLAMP_STATS["float_tensors"] += 1
    flat = tensor.reshape(-1)
    step = 1 << 26
    for i in range(0, flat.numel(), step):
        if torch.isinf(flat[i:i + step]).any():
            _CLAMP_STATS["clamped"] += 1
            return _orig_clamp_inf(tensor)
    return tensor


_ic._clamp_inf_values = _clamp_inf_values_readonly
atexit.register(lambda: print(f"Q32 clamp check: {_CLAMP_STATS}", flush=True))

# litert-torch 0.10.0 model_ext/qwen3/patch.py builds FusedQwen3MLP.gate_up_proj with
# torch.cat([gate_proj.weight.data, up_proj.weight.data]) = a new fp32 tensor per layer in anonymous memory (1.05 GB per
# layer for Qwen3-32B; the q/k/v cat of FusedQwen3Attention adds 0.21 GB). A 64-layer export was stopped at a
# 36 GB footprint, still growing. While FusedQwen3MLP / FusedQwen3Attention.__init__ run, the patch module's
# `torch` is a proxy whose `cat`, when its parts are exactly that module's weights (same data_ptr, contiguous 2-D fp32),
# returns a tensor over existing bytes instead of a new one:
#   * a view of the parts' own span when they are adjacent in memory (transformers' fp32 params are views of the F32
#     copy's file mapping; inside a shard a layer's gate_proj is followed by its up_proj because safetensors sorts keys);
#   * otherwise (in Qwen/Qwen3-32B's sharding every shard boundary falls between a layer's gate_proj and up_proj, and
#     q / k / v are never adjacent: 35.98 GB and rising) a view of Q32_PREFUSED, prefuse_qwen3.py's F32 file of the
#     same concatenations, mapped MAP_PRIVATE (clean file pages): qkv by the attention module's layer_idx, gate_up by
#     content (the one prefused gate_up whose first and last rows equal the parts' first and last rows);
#   * every view has the first and last row of each part checked against it; a mismatch stops the export.
# Any other call goes to torch.cat (counted as copies). Q32_PREFUSED_FIRST=1 serves the prefused file even where the parts
# are adjacent (a check of that path). Q32_CATVIEW=0 turns all of this off.
import ctypes  # noqa: E402
import mmap  # noqa: E402
import struct  # noqa: E402
import types  # noqa: E402

_FUSE = {"views": 0, "prefused": 0, "copies": 0, "pending": None, "layer": None, "kind": None}
_FUSE_KEEP = []
_PREFUSED = None
_PREFUSED_FIRST = os.environ.get("Q32_PREFUSED_FIRST") == "1"


def _open_prefused(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    hdr.pop("__metadata__", None)
    return {"path": path, "mm": mm, "hdr": hdr, "base": 8 + n}


def _prefused_view(name):
    if _PREFUSED is None or name not in _PREFUSED["hdr"]:
        return None
    i = _PREFUSED["hdr"][name]
    if i["dtype"] != "F32":
        raise RuntimeError(f"Q32 prefused: {name} dtype {i['dtype']}")
    s, e = i["data_offsets"]
    return torch.frombuffer(_PREFUSED["mm"], dtype=torch.float32, count=(e - s) // 4,
                            offset=_PREFUSED["base"] + s).view(*i["shape"])


def _rows_match(v, ts):
    if tuple(v.shape) != (sum(t.shape[0] for t in ts), ts[0].shape[1]):
        return False
    r = 0
    for t in ts:
        if not (torch.equal(v[r], t[0]) and torch.equal(v[r + t.shape[0] - 1], t[-1])):
            return False
        r += t.shape[0]
    return True


def _prefused_for(ts):
    if _PREFUSED is None:
        return None
    if _FUSE["kind"] == "qkv":
        v = _prefused_view(f"model.layers.{_FUSE['layer']}.self_attn.qkv_proj.weight")
        return v if v is not None and _rows_match(v, ts) else None
    hits = []
    for name in _PREFUSED["hdr"]:
        if name.endswith(".mlp.gate_up_proj.weight"):
            v = _prefused_view(name)
            if _rows_match(v, ts):
                hits.append((name, v))
    if len(hits) > 1:
        raise RuntimeError(f"Q32 prefused: {len(hits)} gate_up tensors match one module: {[h[0] for h in hits]}")
    return hits[0][1] if hits else None


def _cat_view(tensors, dim=0, *args, **kwargs):
    pend = _FUSE["pending"]
    if pend is not None and dim == 0 and not args and not kwargs and len(tensors) == len(pend):
        ts = list(tensors)
        ok = all(a.data_ptr() == b.data_ptr() for a, b in zip(ts, pend))
        ok = ok and all(t.dtype == torch.float32 and t.dim() == 2 and t.is_contiguous() and t.shape[1] == ts[0].shape[1]
                        for t in ts)
        if ok:
            adjacent = all(ts[i].data_ptr() + ts[i].nbytes == ts[i + 1].data_ptr() for i in range(len(ts) - 1))
            if not (adjacent and not _PREFUSED_FIRST):
                v = _prefused_for(ts)
                if v is not None:
                    _FUSE["prefused"] += 1
                    return v
            if adjacent:
                n = sum(t.nbytes for t in ts)
                buf = (ctypes.c_char * n).from_address(ts[0].data_ptr())
                v = torch.frombuffer(buf, dtype=torch.float32).view(sum(t.shape[0] for t in ts), ts[0].shape[1])
                if not _rows_match(v, ts):
                    raise RuntimeError("Q32 cat view: rows differ from the parts")
                _FUSE_KEEP.append(ts)  # the parts hold the file mapping the view points into
                _FUSE["views"] += 1
                return v
        _FUSE["copies"] += 1
    return torch.cat(tensors, dim, *args, **kwargs)


if not PLAIN and os.environ.get("Q32_CATVIEW", "1") == "1":
    from litert_torch.generative.export_hf.model_ext.qwen3 import patch as _qp  # noqa: E402

    if os.environ.get("Q32_PREFUSED"):
        _PREFUSED = _open_prefused(os.environ["Q32_PREFUSED"])
        print(f"Q32 prefused: {len(_PREFUSED['hdr'])} tensors from {_PREFUSED['path']}", flush=True)

    class _TorchProxy(types.ModuleType):
        def __getattr__(self, name):
            return getattr(torch, name)

    _PROXY = _TorchProxy("q32_torch_proxy")
    _PROXY.cat = _cat_view

    def _wrap_init(cls, kind, parts):
        orig = cls.__init__

        def init(self, original, *a, **k):
            _FUSE["pending"] = parts(original)
            _FUSE["kind"] = kind
            _FUSE["layer"] = getattr(original, "layer_idx", None)
            _qp.torch = _PROXY
            try:
                orig(self, original, *a, **k)
            finally:
                _qp.torch = torch
                _FUSE["pending"] = _FUSE["kind"] = _FUSE["layer"] = None

        cls.__init__ = init

    _wrap_init(_qp.FusedQwen3MLP, "gate_up", lambda m: (m.gate_proj.weight.data, m.up_proj.weight.data))
    _wrap_init(_qp.FusedQwen3Attention, "qkv",
               lambda m: (m.q_proj.weight.data, m.k_proj.weight.data, m.v_proj.weight.data))
    atexit.register(lambda: print(f"Q32 cat view: views={_FUSE['views']} prefused={_FUSE['prefused']} "
                                  f"copies={_FUSE['copies']}", flush=True))

from litert_torch.cli import main  # noqa: E402
import importlib.metadata as _md  # noqa: E402

os.makedirs(outdir, exist_ok=True)
json.dump({"argv": argv, "plain": PLAIN, "env": {k: os.environ.get(k) for k in (
    "RESOURCE_CONSTANT_NUMEL_THRESHOLD", "Q32_CACHE", "Q32_PACKAGE_BYTES", "Q32_PLAIN", "Q32_CATVIEW", "Q32_PREFUSED",
    "Q32_PREFUSED_FIRST", "HF_HUB_OFFLINE", "TMPDIR")},
    "versions": {p: _md.version(p) for p in ("litert-torch", "ai-edge-quantizer", "ai-edge-litert", "litert-converter",
                                               "litert-lm-builder", "torch", "transformers", "safetensors")}},
          open(os.path.join(outdir, "export_flags.json"), "w"), indent=1)
print("Q32 export argv:", argv, flush=True)
sys.argv = argv
rc = main()
if rc:
    sys.exit(rc)
print("DONE:", os.path.join(outdir, "export_artifacts.json"), flush=True)
sys.exit(0)
