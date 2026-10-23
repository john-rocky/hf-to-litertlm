#!/usr/bin/env python3
"""convert_qwen38_hybrid_lw.py MODEL OUTDIR - the float export of convert_qwen35_hybrid_lw.py for a Qwen3.5-architecture hybrid whose
float .litertlm would not fit on disk next to its TFLite (Qwen/Qwen3.8-27B: a 108 GB float32 TFLite); one step of
convert_qwen38_27b.sh. MODEL = an HF id or a local checkpoint directory (convert_qwen38_27b.sh passes the F32 copy made by
upcast_fp32.py). Changes from convert_qwen35_hybrid_lw.py (everything else as there: QWEN35_PREFILL_LADDER, cache 4096,
RESOURCE_CONSTANT_NUMEL_THRESHOLD default 1e8 bytes, --quantization_recipe "", --experimental_lightweight_conversion=True):
  * export only: the int8 / template / metadata steps 2-4 of convert_qwen35_hybrid_lw.py are not here; make_int4_27b.py builds the
    int4 file from this step's TFLite;
  * --keep_temporary_files=True: the work dir is OUTDIR itself (without it export() makes a temporary dir under OUTDIR and deletes
    it at the end, with the TFLite in it);
  * package_model runs as written but its builder writes no bytes: the packager still selects the chat template and builds its
    LlmMetadata (OUTDIR/llm_metadata.pb), LitertLmFileBuilder.build is replaced for that call by a recorder, the empty
    OUTDIR/model.litertlm it leaves is removed, and OUTDIR/export_artifacts.json lists the sections the bundle would have had
    (type + metadata items) and the exported artifact paths (prefill_decode TFLite, tokenizer); the LlmMetadata is also written as
    OUTDIR/llm_metadata.pbtext (text_format);
  * litert-torch's backend/inline_consts.py `_clamp_inf_values` is replaced by a read-only +-inf check that calls the original only
    when a tensor holds +-inf (see the note at the replacement); the constants are unchanged.
PYTHONPATH = litert-torch at 115a13607c730c81018bb9789138a3e5e5119e3d with qwen35_hybrid_litert_torch.patch applied (REPRODUCE.md);
the packager imports litert_lm_builder.
"""
import dataclasses
import json
import os
import sys

if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    sys.exit(2)
model, outdir = sys.argv[1], sys.argv[2]
os.environ.setdefault("RESOURCE_CONSTANT_NUMEL_THRESHOLD", "100000000")  # bytes; see convert_qwen35_hybrid_lw.py

# Float export, the prefill ladder of QWEN35_PREFILL_LADDER (convert_qwen38_27b.sh: 1024,256,64,16,4,1), 4096-token KV for the
# full-attention layers (the GatedDeltaNet layers hold a constant-size conv + recurrent state).
argv = [
    "litert-torch", "export_hf",
    "--model", model,
    "--output_dir", outdir,
    "--prefill_lengths",
    os.environ.get("QWEN35_PREFILL_LADDER", "1024,512,256,128,64,32,16,8,4,2,1"),
    "--cache_length", "4096",
    "--quantization_recipe", "",
    "--experimental_lightweight_conversion=True",
    "--keep_temporary_files=True",  # the work dir = OUTDIR (the float TFLite stays)
]

# package_model with a builder that writes no bytes (see the module docstring).
import litert_lm_builder as _lb  # noqa: E402
from litert_torch.generative.export_hf.core import litert_lm_builder as _llb  # noqa: E402

_orig_package = _llb.package_model
_REC = {}


def _record_build(self, stream):
    _REC["sections"] = [{"data_type": str(getattr(x, "data_type", None)),
                         "metadata": [[str(m.key), str(m.value)] for m in getattr(x, "metadata", [])]}
                        for x in self._sections]
    print("builder.build skipped (no bundle bytes);", len(_REC["sections"]), "sections recorded", flush=True)


def _package_without_bytes(source_model_artifacts, export_config, exported_model_artifacts):
    orig_build = _lb.LitertLmFileBuilder.build
    _lb.LitertLmFileBuilder.build = _record_build
    try:
        res = _orig_package(source_model_artifacts, export_config, exported_model_artifacts)
    finally:
        _lb.LitertLmFileBuilder.build = orig_build
    empty = res.litert_lm_model_path
    if empty and os.path.exists(empty) and os.path.getsize(empty) == 0:
        os.remove(empty)
    from google.protobuf import text_format
    from litert_lm_builder.runtime.proto import llm_metadata_pb2
    pb = os.path.join(export_config.work_dir, "llm_metadata.pb")
    md = llm_metadata_pb2.LlmMetadata()
    md.ParseFromString(open(pb, "rb").read())
    open(os.path.join(export_config.output_dir, "llm_metadata.pbtext"), "w", encoding="utf-8").write(
        text_format.MessageToString(md, as_utf8=True))
    art = {f.name: getattr(res, f.name) for f in dataclasses.fields(res)}
    json.dump({"artifacts": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)) for k, v in art.items()},
               "llm_metadata_pb": pb, "sections": _REC.get("sections")},
              open(os.path.join(export_config.output_dir, "export_artifacts.json"), "w"), indent=1)
    return dataclasses.replace(res, litert_lm_model_path=None)


_llb.package_model = _package_without_bytes

# litert-torch's backend/inline_consts.py `_clamp_inf_values` runs `tensor.clamp_(finfo.min, finfo.max)` IN PLACE on every float
# constant while the MLIR module is created. On a memory-mapped (copy-on-write) F32 safetensors weight that write dirties every page,
# so the whole checkpoint becomes anonymous memory (a 4-layer cut of Qwen3.8-27B: peak memory footprint 17.7 GB with the original,
# 1.6 GB with this check, the same TFLite byte for byte). clamp(finfo.min, finfo.max) changes only +-inf, so the replacement looks for
# +-inf read-only in chunks of 2^26 elements and calls the original only when one is there: the same constants, the weights stay
# file pages.
import atexit  # noqa: E402

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
atexit.register(lambda: print(f"clamp check: {_CLAMP_STATS}", flush=True))

from litert_torch.cli import main  # noqa: E402  (after PYTHONPATH is set by the caller)

print("RESOURCE_CONSTANT_NUMEL_THRESHOLD =", os.environ["RESOURCE_CONSTANT_NUMEL_THRESHOLD"], flush=True)
sys.argv = argv
rc = main()
if rc:
    sys.exit(rc)
print("DONE (float export only, no bundle):", os.path.join(outdir, "export_artifacts.json"))
sys.exit(0)
