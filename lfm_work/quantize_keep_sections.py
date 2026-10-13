#!/usr/bin/env python3
"""quantize_keep_sections.py IN.litertlm OUT.litertlm --recipe wi8fc|wi4b32_wi8 [--algo minmax|octav] --report REPORT.json
    [--cli 'litert-lm CLI'] - post-hoc quantization of an unquantized .litertlm that keeps EVERY section.

Written for the 4-section bundles litert-torch 0.9.4 writes for LFM2 / LFM2.5 (LlmMetadata, ExecutorMetadata,
HF tokenizer, TFLite model). minicpm_work/quantize_litertlm.py `apply` rebuilds a bundle from three sections only
(llm_metadata + tokenizer + tflite_model) and takes "the" metadata as the LAST *.pbtext that os.listdir returns from
the builder's peek dump. On a 4-section bundle that dump holds LlmMetadataProto.pbtext and ExecutorMetadataProto.pbtext,
so the tool either drops the ExecutorMetadata section (the hybrid's state-buffer binding, without which litert-lm
0.15+ cannot run the model) or hands the ExecutorMetadata text to the builder as llm_metadata (macOS / APFS lists
ExecutorMetadataProto.pbtext last; measured 2026-10-12 on LFM2.5-350M):
    ValueError: Failed to parse LlmMetadata from .../ExecutorMetadataProto.pbtext. Exception: 1:1 : Message type
    "litert.lm.proto.LlmMetadata" has no field named "llm_executor_metadata".

What this tool does instead (the recipe is quantize_litertlm.py's build_recipe, imported by path, unchanged):
  1. `CLI unpack IN` (litert-lm 0.18.0) -> model.toml + every section file
  2. quantize the one TFLiteModel section with ai-edge-quantizer + quantize_litertlm.build_recipe(recipe, algo)
  3. put the quantized tflite in place of the unpacked one (same file name), `CLI pack` the same model.toml
  4. verify OUT against IN section by section: same section list (count, order, type, per-section metadata items);
     LlmMetadata / ExecutorMetadata parsed-identical; HF tokenizer zlib-decoded identical; the TFLiteModel section
     byte-identical to the quantizer's output file (sha256); IN unchanged.
Exit 0 = PASS, 1 = a check failed or a step errored (REPORT says which), 2 = refused before writing.
Runs on a python with ai-edge-quantizer + litert-lm-builder (the litert-torch 0.9.4 environment); the litert-lm CLI
(pip install litert-lm==0.18.0) does unpack / pack: on PATH as `litert-lm`, or pass --cli.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

from google.protobuf import text_format  # noqa: F401  (keeps the proto runtime import explicit)
from litert_lm_builder import litertlm_core, litertlm_peek
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import executor_metadata_pb2, llm_metadata_pb2

MIRROR_TOOL = str(Path(__file__).resolve().parent.parent / "minicpm_work" / "quantize_litertlm.py")
DEFAULT_CLI = os.environ.get("LITERT_LM", "litert-lm")
CHUNK = 4 << 20


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(CHUNK), b""):
            h.update(c)
    return h.hexdigest()


def sections(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        prefix = f.read(32)
        assert prefix[:8] == litertlm_core.HEADER_MAGIC_BYTES, path
        end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                    litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        f.seek(0)
        head = f.read(end)
    md = schema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    out = []
    for i in range(md.SectionMetadata().ObjectsLength()):
        s = md.SectionMetadata().Objects(i)
        out.append({"index": i, "type": litertlm_core.any_section_data_type_to_string(s.DataType()),
                    "begin": s.BeginOffset(), "end": s.EndOffset(), "bytes": s.EndOffset() - s.BeginOffset(),
                    "items": [litertlm_peek.kvp_to_dict(s.Items(j)) for j in range(s.ItemsLength())]})
    assert all(s["end"] <= size for s in out)
    return out


def read(path, s):
    with open(path, "rb") as f:
        f.seek(s["begin"])
        return f.read(s["bytes"])


def section_sha(path, s):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(s["begin"])
        left = s["bytes"]
        while left:
            c = f.read(min(CHUNK, left))
            h.update(c)
            left -= len(c)
    return h.hexdigest()


def canon(m):
    return m.SerializeToString(deterministic=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--recipe", required=True, choices=["wi8fc", "wi4b32_wi8", "wi8"])
    ap.add_argument("--algo", default="minmax", choices=["minmax", "octav"])
    ap.add_argument("--report", required=True)
    ap.add_argument("--cli", default=DEFAULT_CLI)
    args = ap.parse_args()
    src, dst, rep = Path(args.source).absolute(), Path(args.output).absolute(), Path(args.report).absolute()
    for p in (dst, rep):
        if p.exists():
            ap.error(f"refusing to overwrite {p}")
    if not src.is_file():
        ap.error(f"missing {src}")
    cli = shlex.split(args.cli)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rep.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".quant_scratch_", dir=dst.parent))
    rec = {"tool": "quantize_keep_sections.py", "status": "RUNNING", "source": str(src), "output": str(dst),
           "recipe": args.recipe, "algo": args.algo, "mirror_tool": MIRROR_TOOL, "mirror_tool_sha256": sha_file(MIRROR_TOOL),
           "versions": {p: importlib.metadata.version(p) for p in ("ai-edge-quantizer", "litert-lm-builder", "ai-edge-litert")},
           "cli": args.cli, "start_utc": datetime.now(timezone.utc).isoformat(), "timings_s": {}, "checks": {}}

    def save():
        rep.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")

    def run(name, argv):
        t = time.monotonic()
        p = subprocess.run([*cli, *argv], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3600)
        text = p.stdout + p.stderr
        rec["timings_s"][name] = float("%.1f" % (time.monotonic() - t))
        rec.setdefault("commands", []).append({"name": name, "argv": [*cli, *argv], "exit": p.returncode,
                                               "output_tail": text[-1500:]})
        save()
        # the 0.18.0 CLI can print an error and still exit 0: require its success line
        if p.returncode != 0 or "Successfully" not in text or "Traceback (most recent call last)" in text:
            raise RuntimeError(f"{name} failed: exit {p.returncode}: {text[-600:]}")

    try:
        rec["source_sha256"] = sha_file(src)
        sa = sections(src)
        rec["source_sections"] = [{k: s[k] for k in ("index", "type", "bytes", "items")} for s in sa]
        tfl = [s for s in sa if s["type"] == "TFLiteModel"]
        if len(tfl) != 1:
            raise ValueError(f"expected one TFLiteModel section, got {len(tfl)}")
        unpacked = scratch / "unpacked"
        run("unpack", ["unpack", str(src), "--output-dir", str(unpacked)])
        toml_before = (unpacked / "model.toml").read_bytes()
        tfl_files = sorted(p for p in unpacked.iterdir() if p.suffix == ".tflite")
        if len(tfl_files) != 1:
            raise ValueError(f"expected one unpacked .tflite, got {[p.name for p in tfl_files]}")
        tfl_file = tfl_files[0]
        rec["unpacked_files"] = sorted(p.name for p in unpacked.iterdir())
        rec["checks"]["unpacked_tflite_equals_source_section"] = sha_file(tfl_file) == section_sha(src, tfl[0])

        spec = importlib.util.spec_from_file_location("quantize_litertlm", MIRROR_TOOL)
        qmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qmod)
        from ai_edge_quantizer import quantizer
        t = time.monotonic()
        recipe = qmod.build_recipe(args.recipe, args.algo)
        rec["recipe_entries"] = recipe
        qt = quantizer.Quantizer(str(tfl_file), recipe)
        if qt.need_calibration:
            raise RuntimeError("recipe unexpectedly needs calibration")
        res = qt.quantize()
        qpath = scratch / "model_quant.tflite"
        res.export_model(str(qpath))
        rec["timings_s"]["quantize"] = float("%.1f" % (time.monotonic() - t))
        rec["quantized_tflite_bytes"] = qpath.stat().st_size
        rec["quantized_tflite_sha256"] = sha_file(qpath)
        save()
        shutil.move(str(qpath), str(tfl_file))
        if (unpacked / "model.toml").read_bytes() != toml_before:
            raise RuntimeError("model.toml changed")
        run("pack", ["pack", str(unpacked), "--output", str(dst)])

        sb = sections(dst)
        fails = []
        same_list = [s["type"] for s in sa] == [s["type"] for s in sb]
        rec["checks"]["section_list_unchanged"] = same_list
        if not same_list:
            fails.append(f"section list {[s['type'] for s in sa]} -> {[s['type'] for s in sb]}")
        rows = []
        for a, b in zip(sa, sb):
            row = {"index": a["index"], "type": a["type"], "source_bytes": a["bytes"], "output_bytes": b["bytes"],
                   "items_unchanged": a["items"] == b["items"], "output_sha256": section_sha(dst, b)}
            if a["type"] == "LlmMetadataProto":
                ma, mb = llm_metadata_pb2.LlmMetadata(), llm_metadata_pb2.LlmMetadata()
                ma.ParseFromString(read(src, a)); mb.ParseFromString(read(dst, b))
                row["check"], row["PASS"] = "parsed LlmMetadata identical", canon(ma) == canon(mb)
            elif a["type"] == "ExecutorMetadataProto":
                ma, mb = executor_metadata_pb2.ExecutorMetadata(), executor_metadata_pb2.ExecutorMetadata()
                ma.ParseFromString(read(src, a)); mb.ParseFromString(read(dst, b))
                row["check"], row["PASS"] = "parsed ExecutorMetadata identical", canon(ma) == canon(mb)
                row["state_buffers"] = len(mb.llm_executor_metadata.state_buffers)
            elif a["type"] == "HF_Tokenizer_Zlib":
                da, db = zlib.decompress(read(src, a)[8:]), zlib.decompress(read(dst, b)[8:])
                row["check"], row["PASS"] = "zlib-decoded tokenizer identical", da == db
            elif a["type"] == "TFLiteModel":
                row["check"] = "section sha256 == quantizer output sha256"
                row["PASS"] = row["output_sha256"] == rec["quantized_tflite_sha256"]
            else:
                row["check"], row["PASS"] = "raw bytes identical", section_sha(src, a) == row["output_sha256"]
            row["PASS"] = bool(row["PASS"] and row["items_unchanged"])
            if not row["PASS"]:
                fails.append(f"section {a['index']} {a['type']}: {row['check']} failed or items changed")
            rows.append(row)
        rec["sections"] = rows
        rec["checks"]["source_unchanged"] = sha_file(src) == rec["source_sha256"]
        if not rec["checks"]["source_unchanged"]:
            fails.append("source changed")
        if not rec["checks"]["unpacked_tflite_equals_source_section"]:
            fails.append("unpacked tflite != source TFLiteModel section")
        rec["output_bytes"] = dst.stat().st_size
        rec["output_sha256"] = sha_file(dst)
        rec["failures"] = fails
        rec["status"] = "PASS" if not fails else "FAIL"
        print(f"{rec['status']}: {src.name} -> {dst} ({rec['output_bytes']} B, sha256 {rec['output_sha256']})")
        for r in rows:
            print(f"  [{r['index']}] {r['type']}: {r['source_bytes']} -> {r['output_bytes']} B, {r['check']}: PASS={r['PASS']}"
                  + (f" ({r['state_buffers']} state buffers)" if "state_buffers" in r else ""))
        for f in fails:
            print("  FAIL:", f)
        return 0 if not fails else 1
    except Exception as e:  # noqa: BLE001
        rec["status"] = "ERROR"
        rec["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        rec["scratch_removed"] = not scratch.exists()
        rec["end_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    raise SystemExit(main())
