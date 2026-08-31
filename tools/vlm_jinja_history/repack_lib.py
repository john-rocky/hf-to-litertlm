#!/usr/bin/env python3
"""Metadata-only repack of a published .litertlm, with the two proofs the upload gate needs.

repack(src, dst, jinja, drop_start_token, system_prefix):
    litert-lm unpack -> parse LlmMetadataProto.pbtext as a proto -> set fields -> write ->
    litert-lm pack. The proto round-trip (not string surgery) keeps every other field intact.
section_compare(before, after):
    sha256 of every section; everything except LlmMetadataProto must match byte for byte.
proto_diff(before_pbtext, after_pbtext):
    the exact set of LlmMetadata fields that changed, so the upload note can say "only X".
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from google.protobuf import text_format, json_format
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2

LLM = os.environ.get("LITERT_LM_CLI", "litert-lm")  # the litert-lm CLI (pip install litert-lm-builder)
HDR_BEGIN = litertlm_core.HEADER_BEGIN_BYTE_OFFSET
HDR_END_LOC = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
SCRATCH = Path(os.environ.get("REPACK_SCRATCH", tempfile.gettempdir())) / "litertlm_repack"


def read_meta(pbtext_path):
    m = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(Path(pbtext_path).read_text(), m)
    return m


def repack(src, dst, jinja=None, drop_start_token=False, system_prefix=None):
    src, dst = Path(src), Path(dst)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="rp_", dir=str(SCRATCH)))
    try:
        subprocess.run([LLM, "unpack", str(src), "--output-dir", str(work / "u")],
                       check=True, capture_output=True)
        pb = work / "u" / "LlmMetadataProto.pbtext"
        before_text = pb.read_text()
        m = read_meta(pb)
        notes = []
        if jinja is not None:
            notes.append("jinja_prompt_template " + ("replaced" if m.jinja_prompt_template else "added"))
            m.jinja_prompt_template = jinja
        if drop_start_token:
            assert m.HasField("start_token"), "no start_token to drop"
            notes.append("start_token dropped: " + text_format.MessageToString(m.start_token, as_one_line=True))
            m.ClearField("start_token")
        if system_prefix is not None:
            notes.append(f"prompt_templates.system.prefix {m.prompt_templates.system.prefix!r} -> {system_prefix!r}")
            m.prompt_templates.system.prefix = system_prefix
        after_text = text_format.MessageToString(m)
        pb.write_text(after_text)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        subprocess.run([LLM, "pack", str(work / "u"), "--output", str(dst)],
                       check=True, capture_output=True)
        return {"notes": notes, "pbtext_before": before_text, "pbtext_after": after_text}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def sections(p):
    def fetch(s, e):
        with open(p, "rb") as f:
            f.seek(s); return f.read(e - s + 1)
    head = fetch(0, 4095)
    he = int.from_bytes(head[HDR_END_LOC:HDR_END_LOC + 8], "little")
    if he > len(head):
        head = fetch(0, he + 64)
    meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[HDR_BEGIN:he]), 0)
    out = []
    for i in range(meta.SectionMetadata().ObjectsLength()):
        so = meta.SectionMetadata().Objects(i)
        dt = litertlm_core.any_section_data_type_to_string(so.DataType())
        h = hashlib.sha256()
        off, end = so.BeginOffset(), so.EndOffset()
        with open(p, "rb") as f:
            f.seek(off); left = end - off
            while left > 0:
                b = f.read(min(1 << 22, left)); h.update(b); left -= len(b)
        out.append((dt, end - off, h.hexdigest()))
    return out


def section_compare(before, after):
    A, B = sections(before), sections(after)
    ok = len(A) == len(B)
    rows = []
    for (da, sa, ha), (db, sb, hb) in zip(A, B):
        same = (da == db and ha == hb and sa == sb)
        if da != "LlmMetadataProto" and not same:
            ok = False
        rows.append({"type": da, "bytes_before": sa, "bytes_after": sb, "same": same})
    return ok, rows


def meta_from_file(p):
    """Read LlmMetadata straight out of a packed bundle (what the runtime will see)."""
    def fetch(s, e):
        with open(p, "rb") as f:
            f.seek(s); return f.read(e - s + 1)
    head = fetch(0, 4095)
    he = int.from_bytes(head[HDR_END_LOC:HDR_END_LOC + 8], "little")
    if he > len(head):
        head = fetch(0, he + 64)
    meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[HDR_BEGIN:he]), 0)
    for i in range(meta.SectionMetadata().ObjectsLength()):
        so = meta.SectionMetadata().Objects(i)
        if litertlm_core.any_section_data_type_to_string(so.DataType()) == "LlmMetadataProto":
            m = llm_metadata_pb2.LlmMetadata()
            m.ParseFromString(fetch(so.BeginOffset(), so.EndOffset() - 1))
            return m
    return None


def proto_diff(a, b):
    """Top-level LlmMetadata fields whose value differs (json view, so nested prompt_templates
    shows up as one field)."""
    da = json_format.MessageToDict(a, preserving_proto_field_name=True)
    db = json_format.MessageToDict(b, preserving_proto_field_name=True)
    return sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
