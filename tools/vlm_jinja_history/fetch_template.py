#!/usr/bin/env python3
"""v0: range-read the LlmMetadata section of published .litertlm bundles and print
their jinja_prompt_template role handling.  Targets: the Google-published FastVLM
bundles (the template our VLM bundles were copied from) and, for comparison, any
bundle passed on the command line as org/repo/file.litertlm.

Only the header + LlmMetadata bytes are downloaded (no weights).  Facts go to
v0_templates.json; raw template text to templates/<name>.jinja.
"""
import json, os, re, sys, time, urllib.request
from pathlib import Path
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2

HERE = Path(__file__).parent
TPL = HERE / "templates"; TPL.mkdir(exist_ok=True)
TOKF = os.path.expanduser("~/.cache/huggingface/token")
TOKEN = os.environ.get("HF_TOKEN") or (open(TOKF).read().strip() if os.path.exists(TOKF) else None)
HDR_BEGIN = litertlm_core.HEADER_BEGIN_BYTE_OFFSET
HDR_END_LOC = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET

DEFAULT_TARGETS = [
    "litert-community/FastVLM-0.5B/FastVLM-0.5B.litertlm",
]


def req(url, rng=None):
    r = urllib.request.Request(url)
    if TOKEN: r.add_header("Authorization", f"Bearer {TOKEN}")
    if rng: r.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    return r


def http_range(url, s, e):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req(url, (s, e)), timeout=180) as r:
                data = r.read()
            if len(data) != e - s + 1:
                raise IOError(f"short range read {len(data)} != {e - s + 1}")
            return data
        except Exception:
            if attempt == 3: raise
            time.sleep(2 * (attempt + 1))


def analyze(target):
    org, repo, fname = target.split("/", 2)
    url = f"https://huggingface.co/{org}/{repo}/resolve/main/{fname}"
    head = http_range(url, 0, 4095)
    if head[:8] != litertlm_core.HEADER_MAGIC_BYTES: raise ValueError("bad magic")
    hdr_end = int.from_bytes(head[HDR_END_LOC:HDR_END_LOC + 8], "little")
    if hdr_end > len(head): head = http_range(url, 0, hdr_end + 64)
    root = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[HDR_BEGIN:hdr_end]), 0)
    rec = {"target": target, "sections": []}
    for i in range(root.SectionMetadata().ObjectsLength()):
        so = root.SectionMetadata().Objects(i)
        dtype = litertlm_core.any_section_data_type_to_string(so.DataType())
        rec["sections"].append({"type": dtype,
                                "begin": so.BeginOffset(), "end": so.EndOffset()})
        if dtype == "LlmMetadataProto":
            raw = http_range(url, so.BeginOffset(), so.EndOffset() - 1)
            md = llm_metadata_pb2.LlmMetadata(); md.ParseFromString(raw)
            tpl = md.jinja_prompt_template
            rec["jinja_len"] = len(tpl)
            rec["has_jinja"] = bool(tpl)
            if tpl:
                out = TPL / (target.replace("/", "__") + ".jinja")
                out.write_text(tpl)
                rec["jinja_file"] = str(out.relative_to(HERE))
                rec["mentions_role_model"] = "'model'" in tpl or '"model"' in tpl
                rec["mentions_role_assistant_in_condition"] = bool(
                    re.search(r"role\s*==\s*['\"]assistant['\"]", tpl))
            rec["model_type"] = md.llm_model_type.WhichOneof("model_type")
            gm = md.llm_model_type.generic_model if md.llm_model_type.HasField("generic_model") else None
            if gm is not None and gm.HasField("model_role"):
                rec["generic_model_role"] = gm.model_role
    return rec


def main():
    targets = sys.argv[1:] or DEFAULT_TARGETS
    out = []
    for t in targets:
        try:
            rec = analyze(t)
        except Exception as e:
            rec = {"target": t, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(rec, indent=1))
        out.append(rec)
    (HERE / "v0_templates.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
