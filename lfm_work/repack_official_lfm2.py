#!/usr/bin/env python3
"""repack_official_lfm2.py IN.litertlm OUT.litertlm [--max-num-tokens 4096] [--supports-thinking false|true]
    [--cli litert-lm] [--pbtext FILE --jinja FILE]

Replace the whole LlmMetadata of an LFM2 / LFM2.5 bundle with the LiteRT-LM `models/lfm2` configuration at a pinned
commit, then prove that every other section survived.

The configuration (LlmMetadataProto.pbtext, which carries the chat template inline, and chat_template.jinja) is read
at commit 86a7e6d098c3a6578208fc8538228fa704ca2e8b of github.com/google-ai-edge/LiteRT-LM and checked by sha256.
One or two lines are changed, nothing else:
    max_num_tokens: 32768     -> max_num_tokens: <the bundle's KV cache, 4096 for the LFM2.5-350M and LFM2-2.6B files>
    supports_thinking: true   -> supports_thinking: false   with --supports-thinking false (the default): LFM2.5-350M is
                                                              an instruct model; a comment in the pbtext says to set it
                                                              to false for the Instruct model
    supports_thinking: true      kept                       with --supports-thinking true: LFM2-2.6B reasons
                                                              between <think> and </think> for some prompts
                                                              (Liquid AI's model card: "dynamic hybrid reasoning"), and
                                                              the configuration's `thought` channel takes that text out
                                                              of the reply
Why: the vendor chat template that litert-torch embeds does not run on litert-lm's minijinja (LFM2.5-350M: the
`{% generation %}` markers fail at template creation; LFM2-2.6B: `message.get("content")` fails at render, "map has no
method named get"), and the exporter's stop list is the end-of-turn id plus punctuation strings. The lfm2
configuration renders the conversation with content as a list of parts, which litert-lm 0.18.0 passes (0.17.x passes
a string and the user turn renders empty).

Steps: `CLI unpack IN` -> put the edited pbtext in place of LlmMetadataProto.pbtext -> `CLI pack` the same model.toml.
Checks (exit 1 when any fails): same section list (count, order, type); TFLiteModel raw bytes identical;
ExecutorMetadata parsed-identical; HF tokenizer zlib-decoded identical; OUT's LlmMetadata == the edited pbtext (parsed);
OUT's embedded chat template == chat_template.jinja at the pinned commit (bytes); IN unchanged. Prints the sha256 of
every OUT section. Environment: a python with litert-lm==0.18.0 (it brings litert-lm-builder 0.18.0 and the CLI).
"""
import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zlib
from pathlib import Path

from google.protobuf import text_format
from litert_lm_builder import litertlm_core, litertlm_peek
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import executor_metadata_pb2, llm_metadata_pb2

COMMIT = "86a7e6d098c3a6578208fc8538228fa704ca2e8b"
RAW = f"https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/{COMMIT}/models/lfm2/"
PBTEXT_SHA256 = "6640540f0ceb7b8bc6ebdde798834fc1f79f359de669ddfc482bc6b019176b91"
JINJA_SHA256 = "153fa3c9775a98047ef55c37ba5bf003fa9103de284202de5e407121b6c34a37"
CHUNK = 4 << 20


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(CHUNK), b""):
            h.update(c)
    return h.hexdigest()


def fetch(name, local):
    data = Path(local).read_bytes() if local else urllib.request.urlopen(RAW + name, timeout=60).read()
    return data


def sections(path):
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
    return out


def read(path, s):
    with open(path, "rb") as f:
        f.seek(s["begin"])
        return f.read(s["bytes"])


def canon(m):
    return m.SerializeToString(deterministic=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--max-num-tokens", type=int, default=4096)
    ap.add_argument("--supports-thinking", choices=("false", "true"), default="false",
                    help="false (default) edits the configuration's supports_thinking line; true keeps it")
    ap.add_argument("--cli", default=os.environ.get("LITERT_LM", "litert-lm"))
    ap.add_argument("--pbtext", default=None, help="a local copy of models/lfm2/LlmMetadataProto.pbtext (sha256-checked)")
    ap.add_argument("--jinja", default=None, help="a local copy of models/lfm2/chat_template.jinja (sha256-checked)")
    a = ap.parse_args()
    src, dst = Path(a.source).absolute(), Path(a.output).absolute()
    if dst.exists():
        ap.error(f"refusing to overwrite {dst}")
    cli = shlex.split(a.cli)

    pbtext = fetch("LlmMetadataProto.pbtext", a.pbtext)
    jinja = fetch("chat_template.jinja", a.jinja)
    if sha_bytes(pbtext) != PBTEXT_SHA256 or sha_bytes(jinja) != JINJA_SHA256:
        sys.exit(f"pinned files differ: pbtext {sha_bytes(pbtext)}, jinja {sha_bytes(jinja)}")
    lines = pbtext.decode().split("\n")
    edits = {"max_num_tokens: 32768": f"max_num_tokens: {a.max_num_tokens}"}
    if a.supports_thinking == "false":
        edits["supports_thinking: true"] = "supports_thinking: false"
    new_lines = [edits.get(l, l) for l in lines]
    changed = [(o, n) for o, n in zip(lines, new_lines) if o != n]
    if len(changed) != len(edits):
        sys.exit(f"expected exactly {len(edits)} edited lines, got {changed}")
    edited = "\n".join(new_lines)
    want = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(edited, want)
    print(f"lfm2 configuration @ {COMMIT[:8]}: pbtext sha256 {PBTEXT_SHA256[:16]}…, jinja sha256 {JINJA_SHA256[:16]}…; "
          f"edited: {changed}")

    src_sha = sha_file(src)
    scratch = Path(tempfile.mkdtemp(prefix=".repack_lfm2_", dir=dst.parent))
    try:
        unpacked = scratch / "unpacked"
        for name, argv in (("unpack", ["unpack", str(src), "--output-dir", str(unpacked)]), ("pack", None)):
            if name == "pack":
                (unpacked / "LlmMetadataProto.pbtext").write_text(edited)
                argv = ["pack", str(unpacked), "--output", str(dst)]
            p = subprocess.run([*cli, *argv], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800)
            text = p.stdout + p.stderr
            # the 0.18.0 CLI can print an error and still exit 0: require its success line
            if p.returncode != 0 or "Successfully" not in text or "Traceback (most recent call last)" in text:
                sys.exit(f"{name} failed: exit {p.returncode}: {text[-800:]}")
            if name == "unpack" and not (unpacked / "LlmMetadataProto.pbtext").exists():
                sys.exit("unpack wrote no LlmMetadataProto.pbtext")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    sa, sb = sections(src), sections(dst)
    fails = []
    if [s["type"] for s in sa] != [s["type"] for s in sb]:
        fails.append(f"section list {[s['type'] for s in sa]} -> {[s['type'] for s in sb]}")
    for x, y in zip(sa, sb):
        t = x["type"]
        if t == "LlmMetadataProto":
            got = llm_metadata_pb2.LlmMetadata()
            got.ParseFromString(read(dst, y))
            ok = canon(got) == canon(want) and got.jinja_prompt_template.encode() == jinja
            how = "LlmMetadata == edited pbtext, template == chat_template.jinja"
        elif t == "ExecutorMetadataProto":
            ma, mb = executor_metadata_pb2.ExecutorMetadata(), executor_metadata_pb2.ExecutorMetadata()
            ma.ParseFromString(read(src, x))
            mb.ParseFromString(read(dst, y))
            ok, how = canon(ma) == canon(mb), "parsed identical"
        elif t == "HF_Tokenizer_Zlib":
            ok, how = zlib.decompress(read(src, x)[8:]) == zlib.decompress(read(dst, y)[8:]), "zlib-decoded identical"
        else:
            ok, how = read(src, x) == read(dst, y), "raw bytes identical"
        print(f"  [{y['index']}] {t}: {y['bytes']} B sha256 {sha_bytes(read(dst, y))} ({how}: {ok})")
        if not ok:
            fails.append(f"section {y['index']} {t}: {how} failed")
    if sha_file(src) != src_sha:
        fails.append("input changed")
    print(f"{'PASS' if not fails else 'FAIL'}: {dst.name} {dst.stat().st_size} B sha256 {sha_file(dst)}")
    for f in fails:
        print("  FAIL:", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
