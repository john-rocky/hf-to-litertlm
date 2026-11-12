#!/usr/bin/env python3
"""make_bundle.py --tflite MODEL.tflite --tokenizer tokenizer.json --out OUT.litertlm [--max-tokens N]
                  [--act fp32|fp16|none] [--pbtext P] [--tokenizer-sha256 SHA|any] [--ids-from FILE]

Packs the writer's prefill / decode .tflite into a .litertlm with litert-lm-builder (0.19.x):
- LlmMetadata = template/gptoss_litertlm.pbtext (template/make_pbtext.py: the ship jinja inline, stops 200002 /
  199999 / 200012, generic_model, the channels, sampler TOP_P k 50 p 1.0 T 1.0, max_num_tokens 4096); --max-tokens
  replaces max_num_tokens only, for a graph whose KV cache is shorter;
- the tokenizer section = the checkpoint's own tokenizer.json bytes (its sha256 must be --tokenizer-sha256, by
  default the one of openai/gpt-oss-120b at b5c939de = openai/gpt-oss-20b at 6cee5e81's LFS oid: 0614fe83...);
- one TFLiteModel section (prefill_decode) that declares prefer_activation_type (fp32 by default; fp16 for the GPU
  file, whose graph is rescaled for fp16 activations (README): a GPU build with fp16 activations and without that
  rescaling answered "!" from the first token).
The builder writes a new uuid and creation_timestamp into the header at every build; --ids-from FILE writes FILE's
instead (FILE = a .litertlm, or its first bytes through the end of its header, e.g. `curl -r 0-65535`), so a rebuild
of the same sections can be compared with the published file byte for byte.
Writes OUT.llm_metadata.pb (the serialized metadata the bundle holds) and OUT.build.json (paths and sha256).

--stream (added 2026-11-10 for the 77 GB gpt-oss-120b graph on a shared USB HDD): the same bytes in one pass. The
TFLiteModel section is the last one, so everything before it (header, LlmMetadata, tokenizer, the zero padding up to
its 16 KiB-aligned begin) depends on the tflite's size only. The builder runs once into memory with the tflite
section's writer replaced by a seek over its size (BytesIO zero-fills a gap as the file's hole reads as zeros); the
bytes before the section's begin offset are the head. Then OUT = head + the tflite, copied in --copy-mib pieces and
hashed on the way (tflite_sha256 and bundle_sha256 of the record come from that pass). The builder itself copies in
1 MiB pieces (os.sendfile fails on macOS for a file target, then shutil.copyfileobj) and the record reads the tflite
and OUT once more each: four passes over the HDD instead of two.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_SHA = "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3"


def _str(v):
    return v.decode() if isinstance(v, bytes) else v


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(64 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--tokenizer", required=True, help="the checkpoint's tokenizer.json (the path must end in .json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--act", default="fp32", help="prefer_activation_type of the TFLiteModel section, or none")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--pbtext", default=os.path.join(os.path.dirname(HERE), "template/gptoss_litertlm.pbtext"))
    ap.add_argument("--tokenizer-sha256", default=TOK_SHA, help="expected sha256 of --tokenizer, or 'any'")
    ap.add_argument("--ids-from", metavar="FILE", help="copy uuid and creation_timestamp from this .litertlm (header)")
    ap.add_argument("--stream", action="store_true", help="head + one hashed pass over the tflite (see above)")
    ap.add_argument("--copy-mib", type=int, default=256, help="--stream: the copy piece in MiB")
    ap.add_argument("--head-only", action="store_true",
                    help="--stream: write only the head (OUT = the bundle's bytes before the tflite), for a check")
    a = ap.parse_args()
    import litert_lm_builder as lb
    import litert_lm_builder.litertlm_builder as lbm
    from google.protobuf import text_format
    from litert_lm_builder.runtime.proto import llm_metadata_pb2

    tok_sha = sha256(a.tokenizer)
    if a.tokenizer_sha256 != "any" and tok_sha != a.tokenizer_sha256:
        raise SystemExit(f"{a.tokenizer}: sha256 {tok_sha} != {a.tokenizer_sha256}")

    md = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(open(a.pbtext).read(), md)
    if a.max_tokens:
        md.max_num_tokens = a.max_tokens
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    md_path = a.out + ".llm_metadata.pb"
    with open(md_path, "wb") as f:
        f.write(md.SerializeToString())
    jinja = md.jinja_prompt_template

    ids = None
    if a.ids_from:
        spec = importlib.util.spec_from_file_location("bundle_sections", os.path.join(HERE, "bundle_sections.py"))
        bs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bs)
        ids = {m.key: m for m in map(lbm.Metadata.from_key_value_pair,
                                     bs.read_header(a.ids_from).systemMetadata.entries or [])
               if m.key in ("uuid", "creation_timestamp")}
        if len(ids) != 2:
            raise SystemExit(f"--ids-from {a.ids_from}: no uuid and creation_timestamp in its system metadata")
        copied = [ids["uuid"], ids["creation_timestamp"]]
        # the builder appends these two at build time (populate_system_metadata, uuid first): write FILE's there
        lbm.populate_system_metadata = lambda entries: [m for m in entries if m.key not in ids] + copied

    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_llm_metadata(md_path)
    b.add_hf_tokenizer(a.tokenizer)
    b.add_tflite_model(a.tflite, "prefill_decode", prefer_activation_type=(None if a.act == "none" else a.act))
    stream = None
    if not a.stream:
        with open(a.out, "wb") as f:
            b.build(f)
        tfl_sha, out_sha = sha256(a.tflite), sha256(a.out)
    else:
        size = os.path.getsize(a.tflite)
        sec = b._sections[-1]
        if sec.data_type != lbm.schema.AnySectionDataType.TFLiteModel or sec.source_path != a.tflite:
            raise SystemExit("--stream: the last section is not the tflite")
        begin = []

        def seek_over(f):
            begin.append(f.tell())
            f.seek(begin[0] + size)

        sec.data_writer = seek_over
        mem = io.BytesIO()
        b.build(mem)
        written = mem.getvalue()
        if len(begin) != 1 or len(written) > begin[0]:
            raise SystemExit(f"--stream: tflite begin {begin}, head bytes written {len(written)}")
        head = written.ljust(begin[0], b"\0")
        h_t, h_b = hashlib.sha256(), hashlib.sha256(head)

        def write_all(f, view):
            while len(view):
                view = view[f.write(view):]

        with open(a.out, "wb", buffering=0) as dst:
            write_all(dst, memoryview(head))
            n_tfl = 0
            if not a.head_only:
                buf = memoryview(bytearray(a.copy_mib << 20))
                with open(a.tflite, "rb", buffering=0) as src:
                    while n := src.readinto(buf):
                        h_t.update(buf[:n])
                        h_b.update(buf[:n])
                        write_all(dst, buf[:n])
                        n_tfl += n
        if not a.head_only and (n_tfl != size or os.path.getsize(a.out) != len(head) + size):
            raise SystemExit(f"--stream: copied {n_tfl} of {size} B, OUT {os.path.getsize(a.out)} B")
        tfl_sha, out_sha = (None, None) if a.head_only else (h_t.hexdigest(), h_b.hexdigest())
        stream = {"head_bytes": len(head), "tflite_begin": begin[0], "tflite_bytes": size, "copy_mib": a.copy_mib,
                  "head_only": a.head_only, "head_sha256": hashlib.sha256(head).hexdigest()}
    rec = {"bundle": os.path.abspath(a.out), "bytes": os.path.getsize(a.out), "tflite": os.path.abspath(a.tflite),
           "tflite_sha256": tfl_sha, "tokenizer": os.path.abspath(a.tokenizer), "tokenizer_sha256": tok_sha,
           "pbtext": os.path.abspath(a.pbtext), "pbtext_sha256": sha256(a.pbtext),
           "jinja_sha256": hashlib.sha256(jinja.encode()).hexdigest(), "prefer_activation_type": a.act,
           "max_num_tokens": md.max_num_tokens,
           "ids_from": os.path.abspath(a.ids_from) if a.ids_from else None,
           "uuid": _str(ids["uuid"].value) if ids else None,
           "creation_timestamp": _str(ids["creation_timestamp"].value) if ids else None,
           "bundle_sha256": out_sha, "stream": stream}
    json.dump(rec, open(a.out + ".build.json", "w"), indent=1)
    print("BUNDLE_DONE", a.out, rec["bytes"], "B", rec["bundle_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
