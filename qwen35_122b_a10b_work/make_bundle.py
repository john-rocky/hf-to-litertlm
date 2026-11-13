#!/usr/bin/env python3
"""make_bundle.py --tflite MODEL.tflite --tokenizer tokenizer.json --out OUT.litertlm [--max-tokens N]
                  [--act fp32|none] [--pbtext P] [--tokenizer-sha256 SHA|any] [--ids-from FILE] [--embedder E.tflite]
make_bundle.py --plan-offset --tokenizer tokenizer.json --out OUT.litertlm [same section flags]   -> TFLITE_OFFSET <X>
make_bundle.py --head-only --tflite-offset X --tokenizer tokenizer.json --out OUT.litertlm [same section flags]

Packs the writer's prefill / decode graph into a .litertlm with litert-lm-builder (0.19.0):
- LlmMetadata = LiteRT-LM's official models/qwen3_5/LlmMetadataProto.pbtext at commit dab2e708, verbatim
  (template/qwen3_5_official_dab2e708.pbtext, sha256 2bdb4bcb...: the official chat template inline, stop <|im_end|>,
  pad <|endoftext|>, the generic_model type, a thought channel, a tool_call_config for <tool_call> / <function=...>
  blocks, the sampler TOP_P k 20 p 0.95 T 0.6, max_num_tokens 4096); --max-tokens replaces max_num_tokens only, for a graph whose KV cache is shorter;
- the tokenizer section = the checkpoint's own tokenizer.json bytes (its sha256 must be --tokenizer-sha256, by
  default the one of Qwen/Qwen3.5-122B-A10B at revision dc4d3484: 5f9e4d49...);
- one TFLiteModel section (prefill_decode) that declares prefer_activation_type (fp32 by default);
- with --embedder E.tflite, a TFLiteModel section of model type embedder in front of it: the token table as a model of
  its own (the writer's --embedder_only --embedder_out), for a graph written with --external_embedder, whose prefill /
  decode signatures take the embeddings in place of the token ids.
The graph finds its real prefill slots from input_pos, not from the pad token, so the official pad_token is kept.

In place, for a graph too big to copy (the writer's --out_offset): the prefill_decode section is the last one and runs
to the end of the file, so its offset X depends only on the sections in front of it. --plan-offset builds the bundle in
memory with an empty prefill_decode section (and the embedder section's size skipped, not read) and prints X; the
writer then writes the graph into OUT from byte X on; then --head-only writes the header, the metadata, the tokenizer
section and the embedder section in front of it (OUT opened for update, not truncated; the prefill_decode section's
writer only checks that the section starts at X). The result has the bytes of a bundle built from the plain .tflite
with --tflite.

The builder writes a new uuid and creation_timestamp into the header at every build; --ids-from FILE writes FILE's
instead (FILE = a .litertlm, or its first bytes through the end of its header, e.g. `curl -r 0-65535`), so a rebuild
of the same sections can be compared with the published file byte for byte (read by
../qwen3_30b_a3b_work/bundle_sections.py). Writes OUT.llm_metadata.pb (the serialized metadata the bundle holds) and
OUT.build.json (paths and sha256; with --head-only the bundle's sha256 reads the whole file once).
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_SHA = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"


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
    ap.add_argument("--tflite")
    ap.add_argument("--plan-offset", action="store_true")
    ap.add_argument("--head-only", action="store_true")
    ap.add_argument("--tflite-offset", type=int)
    ap.add_argument("--tokenizer", required=True, help="the checkpoint's tokenizer.json (the path must end in .json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--act", default="fp32", help="prefer_activation_type of the TFLiteModel section, or none")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--pbtext", default=os.path.join(HERE, "template/qwen3_5_official_dab2e708.pbtext"))
    ap.add_argument("--tokenizer-sha256", default=TOK_SHA, help="expected sha256 of --tokenizer, or 'any'")
    ap.add_argument("--ids-from", metavar="FILE", help="copy uuid and creation_timestamp from this .litertlm (header)")
    ap.add_argument("--embedder", metavar="E.tflite", help="the writer's --embedder_out model, for an --external_embedder graph")
    a = ap.parse_args()
    if a.plan_offset + a.head_only + bool(a.tflite) != 1:
        raise SystemExit("one of --tflite, --plan-offset, --head-only")
    if a.head_only and not a.tflite_offset:
        raise SystemExit("--head-only needs --tflite-offset (from --plan-offset)")
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
        spec = importlib.util.spec_from_file_location(
            "bundle_sections", os.path.join(HERE, "..", "qwen3_30b_a3b_work", "bundle_sections.py"))
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
    if a.embedder:
        b.add_tflite_model(a.embedder, "embedder")
        if a.plan_offset:
            # the plan needs only the section's size: move the stream past it instead of copying 3 GB into memory
            # (the builder seeks to each section's offset and takes its end from tell() after the section's writer)
            esize = os.path.getsize(a.embedder)
            b._sections[-1].data_writer = lambda stream: stream.seek(stream.tell() + esize)
    # --plan-offset / --head-only: the TFLiteModel section's metadata does not depend on the file (model_type and
    # prefer_activation_type only); any existing path stands in for the builder's existence check, and its writer is
    # replaced, so the stand-in is never read.
    b.add_tflite_model(a.tflite or a.pbtext, "prefill_decode", prefer_activation_type=(None if a.act == "none" else a.act))
    sec = b._sections[-1]
    if a.plan_offset:
        seen = []
        sec.data_writer = lambda stream: seen.append(stream.tell())
        b.build(io.BytesIO())
        assert len(seen) == 1 and seen[0] % 16384 == 0, seen
        print("TFLITE_OFFSET", seen[0])
        return 0
    if a.head_only:
        end = os.path.getsize(a.out)
        assert end > a.tflite_offset, (end, a.tflite_offset)
        with open(a.out, "rb") as f:
            f.seek(a.tflite_offset + 4)
            ident = f.read(4)
        assert ident == b"TFL3", f"no TFLite model at offset {a.tflite_offset} of {a.out}: {ident!r}"

        def in_place(stream):
            assert stream.tell() == a.tflite_offset, (stream.tell(), a.tflite_offset)
            stream.seek(end)
        sec.data_writer = in_place
        with open(a.out, "r+b") as f:
            b.build(f)
        assert os.path.getsize(a.out) == end, (os.path.getsize(a.out), end)
    else:
        with open(a.out, "wb") as f:
            b.build(f)
    rec = {"bundle": os.path.abspath(a.out), "bytes": os.path.getsize(a.out),
           "tflite": (f"in place from byte {a.tflite_offset}" if a.head_only else os.path.abspath(a.tflite)),
           "tflite_sha256": None if a.head_only else sha256(a.tflite),
           "tokenizer": os.path.abspath(a.tokenizer), "tokenizer_sha256": tok_sha,
           "embedder": os.path.abspath(a.embedder) if a.embedder else None,
           "embedder_sha256": sha256(a.embedder) if a.embedder else None,
           "pbtext": os.path.abspath(a.pbtext), "pbtext_sha256": sha256(a.pbtext),
           "jinja_sha256": hashlib.sha256(jinja.encode()).hexdigest(), "prefer_activation_type": a.act,
           "max_num_tokens": md.max_num_tokens,
           "ids_from": os.path.abspath(a.ids_from) if a.ids_from else None,
           "uuid": _str(ids["uuid"].value) if ids else None,
           "creation_timestamp": _str(ids["creation_timestamp"].value) if ids else None,
           "bundle_sha256": sha256(a.out)}
    json.dump(rec, open(a.out + ".build.json", "w"), indent=1)
    print("BUNDLE_DONE", a.out, rec["bytes"], "B", rec["bundle_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
