#!/usr/bin/env python3
"""make_bundle.py --tflite MODEL.tflite --tokenizer tokenizer.json --out OUT.litertlm [--max-tokens N]
                  [--act fp32|none] [--pbtext P] [--tokenizer-sha256 SHA|any] [--ids-from FILE]

Packs the writer's prefill / decode .tflite into a .litertlm with litert-lm-builder (0.19.0):
- LlmMetadata = LiteRT-LM's official models/qwen3/LlmMetadataProto.pbtext at commit dab2e708, verbatim
  (template/qwen3_official_dab2e708.pbtext, sha256 f132370a...: the official chat template inline, stop <|im_end|>,
  the qwen3 model type, a thought channel, the sampler TOP_P k 20 p 0.95 T 0.6, max_num_tokens 4096); --max-tokens
  replaces max_num_tokens only, for a graph whose KV cache is shorter;
- the tokenizer section = the checkpoint's own tokenizer.json bytes (its sha256 must be --tokenizer-sha256, by
  default the one of Qwen/Qwen3-30B-A3B at revision ad44e777: aeb13307...);
- one TFLiteModel section (prefill_decode) that declares prefer_activation_type (fp32 by default: with fp16
  activations on the GPU the model's greedy thinking loops instead of answering).
The builder writes a new uuid and creation_timestamp into the header at every build; --ids-from FILE writes FILE's
instead (FILE = a .litertlm, or its first bytes through the end of its header, e.g. `curl -r 0-65535`), so a rebuild
of the same sections can be compared with the published file byte for byte.
Writes OUT.llm_metadata.pb (the serialized metadata the bundle holds) and OUT.build.json (paths and sha256).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


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
    ap.add_argument("--pbtext", default=os.path.join(HERE, "template/qwen3_official_dab2e708.pbtext"))
    ap.add_argument("--tokenizer-sha256", default=TOK_SHA, help="expected sha256 of --tokenizer, or 'any'")
    ap.add_argument("--ids-from", metavar="FILE", help="copy uuid and creation_timestamp from this .litertlm (header)")
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
    with open(a.out, "wb") as f:
        b.build(f)
    rec = {"bundle": os.path.abspath(a.out), "bytes": os.path.getsize(a.out), "tflite": os.path.abspath(a.tflite),
           "tflite_sha256": sha256(a.tflite), "tokenizer": os.path.abspath(a.tokenizer), "tokenizer_sha256": tok_sha,
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
