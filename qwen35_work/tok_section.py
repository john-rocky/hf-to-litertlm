#!/usr/bin/env python3
"""tok_section.py BUNDLE OUT_PREFIX REPO_TOKENIZER_JSON - check the tokenizer a .litertlm carries against a model repo's
tokenizer.json. Reads the HF_Tokenizer_Zlib section (an 8-byte little-endian size + a zlib stream) through the litert_lm_builder
header, checks the size field, writes OUT_PREFIX.tokenizer.json (the inflated section) and OUT_PREFIX.json: the section's offsets and
sha256, the json bytes' sha256, the tokenizers backend JSON's sha256 (tokenizers.Tokenizer.from_str(...).to_str()), the normalizer /
pre_tokenizer / added-token count of the section and of the repo file, and whether the section's json bytes equal the repo file's.
`AutoTokenizer.save_pretrained` can write a pipeline built from a tokenizer class instead of the repo's file (for Qwen3.5-9B: a
pre-tokenizer regex without \\p{M}), which tokenizes some scripts differently; compare the section with the repo file at the pinned
revision. Read only on the bundle; refuses to overwrite. Needs litert-lm-builder and tokenizers.
"""
import hashlib
import json
import os
import sys
import zlib

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from tokenizers import Tokenizer


def section(path):
    with open(path, "rb") as f:
        head = f.read(65536)
        if head[:8] != litertlm_core.HEADER_MAGIC_BYTES:
            raise ValueError(f"{path}: bad magic")
        loc = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
        hdr_end = int.from_bytes(head[loc:loc + 8], "little")
        if hdr_end > len(head):
            f.seek(0)
            head = f.read(hdr_end)
        meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:hdr_end]), 0)
        for i in range(meta.SectionMetadata().ObjectsLength()):
            so = meta.SectionMetadata().Objects(i)
            if litertlm_core.any_section_data_type_to_string(so.DataType()) != "HF_Tokenizer_Zlib":
                continue
            f.seek(so.BeginOffset())
            raw = f.read(so.EndOffset() - so.BeginOffset())
            if len(raw) != so.EndOffset() - so.BeginOffset():
                raise ValueError(f"{path}: short read {len(raw)}")
            return i, so.BeginOffset(), so.EndOffset(), raw
    raise ValueError(f"{path}: no HF_Tokenizer_Zlib section")


def summary(js):
    t = Tokenizer.from_str(js)
    j = json.loads(t.to_str())
    return {"backend_json_sha256": hashlib.sha256(t.to_str().encode()).hexdigest(), "normalizer": j.get("normalizer"),
            "pre_tokenizer": j.get("pre_tokenizer"), "added_tokens": len(j.get("added_tokens") or []),
            "vocab_size_with_added": t.get_vocab_size(with_added_tokens=True)}


def main(bundle, prefix, repo_tok):
    tj, mj = prefix + ".tokenizer.json", prefix + ".json"
    for p in (tj, mj):
        if os.path.exists(p):
            raise FileExistsError(p)
    i, b, e, raw = section(bundle)
    size = int.from_bytes(raw[:8], "little")
    body = zlib.decompress(raw[8:])
    if len(body) != size:
        raise ValueError(f"size field {size} != {len(body)}")
    with open(tj, "wb") as f:
        f.write(body)
    repo = open(repo_tok, "rb").read()
    doc = {"bundle": os.path.abspath(bundle), "section_index": i, "begin": b, "end": e,
           "section_sha256": hashlib.sha256(raw).hexdigest(), "json_bytes": size,
           "json_sha256": hashlib.sha256(body).hexdigest(), "repo_tokenizer_json": os.path.abspath(repo_tok),
           "repo_json_bytes": len(repo), "repo_json_sha256": hashlib.sha256(repo).hexdigest(),
           "json_bytes_equal_repo": body == repo,
           "section": summary(body.decode("utf-8")), "repo": summary(repo.decode("utf-8"))}
    doc["backend_equal_repo"] = doc["section"]["backend_json_sha256"] == doc["repo"]["backend_json_sha256"]
    doc["pre_tokenizer_equal_repo"] = doc["section"]["pre_tokenizer"] == doc["repo"]["pre_tokenizer"]
    with open(mj, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"section {i} {e - b} B -> tokenizer.json {size} B sha256 {doc['json_sha256'][:16]} backend "
          f"{doc['section']['backend_json_sha256'][:16]} added {doc['section']['added_tokens']} | repo {len(repo)} B "
          f"{doc['repo_json_sha256'][:16]} backend {doc['repo']['backend_json_sha256'][:16]} added {doc['repo']['added_tokens']} | "
          f"bytes_equal {doc['json_bytes_equal_repo']} backend_equal {doc['backend_equal_repo']} "
          f"pre_tokenizer_equal {doc['pre_tokenizer_equal_repo']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
