#!/usr/bin/env python3
"""tok_section.py BUNDLE.litertlm REPO_TOKENIZER.json [OUT.json] - is the bundle's HF tokenizer section the repo's
tokenizer.json, byte for byte? Header read (litert_lm_builder 0.18.0), the HF_Tokenizer_Zlib section = uint64-LE size prefix + zlib
stream (as repack_metadata.py decodes it), inflated and compared with the repo file: sha256 / size, and when they differ the top-level
JSON keys that differ (model.vocab / model.merges compared as values). litert-torch's export writes the tokenizer through transformers'
save_pretrained, which re-serializes it (for this model only the post_processor differs); convert_olmo3_7b_think.sh then swaps the
repo's bytes in (tools/tokenizer_parity/swap_tokenizer_section.py) and checks again. Read only. Exit 0 = identical bytes,
1 = different, 2 = no HF tokenizer section."""
import hashlib
import json
import struct
import sys
import zlib

from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema


def sections(path):
    with open(path, "rb") as f:
        prefix = f.read(32)
        end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                    litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        f.seek(0)
        head = f.read(end)
    md = schema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    return [(litertlm_core.any_section_data_type_to_string(md.SectionMetadata().Objects(i).DataType()),
             md.SectionMetadata().Objects(i).BeginOffset(), md.SectionMetadata().Objects(i).EndOffset())
            for i in range(md.SectionMetadata().ObjectsLength())]


def main(bundle, repo_json, out=None):
    hits = [s for s in sections(bundle) if s[0] == "HF_Tokenizer_Zlib"]
    if not hits:
        print("no HF_Tokenizer_Zlib section", [s[0] for s in sections(bundle)])
        return 2
    _, b, e = hits[0]
    with open(bundle, "rb") as f:
        f.seek(b)
        raw = f.read(e - b)
    size = struct.unpack("<Q", raw[:8])[0]
    sec = zlib.decompress(raw[8:])
    repo = open(repo_json, "rb").read()
    rec = {"bundle": bundle, "repo_tokenizer_json": repo_json, "section_raw_bytes": e - b, "size_prefix": size,
           "section_inflated_bytes": len(sec), "section_sha256": hashlib.sha256(sec).hexdigest(),
           "repo_bytes": len(repo), "repo_sha256": hashlib.sha256(repo).hexdigest()}
    rec["identical"] = sec == repo
    if not rec["identical"]:
        a, r = json.loads(sec), json.loads(repo)
        rec["differing_top_keys"] = sorted(k for k in set(a) | set(r) if a.get(k) != r.get(k))
        if a.get("model") != r.get("model"):
            rec["differing_model_keys"] = sorted(k for k in set(a["model"]) | set(r["model"])
                                                 if a["model"].get(k) != r["model"].get(k))
    if out:
        json.dump(rec, open(out, "w"), indent=1)
    print(f"tok_section {bundle}: {'IDENTICAL' if rec['identical'] else 'DIFFERENT'} to the repo file "
          f"(section {rec['section_inflated_bytes']} B sha256 {rec['section_sha256'][:16]}… vs repo {rec['repo_bytes']} B "
          f"{rec['repo_sha256'][:16]}…)" + ("" if rec["identical"] else f"; differing keys {rec.get('differing_top_keys')} "
                                            f"model {rec.get('differing_model_keys')}"))
    return 0 if rec["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
