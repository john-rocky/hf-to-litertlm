"""List a .litertlm bundle's sections (type, bytes, sha256) and check the HF tokenizer section against a tokenizer.json
(memory tok-json-verbatim: the bundle must carry the repo's own bytes). Uses the section reader of
hf-to-litertlm/tools/tokenizer_parity/swap_tokenizer_section.py by file path (needs litert_lm_builder).

    python3 scripts/bundle_sections.py exports/engine/<bundle>.litertlm hf/tokenizer.json
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
SWAP = W.parent / "tools" / "tokenizer_parity" / "swap_tokenizer_section.py"


def load(path):
    spec = importlib.util.spec_from_file_location("swap_tokenizer_section", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def section_bytes(mod, bundle, index):
    """The raw bytes of section `index` (the same header walk as the swap tool's sections())."""
    lc, schema = mod.litertlm_core, mod.schema
    with open(bundle, "rb") as f:
        head = f.read(4096)
        he = int.from_bytes(head[lc.HEADER_END_LOCATION_BYTE_OFFSET:lc.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        if he > len(head):
            f.seek(0); head = f.read(he + 64)
        meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[lc.HEADER_BEGIN_BYTE_OFFSET:he]), 0)
        so = meta.SectionMetadata().Objects(index)
        f.seek(so.BeginOffset())
        return f.read(so.EndOffset() - so.BeginOffset())


def main():
    import zlib
    bundle, tok = sys.argv[1], sys.argv[2]
    mod = load(SWAP)
    secs = mod.sections(bundle)
    tok_sha = sha256_file(tok)
    ok = False
    for i, (dt, size, sha) in enumerate(secs):
        flag = ""
        if "okenizer" in dt or "HF" in dt.upper():
            raw = section_bytes(mod, bundle, i)
            if "Zlib" in dt or "zlib" in dt:
                print(f"  zlib section head: {raw[:16].hex()}")
                out = None
                for off in range(0, 17):           # a small length prefix before the zlib stream, if any
                    try:
                        out = zlib.decompress(raw[off:]); print(f"  zlib stream at offset {off}"); break
                    except zlib.error:
                        continue
                if out is None:
                    out = zlib.decompressobj(-15).decompress(raw); print("  raw deflate")
                raw = out
            s = hashlib.sha256(raw).hexdigest()
            same = s == tok_sha
            flag = f"  unpacked {len(raw)} B sha {s[:16]}… {'== tokenizer.json' if same else '!= tokenizer.json'}"
            ok = ok or same
        print(f"{dt:40s} {size:>12d} B  {sha[:16]}…{flag}")
    print(f"tokenizer.json sha256 {tok_sha[:16]}…  section match: {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
