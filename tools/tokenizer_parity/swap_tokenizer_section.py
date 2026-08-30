#!/usr/bin/env python3
"""Replace ONLY the tokenizer section of a .litertlm bundle with an HF tokenizer.json, leaving every
other section byte-identical (checked). Needs the `litert-lm` CLI (pip install litert-lm) on PATH
or in LITERT_LM_CLI, and litert-lm-builder for the section check.

    python swap_tokenizer_section.py in.litertlm tokenizer.json out.litertlm
"""
import hashlib, os, re, shutil, subprocess, sys, tempfile
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
CLI = os.environ.get("LITERT_LM_CLI", "litert-lm")


def sections(p):
    with open(p, "rb") as f:
        head = f.read(4096); he = int.from_bytes(head[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], "little")
        if he > len(head): f.seek(0); head = f.read(he + 64)
        meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:he]), 0)
        out = []
        for i in range(meta.SectionMetadata().ObjectsLength()):
            so = meta.SectionMetadata().Objects(i); dt = litertlm_core.any_section_data_type_to_string(so.DataType())
            f.seek(so.BeginOffset()); left = so.EndOffset() - so.BeginOffset(); h = hashlib.sha256()
            while left > 0:
                b = f.read(min(1 << 22, left)); h.update(b); left -= len(b)
            out.append((dt, so.EndOffset() - so.BeginOffset(), h.hexdigest()))
    return out


def main(src, tok_json, dst):
    work = tempfile.mkdtemp(prefix="swaptok_")
    try:
        u = os.path.join(work, "u")
        subprocess.run([CLI, "unpack", src, "--output-dir", u], check=True)
        shutil.copy(tok_json, os.path.join(u, "tokenizer.json"))
        toml = open(os.path.join(u, "model.toml")).read()
        new = re.sub(r'section_type = "(SP_Tokenizer|HF_Tokenizer)"\ndata_path = "[^"]+"', 'section_type = "HF_Tokenizer"\ndata_path = "tokenizer.json"', toml)
        assert new != toml and new.count('data_path = "tokenizer.json"') == 1, "tokenizer section not found in model.toml"
        open(os.path.join(u, "model.toml"), "w").write(new)
        subprocess.run([CLI, "pack", u, "--output", dst, "--allow-overwrite"], check=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    A, B = sections(src), sections(dst)
    assert len(A) == len(B), "section count changed"
    for (ta, na, ha), (tb, nb, hb) in zip(A, B):
        if "Tokenizer" in ta:
            print(f"  tokenizer: {ta} ({na} B) -> {tb} ({nb} B)")
        else:
            assert (ta, ha) == (tb, hb), f"section {ta} changed"
            print(f"  {ta:22} {na:>12} B  identical")
    print("OK", dst)


if __name__ == "__main__":
    main(*sys.argv[1:4])
