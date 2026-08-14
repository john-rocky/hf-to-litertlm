#!/usr/bin/env python3
"""Drop the Strip decoder from a bundled HF tokenizer (metaspace models).

The pinned litert-lm runtimes (0.15.0/0.16.0) run the HF tokenizer's decoder
pipeline PER TOKEN while streaming. Metaspace-style tokenizers (Llama/Mistral
SP-BPE serialized as tokenizer.json — Zamba2 is the first such model on this
rail) end their decoder chain with `Strip(" ", start=1)`, which is meant to
remove the ONE artificial leading space the `Prepend("▁")` normalizer adds to
the whole sequence. Streamed per token, it instead eats the leading space of
EVERY piece — every word starts with ▁ in SP-BPE — so generations come out
with all spaces missing ("Thesumof17and25is42."). Byte-level-BPE models
(falcon/qwen/llama3-style Ġ pieces) are unaffected: their spaces live inside
the piece bytes and their decoders carry no Strip.

Removing Strip preserves every interior space under per-token decode; the one
behavior change is a cosmetic leading space at sequence start, which the
runtime/UI already trims.

  python fix_tokenizer_strip.py in.litertlm out.litertlm
"""
import json
import struct
import subprocess
import sys
import tempfile
import zlib
import os


def fix(src, dst, litert_lm="litert-lm"):
    with tempfile.TemporaryDirectory() as td:
        up = os.path.join(td, "unpacked")
        subprocess.run([litert_lm, "unpack", src, "--output-dir", up,
                        "--allow-overwrite"], check=True, capture_output=True)
        tok_path = None
        for name in os.listdir(up):
            if "HF_Tokenizer" in name and name.endswith(".zlib"):
                tok_path = os.path.join(up, name)
        assert tok_path, "no HF tokenizer section found"
        buf = open(tok_path, "rb").read()
        size = struct.unpack("<Q", buf[:8])[0]
        raw = zlib.decompress(buf[8:])
        assert len(raw) == size
        t = json.loads(raw)
        decs = t.get("decoder", {}).get("decoders", [])
        kept = [d for d in decs if d.get("type") != "Strip"]
        if len(kept) == len(decs):
            print("no Strip decoder present - nothing to do")
        t["decoder"]["decoders"] = kept
        out = json.dumps(t, ensure_ascii=False).encode()
        open(tok_path, "wb").write(struct.pack("<Q", len(out)) + zlib.compress(out, 9))
        subprocess.run([litert_lm, "pack", up, "--output", dst],
                       check=True, capture_output=True)
    print(f"OK: {dst} (decoder: {[d['type'] for d in kept]})")


if __name__ == "__main__":
    fix(sys.argv[1], sys.argv[2])
