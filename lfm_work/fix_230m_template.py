#!/usr/bin/env python3
"""Post-hoc template repair for LFM2.5-230M bundles (weight-identical).

The 230M checkpoint's chat_template.jinja carries HF's `{% generation %}` /
`{% endgeneration %}` assistant-span markers; litert-lm's minijinja fails
engine start on them ("unknown statement generation"). Rendering is proven
byte-identical without them (make_metadata_230m.py, 6/6 cases), so this
script unpacks the bundle, strips the two statements from the embedded
jinja_prompt_template, adds stop id 2 (<|endoftext|>, family precedent —
the exporter only derives stop 7), and repacks. Everything else the
exporter derived (string stops, thought channel, start token) is kept.

    python3 fix_230m_template.py in.litertlm out.litertlm
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

from litert_lm_builder import pack_litertlm_file, unpack_litertlm_file

STMT = re.compile(r"\{%-?\s*(?:end)?generation\s*-?%\}")
# minijinja has no map .get method ("unknown method: map has no method named
# get"); plain indexing is semantically identical here (missing key is a
# falsy undefined in both engines). Proven byte-equal: make_metadata_230m.py.
GET = re.compile(r'(messages\[0\]|message)\.get\(\\"content\\"\)')
STOP2 = "stop_tokens {\n  token_ids {\n    ids: 2\n  }\n}"

def main(src, dst):
    src, dst = Path(src), Path(dst)
    with tempfile.TemporaryDirectory(dir=src.parent) as td:
        unpack_litertlm_file(str(src), td)
        pb = Path(td) / "LlmMetadataProto.pbtext"
        text = pb.read_text()

        hits = STMT.findall(text)
        if len(hits) != 2:
            sys.exit(f"expected exactly 2 generation statements, found {len(hits)}")
        text = STMT.sub("", text)
        assert not STMT.search(text)

        n_get = len(GET.findall(text))
        if n_get != 2:
            sys.exit(f'expected 2 .get(\\"content\\") sites in the escaped template, found {n_get}')
        text = GET.sub(r'\1[\\"content\\"]', text)

        if not re.search(r"ids:\s*2\b", text):
            m = re.search(r"stop_tokens \{\n  token_ids \{\n    ids: 7\n  \}\n\}", text)
            if not m:
                sys.exit("stop id 7 block not found — layout changed, refusing to edit")
            text = text.replace(m.group(0), m.group(0) + "\n" + STOP2, 1)

        pb.write_text(text)
        if dst.exists():
            dst.unlink()  # litert-lm pack exits 0 without writing when the output exists
        pack_litertlm_file(str(Path(td) / "model.toml"), str(dst))
    print(f"wrote {dst} ({dst.stat().st_size} B) from {src.name} ({src.stat().st_size} B)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
