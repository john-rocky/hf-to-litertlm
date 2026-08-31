#!/usr/bin/env python3
"""Measure the one-line template fix on a local copy of a bundle whose jinja
template matches history turns only on role 'model'.

Takes the bundle's jinja_prompt_template, widens every role=='model' condition to
role=='model' or role=='assistant' (nothing else changes), and repacks
metadata-only (repack_lib: litert-lm unpack -> proto edit -> pack; every other
section stays byte-identical).  Re-run repro_history_roles.py on the output to
see both arms render identically.

Usage: python3 fix_template.py /path/to/bundle.litertlm
Needs: litert-lm-builder (the litert-lm CLI + litert_lm_builder package), protobuf.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import repack_lib as rl

SRC = Path(sys.argv[1]).expanduser()
DST = HERE / (SRC.stem + ".assistant_fix.litertlm")

meta = rl.meta_from_file(SRC)
tpl = meta.jinja_prompt_template
assert tpl, "no jinja template in bundle"
old = "message.role == 'model'"
new = "message.role == 'model' or message.role == 'assistant'"
assert old in tpl and "role == 'assistant'" not in tpl
patched = tpl.replace(old, new)
print(f"conditions widened: {tpl.count(old)}")

rec = rl.repack(SRC, DST, jinja=patched)
ok, rows = rl.section_compare(SRC, DST)
rec["sections_equal_except_metadata"] = ok
back = rl.meta_from_file(DST)
rec["jinja_readback_matches"] = back.jinja_prompt_template == patched
print(json.dumps({k: v for k, v in rec.items() if not k.startswith("pbtext")}, indent=1, default=str))
print(f"patched bundle -> {DST}")
