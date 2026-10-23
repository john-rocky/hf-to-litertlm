"""set_loops.py SNAPSHOT_DIR MODEL_DIR LOOPS - a model dir for export_dory.py that differs from the pinned checkpoint in
one config key, n_recurrent_loops (how many times the 20 recurrent attention/MLP pairs run per token; dory_modeling.py
reads it from config.json, there is no flag).

MODEL_DIR/config.json = SNAPSHOT_DIR/config.json with n_recurrent_loops set to LOOPS, written in the same key order and
indent (the pinned file is json.dumps(indent=2) of itself plus a newline, which is checked, so the two files differ in
that one line); every other key compares equal. The other six files the export reads are symlinks to the snapshot.
The line diff of the two config.json files is printed.

Usage: python dory_work/set_loops.py <snapshot dir> <model dir> <loops>
"""
import difflib
import json
import os
import shutil
import sys

FILES = ("generation_config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
         "chat_template.jinja", "model.safetensors")


def main(snap, mdir, loops):
  loops = int(loops)
  if loops < 1:
    raise SystemExit("set_loops: LOOPS must be a positive integer")
  src_text = open(os.path.join(snap, "config.json"), encoding="utf-8").read()
  cfg = json.loads(src_text)
  if not isinstance(cfg.get("n_recurrent_loops"), int):
    raise SystemExit(f"set_loops: the pinned config has no integer n_recurrent_loops ({cfg.get('n_recurrent_loops')!r})")
  if json.dumps(cfg, indent=2) + "\n" != src_text:
    raise SystemExit("set_loops: the pinned config.json is not json.dumps(indent=2) of itself; a re-dump would change "
                     "more than one line")
  new = dict(cfg)
  new["n_recurrent_loops"] = loops
  changed = sorted(k for k in set(cfg) | set(new) if cfg.get(k) != new.get(k))
  assert changed in ([], ["n_recurrent_loops"]), changed
  assert list(new) == list(cfg)
  new_text = json.dumps(new, indent=2) + "\n"
  if os.path.lexists(mdir):
    shutil.rmtree(mdir)
  os.makedirs(mdir)
  with open(os.path.join(mdir, "config.json"), "w", encoding="utf-8") as f:
    f.write(new_text)
  for name in FILES:
    src = os.path.abspath(os.path.join(snap, name))
    if not os.path.exists(src):
      raise SystemExit(f"set_loops: missing {src}")
    os.symlink(src, os.path.join(mdir, name))
  diff = [l.rstrip("\n") for l in difflib.unified_diff(src_text.splitlines(True), new_text.splitlines(True),
                                                        "pinned/config.json", "model dir/config.json", n=0)]
  print("CONFIG_DIFF\n" + "\n".join(diff), flush=True)
  print(f"SET_LOOPS n_recurrent_loops {cfg['n_recurrent_loops']} -> {loops}: {mdir} (config.json + {len(FILES)} "
        f"symlinks to {snap})", flush=True)
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 4:
    raise SystemExit(__doc__)
  sys.exit(main(*sys.argv[1:4]))
