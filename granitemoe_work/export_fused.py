"""export_one.py with the experts of the litert_moe_sequential route fused into two FCs.

  export_fused.py [--routing bcast|loop] [--down-groups G] [--act model|gelu_tanh] <export_one.py arguments>

Installs fused_experts.py (its experts forward registered as "litert_moe_sequential" in transformers' experts
interface, export_lib.pre_split_model_experts pointed at its pre-fuse step), then runs export_one.py's main()
(loaded by path, unchanged) with the remaining arguments; the impl argument must be litert_moe_sequential. On exit,
<out>/FUSED.json records the install (what was replaced), the fused modules and the number of traced forward calls.
"""
import argparse
import atexit
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument("--routing", choices=["bcast", "loop"], default="bcast")
  ap.add_argument("--down-groups", type=int, default=1)
  ap.add_argument("--act", choices=["model", "gelu_tanh"], default="model")
  a, rest = ap.parse_known_args()
  if "litert_moe_sequential" not in rest:
    raise SystemExit("export_fused.py: pass impl litert_moe_sequential (the route whose experts are fused)")
  positional = [x for x in rest if not x.startswith("--")]
  out_dir = positional[1] if len(positional) > 1 else None
  fe = _load("fused_experts", os.path.join(HERE, "fused_experts.py"))
  info = fe.install(routing=a.routing, down_groups=a.down_groups, act=None if a.act == "model" else a.act)

  def write_record():
    if out_dir and os.path.isdir(out_dir):
      rec = {"install": info, "config": dict(fe.CONFIG), "modules": fe.RECORD["modules"],
             "forward_calls": fe.RECORD["forward_calls"], "argv": sys.argv}
      with open(os.path.join(out_dir, "FUSED.json"), "w") as f:
        json.dump(rec, f, indent=1)
      print("FUSED_RECORD", json.dumps({"modules": len(rec["modules"]), "forward_calls": rec["forward_calls"],
                                        "routing": fe.CONFIG["routing"], "down_groups": fe.CONFIG["down_groups"],
                                        "act": fe.CONFIG["act"]}), flush=True)

  atexit.register(write_record)
  eo = _load("export_one", os.path.join(HERE, "export_one.py"))
  sys.argv = [os.path.join(HERE, "export_one.py")] + rest
  eo.main()


if __name__ == "__main__":
  main()
