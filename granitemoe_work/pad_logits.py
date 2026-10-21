"""export_fused.py with the decode signature's logits padded to a multiple of 4 columns.

  pad_logits.py <export_fused.py arguments>      (the model dir and the out dir first)

Wraps litert_torch's core decode module (export_hf/core/exportable_module.py, class
LiteRTExportableModuleForDecoderOnlyLMGenerate, forward at L554-578 of git 0d3869c; granitemoe takes this core class:
export_lib.get_prefill_decode_exportable_cls L595-622): the original forward runs unchanged, then the "logits" entry of
the dict it returns, [B, T, V] float32, gets -V % 4 columns of the constant -1e4 appended on the last axis:
  torch.cat([logits, logits.new_full((*logits.shape[:-1], pad), -1e4)], dim=-1)
For this model V = 49,155, so the decode logits get one column (49,156). With 49,155 columns every LiteRT-LM GPU
run logs "TopK requires src tensor C dimension to be divisible by 4.", and where the CPU started a reply with
<|tool_call|> (id 49154, the last of the three ids above 49,151) the GPU started with other text; with the padded
column the GPU started with <|tool_call|> as well. The padded column holds -1e4. The weights, the prefill module (it
returns no logits in this export) and every other output are untouched. Then export_fused.py's main() runs (loaded by path, unchanged)
with the same arguments. On exit <out>/PAD.json records what was wrapped, the number of decode forward calls and the
logits shape / pad seen at each call.
"""
import argparse
import atexit
import functools
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = {"calls": 0, "seen": []}


def _load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def install():
  import torch
  from litert_torch.generative.export_hf.core import exportable_module as em
  cls = em.LiteRTExportableModuleForDecoderOnlyLMGenerate
  orig = cls.forward

  @functools.wraps(orig)  # keeps inspect.signature = the original's (torch.export binds dynamic_shapes by name)
  def forward(self, *args, **kwargs):
    out = orig(self, *args, **kwargs)
    logits = out["logits"]
    pad = -logits.shape[-1] % 4
    RECORD["calls"] += 1
    RECORD["seen"].append({"logits_in": [int(x) for x in logits.shape], "dtype": str(logits.dtype), "pad": int(pad)})
    if pad:
      out = dict(out)
      out["logits"] = torch.cat([logits, logits.new_full((*logits.shape[:-1], pad), -1e4)], dim=-1)
    return out

  cls.forward = forward
  return {"class": f"{cls.__module__}.{cls.__qualname__}", "module_file": em.__file__,
          "original_forward": f"{orig.__module__}.{orig.__qualname__}", "pad_value": -1e4,
          "expression": "torch.cat([logits, logits.new_full((*logits.shape[:-1], -V % 4), -1e4)], dim=-1)"}


def main():
  ap = argparse.ArgumentParser(add_help=False)  # export_fused.py's options, only to find the out dir
  ap.add_argument("--routing")
  ap.add_argument("--down-groups")
  ap.add_argument("--act")
  _, rest = ap.parse_known_args()
  positional = [x for x in rest if not x.startswith("--")]
  out_dir = positional[1] if len(positional) > 1 else None
  info = install()
  print("PAD_INSTALLED", json.dumps(info), flush=True)

  def write_record():
    if out_dir and os.path.isdir(out_dir):
      rec = {"install": info, "calls": RECORD["calls"], "seen": RECORD["seen"], "argv": sys.argv,
             "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
      with open(os.path.join(out_dir, "PAD.json"), "w") as f:
        json.dump(rec, f, indent=1)
      print("PAD_RECORD", json.dumps({"calls": RECORD["calls"], "seen": RECORD["seen"][:4]}), flush=True)

  atexit.register(write_record)
  ef = _load("export_fused", os.path.join(HERE, "export_fused.py"))
  sys.argv = [os.path.join(HERE, "export_fused.py")] + sys.argv[1:]
  ef.main()


if __name__ == "__main__":
  main()
