#!/usr/bin/env python3
"""line1_module.py HF_DIR ORACLE_DIR OUT_JSON [--cache 2048] [--only NAME,...] - the module side of line 1.

The exporter's PyTorch module (`lfm3_ext.loader` with key prefix `model.` = the loader the VL / Omni / Audio exports use,
fp32, `lfm3_ext.patch` + the `lrt_transposed_attention` interface) driven by `lfm3_ext.harness` the way LiteRT-LM
drives the exported signatures, against ORACLE_DIR/hf_fp32/<name>.npy (line1_hf_oracle.py). Paths: (i) one prefill
call, (ii) decode token by token, (iii) prefill in 128-token calls with the last call padded to 128 as the runtime
pads it (position 0, pad token); extra (iv): the prompt as (iii), then the continuation decoded (the padded call's
conv state is read). For the long sequence also (iii)/(iv) with embeddings input (externalized embedder) and noise
in the padded slots. Line 1 PASS (the bar was set before the first run): every position max|dlogit| <= 5e-3 and top-1 equal, positions
whose HF fp32 top-2 gap < 1e-3 not counted (their number is recorded); (ii) and (iii) the same bar.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lfm3_ext import harness  # noqa: E402
from lfm3_ext import loader  # noqa: E402

BAR = 5e-3


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("hf_dir")
  ap.add_argument("oracle_dir")
  ap.add_argument("out_json")
  ap.add_argument("--cache", type=int, default=2048)
  ap.add_argument("--only", default="")
  a = ap.parse_args()
  torch.set_grad_enabled(False)
  meta = json.load(open(os.path.join(a.oracle_dir, "inputs.json")))
  seqs = meta["sequences"]
  if a.only:
    keep = set(a.only.split(","))
    seqs = [s for s in seqs if s["name"] in keep]
  t0 = time.time()
  model = loader.load_lfm3_causal_lm(a.hf_dir, key_prefix="model.")
  pad = int(model.config.pad_token_id)
  print(f"module loaded {time.time() - t0:.1f}s (pad {pad})", flush=True)
  results = {"cache_length": a.cache, "bar_max_abs_dlogit": BAR, "near_tie_gap": 1e-3, "sequences": {}}
  for emb in (False, True):
    config = harness.make_export_config(a.cache, externalize_embedder=emb)
    with harness.patched(model, config):
      runner = harness.Runner(model, config, pad_token_id=pad)
      for s in seqs:
        if emb and s["name"] != "long_license":
          continue
        ref = torch.from_numpy(np.load(os.path.join(a.oracle_dir, "hf_fp32", s["name"] + ".npy")))
        t1 = time.time()
        if emb:
          runner.reset()
          parts = [runner.prefill(s["ids"][i:i + 128], i, pad_to=128) for i in range(0, len(s["ids"]), 128)]
          paths = {"iii_chunked_emb": torch.cat(parts)}
          runner.reset()
          split = len(s["prompt_ids"])
          parts = [runner.prefill(s["ids"][i:min(i + 128, split)], i, pad_to=128) for i in range(0, split, 128)]
          parts.append(torch.stack([runner.decode(s["ids"][p], p) for p in range(split, len(s["ids"]))]))
          paths["iv_chunked_then_decode_emb"] = torch.cat(parts)
        else:
          paths = harness.run_paths(runner, s["ids"], chunk=128, split=len(s["prompt_ids"]))
        row = results["sequences"].setdefault(s["name"], {})
        for name, logits in paths.items():
          r = harness.compare(ref, logits)
          r["pass"] = r["max_abs_dlogit"] <= BAR and r["top1_agree"] == r["top1_of"] and r["finite"]
          row[name] = r
          print(f"{s['name']:13s} {name:28s} max|d|={r['max_abs_dlogit']:.2e} (all {r['max_abs_dlogit_all']:.2e})"
                f" top1 {r['top1_agree']}/{r['top1_of']} near-tie {r['excluded_near_tie']}"
                f" {'PASS' if r['pass'] else 'FAIL'}", flush=True)
        print(f"  ({time.time() - t1:.1f}s)", flush=True)
  verdict = all(r["pass"] for row in results["sequences"].values() for r in row.values())
  results["pass"] = verdict
  with open(a.out_json, "w") as f:
    json.dump(results, f, indent=1)
  print(f"DONE line1_module {'PASS' if verdict else 'FAIL'}", flush=True)


if __name__ == "__main__":
  main()
