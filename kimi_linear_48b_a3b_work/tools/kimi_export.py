#!/usr/bin/env python3
"""kimi_export.py CUT_DIR OUT_TFLITE [--prefill 128[,1024]] [--cache 1024] - convert a Kimi-Linear checkpoint (a cut or
the full model) with kimi_ext.py to one .tflite holding the signatures prefill_<T> ... and decode, fp32 weights.

The legacy converter path that export_hf takes by default (use_v2 False, no lightweight conversion; litert-torch
67f11e89): the parameters are materialized in fp32 from the checkpoint by kimi_ext.CheckpointLoader,
the signatures are traced with real sample inputs, then mu_pass_lib.update_model (export_hf's post pass: sum -> mean
fusion, the fp32-cast / FC / fp16-cast fusion) and LiteRTModel.export. With KIMI_EXPERTS=graft the routed experts are
4-element placeholders (graft_kimi.py swaps the quantized experts in), so the full model is exported without
materializing its 47.1B expert parameters. Writes OUT_TFLITE and <OUT_TFLITE>.export.json (arguments, env switches, wall
time, the loader's served names and shapes).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("kimi_ext", os.path.join(HERE, "kimi_ext.py"))
ke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ke)

from litert_torch._convert import interface as converter_utils  # noqa: E402
from litert_torch.generative.export_hf.core.mu import mu_pass_lib  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cut")
  ap.add_argument("out")
  ap.add_argument("--prefill", default="128")
  ap.add_argument("--cache", type=int, default=1024)
  a = ap.parse_args()
  prefill = [int(x) for x in a.prefill.split(",")]
  out = os.path.abspath(a.out)
  os.makedirs(os.path.dirname(out), exist_ok=True)
  t0 = time.time()
  cfg = ke.load_config(a.cut)
  model = ke.KimiStatic(cfg, prefill, device="meta").eval()
  loader = ke.CheckpointLoader(a.cut, cfg)
  ke.materialize(model, loader)
  loader.close()
  t_load = round(time.time() - t0, 1)
  pre, dec = ke.Prefill(model).eval(), ke.Decode(model).eval()
  conv = converter_utils.Converter()
  for t in prefill:
    conv.add_signature(f"prefill_{t}", pre, sample_kwargs=ke.sample_inputs(model, t, a.cache, "cpu"))
  conv.add_signature("decode", dec, sample_kwargs=ke.sample_inputs(model, 1, a.cache, "cpu"))
  with torch.no_grad():
    lrt = conv.convert(strict_export=False)
  lrt = mu_pass_lib.update_model(lrt)
  lrt.export(out)
  wall = round(time.time() - t0, 1)
  rec = {"cut": os.path.abspath(a.cut), "out": out, "bytes": os.path.getsize(out), "prefill": prefill,
         "cache": a.cache, "chunk": ke.CHUNK, "fusion_guard": ke.FUSION_GUARD, "router_gather": ke.ROUTER_GATHER,
         "experts": ke.EXPERTS,
         "pad_id": ke.PAD_ID, "load_s": t_load, "wall_s": wall, "torch": torch.__version__, "served": loader.served}
  json.dump(rec, open(out + ".export.json", "w"), indent=1)
  print("KIMI_EXPORT_DONE", json.dumps({k: rec[k] for k in ("out", "bytes", "prefill", "cache", "wall_s")}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
