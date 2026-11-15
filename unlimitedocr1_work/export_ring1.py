#!/usr/bin/env python3
"""Export the 12-layer Unlimited-OCR (v1) decoder with the pinned + ring KV (uocr1_decoder.py) and the checkpoint's
weights.

  MODEL=<checkpoint dir> PYTHONHASHSEED=0 python export_ring1.py OUT_DIR [--prefill 278] [--ctx 4096] [--embedder]
        [--rope-theta 10000] [--P 277]           (litert-torch 0.9.3, litert-converter 0.4.0, torch 2.13.0)
V1Decoder: KV kv_cache_k_<i> [1, 10, P + 128, 128] and kv_cache_v_<i> [1, 10, 128, P + 128], the mask input
[1, 1, T, ctx] and the cos / sin table of ctx rows (ctx = the runtime's context, 4,096). Signatures prefill_<T> and
decode; the default T = P + 1 = 278 takes the 277-id one-page prompt in one prefill call (why not P - 1 = 276: the
docstring of uocr1_decoder.py). Writes OUT_DIR/decoder_fp32.tflite and OUT_DIR/export.json (versions, cfg, seconds,
bytes, weights read); with --embedder also OUT_DIR/embedder_fp32.tflite (signature `embedder`, token_ids [1, 1] ->
embeddings [1, 1, 1280], exported after the decoder in the same process). Quantization: quant_decoder.py.
Adapted from ../unlimitedocr2_work/export_ring.py with the DASH ring decoder replaced by uocr1_decoder.V1Decoder.
"""
import argparse
import importlib.metadata as md
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uocr1_decoder as U  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--prefill", default=str(U.PROMPT_LEN + 1))
  ap.add_argument("--ctx", type=int, default=4096)
  ap.add_argument("--embedder", action="store_true", help="also export the embedder TFLite")
  ap.add_argument("--rope-theta", type=float, default=None, help="default: the config's (DeepseekV2Config 10000)")
  ap.add_argument("--P", type=int, default=U.PROMPT_LEN)
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  assert os.environ.get("PYTHONHASHSEED") == "0", "export with PYTHONHASHSEED=0"
  torch.manual_seed(0)
  torch.set_num_threads(4)
  t0 = time.time()
  cfg, dec, emb, used = U.build(a.ctx, with_embedder=a.embedder, P=a.P, rope_theta=a.rope_theta)
  t_load = time.time() - t0
  prefills = [int(x) for x in a.prefill.split(",")]
  dpath = os.path.join(a.out, "decoder_fp32.tflite")
  epath = os.path.join(a.out, "embedder_fp32.tflite") if emb is not None else None
  t1 = time.time()
  U.export_tflite(dec, emb, cfg, prefills, dpath, epath)
  rec = {"attn": "v1_mha_pinned_ring", "pin": dec.P, "ring": dec.ring, "slots": dec.slots, "ctx": a.ctx,
         "prefills": prefills, "cfg": cfg.__dict__,
         "versions": {p: md.version(p) for p in ("litert-torch", "litert-converter", "ai-edge-litert", "torch")},
         "load_s": round(t_load, 1), "export_s": round(time.time() - t1, 1),
         "bytes": {"decoder_fp32": os.path.getsize(dpath),
                   **({"embedder_fp32": os.path.getsize(epath)} if epath else {})}, "weights_read": len(used),
         "params_decoder": sum(p.numel() for p in dec.parameters())}
  json.dump(rec, open(os.path.join(a.out, "export.json"), "w"), indent=1)
  print("EXPORT_RING1_DONE", json.dumps({k: rec[k] for k in ("load_s", "export_s", "bytes", "params_decoder")}),
        flush=True)


if __name__ == "__main__":
  main()
