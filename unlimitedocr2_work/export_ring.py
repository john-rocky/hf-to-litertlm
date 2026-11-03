#!/usr/bin/env python3
"""Export the 12-layer Unlimited-OCR-2 decoder with the pinned + ring KV (uocr_ring.py) and the checkpoint's weights.

  MODEL=<checkpoint dir> PYTHONHASHSEED=0 python export_ring.py OUT_DIR [--prefill 280] [--ctx 4096] [--embedder]
   (litert-torch 0.9.3, litert-converter 0.4.0, torch 2.13.0)
RingDecoder: the weights and math of uocr_dash.py, KV kv_cache_k_<i> [1, 11, 416, 128] (head 10 = the router key)
and kv_cache_v_<i> [1, 10, 128, 416], the mask input [1, 1, T, ctx] and the f(p) cos / sin table of ctx rows (ctx =
the runtime's context, 4,096). One signature prefill_280 (the fixed 279-id prompt is one call: LiteRT-LM prefills all
ids but the last) and decode. Writes OUT_DIR/decoder_fp32.tflite and OUT_DIR/export.json (versions, cfg, seconds,
bytes, weights read); with --embedder also OUT_DIR/embedder_fp32.tflite (signature `embedder`, token_ids [1, 1] ->
embeddings [1, 1, 1280], exported after the decoder in the same process). Quantization: quant_decoder.py.
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
import uocr_ring as RG  # noqa: E402

ud, UD = RG.ud, RG.UD


def build(ctx, n_layers=None, with_embedder=False):
  cfg = ud.Cfg.from_hf(cache=ctx, extra_k_heads=1)
  get, _ = ud.hf_tensor_reader()
  dec = RG.RingDecoder(cfg, n_layers).eval()
  used = []
  for i, layer in enumerate(dec.layers):
    used += UD.load_dash_layer(layer, cfg, i, get)
  with torch.no_grad():
    dec.norm.copy_(get("model.norm.weight").float())
    dec.lm_head.weight.copy_(get("lm_head.weight").float())
  used += ["model.norm.weight", "lm_head.weight"]
  emb = None
  if with_embedder:
    emb = ud.Embedder(cfg).eval()
    with torch.no_grad():
      emb.embed_tokens.weight.copy_(get("model.embed_tokens.weight").float())
    used.append("model.embed_tokens.weight")
  return cfg, dec, emb, used


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--prefill", default="280")
  ap.add_argument("--ctx", type=int, default=4096)
  ap.add_argument("--embedder", action="store_true", help="also export the embedder TFLite")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  assert os.environ.get("PYTHONHASHSEED") == "0", "export with PYTHONHASHSEED=0"
  torch.manual_seed(0)
  torch.set_num_threads(4)
  t0 = time.time()
  cfg, dec, emb, used = build(a.ctx, with_embedder=a.embedder)
  t_load = time.time() - t0
  prefills = [int(x) for x in a.prefill.split(",")]
  dpath = os.path.join(a.out, "decoder_fp32.tflite")
  epath = os.path.join(a.out, "embedder_fp32.tflite") if emb is not None else None
  t1 = time.time()
  RG.export_tflite(dec, emb, cfg, prefills, dpath, epath)
  rec = {"attn": "dash_static_1page_ring", "pin": RG.PIN, "ring": RG.RING, "slots": RG.SLOTS, "ctx": a.ctx,
         "prefills": prefills, "cfg": cfg.__dict__,
         "versions": {p: md.version(p) for p in ("litert-torch", "litert-converter", "ai-edge-litert", "torch")},
         "load_s": round(t_load, 1), "export_s": round(time.time() - t1, 1),
         "bytes": {"decoder_fp32": os.path.getsize(dpath),
                   **({"embedder_fp32": os.path.getsize(epath)} if epath else {})}, "weights_read": len(used),
         "params_decoder": sum(p.numel() for p in dec.parameters())}
  json.dump(rec, open(os.path.join(a.out, "export.json"), "w"), indent=1)
  print("EXPORT_RING_DONE", json.dumps({k: rec[k] for k in ("load_s", "export_s", "bytes", "params_decoder")}),
        flush=True)


if __name__ == "__main__":
  main()
