#!/usr/bin/env python3
"""The ExecutorMetadata text of the Unlimited-OCR-2 bundle: one state buffer per KV cache input of the decode signature,
read from the census that quant_decoder.py --json writes, with the K / V caches declared LOCAL (default) or GLOBAL.

  python executor_pbtext.py QUANT_CENSUS.json OUT.pbtext [--kv-type local|global]

Why LOCAL: the ring decoder (uocr_ring.py) keeps 416 KV slots for a 4,096-token context. LiteRT-LM v0.19.0 sizes the
context of a bundle that carries ExecutorMetadata by the smallest sequence dim of its GLOBAL K / V caches
(runtime/executor/litert/state.cc, MetadataBasedCreate): 416 GLOBAL slots would stop every reply at 416 tokens, with a
warning and no error. LOCAL caches set no limit, so the bundle's LlmMetadata.max_num_tokens (4,096) stays the context.
sequence_axis = the slot axis (the largest dim, as scripts/add_executor_metadata.py reads it); maximum_sequence_length
= the slot count (read by no CPU / GPU path of v0.19.0).
"""
import argparse
import json


def pbtext(buffers, kind):
  """ExecutorMetadata text for [(name, shape)]: K / V caches as LOCAL or GLOBAL types, sequence axis = the slot axis."""
  blocks = []
  for name, shape in buffers:
    kv = name.split("_")[2]
    typ = {("k", "local"): "TYPE_LOCAL_KEY_CACHE", ("v", "local"): "TYPE_LOCAL_VALUE_CACHE",
           ("k", "global"): "TYPE_GLOBAL_KEY_CACHE", ("v", "global"): "TYPE_GLOBAL_VALUE_CACHE"}[(kv, kind)]
    axis = shape.index(max(shape))
    blocks.append("  state_buffers {\n"
                  f'    prefill_input_name: "{name}"\n    prefill_output_name: "{name}"\n'
                  f'    decode_input_name: "{name}"\n    decode_output_name: "{name}"\n'
                  f"    type: {typ}\n    sequence_axis: {axis}\n    maximum_sequence_length: {max(shape)}\n  }}")
  return "llm_executor_metadata {\n  max_history_size: 0\n" + "\n".join(blocks) + "\n}\n"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("census")
  ap.add_argument("out")
  ap.add_argument("--kv-type", default="local", choices=("local", "global"))
  a = ap.parse_args()
  io = json.load(open(a.census))["census"]["signatures"]["decode"]["io"]["inputs"]
  bufs = sorted([n[len("decode_"):] if n.startswith("decode_") else n, v[0]] for n, v in io.items() if "kv_cache_" in n)
  open(a.out, "w").write(pbtext(bufs, a.kv_type))
  print("EXECUTOR_PBTEXT", a.kv_type, len(bufs), json.dumps(bufs[:2]))


if __name__ == "__main__":
  main()
