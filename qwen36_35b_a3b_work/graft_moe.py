#!/usr/bin/env python3
"""graft_moe.py --tflite Q.tflite --ckpt CKPT_DIR --out OUT.tflite [--bits 4] [--block 32]

Swaps the quantized routed experts into a graph exported with Q35MOE_EXPERTS=graft (q35moe_ext.py), either the
float export itself (reproduce.sh: nothing but the experts is quantized) or the export quantized with the experts left
alone (a recipe whose CUSTOM_OP rule is no_quantize). --block 0 quantizes per channel (one scale for each output row
of each expert, the `<name>_scales` tensor [out, E, 1, 1]); --block B > 0 blockwise:
every `moe` custom op there has 7 inputs whose weight slots 3 / 4 / 5 are 4-element fp32 placeholder constants holding
1000 * layer + 10 * slot (slot 0 = gate, 1 = ff1 = up, 2 = linear = down).

For each (layer, slot), in layer order: the checkpoint's `model.language_model.layers.<L>.mlp.experts.gate_up_proj`
[E, 2I, D] / `down_proj` [E, D, I] (bf16 or fp32 safetensors, through model.safetensors.index.json) is read one tensor
at a time, upcast to fp32 (exact for bf16), laid out as the op wants (q35moe_ext_layout.py ohwi_from_checkpoint: gate /
up [I, E, 1, D], down [D, E, 1, I]) and quantized by litert_quantizer's own code, the same calls its MoE path makes:
naive_min_max_quantize.get_tensor_quant_params (min / max per block, the min-max scale with a 2^-14 floor,
uniform_quantize) and quantize_tensor.quantize_tensor (pack_data, the multi-axis BlockwiseQuantization with the FLOAT32
`<name>_scales` [out, E, 1, in/B] tensor, buffer shared by every subgraph that uses it). Then
ModelModifier._update_quantized_moe_ops (7 -> 10 inputs, weight_type int4 / int8) and ModelModifier._serialize_model
(buffers packed after the flatbuffer at 16-byte offsets), i.e. the quantizer's own last two steps.
No fp32 copy of all experts ever exists: one projection of one layer at a time (~2 GB fp32 transient for gate_up).
Writes OUT.graft.json (per layer seconds, scale counts, floored scales, peak RSS, the checkpoint shards read).
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q35_quant_common as qc  # pylint: disable=g-import-not-at-top

SLOT_NAMES = ("gate", "ff1", "lin")


def placeholder_sites(model):
  """{(layer, slot): [(subgraph index, tensor index)]} from the moe ops' placeholder constants."""
  from litert_quantizer.utils import tfl_flatbuffer_utils as tfu
  sites = {}
  for si, sg in enumerate(model.subgraphs):
    for op in sg.operators:
      if tfu.get_custom_code(model.operatorCodes[op.opcodeIndex]) != "moe":
        continue
      if len(op.inputs) != 7:
        raise ValueError(f"moe op with {len(op.inputs)} inputs in subgraph {si}: not a placeholder graph")
      for slot in range(3):
        ti = int(op.inputs[3 + slot])
        t = sg.tensors[ti]
        data = tfu.get_tensor_data(t, model.buffers)
        if data is None or data.size != 4 or not np.all(data == data.reshape(-1)[0]):
          raise ValueError(f"moe input {3 + slot} in subgraph {si} is not a 4-element placeholder")
        v = float(data.reshape(-1)[0])
        layer = int(v // 1000)
        if abs(v - (1000 * layer + 10 * slot)) > 1e-3:
          raise ValueError(f"placeholder value {v} does not match slot {slot}")
        sites.setdefault((layer, slot), []).append((si, ti))
  return sites


class Checkpoint:
  """Reads single tensors of a safetensors checkpoint (index json or one file), torch -> fp32 numpy."""

  def __init__(self, path):
    self.path = path
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
      self.wmap = json.load(open(idx))["weight_map"]
    else:
      from safetensors import safe_open
      with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
        self.wmap = {k: "model.safetensors" for k in f.keys()}
    self.shards_read = set()

  def get(self, name):
    from safetensors import safe_open
    shard = self.wmap[name]
    self.shards_read.add(shard)
    with safe_open(os.path.join(self.path, shard), framework="pt") as f:
      return f.get_tensor(name)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tflite", required=True)
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--bits", type=int, default=4)
  ap.add_argument("--block", type=int, default=32)
  a = ap.parse_args()
  import torch
  from litert_quantizer import model_modifier as mm_lib
  from litert_quantizer.transformations import quantize_tensor as qt_lib
  from litert_quantizer.transformations import transformation_utils as tu
  from litert_quantizer.utils import tfl_flatbuffer_utils as tfu
  from q35moe_ext_layout import ohwi_from_checkpoint  # pylint: disable=g-import-not-at-top  (this directory, on sys.path above)

  t0 = time.time()
  # the quantizer's own reader: get_model_content = a read-only memoryview of the file (read_model refuses an mmap object)
  model = tfu.read_model(tfu.get_model_content(a.tflite))
  sites = placeholder_sites(model)
  layers = sorted({layer for layer, _ in sites})
  ck = Checkpoint(a.ckpt)
  stats = {}
  per_layer = []
  buffer_origin = {}
  done_bids = set()  # buffers of finished (layer, slot) pairs; each must be met once
  with torch.no_grad(), qc.floored_blockwise_scales(stats):
    for layer in layers:
      tl = time.time()
      base = f"model.language_model.layers.{layer}.mlp.experts."
      gate_up = ck.get(base + "gate_up_proj")
      down = ck.get(base + "down_proj")
      for slot in range(3):
        if (layer, slot) not in sites:
          continue
        suffix = (".mlp.experts.w_gate", ".mlp.experts.w_ff1", ".mlp.experts.w_lin")[slot]
        w = ohwi_from_checkpoint(suffix, gate_up, down).numpy()
        qp = qc.expert_quant_params(w, a.bits, a.block)
        bids = set()
        for si, ti in sites[(layer, slot)]:
          sg = model.subgraphs[si]
          sg.tensors[ti].shape = list(w.shape)
          if sg.tensors[ti].buffer in done_bids:
            raise ValueError(f"buffer {sg.tensors[ti].buffer} of layer {layer} slot {slot} was already grafted")
          bids.add(sg.tensors[ti].buffer)
          qt_lib.quantize_tensor(tu.TransformationInput(
              tensor_id=ti, model=model, subgraph=sg, producer=-1, consumers=[], quant_params=qp,
              buffer_origin=buffer_origin))
        # quantize_tensor keeps quant_params in buffer_origin (to pack a buffer shared by several signatures once), and
        # quant_params holds the unpacked int8 values (1 byte per weight): kept for 40 layers that is ~0.8 GB per layer
        # on top of the packed data (the 40-layer graft passed 56 GB of footprint at layer 36, 2026-11-09 23:48). Every
        # tensor of this (layer, slot) is done and its buffer is packed, so the entries are dropped here.
        # The bytes written do not change: quantize_tensor reads buffer_origin only to skip a second packing of the
        # same quant_params into the same buffer, and each (layer, slot) has its own buffers (done_bids checks it).
        for b in bids:
          buffer_origin.pop(b, None)
        done_bids |= bids
        del w, qp
      del gate_up, down
      per_layer.append({"layer": layer, "seconds": round(time.time() - tl, 1),
                        "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)})
      print("GRAFT_LAYER", json.dumps(per_layer[-1]), flush=True)
  mm = mm_lib.ModelModifier(model)
  model = mm._update_quantized_moe_ops(model)  # pylint: disable=protected-access
  packed = mm_lib._PackedBufferData(model)  # pylint: disable=protected-access
  if packed.packed_size < 256 * 1024:
    raise ValueError("unexpectedly small model")
  mm._serialize_model(model, packed, serialize_to_path=a.out)  # pylint: disable=protected-access
  rec = {"tflite": os.path.abspath(a.tflite), "ckpt": os.path.abspath(a.ckpt), "out": os.path.abspath(a.out),
         "out_bytes": os.path.getsize(a.out), "bits": a.bits, "block": a.block, "layers": layers,
         "sites": {f"{l}:{SLOT_NAMES[s]}": v for (l, s), v in sorted(sites.items())},
         "blockwise_int4_calls": stats.get("calls", 0), "block_scales": stats.get("scales", 0),
         "scales_floored": stats.get("floored", 0), "shards_read": sorted(ck.shards_read), "per_layer": per_layer,
         "seconds": round(time.time() - t0, 1),
         "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)}
  json.dump(rec, open(a.out + ".graft.json", "w"), indent=1)
  print("GRAFT_DONE", json.dumps({k: v for k, v in rec.items() if k not in ("sites", "per_layer")}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
