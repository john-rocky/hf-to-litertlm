#!/usr/bin/env python3
"""graft_kimi.py --tflite Q.tflite --ckpt CKPT_DIR --out OUT.tflite [--bits 4] [--block 32]

Swaps the quantized routed experts into a graph exported with KIMI_EXPERTS=graft (kimi_ext.py) and quantized with
the experts left alone (a recipe whose CUSTOM_OP entry is no_quantize): every `moe` custom op has 7 inputs whose weight
slots 3 / 4 / 5 are 4-element fp32 placeholder constants holding 1000 * layer + 10 * slot (slot 0 = gate = w1,
1 = ff1 = up = w3, 2 = linear = down = w2).

Per (layer, slot), in layer order, the 256 experts'
`model.layers.<L>.block_sparse_moe.experts.<e>.w1|w3|w2.weight` (bf16 [I, D] / [D, I],
through model.safetensors.index.json) are read one tensor at a time, stacked and upcast to fp32 (exact for bf16) in
the op's layout (kimi_ext.CheckpointLoader: gate / up [I, E, 1, D], down [D, E, 1, I]) and quantized by litert_quantizer's
own code, the calls its MoE path makes (quant_common.expert_quant_params: min / max per block of `block` inputs,
the min-max scale with the 2^-14 floor, uniform_quantize; quantize_tensor.quantize_tensor packs the data and writes the
FLOAT32 `<name>_scales` [out, E, 1, in / B] tensor, one buffer shared by every subgraph that uses it). Then
ModelModifier._update_quantized_moe_ops (7 -> 10 inputs, weight_type int4 / int8) and ModelModifier._serialize_model,
the quantizer's own last two steps. One projection of one layer exists in fp32 at a time (2.4 GB). The quantizer's
buffer_origin entries of a finished (layer, slot) are dropped (they would keep every finished projection in RAM).
Writes OUT.graft.json (per layer seconds and RSS, scale counts, floored scales, peak RSS, the shards read).

--spill FILE (off by default; the output bytes do not depend on it): after each (layer, slot) its finished expert
weight buffer is written to FILE and the BufferT's data becomes a read-only np.memmap of those bytes, so the quantized
experts do not stay in RAM until the quantizer's own _PackedBufferData / _serialize_model copy them to OUT (the int8
experts of the 27-layer model are 47 GB). FILE may be on another volume than OUT; it is deleted at the end. The
scale buffers stay in RAM (int8 per channel: 116 MB for 27 layers; int4 blockwise-32: 5.9 GB).
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = (("gate", "w1"), ("ff1", "w3"), ("lin", "w2"))


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
        data = tfu.get_tensor_data(sg.tensors[ti], model.buffers)
        if data is None or data.size != 4 or not np.all(data == data.reshape(-1)[0]):
          raise ValueError(f"moe input {3 + slot} in subgraph {si} is not a 4-element placeholder")
        v = float(data.reshape(-1)[0])
        layer = int(v // 1000)
        if abs(v - (1000 * layer + 10 * slot)) > 1e-3:
          raise ValueError(f"placeholder value {v} does not match slot {slot}")
        sites.setdefault((layer, slot), []).append((si, ti))
  return sites


class Checkpoint:
  """Single tensors of a safetensors checkpoint (index json or one file), as torch tensors."""

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
    self.files = {}

  def get(self, name):
    from safetensors import safe_open
    shard = self.wmap[name]
    self.shards_read.add(shard)
    if shard not in self.files:
      self.files[shard] = safe_open(os.path.join(self.path, shard), framework="pt")
    return self.files[shard].get_tensor(name)


class Spill:
  """Moves a BufferT's bytes to the end of one file and leaves a read-only np.memmap of them as the BufferT's data."""

  def __init__(self, path):
    self.path = path
    self.f = open(path, "wb")
    self.off = 0
    self.moved = 0

  def move(self, buf):
    if buf.data is None or isinstance(buf.data, np.memmap):
      return
    d = np.ravel(np.asarray(buf.data)).view(np.uint8)
    pad = (-self.off) % 16384  # page-aligned regions: the serializer's MADV_DONTNEED stays inside one buffer
    if pad:
      self.f.write(b"\0" * pad)
      self.off += pad
    self.f.write(memoryview(np.ascontiguousarray(d)))
    self.f.flush()
    buf.data = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.off, shape=(d.size,))
    self.off += d.size
    self.moved += d.size

  def close(self):
    self.f.close()


def expert_slot(ck, layer, src, n_experts):
  """The op's layout of one projection of one layer, fp32: w1 / w3 [I, D] -> [I, E, 1, D]; w2 [D, I] -> [D, E, 1, I]."""
  import torch
  base = f"model.layers.{layer}.block_sparse_moe.experts."
  w = torch.stack([ck.get(f"{base}{e}.{src}.weight") for e in range(n_experts)], dim=0)  # [E, out, in] bf16
  o, n = w.shape[1], w.shape[2]
  return w.permute(1, 0, 2).to(torch.float32).contiguous().reshape(o, n_experts, 1, n).numpy()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tflite", required=True)
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--bits", type=int, default=4)
  ap.add_argument("--block", type=int, default=32)
  ap.add_argument("--spill", help="file for the finished expert buffers (see the module docstring); off by default")
  a = ap.parse_args()
  import importlib.util
  import torch
  from litert_quantizer import model_modifier as mm_lib
  from litert_quantizer.transformations import quantize_tensor as qt_lib
  from litert_quantizer.transformations import transformation_utils as tu
  from litert_quantizer.utils import tfl_flatbuffer_utils as tfu
  spec = importlib.util.spec_from_file_location("quant_common", os.path.join(HERE, "quant_common.py"))
  qc = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(qc)

  t0 = time.time()
  cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
  n_experts = int(cfg["num_experts"] if "num_experts" in cfg else cfg["num_local_experts"])
  model = tfu.read_model(tfu.get_model_content(a.tflite))
  sites = placeholder_sites(model)
  layers = sorted({layer for layer, _ in sites})
  ck = Checkpoint(a.ckpt)
  spill = Spill(a.spill) if a.spill else None
  stats, per_layer, buffer_origin, done_bids = {}, [], {}, set()
  with torch.no_grad(), qc.floored_blockwise_scales(stats):
    for layer in layers:
      tl = time.time()
      for slot, (_, src) in enumerate(SLOTS):
        if (layer, slot) not in sites:
          continue
        w = expert_slot(ck, layer, src, n_experts)
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
        for b in bids:
          buffer_origin.pop(b, None)
        done_bids |= bids
        if spill is not None:
          for b in sorted(bids):
            spill.move(model.buffers[b])
        del w, qp
      per_layer.append({"layer": layer, "seconds": round(time.time() - tl, 1),
                        "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)})
      print("GRAFT_LAYER", json.dumps(per_layer[-1]), flush=True)
  ck.files.clear()
  mm = mm_lib.ModelModifier(model)
  model = mm._update_quantized_moe_ops(model)  # pylint: disable=protected-access
  packed = mm_lib._PackedBufferData(model)  # pylint: disable=protected-access
  if packed.packed_size < 256 * 1024:
    raise ValueError("unexpectedly small model")
  mm._serialize_model(model, packed, serialize_to_path=a.out)  # pylint: disable=protected-access
  spilled = 0
  if spill is not None:
    spilled = spill.moved
    del packed, model
    spill.close()
    os.remove(a.spill)
  rec = {"tflite": os.path.abspath(a.tflite), "ckpt": os.path.abspath(a.ckpt), "out": os.path.abspath(a.out),
         "out_bytes": os.path.getsize(a.out), "bits": a.bits, "block": a.block, "layers": layers,
         "spill": os.path.abspath(a.spill) if a.spill else None, "spilled_bytes": spilled,
         "sites": {f"{l}:{SLOTS[s][0]}": v for (l, s), v in sorted(sites.items())},
         "blockwise_int4_calls": stats.get("calls", 0), "block_scales": stats.get("scales", 0),
         "scales_floored": stats.get("floored", 0), "shards_read": sorted(ck.shards_read), "per_layer": per_layer,
         "seconds": round(time.time() - t0, 1),
         "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)}
  json.dump(rec, open(a.out + ".graft.json", "w"), indent=1)
  print("GRAFT_DONE", json.dumps({k: v for k, v in rec.items() if k not in ("sites", "per_layer")}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
