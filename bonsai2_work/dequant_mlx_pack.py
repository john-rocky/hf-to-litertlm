#!/usr/bin/env python3
"""Dequantize the PrismML Ternary-Bonsai-2-27B MLX pack (2-bit affine, g128, Hadamard-rotated) into an
HF-layout bf16 checkpoint of Qwen3_5ForConditionalGeneration (text weights only) + the Hadamard sign vectors.

The pack stores, per rotated linear: weight U32 [rows, K/16] (16 x 2-bit codes per word, LSB-first),
scales F16 [rows, K/128], biases F16 (== -scales), signs F32 [K] in {-1, +1}. Codes {0,1,2} decode to
{-s, 0, +s} (q*scale + bias). The stored values are in the ROTATED basis: the runtime applies
x -> H_1024 (x * signs) / sqrt(1024) to every rotated linear's input, and the inverse to the embedding output.
We keep the rotated values verbatim (exact ternary) and carry the signs separately; the export inserts the
activation transform (see bonsai2_work/hadamard_patch.py).

Usage: dequant_mlx_pack.py <pack_dir> <out_dir>   (streams tensor by tensor; ~1 GB RAM)
"""
import json, os, sys, time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

pack, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
src = os.path.join(pack, "model.safetensors")
f = safe_open(src, framework="np")
keys = list(f.keys())
packed = sorted({k[:-len(".weight")] for k in keys if k.endswith(".weight") and (k[:-len(".weight")] + ".scales") in keys})
print(f"{len(keys)} tensors, {len(packed)} packed modules")

def hf_name(k):
    # language_model.model.X -> model.language_model.X ; language_model.lm_head -> lm_head
    if k.startswith("language_model.lm_head"):
        return k.replace("language_model.lm_head", "lm_head")
    assert k.startswith("language_model.model."), k
    return "model.language_model." + k[len("language_model.model."):]

def unpack2(w):  # w: uint32 [rows, K/16] -> int8 codes [rows, K]
    rows, words = w.shape
    q = np.empty((rows, words * 16), dtype=np.int8)
    for lane in range(16):
        q[:, lane::16] = ((w >> (2 * lane)) & 3).astype(np.int8)
    return q

signs = {}
stats = {"q3": 0, "zero": 0, "total": 0, "bias_mismatch": 0}
shard, shard_bytes, shard_id, index = {}, 0, 0, {}
SHARD_LIMIT = 4 * 1024**3
def flush():
    global shard, shard_bytes, shard_id
    if not shard: return
    name = f"model-{shard_id:05d}.safetensors"
    save_file(shard, os.path.join(out, name), metadata={"format": "pt"})
    for k in shard: index[k] = name
    print(f"  wrote {name} ({shard_bytes/1e9:.2f} GB, {len(shard)} tensors) t={time.time()-t0:.0f}s", flush=True)
    shard, shard_bytes, shard_id = {}, 0, shard_id + 1
def put(name, t):
    global shard_bytes
    shard[name] = t.contiguous(); shard_bytes += t.numel() * t.element_size()
    if shard_bytes >= SHARD_LIMIT: flush()

t0 = time.time()
for k in keys:
    if k.startswith("vision_tower."):
        continue  # text-only checkpoint; the vision tower is the stock Qwen3.8 tower, not exported here
    base = k.rsplit(".", 1)[0]
    if base in packed:
        suffix = k.rsplit(".", 1)[1]
        if suffix != "weight":
            continue  # scales/biases/signs consumed with the weight
        w = f.get_tensor(k); sc = f.get_tensor(base + ".scales").astype(np.float32); bi = f.get_tensor(base + ".biases").astype(np.float32)
        sg = f.get_tensor(base + ".signs") if (base + ".signs") in keys else None
        rows, K = w.shape[0], w.shape[1] * 16
        assert sc.shape == (rows, K // 128), (k, sc.shape)
        stats["bias_mismatch"] += int((bi != -sc).sum())
        q = unpack2(w)
        stats["q3"] += int((q == 3).sum()); stats["zero"] += int((q == 1).sum()); stats["total"] += q.size
        vals = (q.astype(np.float32) - 1.0).reshape(rows, K // 128, 128) * sc[..., None]
        put(hf_name(base + ".weight"), torch.from_numpy(vals.reshape(rows, K)).to(torch.bfloat16))
        if sg is not None:
            assert set(np.unique(sg)) <= {-1.0, 1.0}, k
            width = sg.shape[0]
            if width in signs:
                assert np.array_equal(signs[width][1], sg), f"sign vector differs within width {width}: {k} vs {signs[width][0]}"
            else:
                signs[width] = (k, sg.copy())
    else:
        a = f.get_tensor(k)
        if k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "language_model.model.norm.weight",
                       "q_norm.weight", "k_norm.weight")):
            a = a.astype(np.float32) - 1.0  # HF Qwen3_5RMSNorm uses (1 + weight); the mlx pack stores the plain weight
        if k.endswith("conv1d.weight"):
            a = a.transpose(0, 2, 1)  # mlx Conv1d layout [out, kernel, in] -> torch [out, in, kernel]
        put(hf_name(k), torch.from_numpy(np.ascontiguousarray(a).astype(np.float32)).to(torch.bfloat16))
flush()
json.dump({"metadata": {"format": "pt"}, "weight_map": index}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
save_file({f"signs_{w}": torch.from_numpy(v[1]) for w, v in signs.items()}, os.path.join(out, "hadamard_signs.safetensors"))
print("signs widths:", {w: v[0] for w, v in signs.items()})
print("stats:", stats, "| zero fraction %.4f | q==3 count %d | bias!=-scale %d" % (stats["zero"] / max(stats["total"], 1), stats["q3"], stats["bias_mismatch"]))
print("DONE", out, "in %.0fs" % (time.time() - t0))
