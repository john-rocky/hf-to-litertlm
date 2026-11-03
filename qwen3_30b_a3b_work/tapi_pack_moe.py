#!/usr/bin/env python3
"""tapi_pack_moe.py SRC OUT_DIR [--experts fp32|xa8|xb8|xb4] [--block 32] [--fc fp32|int8|int4]
                   [--lm-head fp32|int8|int4] [--layers N] [--dequant-out DIR]

Weight pre-pass for the Tensor API Qwen3-MoE writer (graph/qwen3moe_writer.cc): reads a Qwen3-MoE HF checkpoint
directory SRC (Qwen/Qwen3-30B-A3B's tensor names: per-expert model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight;
bf16 or fp32 safetensors, through model.safetensors.index.json when present) and writes
  OUT_DIR/weights.bin     every tensor the decoder needs, 64-byte aligned
  OUT_DIR/index.tsv       name, kind, shape, data offset, data bytes, scales offset, scales count, block size
  OUT_DIR/config.tsv      dims (hidden, layers, heads, kv heads, head dim, moe_inter, experts, top_k, vocab,
                          norm_topk, eps, rope theta, expert_kind)
  OUT_DIR/moe_options.bin the "moe" custom options (flexbuffer map: num_experts, num_active_experts, model_dim,
                          hidden_dim, activation silu, weight_type fp32|int8|int4, renormalized_top_weights true)
  OUT_DIR/pack.json       what was packed: the source directory, the shards read, the floor counts, bytes, seconds
                          per layer, peak RSS
SRC is a checkpoint directory, e.g. the snapshot directory of
  hf download Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
bf16 tensors are read one at a time and upcast to fp32 (exact).
Experts per layer are stacked in the op's layout: gate / up [I, E, 1, D] (= experts[e].gate_proj[o, i] at [o, e, 0, i]),
down [D, E, 1, I], i.e. row r = o * E + e. Kinds:
  fp32  stacked fp32.
  xb4   int4 blockwise, one scale per (output row, expert, block of --block inputs): ai-edge-quantizer's own min-max
        rule for BLOCKWISE granularity (symmetric, scale = max|w| / 7, rounded to a 7-bit mantissa through fp16),
        block scales raised to at least 2^-14 (the smallest normal fp16: a smaller scale rounds to 0 or a subnormal
        value there, and the block's weights would be divided by it), its uniform_quantize and pack_data (two
        values per byte, low nibble first). Scales float32 [out, E, 1, in/B] in <name>_scales = the form LiteRT
        #11309 writes for MoE experts (multi-axis, quantized dims (0, 1)).
  xb8   int8, one scale per (output row, expert): CHANNELWISE min-max on the [out*E, in] rows, no rounding;
        scales float32 [out, E, 1, 1] in <name>_scales (#11309's int8 MoE form, block = in).
  xa8   int8 with affine params on the tensor: one scale per output row shared by all experts (CHANNELWISE on the
        [out, E*in] rows), so the tensor's affine params (qdim 0) are the true dequantization; the op's
        <name>_scales [out, E, 1, 1] repeats them per expert.
xb4 / xb8 quantize each expert's [out, in] block on its own (every scale is per row, so this equals quantizing the
stacked [out*E, in] rows) into one preallocated int8 array: the peak is one layer-projection's int8 copy plus one
expert in fp32. fp32 and xa8 hold the stacked fp32 tensor.
Attention projections follow --fc (int4 = int4 blockwise 2-D with --fc-block, int8 = int8 per channel), lm_head
--lm-head, router / norms / embedding stay fp32. --dequant-out (small models only, at most 4 layers) writes an HF
checkpoint (config.json + model.safetensors, fp32) whose experts (and quantized FC weights) are the dequantized
values: run Transformers on it to get the reference a correct runtime must match.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

ALIGN = 64
FLOOR = np.float32(2.0 ** -14)


def resolve_src(src):
    """SRC must be a checkpoint directory (with config.json)."""
    path = os.path.abspath(src)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise SystemExit(f"{src}: not a checkpoint directory (no config.json)")
    return path, {"src": path}


def open_checkpoint(src):
    from safetensors import safe_open
    idx = os.path.join(src, "model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
    handles = {}
    read = set()

    def get(name):
        fname = wm[name] if wm else "model.safetensors"
        if fname not in handles:
            handles[fname] = safe_open(os.path.join(src, fname), framework="pt")
        read.add(fname)
        return handles[fname].get_tensor(name).float().numpy()

    return get, read


def gran_of(block):
    from ai_edge_quantizer import qtyping
    return {32: qtyping.QuantGranularity.BLOCKWISE_32, 64: qtyping.QuantGranularity.BLOCKWISE_64,
            128: qtyping.QuantGranularity.BLOCKWISE_128, 256: qtyping.QuantGranularity.BLOCKWISE_256}[block]


def quant_bw4_rows(w2, block):
    """int4 blockwise over the rows of a 2-D array [R, C]: returns (q int8 [R, C], scale f32 [R, C/B], floored)."""
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    R, C = w2.shape
    assert C % block == 0, (w2.shape, block)
    wb = w2.reshape(R, C // block, block)
    _, scale = u.tensor_zp_scale_from_min_max(wb.min(axis=2), wb.max(axis=2), 4, True, gran_of(block))
    low = scale < FLOOR
    floored = int(low.sum())
    if floored:
        scale = np.where(low, FLOOR, scale).astype(np.float32)
    params = qtyping.UniformQuantParams(scale=scale, zero_point=np.zeros_like(scale, dtype=np.int32), num_bits=4,
                                        symmetric=True, quantized_dimension=1, block_size=block)
    q = np.asarray(u.uniform_quantize(w2, params, is_blockwise_quant=True)).astype(np.int8)
    return q, scale.astype(np.float32), floored


def quant_c8_rows(w2):
    """int8 per row of a 2-D array [R, C]: returns (q int8 [R, C], scale f32 [R])."""
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    _, scale = u.tensor_zp_scale_from_min_max(w2.min(axis=1), w2.max(axis=1), 8, True,
                                              qtyping.QuantGranularity.CHANNELWISE)
    params = qtyping.UniformQuantParams(scale=scale[:, None], zero_point=np.zeros((w2.shape[0], 1), np.int32),
                                        num_bits=8, symmetric=True, quantized_dimension=0)
    q = np.asarray(u.uniform_quantize(w2, params)).astype(np.int8)
    return q, scale.astype(np.float32)


def pack4(q):
    from ai_edge_quantizer.transformations import transformation_utils as tu
    return np.asarray(tu.pack_data(4, np.ravel(q).view(np.uint8)), dtype=np.uint8)


def moe_options(experts, top_k, model_dim, hidden_dim, weight_type):
    from flatbuffers import flexbuffers
    return bytes(flexbuffers.Dumps({"num_experts": experts, "num_active_experts": top_k, "model_dim": model_dim,
                                    "hidden_dim": hidden_dim, "activation": "silu", "weight_type": weight_type,
                                    "renormalized_top_weights": True}))


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--experts", default="xb4", choices=["fp32", "xa8", "xb8", "xb4"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--fc", default="fp32", choices=["fp32", "int8", "int4"])
    ap.add_argument("--lm-head", default="fp32", choices=["fp32", "int8", "int4"])
    ap.add_argument("--fc-block", type=int, default=32)
    ap.add_argument("--layers", type=int, help="keep the first N layers")
    ap.add_argument("--dequant-out", help="write an HF checkpoint with the dequantized weights here (<= 4 layers)")
    a = ap.parse_args()

    src, src_info = resolve_src(a.src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    dims = {"hidden": cfg["hidden_size"], "layers": a.layers or cfg["num_hidden_layers"],
            "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"],
            "head_dim": cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            "moe_inter": cfg["moe_intermediate_size"], "experts": cfg["num_experts"],
            "top_k": cfg["num_experts_per_tok"], "vocab": cfg["vocab_size"],
            "norm_topk": int(bool(cfg["norm_topk_prob"])), "eps": cfg["rms_norm_eps"],
            "rope_theta": cfg.get("rope_theta") or (cfg.get("rope_parameters") or {}).get("rope_theta"),
            "expert_kind": a.experts}
    assert cfg["hidden_act"] == "silu" and not cfg.get("mlp_only_layers") and cfg.get("decoder_sparse_step", 1) == 1
    assert not cfg.get("tie_word_embeddings"), "the writer reads a separate lm_head"
    assert dims["layers"] <= cfg["num_hidden_layers"]
    if a.dequant_out and dims["layers"] > 4:
        raise SystemExit("--dequant-out holds every dequantized tensor in memory: at most 4 layers")
    D, I, E = dims["hidden"], dims["moe_inter"], dims["experts"]
    wtype = {"fp32": "fp32", "xa8": "int8", "xb8": "int8", "xb4": "int4"}[a.experts]
    os.makedirs(a.out, exist_ok=True)
    opts = moe_options(E, dims["top_k"], D, I, wtype)
    open(os.path.join(a.out, "moe_options.bin"), "wb").write(opts)
    get, shards_read = open_checkpoint(src)
    deq = {} if a.dequant_out else None

    stats = {"floored_tensors": 0, "floored_scales": 0}
    t0 = time.time()
    layer_seconds = []
    rows, off = [], 0
    with open(os.path.join(a.out, "weights.bin"), "wb") as f:
        def put(arr):
            nonlocal off
            pad = (-off) % ALIGN
            if pad:
                f.write(b"\0" * pad)
                off += pad
            start = off
            arr = np.ascontiguousarray(arr)
            f.write(memoryview(arr).cast("B"))  # no bytes copy of a large array
            off += arr.nbytes
            return start, arr.nbytes

        def put_fp32(name, w):
            d0, dn = put(w.astype(np.float32, copy=False))
            rows.append((name, "fp32", w.shape, d0, dn, 0, 0, 0))

        def note_floor(n):
            if n:
                stats["floored_tensors"] += 1
                stats["floored_scales"] += n

        def put_fc(name, kind):
            w = get(name)
            if kind == "fp32":
                put_fp32(name, w)
                if deq is not None:
                    deq[name] = w
            elif kind == "int4":
                q, s, n = quant_bw4_rows(w, a.fc_block)
                note_floor(n)
                d0, dn = put(pack4(q))
                s0, _ = put(s)
                rows.append((name, "bw4", w.shape, d0, dn, s0, s.size, a.fc_block))
                if deq is not None:
                    deq[name] = (q.astype(np.float32).reshape(w.shape[0], -1, a.fc_block)
                                 * s[:, :, None]).reshape(w.shape)
            else:
                q, s = quant_c8_rows(w)
                d0, dn = put(q)
                s0, _ = put(s)
                rows.append((name, "pc8", w.shape, d0, dn, s0, s.size, 0))
                if deq is not None:
                    deq[name] = q.astype(np.float32) * s[:, None]
            print(f"{name} {kind} {list(w.shape)}", flush=True)

        def put_norm(name):
            w = get(name)
            put_fp32(name, w)
            if deq is not None:
                deq[name] = w

        w = get("model.embed_tokens.weight")
        put_fp32("model.embed_tokens.weight", w)
        if deq is not None:
            deq["model.embed_tokens.weight"] = w
        del w
        for l in range(dims["layers"]):
            tl = time.time()
            p = f"model.layers.{l}."
            put_norm(p + "input_layernorm.weight")
            put_norm(p + "post_attention_layernorm.weight")
            put_norm(p + "self_attn.q_norm.weight")
            put_norm(p + "self_attn.k_norm.weight")
            put_norm(p + "mlp.gate.weight")
            for n in ("q", "k", "v", "o"):
                put_fc(p + f"self_attn.{n}_proj.weight", a.fc)
            for m in ("gate", "up", "down"):
                key = lambda e: p + f"mlp.experts.{e}.{m}_proj.weight"
                w0 = get(key(0))
                out_f, in_f = w0.shape
                name = p + f"mlp.experts.{m}"
                shape4 = (out_f, E, 1, in_f)
                dq = None
                if a.experts in ("fp32", "xa8"):
                    st = np.empty((out_f, E, in_f), np.float32)  # [out, E, in]
                    st[:, 0, :] = w0
                    for e in range(1, E):
                        st[:, e, :] = get(key(e))
                    if a.experts == "fp32":
                        put_fp32(name, st.reshape(shape4))
                        dq = st
                    else:  # xa8
                        q, s = quant_c8_rows(st.reshape(out_f, E * in_f))
                        d0, dn = put(q)
                        s0, _ = put(s)
                        rows.append((name, "xa8", shape4, d0, dn, s0, s.size, 0))
                        put_fp32(name + "_scales", np.repeat(s[:, None], E, axis=1).reshape(out_f, E, 1, 1))
                        if deq is not None:
                            dq = (q.astype(np.float32) * s[:, None]).reshape(out_f, E, in_f)
                    del st
                else:
                    blk = a.block if a.experts == "xb4" else in_f
                    q = np.empty((out_f, E, in_f), np.int8)
                    s = np.empty((out_f, E, in_f // blk), np.float32)
                    floored = 0
                    for e in range(E):
                        we = w0 if e == 0 else get(key(e))
                        if a.experts == "xb4":
                            qe, se, n = quant_bw4_rows(we, blk)
                            floored += n
                        else:
                            qe, se = quant_c8_rows(we)
                            se = se[:, None]
                        q[:, e, :] = qe
                        s[:, e, :] = se
                    note_floor(floored)
                    d0, dn = put(pack4(q) if a.experts == "xb4" else q)
                    rows.append((name, a.experts, shape4, d0, dn, 0, 0, blk))
                    put_fp32(name + "_scales", s.reshape(out_f, E, 1, in_f // blk))
                    if deq is not None:
                        dq = (q.astype(np.float32).reshape(out_f, E, in_f // blk, blk)
                              * s[..., None]).reshape(out_f, E, in_f)
                    del q, s
                if deq is not None:
                    for e in range(E):
                        deq[p + f"mlp.experts.{e}.{m}_proj.weight"] = np.ascontiguousarray(dq[:, e, :])
                del dq, w0
                print(f"{name} {a.experts} {list(shape4)}", flush=True)
            layer_seconds.append(round(time.time() - tl, 2))
            print(f"LAYER {l} {layer_seconds[-1]} s, peak RSS {peak_rss_bytes() / 2**30:.2f} GiB", flush=True)
        put_norm("model.norm.weight")
        put_fc("lm_head.weight", a.lm_head)

    with open(os.path.join(a.out, "index.tsv"), "w") as f:
        for r in rows:
            f.write("\t".join([r[0], r[1], ",".join(str(x) for x in r[2])] + [str(x) for x in r[3:]]) + "\n")
    with open(os.path.join(a.out, "config.tsv"), "w") as f:
        for k, v in dims.items():
            f.write(f"{k}\t{v}\n")
    meta = {"source": src_info, "shards_read": sorted(shards_read),
            "dims": dims, "experts": a.experts, "block": a.block, "fc": a.fc, "lm_head": a.lm_head,
            "fc_block": a.fc_block, "floor": stats, "bytes": off, "moe_options_hex": opts.hex(),
            "seconds": round(time.time() - t0, 1), "layer_seconds": layer_seconds, "tensors": len(rows),
            "peak_rss_bytes": peak_rss_bytes()}
    json.dump(meta, open(os.path.join(a.out, "pack.json"), "w"), indent=1)
    if deq is not None:
        from safetensors.numpy import save_file
        os.makedirs(a.dequant_out, exist_ok=True)
        save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in deq.items()},
                  os.path.join(a.dequant_out, "model.safetensors"), metadata={"format": "pt"})
        c2 = dict(cfg)
        c2["num_hidden_layers"] = dims["layers"]
        json.dump(c2, open(os.path.join(a.dequant_out, "config.json"), "w"), indent=2)
        print(f"DEQUANT_DONE {len(deq)} tensors -> {a.dequant_out}")
    print(f"PACK_DONE {len(rows)} tensors {off} B in {meta['seconds']} s; experts {a.experts}; floor {stats}; "
          f"peak RSS {meta['peak_rss_bytes'] / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
