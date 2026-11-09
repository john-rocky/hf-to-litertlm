#!/usr/bin/env python3
"""nh_pack.py SRC OUT_DIR [--layers 0,1,5] [--experts fp32|xb4|xb8] [--block 32] [--fc fp32|int8|int4] [--fc-block 32]
                  [--lm-head fp32|int8|int4] [--embed fp32|int8] [--relu2 gelu_in|gelu_limit|dense|relu2op]
                  [--gelu-s 256] [--gelu-act gelu_tanh|gelu] [--dequant-out DIR]

Weight pre-pass for the Tensor API NemotronH writer (tensor_api/nemotronh): reads a NemotronH checkpoint in NVIDIA's
layout (backbone.layers.L.mixer.*, per-expert experts.E.{up,down}_proj.weight, bf16 / fp32 safetensors) and writes
  OUT_DIR/weights.bin   every tensor the decoder needs, 64-byte aligned
  OUT_DIR/index.tsv     name, kind, shape, data offset, data bytes, scales offset, scales count, block size
  OUT_DIR/config.tsv    dims + the layer list (original indices) and pattern
  OUT_DIR/moe_options.bin  the "moe" custom options (flexbuffer map)
  OUT_DIR/pack.json     what was packed, the source, floor counts, bytes, seconds, peak RSS
The quantization helpers are those of ../qwen3_30b_a3b_work/tapi_pack_moe.py ( ai-edge-quantizer's own min-max rules,
the 2^-14 block-scale floor, pack_data low nibble first; #11309 multi-axis form for the experts).

Experts are relu2 MLPs without a gate (expert(x) = down(relu(up x)^2)). The runtime's "moe" op computes
act(gate x) * (ff1 x) with act in {gelu, gelu_tanh} on the released runtimes. --relu2 picks how relu2 is written:
  gelu_in     gate = ff1 = up (one tensor, one set of scales), linear = down, the op's input scaled by s and the
              per-expert scale 1/s^2 (the writer does both): act(s u) (s u) / s^2 = gelu(s u) u / s, the same limit and
              bound as gelu_limit below, with no second copy of up's codes or scales; the CPU kernel's single-token
              path reads gate and ff1 of one row back to back, so the second read of the same row is a cache hit.
              --gelu-act picks gelu (erf) or gelu_tanh (default) for both gelu forms.
  gelu_limit  gate = s * up, ff1 = up, linear = down, per-expert scale 1/s, activation gelu_tanh:
              gelu(s u) u / s = u^2 Phi(s u) -> relu(u)^2 as s grows; |error| <= 0.17 / s^2 per hidden unit
              (max_t t^2 Phi(-t) ~ 0.166). s = --gelu-s, a power of two, so s * up is exact in every form (fp32 values;
              for int4 / int8 the codes are up's and the scales are s times up's). gate and ff1 then share the codes:
              index.tsv lists gate with up's data offset ("alias") and its own scales tensor.
  dense       no moe op: every expert in two fully connected layers per layer (up_all [E*I, D], down_all [D, E*I]),
              relu2 and the routing weights in between (the outside-the-op form; all experts computed per token).
  relu2op     gate = ff1 = up (one tensor), activation "relu2" = a runtime that implements act(gate x) * (ff1 x) with
              act = relu, i.e. relu(u) * u = relu(u)^2 (a kernel change; no released runtime has it).
Router weight, correction bias, norms, conv, A (= -exp(A_log)), D, dt_bias stay fp32.
--dequant-out (small slices only) writes an HF-layout checkpoint (config.json + model.safetensors, fp32) with the
dequantized values of every quantized tensor (experts as relu2 MLPs with the dequantized up / down), the reference input.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGN = 64
FLOOR = np.float32(2.0 ** -14)


def open_checkpoint(src):
    if os.environ.get("NH_READ", "buffered") == "buffered":
        # one large read() per tensor (st_read.py; bit-equal to get_tensor().float(), st_read_check.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from st_read import StReader
        r = StReader(src)
        return r.get_np, r.has, r.read_files
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

    def has(name):
        if wm:
            return name in wm
        if "model.safetensors" not in handles:
            handles["model.safetensors"] = safe_open(os.path.join(src, "model.safetensors"), framework="pt")
        return name in handles["model.safetensors"].keys()

    return get, has, read


def gran_of(block):
    from ai_edge_quantizer import qtyping
    return {32: qtyping.QuantGranularity.BLOCKWISE_32, 64: qtyping.QuantGranularity.BLOCKWISE_64,
            128: qtyping.QuantGranularity.BLOCKWISE_128, 256: qtyping.QuantGranularity.BLOCKWISE_256}[block]


def quant_bw4_rows(w2, block):
    """int4 blockwise over the rows of [R, C]: (q int8 [R, C], scale f32 [R, C/B], floored count)."""
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
    """int8 per row of [R, C]: (q int8 [R, C], scale f32 [R])."""
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


def moe_options(experts, top_k, model_dim, hidden_dim, weight_type, activation):
    from flatbuffers import flexbuffers
    return bytes(flexbuffers.Dumps({"num_experts": experts, "num_active_experts": top_k, "model_dim": model_dim,
                                    "hidden_dim": hidden_dim, "activation": activation, "weight_type": weight_type,
                                    "renormalized_top_weights": True}))


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--layers", default="", help="original layer indices to keep, comma separated (default all)")
    ap.add_argument("--experts", default="xb4", choices=["fp32", "xb8", "xb4"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--fc", default="fp32", choices=["fp32", "int8", "int4"])
    ap.add_argument("--fc-block", type=int, default=32)
    ap.add_argument("--lm-head", default="fp32", choices=["fp32", "int8", "int4"])
    ap.add_argument("--embed", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--relu2", default="gelu_limit", choices=["gelu_in", "gelu_limit", "dense", "relu2op"])
    ap.add_argument("--gelu-act", default="gelu_tanh", choices=["gelu_tanh", "gelu"])
    ap.add_argument("--gelu-s", type=float, default=256.0)
    ap.add_argument("--dequant-out")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    pattern = cfg["hybrid_override_pattern"]
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(len(pattern)))
    s = float(a.gelu_s)
    assert s > 0 and 2.0 ** round(np.log2(s)) == s, "--gelu-s must be a power of two"
    H, P, N, G = cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["ssm_state_size"], cfg["n_groups"]
    E, Im, Is = cfg["n_routed_experts"], cfg["moe_intermediate_size"], cfg["moe_shared_expert_intermediate_size"]
    D = cfg["hidden_size"]
    assert cfg.get("mlp_hidden_act") == "relu2" and cfg.get("mamba_hidden_act", "silu") == "silu"
    assert not cfg.get("tie_word_embeddings"), "the writer reads a separate lm_head"
    assert cfg.get("n_group", 1) == 1 and cfg.get("topk_group", 1) == 1
    assert cfg.get("moe_latent_size") is None
    lim = cfg.get("time_step_limit", [0.0, float("inf")])
    assert float(lim[0]) == 0.0 and float(lim[1]) == float("inf"), "the writer follows NVIDIA's default dt limit"
    dims = {"hidden": D, "pattern": "".join(pattern[l] for l in layers), "layer_ids": ",".join(map(str, layers)),
            "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"],
            "m_heads": H, "m_head_dim": P, "m_state": N, "m_groups": G, "m_conv": cfg["conv_kernel"],
            "m_chunk": cfg["chunk_size"], "experts": E, "top_k": cfg["num_experts_per_tok"], "moe_inter": Im,
            "shared_inter": Is, "vocab": cfg["vocab_size"], "eps": cfg.get("layer_norm_epsilon", 1e-5),
            "routed_scale": cfg["routed_scaling_factor"], "norm_topk": int(bool(cfg.get("norm_topk_prob", True))),
            "expert_kind": a.experts if a.relu2 != "dense" else "dense", "relu2": a.relu2, "gelu_s": s}
    wtype = {"fp32": "fp32", "xb8": "int8", "xb4": "int4"}[a.experts]
    act = {"gelu_in": a.gelu_act, "gelu_limit": a.gelu_act, "relu2op": "relu2", "dense": "none"}[a.relu2]
    os.makedirs(a.out, exist_ok=True)
    opts = moe_options(E, dims["top_k"], D, Im, wtype, act)
    open(os.path.join(a.out, "moe_options.bin"), "wb").write(opts)
    get, has, shards_read = open_checkpoint(src)
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
            f.write(memoryview(arr).cast("B"))
            off += arr.nbytes
            return start, arr.nbytes

        def put_fp32(name, w):
            w = np.asarray(w, dtype=np.float32)
            d0, dn = put(w)
            rows.append((name, "fp32", w.shape, d0, dn, 0, 0, 0))

        def note_floor(n):
            if n:
                stats["floored_tensors"] += 1
                stats["floored_scales"] += n

        def put_fc_arr(name, w, kind, deq_name=None):
            if kind == "fp32":
                put_fp32(name, w)
                dq = w
            elif kind == "int4":
                q, sc, n = quant_bw4_rows(w, a.fc_block)
                note_floor(n)
                d0, dn = put(pack4(q))
                s0, _ = put(sc)
                rows.append((name, "bw4", w.shape, d0, dn, s0, sc.size, a.fc_block))
                dq = (q.astype(np.float32).reshape(w.shape[0], -1, a.fc_block) * sc[:, :, None]).reshape(w.shape)
            else:
                q, sc = quant_c8_rows(w)
                d0, dn = put(q)
                s0, _ = put(sc)
                rows.append((name, "pc8", w.shape, d0, dn, s0, sc.size, 0))
                dq = q.astype(np.float32) * sc[:, None]
            if deq is not None and deq_name:
                deq[deq_name] = dq
            print(f"{name} {kind} {list(w.shape)}", flush=True)
            return dq

        def put_fc(name, kind):
            return put_fc_arr(name, get(name), kind, deq_name=name)

        def put_plain(name, w=None):
            w = get(name) if w is None else w
            put_fp32(name, w)
            if deq is not None:
                deq[name] = w

        w = get("backbone.embeddings.weight")
        if a.embed == "fp32":
            put_plain("backbone.embeddings.weight", w)
        else:
            q, sc = quant_c8_rows(w)
            d0, dn = put(q)
            s0, _ = put(sc)
            rows.append(("backbone.embeddings.weight", "pc8", w.shape, d0, dn, s0, sc.size, 0))
            if deq is not None:
                deq["backbone.embeddings.weight"] = q.astype(np.float32) * sc[:, None]
        del w
        for l in layers:
            tl = time.time()
            kd = pattern[l]
            p = f"backbone.layers.{l}."
            m = p + "mixer."
            put_plain(p + "norm.weight")
            if kd == "M":
                put_fc(m + "in_proj.weight", a.fc)
                cw = get(m + "conv1d.weight")
                put_fp32(m + "conv1d.weight2d", cw.reshape(cw.shape[0], -1))
                if deq is not None:
                    deq[m + "conv1d.weight"] = cw
                cb = get(m + "conv1d.bias") if has(m + "conv1d.bias") else np.zeros(cw.shape[0], np.float32)
                put_plain(m + "conv1d.bias", cb)
                put_plain(m + "dt_bias")
                alog = get(m + "A_log")
                put_fp32(m + "A", -np.exp(alog.astype(np.float64)).astype(np.float32))
                if deq is not None:
                    deq[m + "A_log"] = alog
                put_plain(m + "D")
                put_plain(m + "norm.weight")
                put_fc(m + "out_proj.weight", a.fc)
            elif kd == "*":
                for n in ("q", "k", "v", "o"):
                    put_fc(m + f"{n}_proj.weight", a.fc)
            elif kd == "E":
                put_plain(m + "gate.weight")
                put_plain(m + "gate.e_score_correction_bias")
                put_fc(m + "shared_experts.up_proj.weight", a.fc)
                put_fc(m + "shared_experts.down_proj.weight", a.fc)
                if a.relu2 == "dense":
                    up_all = np.concatenate([get(m + f"experts.{e}.up_proj.weight") for e in range(E)], 0)
                    dq = put_fc_arr(m + "experts.up_all", up_all, a.fc)
                    if deq is not None:
                        for e in range(E):
                            deq[m + f"experts.{e}.up_proj.weight"] = np.ascontiguousarray(dq[e * Im:(e + 1) * Im])
                    del up_all, dq
                    down_all = np.concatenate([get(m + f"experts.{e}.down_proj.weight") for e in range(E)], 1)
                    dq = put_fc_arr(m + "experts.down_all", down_all, a.fc)
                    if deq is not None:
                        for e in range(E):
                            deq[m + f"experts.{e}.down_proj.weight"] = np.ascontiguousarray(dq[:, e * Im:(e + 1) * Im])
                    del down_all, dq
                else:
                    for pm in ("up", "down"):
                        key = lambda e: m + f"experts.{e}.{pm}_proj.weight"
                        w0 = get(key(0))
                        out_f, in_f = w0.shape
                        name = m + f"experts.{pm}"
                        shape4 = (out_f, E, 1, in_f)
                        if a.experts == "fp32":
                            st = np.empty((out_f, E, in_f), np.float32)
                            st[:, 0, :] = w0
                            for e in range(1, E):
                                st[:, e, :] = get(key(e))
                            put_fp32(name, st.reshape(shape4))
                            if pm == "up" and a.relu2 == "gelu_limit":
                                put_fp32(m + "experts.gate", (st * np.float32(s)).reshape(shape4))
                            dq = st
                        else:
                            blk = a.block if a.experts == "xb4" else in_f
                            q = np.empty((out_f, E, in_f), np.int8)
                            sc = np.empty((out_f, E, in_f // blk), np.float32)
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
                                sc[:, e, :] = se
                            note_floor(floored)
                            d0, dn = put(pack4(q) if a.experts == "xb4" else q)
                            rows.append((name, a.experts, shape4, d0, dn, 0, 0, blk))
                            put_fp32(name + "_scales", sc.reshape(out_f, E, 1, in_f // blk))
                            if pm == "up" and a.relu2 == "gelu_limit":
                                # gate = s * up: the same codes (an alias of up's data) with s times up's scales
                                rows.append((m + "experts.gate", a.experts, shape4, d0, dn, 0, 0, blk))
                                put_fp32(m + "experts.gate_scales",
                                         (sc * np.float32(s)).reshape(out_f, E, 1, in_f // blk))
                            dq = None
                            if deq is not None:
                                dq = (q.astype(np.float32).reshape(out_f, E, in_f // blk, blk)
                                      * sc[..., None]).reshape(out_f, E, in_f)
                            del q, sc
                        if deq is not None:
                            for e in range(E):
                                deq[key(e)] = np.ascontiguousarray(dq[:, e, :])
                        del dq, w0
                        print(f"{name} {a.experts} {list(shape4)}", flush=True)
            layer_seconds.append(round(time.time() - tl, 2))
            print(f"LAYER {l} ({kd}) {layer_seconds[-1]} s, peak RSS {peak_rss_bytes() / 2**30:.2f} GiB", flush=True)
        put_plain("backbone.norm_f.weight")
        put_fc("lm_head.weight", a.lm_head)

    with open(os.path.join(a.out, "index.tsv"), "w") as f:
        for r in rows:
            f.write("\t".join([r[0], r[1], ",".join(str(x) for x in r[2])] + [str(x) for x in r[3:]]) + "\n")
    with open(os.path.join(a.out, "config.tsv"), "w") as f:
        for k, v in dims.items():
            f.write(f"{k}\t{v}\n")
    meta = {"source": src, "shards_read": sorted(shards_read), "dims": dims, "experts": a.experts, "block": a.block,
            "fc": a.fc, "fc_block": a.fc_block, "lm_head": a.lm_head, "embed": a.embed, "relu2": a.relu2,
            "gelu_s": s, "floor": stats, "bytes": off, "moe_options_hex": opts.hex(),
            "seconds": round(time.time() - t0, 1), "layer_seconds": layer_seconds, "tensors": len(rows),
            "peak_rss_bytes": peak_rss_bytes()}
    json.dump(meta, open(os.path.join(a.out, "pack.json"), "w"), indent=1)
    if deq is not None:
        from safetensors.numpy import save_file
        os.makedirs(a.dequant_out, exist_ok=True)
        save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in deq.items()},
                  os.path.join(a.dequant_out, "model.safetensors"), metadata={"format": "pt"})
        c2 = dict(cfg)
        json.dump(c2, open(os.path.join(a.dequant_out, "config.json"), "w"), indent=2)
        print(f"DEQUANT_DONE {len(deq)} tensors -> {a.dequant_out}")
    print(f"PACK_DONE {len(rows)} tensors {off} B in {meta['seconds']} s; experts {a.experts} relu2 {a.relu2} s {s}; "
          f"floor {stats}; peak RSS {meta['peak_rss_bytes'] / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
