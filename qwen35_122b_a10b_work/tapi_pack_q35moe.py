#!/usr/bin/env python3
"""tapi_pack_q35moe.py SRC OUT_DIR [--layers all|0,3] [--experts xb4|xb8|fp32] [--block 32] [--fc int8|int4|fp32]
                      [--fc-block 32] [--lm-head int8|int4|fp32] [--embed fp32|int8]

Weight pre-pass for the Tensor API Qwen3.5-MoE writer (tensor_api/qwen35moe): reads the Qwen3_5MoeForConditionalGeneration
checkpoint (model.language_model.* text decoder, bf16 safetensors through model.safetensors.index.json) one tensor at a
time - the fused experts one expert at a time through safe_open(...).get_slice - upcasts to fp32 (exact), quantizes, and
appends to OUT_DIR/weights.bin. Nothing holds the whole model: the peak is one expert projection's int8 copy
(out x E x in bytes, 0.8 GB for 1024 x 256 x 3072) plus one expert in fp32. Writes
  OUT_DIR/weights.bin     every tensor the decoder needs, 64-byte aligned
  OUT_DIR/index.tsv       name, kind, shape, data offset, data bytes, scales offset, scales count, block size
                          (the 8 columns of ../qwen3_30b_a3b_work/tapi_pack_moe.py, so the writer's reader is shared)
  OUT_DIR/config.tsv      dims and the packed layers (layer_types = linear|full per packed layer, src_layers = HF index)
  OUT_DIR/moe_options.bin the "moe" custom options (flexbuffer map, activation silu, renormalized_top_weights true)
  OUT_DIR/gdn_options.bin the "gated_delta_update" custom options (flexbuffer map mode 0, state_dtype float32 = the bytes
                          litert-torch main gated_delta_rule.py _gated_delta_update_custom_options writes)
  OUT_DIR/pack.json       what was packed, the source (snapshot, revision, shards read with their manifest sha256), floor
                          counts, bytes, seconds per layer, peak RSS
SRC is a checkpoint directory: config.json, model.safetensors.index.json and the shards it names.

Packed names (i = packed layer index 0..n-1; the HF layer of each is config.tsv src_layers):
  embed [V,D], final_norm [D] = 1 + w, lm_head [V,D]
  L<i>.in_norm, L<i>.post_norm [D] = 1 + w (Qwen3_5MoeRMSNorm is zero-centered: x_hat * (1 + w); folded here in fp32)
  linear_attention: L<i>.qkv [2*Kd+Vd, D], L<i>.z [Vd, D], L<i>.b [Hv, D] fp32, L<i>.a [Hv, D] fp32,
                    L<i>.conv0..3 [2*Kd+Vd] fp32 (tap j = conv1d.weight[:, 0, j]), L<i>.negA [Hv] = -exp(A_log)
                    (fp32, as HF computes it), L<i>.dt_bias [Hv] fp32, L<i>.gnorm [Dv] fp32 (gated norm: plain w),
                    L<i>.out [D, Vd]
  full_attention:   L<i>.q [H*hd, D] and L<i>.qgate [H*hd, D] (q_proj rows split per head: [q 256 | gate 256] x 32),
                    L<i>.k [KV*hd, D], L<i>.v [KV*hd, D], L<i>.qn / L<i>.kn [hd] = 1 + w, L<i>.o [D, H*hd]
  MoE:              L<i>.router [E, D] fp32; L<i>.ex_gate / L<i>.ex_up [I, E, 1, D] and L<i>.ex_down [D, E, 1, I]
                    (+ <name>_scales [out, E, 1, in/B] fp32 for xb4 / xb8) in the moe op's row order r = o * E + e
                    (gate = gate_up_proj[e, :I], up = gate_up_proj[e, I:], down = down_proj[e]);
                    L<i>.sh_gate / L<i>.sh_up [Ish, D], L<i>.sh_down [D, Ish], L<i>.sh_g [1, D] fp32
Kinds (as ../qwen3_30b_a3b_work/tapi_pack_moe.py): fp32; pc8 = int8 per output channel; bw4 = int4 blockwise 2-D (--fc-block);
  xb4 = expert int4 blockwise (ai-edge-quantizer's min-max rule for BLOCKWISE granularity: symmetric, max|w| / 7, 7-bit
  mantissa rounding; block scales below 2^-14 are raised to 2^-14 (none can be 0); pack_data low nibble
  first; scales float32 [out, E, 1, in/B] = LiteRT #11309's MoE form); xb8 = expert int8 per (output row, expert).
FC weights (qkv, z, out, q, qgate, k, v, o, shared expert) follow --fc; the small gating projections b / a, the router,
conv, norms, A / dt_bias stay fp32; lm_head follows --lm-head; embed follows --embed (int8 = pc8 rows, EMBEDDING_LOOKUP
hybrid form).
"""
import argparse
import json
import os
import resource
import struct
import sys
import time

import numpy as np

ALIGN = 64
FLOOR = np.float32(2.0 ** -14)
ROW_CHUNK = 16384  # rows of the embedding / lm_head per read (16,384 x 3,072 fp32 = 201 MB)
PFX = "model.language_model."


def resolve_src(src):
    if src.startswith("hf:") or not os.path.isdir(src):
        raise SystemExit(f"SRC must be a checkpoint directory: {src}")
    return os.path.abspath(src), {"src": os.path.abspath(src)}


class Checkpoint:
    """The checkpoint. Small tensors through safetensors; the big reads (one expert of a fused [E, ...] tensor, row
    chunks of the embedding / lm_head) by explicit preads of their contiguous bytes, PREFETCH bytes at a time.
    safetensors' slices fault the file's pages in, in small reads that a shared USB hard disk served at 6.5 MB/s.
    bf16 / f16 -> fp32 is exact, so the values are the safetensors reader's (checked byte for byte)."""
    PREFETCH = 256 << 20

    def __init__(self, src):
        from safetensors import safe_open
        self._open = safe_open
        self.src = src
        self.wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}
        self.read = set()
        self.meta = {}  # file -> (fd, start of the data, header)
        self.buf = None  # (tensor name, first leading row, rows, raw bytes)

    def _m(self, name):
        fname = self.wm[name]
        if fname not in self.meta:
            fd = os.open(os.path.join(self.src, fname), os.O_RDONLY)
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            self.meta[fname] = (fd, 8 + n, json.loads(os.pread(fd, n, 8)))
        self.read.add(fname)
        fd, start, hdr = self.meta[fname]
        t = hdr[name]
        return fd, start + t["data_offsets"][0], start + t["data_offsets"][1], t["dtype"], tuple(t["shape"])

    @staticmethod
    def _pread(fd, n, off):
        parts, got = [], 0
        while got < n:
            b = os.pread(fd, min(n - got, 1 << 30), off + got)
            if not b:
                raise IOError(f"short read at {off + got}: {got} of {n} B")
            parts.append(b)
            got += len(b)
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def _raw_rows(self, name, r0, r1):
        """The raw bytes of leading rows [r0, r1), served from a buffer of up to PREFETCH bytes read ahead."""
        fd, b0, b1, dt, shape = self._m(name)
        rb = (b1 - b0) // shape[0]
        if self.buf is not None:
            bn, br0, brn, raw = self.buf
            if bn == name and br0 <= r0 and r1 <= br0 + brn:
                return memoryview(raw)[(r0 - br0) * rb:(r1 - br0) * rb], dt, shape
        k = max(r1 - r0, min(shape[0] - r0, max(1, self.PREFETCH // rb)))
        self.buf = None
        raw = self._pread(fd, k * rb, b0 + r0 * rb)
        self.buf = (name, r0, k, raw)
        return memoryview(raw)[:(r1 - r0) * rb], dt, shape

    @staticmethod
    def _f32(raw, dt, shape):
        if dt == "BF16":
            return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "F16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        raise ValueError(f"dtype {dt}")

    def _h(self, name):
        fname = self.wm[name]
        if fname not in self.handles:
            self.handles[fname] = self._open(os.path.join(self.src, fname), framework="pt")
        self.read.add(fname)
        return self.handles[fname]

    def get(self, name):
        return self._h(name).get_tensor(name).float().numpy()

    def get_row(self, name, e):
        """One leading-axis slice (one expert of a fused [E, ...] tensor) as fp32."""
        raw, dt, shape = self._raw_rows(name, e, e + 1)
        return self._f32(raw, dt, shape[1:])

    def get_rows(self, name, r0, r1):
        """Rows [r0, r1) of a 2-D tensor as fp32 (the embedding and lm_head are read in row chunks)."""
        raw, dt, shape = self._raw_rows(name, r0, r1)
        return self._f32(raw, dt, (r1 - r0,) + shape[1:])

    def shape(self, name):
        return tuple(self._h(name).get_slice(name).get_shape())


def gran_of(block):
    from ai_edge_quantizer import qtyping
    return {32: qtyping.QuantGranularity.BLOCKWISE_32, 64: qtyping.QuantGranularity.BLOCKWISE_64,
            128: qtyping.QuantGranularity.BLOCKWISE_128, 256: qtyping.QuantGranularity.BLOCKWISE_256}[block]


def quant_bw4_rows(w2, block):
    """int4 blockwise over the rows of [R, C]: (q int8 [R, C], scale f32 [R, C/B], number of floored scales)."""
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


def moe_options(experts, top_k, model_dim, hidden_dim, weight_type):
    from flatbuffers import flexbuffers
    return bytes(flexbuffers.Dumps({"num_experts": experts, "num_active_experts": top_k, "model_dim": model_dim,
                                    "hidden_dim": hidden_dim, "activation": "silu", "weight_type": weight_type,
                                    "renormalized_top_weights": True}))


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def write_index(a, src_info, dims, flags, part_names, opts, gdn_opts):
    """--part index: every part's pack.json must exist with the same flags and source, and its weights.bin the size it
    recorded; then OUT/parts.txt (the parts in packed order), config.tsv, the two option files and a pack.json that
    sums the parts. Nothing is written when a part is missing or differs."""
    metas, bad = {}, []
    for n in part_names:
        pj = os.path.join(a.out, n, "pack.json")
        if not os.path.exists(pj):
            bad.append(f"{n}: no pack.json")
            continue
        m = json.load(open(pj))
        size = os.path.getsize(os.path.join(a.out, n, "weights.bin"))
        if m.get("flags") != flags:
            bad.append(f"{n}: flags {m.get('flags')} != {flags}")
        if m.get("source") != src_info:
            bad.append(f"{n}: source {m.get('source')} != {src_info}")
        if size != m["bytes"]:
            bad.append(f"{n}: weights.bin {size} B != recorded {m['bytes']} B")
        metas[n] = m
    if bad:
        print("INDEX_REFUSED\n  " + "\n  ".join(bad))
        return 1
    open(os.path.join(a.out, "moe_options.bin"), "wb").write(opts)
    open(os.path.join(a.out, "gdn_options.bin"), "wb").write(gdn_opts)
    with open(os.path.join(a.out, "config.tsv"), "w") as f:
        for k, v in dims.items():
            f.write(f"{k}\t{v}\n")
    with open(os.path.join(a.out, "parts.txt"), "w") as f:
        f.write("".join(n + "\n" for n in part_names))
    floor = {"floored_tensors": sum(m["floor"]["floored_tensors"] for m in metas.values()),
             "floored_scales": sum(m["floor"]["floored_scales"] for m in metas.values())}
    meta = {"source": src_info, "parts": part_names, "dims": dims, "flags": flags, **flags, "floor": floor,
            "bytes": sum(m["bytes"] for m in metas.values()),
            "tensors": sum(m["tensors"] for m in metas.values()),
            "shards_read": sorted({s for m in metas.values() for s in m["shards_read"]}),
            "part_seconds": {n: m["seconds"] for n, m in metas.items()},
            "part_peak_rss_bytes": {n: m["peak_rss_bytes"] for n, m in metas.items()},
            "moe_options_hex": opts.hex(), "gdn_options_hex": gdn_opts.hex()}
    json.dump(meta, open(os.path.join(a.out, "pack.json"), "w"), indent=1)
    print(f"INDEX_DONE {len(part_names)} parts, {meta['tensors']} tensors, {meta['bytes']} B; floor {floor}; "
          f"max part peak RSS {max(meta['part_peak_rss_bytes'].values()) / 2**30:.2f} GiB")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated HF layer indices, e.g. 0,3")
    ap.add_argument("--experts", default="xb4", choices=["fp32", "xb8", "xb4"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--fc", default="int8", choices=["fp32", "int8", "int4"])
    ap.add_argument("--fc-block", type=int, default=32)
    ap.add_argument("--lm-head", default="int8", choices=["fp32", "int8", "int4"])
    ap.add_argument("--embed", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--part", default=None,
                    help="per-layer parts instead of one weights.bin: 'globals' (embedding, final norm, lm_head -> "
                         "OUT/globals/), 'layer:<HF index>' (one layer -> OUT/L<HF index, 2 digits>/, tensors named by "
                         "its packed index), 'index' (checks every part's pack.json, writes OUT/parts.txt, config.tsv, "
                         "the options and pack.json); the writer reads OUT/parts.txt")
    a = ap.parse_args()

    src, src_info = resolve_src(a.src)
    full_cfg = json.load(open(os.path.join(src, "config.json")))
    cfg = full_cfg["text_config"]
    n_all = cfg["num_hidden_layers"]
    layers = list(range(n_all)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    assert all(0 <= l < n_all for l in layers), layers
    rp = cfg["rope_parameters"]
    hd = cfg["head_dim"]
    dims = {"hidden": cfg["hidden_size"], "layers": len(layers),
            "layer_types": ",".join(cfg["layer_types"][l].replace("_attention", "") for l in layers),
            "src_layers": ",".join(str(l) for l in layers),
            "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"], "head_dim": hd,
            "rotary_dim": int(hd * rp.get("partial_rotary_factor", 1.0)), "rope_theta": rp["rope_theta"],
            "eps": cfg["rms_norm_eps"], "lin_k_heads": cfg["linear_num_key_heads"],
            "lin_v_heads": cfg["linear_num_value_heads"], "lin_k_dim": cfg["linear_key_head_dim"],
            "lin_v_dim": cfg["linear_value_head_dim"], "conv_k": cfg["linear_conv_kernel_dim"],
            "moe_inter": cfg["moe_intermediate_size"], "experts": cfg["num_experts"],
            "top_k": cfg["num_experts_per_tok"], "shared_inter": cfg["shared_expert_intermediate_size"],
            "vocab": cfg["vocab_size"], "expert_kind": a.experts}
    assert cfg["hidden_act"] == "silu" and not cfg.get("mlp_only_layers") and cfg.get("attn_output_gate")
    assert not full_cfg.get("tie_word_embeddings") and not cfg.get("attention_bias")
    assert rp.get("rope_type", "default") == "default", rp
    D, I, E, Ish = dims["hidden"], dims["moe_inter"], dims["experts"], dims["shared_inter"]
    Hq, hdim = dims["heads"], hd
    wtype = {"fp32": "fp32", "xb8": "int8", "xb4": "int4"}[a.experts]
    os.makedirs(a.out, exist_ok=True)
    opts = moe_options(E, dims["top_k"], D, I, wtype)
    from flatbuffers import flexbuffers
    gdn_opts = bytes(flexbuffers.Dumps({"mode": 0, "state_dtype": "float32"}))  # = litert-torch gated_delta_rule.py
    flags = {"experts": a.experts, "block": a.block, "fc": a.fc, "fc_block": a.fc_block, "lm_head": a.lm_head,
             "embed": a.embed}
    part_names = ["globals"] + [f"L{l:02d}" for l in layers]
    if a.part == "index":
        return write_index(a, src_info, dims, flags, part_names, opts, gdn_opts)
    if a.part is None:
        out_dir = a.out
        open(os.path.join(a.out, "moe_options.bin"), "wb").write(opts)
        open(os.path.join(a.out, "gdn_options.bin"), "wb").write(gdn_opts)
        todo_layers = list(enumerate(layers))
    elif a.part == "globals":
        out_dir = os.path.join(a.out, "globals")
        todo_layers = []
    elif a.part.startswith("layer:"):
        hf = int(a.part.split(":", 1)[1])
        assert hf in layers, (hf, layers)
        out_dir = os.path.join(a.out, f"L{hf:02d}")
        todo_layers = [(layers.index(hf), hf)]
    else:
        raise SystemExit(f"--part {a.part}?")
    os.makedirs(out_dir, exist_ok=True)
    ck = Checkpoint(src)

    stats = {"floored_tensors": 0, "floored_scales": 0}
    t0 = time.time()
    layer_seconds = []
    rows, off = [], 0
    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
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
            w = np.ascontiguousarray(w, dtype=np.float32)
            d0, dn = put(w)
            rows.append((name, "fp32", w.shape, d0, dn, 0, 0, 0))

        def put_stream(chunks):
            """One aligned start, then the chunks back to back = the same bytes as put(concatenate(chunks))."""
            nonlocal off
            pad = (-off) % ALIGN
            if pad:
                f.write(b"\0" * pad)
                off += pad
            start = off
            for c in chunks:
                c = np.ascontiguousarray(c)
                f.write(memoryview(c).cast("B"))
                off += c.nbytes
            return start, off - start

        def put_rows_chunked(name, src_name, kind, block):
            """A 2-D [R, C] tensor (embedding, lm_head) read and quantized ROW_CHUNK rows at a time. Every kind here is
            per row (fp32; pc8 = one scale per row; bw4 = blocks inside a row), so the bytes equal the whole-tensor
            path's; the peak is one chunk instead of the whole fp32 tensor (3.05 GB for 248,320 x 3,072)."""
            R, C = ck.shape(src_name)
            spans = [(r, min(r + ROW_CHUNK, R)) for r in range(0, R, ROW_CHUNK)]
            if kind == "fp32":
                d0, dn = put_stream(ck.get_rows(src_name, r0, r1) for r0, r1 in spans)
                rows.append((name, "fp32", (R, C), d0, dn, 0, 0, 0))
                return
            scales = []
            floored = [0]

            def qchunks():
                for r0, r1 in spans:
                    w = ck.get_rows(src_name, r0, r1)
                    if kind == "pc8":
                        q, sc = quant_c8_rows(w)
                    else:
                        q, sc, n = quant_bw4_rows(w, block)
                        floored[0] += n
                        q = pack4(q)
                    scales.append(sc)
                    yield q

            if kind == "bw4":
                assert (C % 2 == 0) and (C % block == 0), (src_name, C, block)
            d0, dn = put_stream(qchunks())
            note_floor(floored[0])
            s = np.concatenate(scales, axis=0)
            s0, _ = put(s)
            rows.append((name, kind, (R, C), d0, dn, s0, s.size, block if kind == "bw4" else 0))

        def note_floor(n):
            if n:
                stats["floored_tensors"] += 1
                stats["floored_scales"] += n

        def put_fc(name, w, kind):
            if kind == "fp32":
                put_fp32(name, w)
            elif kind == "int4":
                q, s, n = quant_bw4_rows(w, a.fc_block)
                note_floor(n)
                d0, dn = put(pack4(q))
                s0, _ = put(s)
                rows.append((name, "bw4", w.shape, d0, dn, s0, s.size, a.fc_block))
            else:
                q, s = quant_c8_rows(w)
                d0, dn = put(q)
                s0, _ = put(s)
                rows.append((name, "pc8", w.shape, d0, dn, s0, s.size, 0))

        def put_experts_pair(names, src_name, row_slices, out_f, in_f):
            """Several [out_f, in_f] row slices of each expert of src_name (gate = gate_up_proj[e, :I] and
            up = gate_up_proj[e, I:]) from ONE read of that expert, quantized as put_experts quantizes each slice, then
            written in the order of names = the bytes of put_experts(names[0]) ... put_experts(names[-1])."""
            shape4 = (out_f, E, 1, in_f)
            blk = a.block if a.experts == "xb4" else in_f
            if a.experts == "fp32":
                sts = [np.empty((out_f, E, in_f), np.float32) for _ in names]
            else:
                qs = [np.empty((out_f, E, in_f), np.int8) for _ in names]
                ss = [np.empty((out_f, E, in_f // blk), np.float32) for _ in names]
                floored = [0 for _ in names]
            for e in range(E):
                full = ck.get_row(src_name, e)
                for k, rs in enumerate(row_slices):
                    we = full[rs]
                    assert we.shape == (out_f, in_f), (src_name, we.shape)
                    if a.experts == "fp32":
                        sts[k][:, e, :] = we
                    elif a.experts == "xb4":
                        qe, se, n = quant_bw4_rows(np.ascontiguousarray(we), blk)
                        floored[k] += n
                        qs[k][:, e, :] = qe
                        ss[k][:, e, :] = se
                    else:
                        qe, se = quant_c8_rows(np.ascontiguousarray(we))
                        qs[k][:, e, :] = qe
                        ss[k][:, e, :] = se[:, None]
                del full
            for k, name in enumerate(names):
                if a.experts == "fp32":
                    put_fp32(name, sts[k].reshape(shape4))
                    sts[k] = None
                    continue
                note_floor(floored[k])
                d0, dn = put(pack4(qs[k]) if a.experts == "xb4" else qs[k])
                rows.append((name, a.experts, shape4, d0, dn, 0, 0, blk))
                put_fp32(name + "_scales", ss[k].reshape(out_f, E, 1, in_f // blk))
                qs[k] = ss[k] = None

        def put_experts(name, src_name, row_slice, out_f, in_f):
            """Stack E experts of src_name[e][row_slice] ([out_f, in_f] each) in the moe op's [out, E, 1, in] layout."""
            shape4 = (out_f, E, 1, in_f)
            if a.experts == "fp32":
                st = np.empty((out_f, E, in_f), np.float32)
                for e in range(E):
                    st[:, e, :] = ck.get_row(src_name, e)[row_slice]
                put_fp32(name, st.reshape(shape4))
                return
            blk = a.block if a.experts == "xb4" else in_f
            q = np.empty((out_f, E, in_f), np.int8)
            s = np.empty((out_f, E, in_f // blk), np.float32)
            floored = 0
            for e in range(E):
                we = ck.get_row(src_name, e)[row_slice]
                assert we.shape == (out_f, in_f), (src_name, we.shape)
                if a.experts == "xb4":
                    qe, se, n = quant_bw4_rows(np.ascontiguousarray(we), blk)
                    floored += n
                else:
                    qe, se = quant_c8_rows(np.ascontiguousarray(we))
                    se = se[:, None]
                q[:, e, :] = qe
                s[:, e, :] = se
            note_floor(floored)
            d0, dn = put(pack4(q) if a.experts == "xb4" else q)
            rows.append((name, a.experts, shape4, d0, dn, 0, 0, blk))
            put_fp32(name + "_scales", s.reshape(out_f, E, 1, in_f // blk))
            del q, s

        if a.part in (None, "globals"):
            put_rows_chunked("embed", PFX + "embed_tokens.weight", "fp32" if a.embed == "fp32" else "pc8", 0)
            print(f"embed {a.embed}, peak RSS {peak_rss_bytes() / 2**30:.2f} GiB", flush=True)
        for i, l in todo_layers:
            tl = time.time()
            p = f"{PFX}layers.{l}."
            o = f"L{i}."
            put_fp32(o + "in_norm", 1.0 + ck.get(p + "input_layernorm.weight"))
            put_fp32(o + "post_norm", 1.0 + ck.get(p + "post_attention_layernorm.weight"))
            if cfg["layer_types"][l] == "linear_attention":
                q = p + "linear_attn."
                put_fc(o + "qkv", ck.get(q + "in_proj_qkv.weight"), a.fc)
                put_fc(o + "z", ck.get(q + "in_proj_z.weight"), a.fc)
                put_fp32(o + "b", ck.get(q + "in_proj_b.weight"))
                put_fp32(o + "a", ck.get(q + "in_proj_a.weight"))
                cw = ck.get(q + "conv1d.weight")  # [C, 1, k]
                assert cw.shape[1] == 1 and cw.shape[2] == dims["conv_k"], cw.shape
                for j in range(dims["conv_k"]):
                    put_fp32(o + f"conv{j}", np.ascontiguousarray(cw[:, 0, j]))
                put_fp32(o + "negA", -np.exp(ck.get(q + "A_log").astype(np.float32)))
                put_fp32(o + "dt_bias", ck.get(q + "dt_bias"))
                put_fp32(o + "gnorm", ck.get(q + "norm.weight"))
                put_fc(o + "out", ck.get(q + "out_proj.weight"), a.fc)
            else:
                q = p + "self_attn."
                wq = ck.get(q + "q_proj.weight").reshape(Hq, 2 * hdim, D)
                put_fc(o + "q", np.ascontiguousarray(wq[:, :hdim, :].reshape(Hq * hdim, D)), a.fc)
                put_fc(o + "qgate", np.ascontiguousarray(wq[:, hdim:, :].reshape(Hq * hdim, D)), a.fc)
                del wq
                put_fc(o + "k", ck.get(q + "k_proj.weight"), a.fc)
                put_fc(o + "v", ck.get(q + "v_proj.weight"), a.fc)
                put_fp32(o + "qn", 1.0 + ck.get(q + "q_norm.weight"))
                put_fp32(o + "kn", 1.0 + ck.get(q + "k_norm.weight"))
                put_fc(o + "o", ck.get(q + "o_proj.weight"), a.fc)
            m = p + "mlp."
            put_fp32(o + "router", ck.get(m + "gate.weight"))
            gu = m + "experts.gate_up_proj"
            dn = m + "experts.down_proj"
            assert ck.shape(gu) == (E, 2 * I, D) and ck.shape(dn) == (E, D, I), (ck.shape(gu), ck.shape(dn))
            put_experts_pair([o + "ex_gate", o + "ex_up"], gu, [slice(0, I), slice(I, 2 * I)], I, D)
            put_experts(o + "ex_down", dn, slice(None), D, I)
            put_fc(o + "sh_gate", ck.get(m + "shared_expert.gate_proj.weight"), a.fc)
            put_fc(o + "sh_up", ck.get(m + "shared_expert.up_proj.weight"), a.fc)
            put_fc(o + "sh_down", ck.get(m + "shared_expert.down_proj.weight"), a.fc)
            put_fp32(o + "sh_g", ck.get(m + "shared_expert_gate.weight"))
            layer_seconds.append(round(time.time() - tl, 2))
            print(f"LAYER {i} (HF {l}, {cfg['layer_types'][l]}) {layer_seconds[-1]} s, file {off / 2**30:.2f} GiB, "
                  f"peak RSS {peak_rss_bytes() / 2**30:.2f} GiB", flush=True)
        if a.part in (None, "globals"):
            put_fp32("final_norm", 1.0 + ck.get(PFX + "norm.weight"))
            put_rows_chunked("lm_head", "lm_head.weight", {"fp32": "fp32", "int8": "pc8", "int4": "bw4"}[a.lm_head],
                             a.fc_block)

    with open(os.path.join(out_dir, "index.tsv"), "w") as f:
        for r in rows:
            f.write("\t".join([r[0], r[1], ",".join(str(x) for x in r[2])] + [str(x) for x in r[3:]]) + "\n")
    if a.part is None:
        with open(os.path.join(a.out, "config.tsv"), "w") as f:
            for k, v in dims.items():
                f.write(f"{k}\t{v}\n")
    meta = {"source": src_info, "shards_read": sorted(ck.read),
            "dims": dims, "experts": a.experts, "block": a.block, "fc": a.fc, "fc_block": a.fc_block,
            "lm_head": a.lm_head, "embed": a.embed, "floor": stats, "bytes": off, "moe_options_hex": opts.hex(),
            "gdn_options_hex": gdn_opts.hex(), "seconds": round(time.time() - t0, 1), "layer_seconds": layer_seconds,
            "tensors": len(rows), "part": a.part, "flags": flags, "layers": [l for _, l in todo_layers],
            "peak_rss_bytes": peak_rss_bytes()}
    json.dump(meta, open(os.path.join(out_dir, "pack.json"), "w"), indent=1)
    print(f"PACK_DONE {len(rows)} tensors {off} B in {meta['seconds']} s; experts {a.experts} b{a.block}; fc {a.fc}; "
          f"floor {stats}; peak RSS {meta['peak_rss_bytes'] / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
