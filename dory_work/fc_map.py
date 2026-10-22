"""fc_map.py INT4_TFLITE INT8_TFLITE MODEL_DIR OUT_DIR MAP_JSON - the FULLY_CONNECTED weights of the data-free int4 and
the int8 TFLite (quantize_dory.py's model_quantized.tflite, both from the same float export), mapped to the checkpoint
tensors, and the int4 file's codes and block scales written out per checkpoint tensor for gptq_dory.py.

For every FULLY_CONNECTED op of every subgraph: the op scope ai-edge-quantizer matches a recipe regex against (output
tensor names joined by ';' plus a trailing ';'), its weight tensor (input 1) and buffer. Every distinct weight buffer is
mapped to one checkpoint tensor by value (row directions of the first rows, so a per-row factor folded in by the
converter does not matter), and then checked whole:
  fold   per row r: ratio_r = <deq_r, W_r> / <W_r, W_r> (int8 rows); a tensor is "folded" when ratio_r = c_r, the
         per-row factor the model multiplies after this projection (g: (s_gate * g0^0.5 / std)^2, gate / up:
         suv * H^0.5, lm_head: logit_scale / std; dory_ref.py), "plain" when ratio_r = 1.
  bound  |deq - W_fold| <= 0.5 * scale + tol for every element (int4 blockwise and int8 channelwise), W_fold = W * c.
         This also fixes the int4 nibble order (both orders are tried on the first tensor).
  emul   the quantizer's own arithmetic re-done on W_fold in numpy (int4: block absmax / 7, rounded to bf16 then
         fp16, floored at 2^-14, rint, clip [-8, 7]; int8: row absmax / 127, rint, clip [-127, 127]) against the
         file's codes: the count of differing codes. Also int4 on the UNFOLDED W: codes that differ from the file's =
         what folding changes (the bf16 / fp16 rounding of the block scale is relative, so a row factor changes it).
Outputs: OUT_DIR/int4/<name>.codes.npy (uint8, packed as in the TFLite) + .scales.npy (fp16 [out, in / 32]) and the
map JSON (per tensor: kind, shape, the int4 buffer index, the checks).

Usage: python dory_work/fc_map.py <int4 tflite> <int8 tflite> <model dir> <out dir> <map json>
"""
import collections
import json
import math
import mmap
import os
import sys
import time

import ml_dtypes
import numpy as np
import torch
from ai_edge_litert import schema_py_generated as ts
from safetensors import safe_open

DTYPES = {v: k for k, v in vars(ts.TensorType).items() if not k.startswith("_")}
FC = ts.BuiltinOperator.FULLY_CONNECTED
FLOOR = np.float32(2.0 ** -14)


def tensor_bytes(m, view, buf_idx):
  buf = m.Buffers(buf_idx)
  if buf is None:
    return None
  if buf.DataLength():
    off = buf._tab.Vector(buf._tab.Offset(4))  # pylint: disable=protected-access
    return view[off:off + buf.DataLength()]
  if buf.Size():
    return view[buf.Offset():buf.Offset() + buf.Size()]
  return None


def read_fcs(path):
  """-> (model, view, mm, file handle, ops, weights): ops = every FC op (subgraph, signature, op index, scope, weight
  tensor name, buffer); weights = distinct weight buffers in first-use order with their quantization."""
  fh = open(path, "rb")
  mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
  view = memoryview(mm)
  m = ts.Model.GetRootAs(view, 0)
  sig_of = {}
  for i in range(m.SignatureDefsLength()):
    sd = m.SignatureDefs(i)
    sig_of[sd.SubgraphIndex()] = sd.SignatureKey().decode()
  ops, weights = [], collections.OrderedDict()
  for si in range(m.SubgraphsLength()):
    sg = m.Subgraphs(si)
    for oi in range(sg.OperatorsLength()):
      op = sg.Operators(oi)
      oc = m.OperatorCodes(op.OpcodeIndex())
      if max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()) != FC:
        continue
      outs = [sg.Tensors(op.Outputs(k)).Name().decode(errors="replace") for k in range(op.OutputsLength())
              if op.Outputs(k) != -1]
      outs = [n for n in outs if n]
      scope = ";".join(outs) + (";" if outs else "")
      wt = sg.Tensors(op.Inputs(1))
      b = wt.Buffer()
      ops.append(dict(subgraph=si, signature=sig_of.get(si, f"subgraph{si}"), op=oi, scope=scope,
                      weight=wt.Name().decode(errors="replace"), buffer=b))
      if b in weights:
        continue
      q = wt.Quantization()
      info = dict(buffer=b, name=wt.Name().decode(errors="replace"), dtype=DTYPES.get(wt.Type(), str(wt.Type())),
                  shape=[int(x) for x in wt.ShapeAsNumpy()], form="float")
      if q is not None and q.DetailsType() == ts.QuantizationDetails.BlockwiseQuantization:
        bq = ts.BlockwiseQuantization()
        tab = q.Details()
        bq.Init(tab.Bytes, tab.Pos)
        st = sg.Tensors(bq.Scales())
        info.update(form="blockwise", block=bq.BlockSize(), scale_buffer=st.Buffer(),
                    scale_dtype=DTYPES.get(st.Type(), str(st.Type())), scale_shape=[int(x) for x in st.ShapeAsNumpy()],
                    zp_tensor=bq.ZeroPoints())
      elif q is not None and q.ScaleLength() > 0:
        info.update(form="channelwise" if q.ScaleLength() > 1 else "per_tensor", qdim=q.QuantizedDimension(),
                    scales=q.ScaleAsNumpy().astype(np.float32),
                    zp_max=int(np.abs(q.ZeroPointAsNumpy()).max()) if q.ZeroPointLength() else 0)
      weights[b] = info
  return m, view, mm, fh, ops, weights


def scales4(m, view, info):
  """The fp16 block scales of a blockwise weight as [out, in / block], whichever way the scale tensor is laid out."""
  o, i = info["shape"]
  nb = i // info["block"]
  sc = np.frombuffer(tensor_bytes(m, view, info["scale_buffer"]), dtype=np.float16)
  shp = info["scale_shape"]
  if shp == [o, nb]:
    return sc.reshape(o, nb)
  if shp == [nb, o]:
    return np.ascontiguousarray(sc.reshape(nb, o).T)
  raise SystemExit(f"unexpected block-scale shape {shp} for weight {info['name']} {info['shape']}")


def unpack_int4(raw, n, low_first=True):
  b = np.frombuffer(raw, dtype=np.uint8)
  assert b.size * 2 == n, (b.size, n)
  lo = (b & 0x0F).astype(np.int8)
  hi = (b >> 4).astype(np.int8)
  out = np.empty(n, dtype=np.int8)
  out[0::2], out[1::2] = (lo, hi) if low_first else (hi, lo)
  return ((out ^ 8) - 8).astype(np.int8)  # sign-extend the 4-bit two's complement


def bf16_fp16(x):
  return x.astype(ml_dtypes.bfloat16).astype(np.float16).astype(np.float32)


def emul_int4(wf, block=32):
  """ai-edge-quantizer 0.9.0 blockwise-32 int4 on a float [out, in] array + the 2^-14 floor -> (codes int8, scale fp32 [out, in/b])."""
  o, i = wf.shape
  wb = wf.reshape(o, i // block, block)
  bound = np.maximum(np.abs(wb).max(axis=-1), np.float32(1e-9))
  s = bf16_fp16((bound / np.float32(7.0)).astype(np.float32))
  s = np.where(s < FLOOR, FLOOR, s).astype(np.float32)
  q = np.clip(np.rint(wb / s[..., None]), -8, 7).astype(np.int8)
  return q.reshape(o, i), s


def emul_int8(wf):
  bound = np.maximum(np.abs(wf).max(axis=1), np.float32(1e-9))
  s = (bound / np.float32(127.0)).astype(np.float32)
  q = np.clip(np.rint(wf / s[:, None]), -127, 127).astype(np.int8)
  return q, s


def ckpt_names(cfg):
  out = []
  for i, t in enumerate(cfg["layer_types"]):
    p = f"backbone.layers.{i}.mixer."
    projs = ("gate", "up", "down") if t == "mlp" else ("q", "k", "v", "g", "o")
    out += [(p + f"{x}_proj.weight", i, x) for x in projs]
  out.append(("lm_head.weight", None, "lm_head"))
  return out


def fold_factors(f, cfg):
  """The model's per-row factors (dory_ref.py): name -> fp32 numpy vector, for the projections followed by one."""
  H, std = cfg["hidden_size"], cfg["initializer_range"]
  g0 = cfg["attention_output_gate_logit_scale_init"]
  if g0 is None:
    g0 = H ** 0.5
  sgf, suvf = g0 ** 0.5 / std, H ** 0.5
  out = {}
  for i, t in enumerate(cfg["layer_types"]):
    p = f"backbone.layers.{i}."
    if t == "mlp":
      out[p + "mixer.gate_proj.weight"] = (f.get_tensor(p + "mixer.suv_gate").float() * suvf).numpy()
      out[p + "mixer.up_proj.weight"] = (f.get_tensor(p + "mixer.suv_up").float() * suvf).numpy()
    else:
      out[p + "mixer.g_proj.weight"] = (f.get_tensor(p + "mixer.s_gate").float() * sgf).square().numpy()
  out["lm_head.weight"] = (f.get_tensor("lm_head.logit_scale").float() / std).numpy()
  return out


def rows_unit(x):
  n = np.linalg.norm(x, axis=1, keepdims=True)
  ok = n[:, 0] > 0
  return np.where(n > 0, x / np.where(n > 0, n, 1), 0), ok


def main(p4, p8, ckpt, out_dir, map_json):
  with safe_open(os.path.join(ckpt, "model.safetensors"), framework="pt") as f:
    return run(f, p4, p8, ckpt, out_dir, map_json)


def run(f, p4, p8, ckpt, out_dir, map_json):
  t0 = time.time()
  cfg = json.load(open(os.path.join(ckpt, "config.json")))
  os.makedirs(os.path.join(out_dir, "int4"), exist_ok=True)
  m4, v4, mm4, fh4, ops4, w4 = read_fcs(p4)
  m8, v8, mm8, fh8, ops8, w8 = read_fcs(p8)
  print(f"FC ops int4 {len(ops4)} int8 {len(ops8)}; distinct weights int4 {len(w4)} int8 {len(w8)}", flush=True)
  names = ckpt_names(cfg)
  shape_of = {n: list(f.get_slice(n).get_shape()) for n, _, _ in names}
  R = 16  # rows used for the value fingerprint
  fp = {n: rows_unit(f.get_slice(n)[0:R].float().numpy()) for n, _, _ in names}
  by_shape = collections.defaultdict(list)
  for n, _, _ in names:
    by_shape[tuple(shape_of[n])].append(n)

  def match(info, deq_rows):
    cands = by_shape[tuple(info["shape"])]
    u, ok = rows_unit(deq_rows)
    best = []
    for n in cands:
      cu, cok = fp[n]
      both = ok & cok
      cos = float((u[both] * cu[both]).sum(axis=1).mean()) if both.any() else -1.0
      best.append((cos, n))
    best.sort(reverse=True)
    return best[0][1], best[0][0], (best[1][0] if len(best) > 1 else None)

  folds = fold_factors(f, cfg)
  # int8: fingerprint from the first rows
  map8 = {}
  for b, info in w8.items():
    raw = tensor_bytes(m8, v8, b)
    codes = np.frombuffer(raw, dtype=np.int8).reshape(info["shape"])
    deq = codes[:R].astype(np.float32) * info["scales"][:R, None]
    n, c1, c2 = match(info, deq)
    map8[n] = dict(buffer=b, tensor=info["name"], cos=c1, cos_second=c2)
  # int4: nibble order from the first tensor (the bound check below confirms it on every tensor)
  low_first = None
  map4 = {}
  for b, info in w4.items():
    raw = tensor_bytes(m4, v4, b)
    o, i = info["shape"]
    assert info["block"] == 32 and info["dtype"] == "INT4", info
    sc = scales4(m4, v4, info).astype(np.float32)
    head = raw[:R * i // 2]  # the first R rows (row-major, two codes a byte)
    if low_first is None:
      tries = {}
      for lf in (True, False):
        codes = unpack_int4(head, R * i, lf).reshape(R, i)
        deq = codes.astype(np.float32) * np.repeat(sc[:R], 32, axis=1)
        tries[lf] = match(info, deq)[1]
      low_first = tries[True] >= tries[False]
      print(f"int4 nibble order: low nibble first = {low_first} (cos {tries})", flush=True)
    codes = unpack_int4(head, R * i, low_first).reshape(R, i)
    deq = codes.astype(np.float32) * np.repeat(sc[:R], 32, axis=1)
    n, c1, c2 = match(info, deq)
    map4[n] = dict(buffer=b, tensor=info["name"], cos=c1, cos_second=c2)
  missing = [n for n, _, _ in names if n not in map4 or n not in map8]
  dup4 = len(map4) != len(w4)
  dup8 = len(map8) != len(w8)
  print(f"mapped int4 {len(map4)}/{len(w4)} int8 {len(map8)}/{len(w8)}; missing {missing[:5]}; "
        f"min cos int4 {min(v['cos'] for v in map4.values()):.4f} int8 {min(v['cos'] for v in map8.values()):.6f}; "
        f"max second cos int4 {max((v['cos_second'] or -1) for v in map4.values()):.4f}", flush=True)
  if missing or dup4 or dup8:
    raise SystemExit(f"MAP_FAIL missing={missing} dup4={dup4} dup8={dup8}")

  # whole-tensor checks and the code files
  scopes4 = collections.defaultdict(list)
  for o in ops4:
    scopes4[o["buffer"]].append(dict(signature=o["signature"], scope=o["scope"], weight=o["weight"]))
  scopes8 = collections.defaultdict(list)
  for o in ops8:
    scopes8[o["buffer"]].append(dict(signature=o["signature"], scope=o["scope"], weight=o["weight"]))
  rows = []
  tot = collections.Counter()
  for n, blk, proj in names:
    W = f.get_tensor(n).float().numpy()
    c = folds.get(n)
    i4, i8 = w4[map4[n]["buffer"]], w8[map8[n]["buffer"]]
    o, i = i4["shape"]
    raw4 = tensor_bytes(m4, v4, i4["buffer"])
    codes4 = unpack_int4(raw4, o * i, low_first).reshape(o, i)
    sc16 = scales4(m4, v4, i4)
    sc4 = sc16.astype(np.float32)
    codes8 = np.frombuffer(tensor_bytes(m8, v8, i8["buffer"]), dtype=np.int8).reshape(o, i)
    sc8 = i8["scales"]
    deq8 = codes8.astype(np.float32) * sc8[:, None]
    num = (deq8 * W).sum(axis=1, dtype=np.float64)
    den = (W.astype(np.float64) ** 2).sum(axis=1)
    ratio = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    live = np.isfinite(ratio) & (np.abs(ratio) > 0)
    plain_err = float(np.nanmedian(np.abs(ratio[live] - 1))) if live.any() else None
    fold_err = None
    if c is not None:
      lc = live & (c != 0)
      fold_err = float(np.nanmedian(np.abs(ratio[lc] / c[lc] - 1))) if lc.any() else None
    if c is not None and fold_err is not None and fold_err < 1e-2:
      kind = "folded"
    elif plain_err is not None and plain_err < 1e-2:
      kind = "plain"
    else:
      kind = "unknown"
    wf = (W * c[:, None]).astype(np.float32) if kind == "folded" else W
    deq4 = codes4.astype(np.float32) * np.repeat(sc4, 32, axis=1)
    b4 = float(np.max(np.abs(deq4 - wf) / np.repeat(sc4, 32, axis=1)))
    b8 = float(np.max(np.abs(deq8 - wf) / np.maximum(sc8[:, None], np.float32(1e-30))))
    e4, es4 = emul_int4(wf)
    e8, es8 = emul_int8(wf)
    mis4 = int(np.count_nonzero(e4 != codes4))
    mis4s = int(np.count_nonzero(es4 != sc4))
    mis8 = int(np.count_nonzero(e8 != codes8))
    u4, us4 = emul_int4(W) if kind == "folded" else (e4, es4)
    mis4u = int(np.count_nonzero(u4 != codes4)) if kind == "folded" else mis4
    floor_n = int(np.count_nonzero(sc16 == np.float16(2.0 ** -14)))
    np.save(os.path.join(out_dir, "int4", n + ".codes.npy"), np.frombuffer(raw4, dtype=np.uint8))
    np.save(os.path.join(out_dir, "int4", n + ".scales.npy"), sc16)
    zero_rows = int(np.count_nonzero(np.abs(W).max(axis=1) == 0))
    czero = int(np.count_nonzero(c == 0)) if c is not None else 0
    rows.append(dict(name=n, block=blk, proj=proj, shape=[o, i], params=o * i, kind=kind,
                     ratio_vs_1_median=plain_err, ratio_vs_c_median=fold_err, factor_zero_rows=czero,
                     weight_zero_rows=zero_rows, int4_tensor=i4["name"], int8_tensor=i8["name"],
                     int4_buffer=i4["buffer"], int8_buffer=i8["buffer"], map_cos_int4=map4[n]["cos"],
                     map_cos_int8=map8[n]["cos"], bound_int4=b4, bound_int8=b8, emul_int4_code_mismatch=mis4,
                     emul_int4_scale_mismatch=mis4s, emul_int8_code_mismatch=mis8, unfolded_int4_code_mismatch=mis4u,
                     int4_floor_scales=floor_n, int4_scale_min=float(sc4.min()),
                     scopes_int4=scopes4[i4["buffer"]], scopes_int8=scopes8[i8["buffer"]]))
    tot["params"] += o * i
    tot[f"kind_{kind}"] += 1
    tot["emul4_mis"] += mis4
    tot["emul8_mis"] += mis8
    tot["unfold4_mis"] += mis4u
    tot["floor"] += floor_n
    print(f"{n}: {kind} (vs1 {plain_err}, vs c {fold_err}) bound4 {b4:.4f} bound8 {b8:.4f} emul4 {mis4} emul8 {mis8} "
          f"unfolded4 {mis4u} floor {floor_n} ops {len(scopes4[i4['buffer']])}", flush=True)
  ok = (tot["kind_unknown"] == 0 and all(r["bound_int4"] <= 0.5001 and r["bound_int8"] <= 0.5001 for r in rows))
  doc = dict(int4_tflite=p4, int8_tflite=p8, checkpoint=ckpt, int4_low_nibble_first=bool(low_first),
             n_fc_ops_int4=len(ops4), n_fc_ops_int8=len(ops8), n_weights=len(rows), totals=dict(tot),
             checks_pass=ok, seconds=round(time.time() - t0, 1), tensors=rows)
  json.dump(doc, open(map_json, "w"), indent=1)
  print(f"WQ_DONE pass={ok} {dict(tot)} in {time.time() - t0:.0f} s", flush=True)
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main(*sys.argv[1:6]))
