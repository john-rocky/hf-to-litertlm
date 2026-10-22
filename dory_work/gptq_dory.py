"""gptq_dory.py MODEL_DIR HESS_DIR WQ_DIR MAP_JSON OUT_DIR [--consume-h] - GPTQ int4 codes (group 32 = LiteRT's block
along the input axis) for the 289 FULLY_CONNECTED weights of the int4 TFLite, keeping its block scales.

The weights are the TFLite's own (MAP_JSON from fc_map.py): checkpoint bf16 -> fp32, and for the folded projections
(g, gate, up) times the model's per-row factor c in fp32 = the values the converter quantized. A problem = the weights
that share one Hessian: q k v g of an attention block (rows stacked, H qkvg<i>), its o (H o<i>), gate up of an MLP block
(H gu<i>), its down (H down<i>), lm_head (H lm). H = gptq_hessians.py's raw sum; damp = 0.01 * mean(diag H) on the
diagonal, a zero diagonal entry -> 1; the Cholesky of H, its inverse and the upper Cholesky U of the inverse in float32
on the Apple GPU (MPS; torch's CPU Cholesky on macOS was seen to return wrong factors); the column loop in float32 on
MPS: columns in their natural order (no act-order), lazy batches of 128 columns, q = clamp(round(w / s), -7, 7) with the
block scale s of the column (block = 32 consecutive input columns), err = (w - q s) / U_jj, the remaining columns of the
batch updated by err * U_j,j:, the later columns by Err @ U_batch,rest (Frantar et al. 2022, the reference
implementation's order). s = the int4 TFLite's fp16 block scales (WQ_DIR/int4/<name>.scales.npy = absmax / 7 rounded
bf16 -> fp16, floor 2^-14): only the codes change.
Out: OUT_DIR/int4/<name>.codes.npy (uint8, two codes a byte, low nibble first = the TFLite layout) and
OUT_DIR/int4/<name>.scales.npy (fp16 [out, in / 32], the scales used); OUT_DIR/gptq.json per problem: seconds, codes
changed against the data-free codes, the proxy error tr(dW H dW^T) of the data-free codes and of these codes
(dW = dequantized - W, raw H). Resumable per problem; OUT_DIR/gptq_done at the end.
  --consume-h: delete each problem's H_<key>.npy once its codes and record are saved (disk)

Usage: python dory_work/gptq_dory.py <model dir> <hessian dir> <wq dir> <map json> <out dir> [--consume-h]
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

QMIN, QMAX, BATCH, BLOCK = -7, 7, 128, 32


def unpack(packed):
  b = packed.astype(np.uint8)
  out = np.empty(b.size * 2, dtype=np.int8)
  out[0::2] = (b & 0x0F).astype(np.int8)
  out[1::2] = (b >> 4).astype(np.int8)
  return (out ^ 8) - 8


def pack(codes):
  c = (codes.astype(np.int16) & 0x0F).astype(np.uint8).reshape(-1)
  return (c[0::2] | (c[1::2] << 4)).astype(np.uint8)


def factors(f, cfg):
  """The model's per-row factors after g / gate / up (the same arithmetic as fc_map.fold_factors and dory_ref.py)."""
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
  return out


def problems(cfg):
  out = []
  for i, t in enumerate(cfg["layer_types"]):
    p = f"backbone.layers.{i}.mixer."
    if t == "mlp":
      out.append((f"gu{i}", [p + "gate_proj.weight", p + "up_proj.weight"]))
      out.append((f"down{i}", [p + "down_proj.weight"]))
    else:
      out.append((f"qkvg{i}", [p + x + "_proj.weight" for x in ("q", "k", "v", "g")]))
      out.append((f"o{i}", [p + "o_proj.weight"]))
  out.append(("lm", ["lm_head.weight"]))
  return out


def factor_mps(H):
  """U = the upper Cholesky factor of H^-1 (H float64 numpy, damped), in float32 on MPS -> float32 CPU tensor."""
  L = torch.linalg.cholesky(torch.from_numpy(H.astype(np.float32)).to("mps"))
  return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True).cpu()


def upper_inv_chol(Hraw):
  """-> (U fp32 CPU, dead columns, damp used). damp 0.01 * mean diag (0.1 and 1.0 only if the factorization fails or
  is not finite)."""
  H0 = Hraw.astype(np.float64).copy()
  dead = np.diag(H0) == 0
  H0[dead, dead] = 1.0
  mean_diag = float(np.mean(np.diag(H0)))
  for frac in (0.01, 0.1, 1.0):
    H = H0.copy()
    H[np.diag_indices_from(H)] += frac * mean_diag
    try:
      U = factor_mps(H)
      if not bool(torch.isfinite(U).all()):
        raise RuntimeError("non-finite factor")
      return U, int(dead.sum()), frac * mean_diag
    except (RuntimeError, np.linalg.LinAlgError) as ex:  # not positive definite at this damp
      print(f"cholesky failed at damp {frac} x mean diag: {str(ex)[:120]}", flush=True)
  raise SystemExit("cholesky failed at every damp")


def gptq(W, S, U, dev):
  """W [R, C] fp32, S [R, C/32] fp32 (fixed), U [C, C] fp32 upper -> codes int8 [R, C]."""
  R, C = W.shape
  W = W.clone()
  Q = torch.zeros(R, C, dtype=torch.int8, device=dev)
  for i1 in range(0, C, BATCH):
    i2 = min(i1 + BATCH, C)
    W1 = W[:, i1:i2].clone()
    E1 = torch.zeros_like(W1)
    U1 = U[i1:i2, i1:i2]
    for j in range(i2 - i1):
      col = i1 + j
      s = S[:, col // BLOCK]
      w = W1[:, j]
      q = torch.clamp(torch.round(w / s), QMIN, QMAX)
      Q[:, col] = q.to(torch.int8)
      err = (w - q * s) / U1[j, j]
      W1[:, j:] -= err[:, None] * U1[j, j:][None, :]
      E1[:, j] = err
    if i2 < C:
      W[:, i2:] -= E1 @ U[i1:i2, i2:]
  return Q


def proxy(dW, H):
  """tr(dW H dW^T) on the device (H raw, float32)."""
  return float(((dW @ H) * dW).sum())


def main(snap, hdir, wq, map_json, out, *flags):
  if [f for f in flags if f != "--consume-h"]:
    raise SystemExit(__doc__)
  consume_h = "--consume-h" in flags
  dev = torch.device("mps")
  os.makedirs(os.path.join(out, "int4"), exist_ok=True)
  if os.path.exists(os.path.join(out, "gptq_done")):
    print("GPTQ already done", flush=True)
    return 0
  tmap = {t["name"]: t for t in json.load(open(map_json))["tensors"]}
  cfg = json.load(open(os.path.join(snap, "config.json")))
  rec_path = os.path.join(out, "gptq.json")
  rec = json.load(open(rec_path)) if os.path.exists(rec_path) else dict(qrange=[QMIN, QMAX], batch=BATCH, block=BLOCK,
                                                                         cholesky="mps_f32", problems={})
  t_start = time.time()
  with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
    fac = factors(f, cfg)
    for key, names in problems(cfg):
      if all(os.path.exists(os.path.join(out, "int4", n + ".codes.npy")) for n in names) and key in rec["problems"]:
        continue
      t0 = time.time()
      Ws, Ss, D, rows = [], [], [], []
      for n in names:
        W = f.get_tensor(n).float().numpy()
        if tmap[n]["kind"] == "folded":
          W = (W * fac[n][:, None]).astype(np.float32)
        Ws.append(W)
        Ss.append(np.load(os.path.join(wq, "int4", n + ".scales.npy")).astype(np.float32))
        o, i = tmap[n]["shape"]
        D.append(unpack(np.load(os.path.join(wq, "int4", n + ".codes.npy"))).reshape(o, i))
        rows.append(o)
      W = torch.from_numpy(np.concatenate(Ws, 0)).to(dev)
      S = torch.from_numpy(np.concatenate(Ss, 0)).to(dev)
      Hraw = np.load(os.path.join(hdir, f"H_{key}.npy"))
      t1 = time.time()
      U, dead, damp = upper_inv_chol(Hraw)
      chol_s = time.time() - t1
      U = U.to(dev)
      Hd = torch.from_numpy(Hraw).to(dev)
      t2 = time.time()
      Q = gptq(W, S.clone(), U, dev)
      torch.mps.synchronize()
      loop_s = time.time() - t2
      deq_g = Q.float() * S.repeat_interleave(BLOCK, dim=1)
      Dq = torch.from_numpy(np.concatenate(D, 0)).to(dev).float()
      deq_d = Dq * S.repeat_interleave(BLOCK, dim=1)
      e_df = proxy(deq_d - W, Hd)
      e_g = proxy(deq_g - W, Hd)
      Qc = Q.cpu().numpy()
      Sc = S.cpu().numpy()
      changed = int((Qc != np.concatenate(D, 0)).sum())
      at = 0
      for n, o in zip(names, rows):
        codes = Qc[at:at + o]
        pk = pack(codes)
        assert np.array_equal(unpack(pk).reshape(codes.shape), codes)
        np.save(os.path.join(out, "int4", n + ".scales.npy"), Sc[at:at + o].astype(np.float16))
        np.save(os.path.join(out, "int4", n + ".codes.npy"), pk)
        at += o
      rec["problems"][key] = dict(weights=names, shape=[int(W.shape[0]), int(W.shape[1])], dead_columns=dead,
                                  damp=damp, chol_s=round(chol_s, 2), loop_s=round(loop_s, 2),
                                  seconds=round(time.time() - t0, 2), codes_changed=changed,
                                  codes_changed_frac=changed / Qc.size, proxy_err_data_free=e_df, proxy_err_gptq=e_g,
                                  proxy_ratio=e_g / e_df if e_df else None)
      json.dump(rec, open(rec_path + ".tmp", "w"), indent=1)
      os.replace(rec_path + ".tmp", rec_path)
      if consume_h:
        os.remove(os.path.join(hdir, f"H_{key}.npy"))
      print(f"GPTQ {key} {list(W.shape)}: chol {chol_s:.1f} s loop {loop_s:.1f} s, codes changed "
            f"{changed / Qc.size:.3%}, proxy err {e_g:.4g} vs data-free {e_df:.4g} (x{e_g / e_df:.3f})", flush=True)
      del W, S, U, Hd, Q, deq_g, deq_d, Dq
      torch.mps.empty_cache()
  tot_df = sum(p["proxy_err_data_free"] for p in rec["problems"].values())
  tot_g = sum(p["proxy_err_gptq"] for p in rec["problems"].values())
  rec["total"] = dict(problems=len(rec["problems"]), proxy_err_data_free=tot_df, proxy_err_gptq=tot_g,
                      codes_changed=sum(p["codes_changed"] for p in rec["problems"].values()),
                      seconds=round(sum(p["seconds"] for p in rec["problems"].values()), 1))
  json.dump(rec, open(rec_path, "w"), indent=1)
  open(os.path.join(out, "gptq_done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
  print(f"GPTQ_DONE {len(rec['problems'])} problems in {rec['total']['seconds']} s (window {time.time() - t_start:.0f} s); "
        f"proxy err sum {tot_g:.4g} vs data-free {tot_df:.4g}", flush=True)
  return 0


if __name__ == "__main__":
  if len(sys.argv) < 6:
    raise SystemExit(__doc__)
  sys.exit(main(*sys.argv[1:]))
