"""gptq_hessians.py MODEL_DIR SAMPLES OUT_DIR - the GPTQ Hessians of every FULLY_CONNECTED weight of
NVIDIA-Nemotron-Labs-Dory1.0-4B, from the reference forward of dory_ref.py on Apple silicon (MPS).

Samples = SAMPLES (calib_dory.py's samples.jsonl, 320 rows of token ids). Each sample = one chunk forward of the
reference (fp32, 2 recurrent loops, per-loop KV, MPS) from position 0 with every position's logits (so lm_head sees every
position). A hook on the reference's Num.lin(x, w) adds x^T x (fp32, on the device) to the weight's Hessian:
  qkvg<i> (2560^2, the common input of q / k / v / g), o<i> (4096^2), gu<i> (2560^2, gate / up), down<i> (7168^2),
  lm (2560^2). A recurrent block's weight is called in loop 0 and loop 1 = both inputs go into the one sum.
Writes OUT_DIR/H_<key>.npy (the raw sums, float32), OUT_DIR/summary.json and OUT_DIR/done.

Usage: python dory_work/gptq_hessians.py <model dir> <samples.jsonl> <out dir>
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MAXLEN = 1024


def load_ref():
  spec = importlib.util.spec_from_file_location("dory_ref", os.path.join(HERE, "dory_ref.py"))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def main(snap, samples_path, out):
  os.makedirs(out, exist_ok=True)
  t0 = time.time()
  dr = load_ref()
  m = dr.DoryRef(snap, mode="fp32", loops=2, kv_mode="per_loop", threads=12, max_len=MAXLEN + 8, device="mps")
  dev = m.dev
  S = [json.loads(line) for line in open(samples_path, encoding="utf-8")]
  owner = {}
  for i, l in enumerate(m.layers):
    if l["kind"] == "mlp":
      owner[id(l["w_gu"])], owner[id(l["w_down"])] = f"gu{i}", f"down{i}"
    else:
      owner[id(l["w_qkvg"])], owner[id(l["w_o"])] = f"qkvg{i}", f"o{i}"
  owner[id(m.lm_head)] = "lm"
  dims = {}
  for i, l in enumerate(m.layers):
    if l["kind"] == "mlp":
      dims[f"gu{i}"], dims[f"down{i}"] = l["w_gu"].shape[1], l["w_down"].shape[1]
    else:
      dims[f"qkvg{i}"], dims[f"o{i}"] = l["w_qkvg"].shape[1], l["w_o"].shape[1]
  dims["lm"] = m.lm_head.shape[1]
  H = {k: torch.zeros(d, d, dtype=torch.float32, device=dev) for k, d in dims.items()}
  rows = {k: 0 for k in dims}
  orig = m.num.lin

  def lin(x, w):
    k = owner.get(id(w))
    if k is not None:
      xf = x.reshape(-1, x.shape[-1]).float()
      H[k].addmm_(xf.T, xf)
      rows[k] += xf.shape[0]
    return orig(x, w)

  m.num.lin = lin
  load_s = round(time.time() - t0, 1)
  print(f"HESS load {load_s} s, samples {len(S)}", flush=True)
  t1 = time.time()
  tokens = 0
  for i, s in enumerate(S):
    ids = s["ids"]
    assert 0 < len(ids) <= MAXLEN, (s["name"], len(ids))
    with torch.inference_mode():
      m.forward(ids, 0, rows=list(range(len(ids))))
    torch.mps.synchronize()
    tokens += len(ids)
    if (i + 1) % 20 == 0:
      print(f"HESS {i + 1}/{len(S)} tokens {tokens}", flush=True)
  forward_s = round(time.time() - t1, 1)
  t2 = time.time()
  for k, h in H.items():
    p = os.path.join(out, f"H_{k}.npy")
    np.save(p + ".tmp.npy", h.cpu().numpy())
    os.replace(p + ".tmp.npy", p)
  by = {}
  for s in S:
    by[s["source"]] = by.get(s["source"], 0) + len(s["ids"])
  json.dump(dict(samples=len(S), tokens=tokens, tokens_by_source=by, rows=rows, dims=dims, load_s=load_s,
                 forward_s=forward_s, save_s=round(time.time() - t2, 1), torch=torch.__version__),
            open(os.path.join(out, "summary.json"), "w"), indent=1)
  open(os.path.join(out, "done"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
  print(f"HESS_DONE {len(S)} samples, {tokens} tokens, forward {forward_s} s, save {time.time() - t2:.0f} s", flush=True)
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 4:
    raise SystemExit(__doc__)
  sys.exit(main(*sys.argv[1:4]))
