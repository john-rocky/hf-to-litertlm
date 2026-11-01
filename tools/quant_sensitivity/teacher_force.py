#!/usr/bin/env python3
"""teacher_force.py - teacher-forced runs of a decoder graph on the LiteRT CPU Interpreter, against a reference graph:
per position the top-1 token, its margin over the top-2, and how far a quantized form moves the reference's logits.

  python -I teacher_force.py --decoder REF.tflite --embedder EMB.tflite --cases cases.json --label ref --save-ref
      --out-dir DIR
  python -I teacher_force.py --decoder FORM.tflite --embedder EMB.tflite --cases cases.json --label FORM --ref ref
      --out-dir DIR [--threads 8] [--only key,...]

The decoder graph = a litert-torch export with an external embedder: signatures prefill_<N> and decode with inputs
`embeddings`, `input_pos`, `mask` and the cache / state tensors (every input whose name starts with kv_cache), output
`logits`; the embedder graph has the signature `embedder` (token_ids -> embeddings).
cases.json = {"pad_id": int, "image_token_id": int or null, "cases": [{"key": str, "group": str, "ids": [prompt ids],
"teacher": [teacher ids], "image_npy": path or null (rows for the image slots, in order), "counted": bool}]}.
Per case, as LiteRT-LM 0.18.1 drives the graph: prompt[:-1] through the prefill signatures that its prefill planner
(GetOptimizedPrefillWorkGroups) picks (unused slots: position 0 and the pad token's embedding), then decode from the
last prompt token through the teacher tokens = logits at every teacher position. Recorded per position: the graph's
top-1, top-1 = teacher, the top-1 / top-2 gap, the teacher token's gap to the top-1; with --ref (a label saved by an
earlier --save-ref run): top-1 = the reference's top-1, max |d logit| and rms d logit over the vocabulary, the
reference's gap and this graph's margin on the reference's top-1 / top-2 pair (pair_margin; pair_margin - ref_gap =
the margin change). Per case: every teacher position keeps the teacher (all_teacher) and the reference's top-1
(all_ref). -> DIR/<label>.json (and DIR/logits_<label>/<key>.npy with --save-ref).
Check the tool before reading a table: on the reference graph every teacher position should keep the teacher token
when the teacher came from the source model and the reference form answers like it.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

MASK_FILL = np.float32(-0.7 * np.finfo(np.float32).max)


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def plan_prefill(sig_lengths, n):
  """LiteRT-LM v0.18.1 GetOptimizedPrefillWorkGroups (the cautious plan) -> [(signature length, tokens), ...]."""
  sigs = sorted(sig_lengths, reverse=True)
  out = []
  for idx, cur in enumerate(sigs):
    if n <= 0:
      break
    full = n // cur
    out += [(cur, cur)] * full
    n -= full * cur
    if n == 0:
      break
    if idx + 1 == len(sigs):
      out.append((cur, n))
      n = 0
      break
    nxt = sigs[idx + 1]
    if nxt * 2 >= cur and n <= nxt:
      continue
    if n * 2 >= cur:
      out.append((cur, n))
      n = 0
      break
  assert n == 0
  return out


def mask(start, steps, rows, cache):
  m = np.full((1, 1, rows, cache), MASK_FILL, dtype=np.float32)
  for i in range(steps):
    m[0, 0, i, : start + i + 1] = 0.0
  return m


class Decoder:
  def __init__(self, path, threads):
    from ai_edge_litert import interpreter as interp
    self.itp = interp.Interpreter(model_path=path, num_threads=threads)
    sigs = self.itp.get_signature_list()
    self.decode = self.itp.get_signature_runner("decode")
    self.prefill = {int(k.split("_")[1]): k for k in sigs if k.startswith("prefill_")}
    self.runners = {}
    det = self.decode.get_input_details()
    self.state_shapes = {k: (tuple(int(x) for x in v["shape"]), v["dtype"]) for k, v in det.items() if k.startswith("kv_cache")}
    self.cache = int(det["mask"]["shape"][-1])

  def runner(self, length):
    name = self.prefill[length]
    if name not in self.runners:
      self.runners[name] = self.itp.get_signature_runner(name)
    return self.runners[name]

  def teacher(self, emb, n_prompt, pad_emb):
    """emb [n_prompt + len(teacher) - 1, hidden]: prompt[:-1] prefilled, then decode from the last prompt slot on."""
    state = {k: np.zeros(s, dtype=d) for k, (s, d) in self.state_shapes.items()}
    pos = 0
    for sig, n in plan_prefill(list(self.prefill), n_prompt - 1):
      x = np.repeat(pad_emb[None, None], sig, axis=1).astype(np.float32)
      x[0, :n] = emb[pos:pos + n]
      ip = np.zeros(sig, np.int32)
      ip[:n] = np.arange(pos, pos + n, dtype=np.int32)
      o = self.runner(sig)(embeddings=x, input_pos=ip, mask=mask(pos, n, sig, self.cache), **state)
      state = {k: o[k] for k in state}
      pos += n
    out = []
    for p in range(n_prompt - 1, len(emb)):
      o = self.decode(embeddings=emb[p][None, None].astype(np.float32), input_pos=np.array([p], np.int32),
                      mask=mask(p, 1, 1, self.cache), **state)
      out.append(o["logits"][0, 0].astype(np.float32))
      state = {k: o[k] for k in state}
    return np.stack(out)


def version():
  import importlib.metadata as md
  for p in ("ai-edge-litert-nightly", "ai-edge-litert"):
    try:
      return f"{p} {md.version(p)}"
    except md.PackageNotFoundError:
      pass
  return "?"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--decoder", required=True)
  ap.add_argument("--embedder", required=True)
  ap.add_argument("--cases", required=True)
  ap.add_argument("--label", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--ref", default="")
  ap.add_argument("--save-ref", action="store_true")
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--only", default="")
  a = ap.parse_args()
  from ai_edge_litert import interpreter as interp
  meta = json.load(open(a.cases))
  base = os.path.dirname(os.path.abspath(a.cases))
  img_id, pad_id = meta.get("image_token_id"), meta["pad_id"]
  cases = meta["cases"]
  if a.only:
    keep = set(a.only.split(","))
    cases = [c for c in cases if c["key"] in keep]
  os.makedirs(a.out_dir, exist_ok=True)
  t0 = time.time()
  e_run = interp.Interpreter(model_path=a.embedder, num_threads=1).get_signature_runner("embedder")
  need = {pad_id}
  for c in cases:
    need |= {t for t in c["ids"] if t != img_id} | set(c["teacher"])
  table = {t: e_run(token_ids=np.array([[t]], np.int32))["embeddings"][0, 0].astype(np.float32) for t in sorted(need)}
  hidden = len(table[pad_id])
  dec = Decoder(a.decoder, a.threads)
  load_s = time.time() - t0
  print(f"loaded {load_s:.1f}s embedder ids {len(table)} hidden {hidden} cache {dec.cache} prefill {sorted(dec.prefill)}",
        flush=True)
  ref_dir = os.path.join(a.out_dir, f"logits_{a.ref}") if a.ref else None
  save_dir = os.path.join(a.out_dir, f"logits_{a.label}") if a.save_ref else None
  if save_dir:
    os.makedirs(save_dir, exist_ok=True)
  vis, rows = {}, []
  for c in cases:
    t1 = time.time()
    seq = c["ids"] + c["teacher"][:-1]
    emb = np.stack([table[t] if t != img_id else np.zeros(hidden, np.float32) for t in seq])
    if c.get("image_npy"):
      p = c["image_npy"] if os.path.isabs(c["image_npy"]) else os.path.join(base, c["image_npy"])
      if p not in vis:
        vis[p] = np.load(p)
      slots = [i for i, t in enumerate(c["ids"]) if t == img_id]
      emb[slots] = vis[p]
    logits = dec.teacher(emb, len(c["ids"]), table[pad_id])
    teach = np.array(c["teacher"])
    top = np.argsort(logits, axis=-1)[:, -2:]
    top1 = top[:, 1]
    idx = np.arange(len(teach))
    gap = logits[idx, top1] - logits[idx, top[:, 0]]
    row = {"key": c["key"], "group": c["group"], "n": int(len(teach)), "counted": bool(c.get("counted", True)),
           "top1": top1.tolist(), "eq_teacher": (top1 == teach).tolist(),
           "first_div_teacher": next((int(i) for i in idx if top1[i] != teach[i]), None),
           "gap": np.round(gap, 4).tolist(),
           "teacher_gap_to_top1": np.round(logits[idx, top1] - logits[idx, teach], 4).tolist()}
    row["all_teacher"] = row["first_div_teacher"] is None
    row["min_gap"] = float(np.min(gap))
    fname = c["key"].replace("|", "__").replace("/", "_") + ".npy"
    if save_dir:
      np.save(os.path.join(save_dir, fname), logits)
    if ref_dir:
      ref = np.load(os.path.join(ref_dir, fname))
      assert ref.shape == logits.shape, (ref.shape, logits.shape)
      d = logits - ref
      rtop1 = ref.argmax(-1)
      row["eq_ref"] = (top1 == rtop1).tolist()
      row["first_div_ref"] = next((int(i) for i in idx if top1[i] != rtop1[i]), None)
      row["all_ref"] = row["first_div_ref"] is None
      row["dmax"] = np.round(np.abs(d).max(-1), 4).tolist()
      row["drms"] = np.round(np.sqrt((d.astype(np.float64) ** 2).mean(-1)), 5).tolist()
      rs = np.sort(ref, axis=-1)[:, -2:]
      row["ref_gap"] = np.round(rs[:, 1] - rs[:, 0], 4).tolist()
      ra = np.argsort(ref, axis=-1)[:, -2:]
      row["pair_margin"] = np.round(logits[idx, ra[:, 1]] - logits[idx, ra[:, 0]], 4).tolist()
    rows.append(row)
    msg = (f"{c['key']:28s} n {row['n']:2d} all_teacher {row['all_teacher']!s:5s} first_div {row['first_div_teacher']}"
           f" min_gap {row['min_gap']:.3f}")
    if ref_dir:
      msg += f" | vs ref all {row['all_ref']!s:5s} first_div {row['first_div_ref']} max|d| {max(row['dmax']):.3f}"
    print(msg + f" ({time.time() - t1:.1f}s)", flush=True)
  summ = {}
  for grp in sorted({r["group"] for r in rows}):
    rr = [r for r in rows if r["group"] == grp]
    s = {"rows": len(rr), "all_teacher": sum(r["all_teacher"] for r in rr), "positions": sum(r["n"] for r in rr),
         "pos_eq_teacher": sum(sum(r["eq_teacher"]) for r in rr), "counted_rows": sum(r["counted"] for r in rr),
         "all_teacher_counted": sum(r["all_teacher"] for r in rr if r["counted"])}
    if ref_dir:
      dm = np.concatenate([r["dmax"] for r in rr])
      dr = np.concatenate([r["drms"] for r in rr])
      s.update({"all_ref": sum(r["all_ref"] for r in rr), "pos_eq_ref": sum(sum(r["eq_ref"]) for r in rr),
                "dmax_max": float(dm.max()), "dmax_p95": float(np.percentile(dm, 95)), "dmax_mean": float(dm.mean()),
                "drms_mean": float(dr.mean()), "all_ref_counted": sum(r["all_ref"] for r in rr if r["counted"])})
    summ[grp] = s
  res = {"label": a.label, "decoder": os.path.abspath(a.decoder), "decoder_sha256": sha256(a.decoder),
         "decoder_bytes": os.path.getsize(a.decoder), "embedder": os.path.abspath(a.embedder),
         "embedder_sha256": sha256(a.embedder), "cases": os.path.abspath(a.cases), "ref": a.ref, "threads": a.threads,
         "load_s": round(load_s, 1), "wall_s": round(time.time() - t0, 1),
         "backend": "LiteRT CPU Interpreter (XNNPACK), " + version(), "summary": summ, "rows": rows}
  json.dump(res, open(os.path.join(a.out_dir, f"{a.label}.json"), "w"), indent=1)
  print("TEACHER_FORCE_DONE " + a.label + " " + json.dumps(summ), flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
