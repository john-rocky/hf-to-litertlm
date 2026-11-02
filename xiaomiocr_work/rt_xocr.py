#!/usr/bin/env python3
"""Gate runner of the Xiaomi-OCR-0 fast_vlm bundle on a desktop (written on a Mac): every case of the set of
make_set.py through the LiteRT-LM Python API, ONE fresh process per case (the hybrid decoder keeps a running state, and a
second conversation on the same Engine continues from it), stdin /dev/null, rows in the shape score_xocr.py reads
(name, mode, text, text_raw, stop, n_gen, seconds), so the bundle rows are judged against HF native and compared
character by character with the HF rows of the same picture (compare_hf.py).

  suite  python rt_xocr.py suite BUNDLE --set SET.json --label L --backend cpu|gpu --vision-backend cpu|gpu \
             --mode p1344x768 --out ROWS.jsonl [--threads N, default 4] [--only a,b] [--tiles TILES.json]
             [--cache-dir DIR | default :nocache]
  one    (the per-case child; the suite starts it)
  conv   python rt_xocr.py conv BUNDLE --set SET.json --backend .. --vision-backend .. --name N --mode M \
             --follow "text" [--follow ..] --out ROWS.jsonl   (one conversation: [image, prompt] then text-only turns)

Per case: the image item(s) first, then the case's prompt (as the HF rows: one user turn [image, text]);
max_output_tokens = the case's cap (set.json max_new_tokens); greedy = the bundle's sampler unless --top-k;
generation through send_message_async (a blocking send_message gives up after 10 minutes). stop = "eos" when the
decode count is below the cap, else "max_new_tokens". Each child's stderr is kept; a GPU child whose stderr holds
`Validation error` or `Shape mismatch` lines is void (LiteRT-LM's WebGPU path on a Mac has printed them in runs that
still exited 0): its log is renamed *.invalid.N and the case is run again in a new process (up to --retries); so is a
child that died without a row or whose reply ended in an exception (a GPU reply cut by a pause over 2 min:
DEADLINE_EXCEEDED). The runtime caches go to --cache-dir (one dir per bundle) or nowhere (:nocache, the default) -
never beside the bundle (the runtime's default place, where they outlive the bundle and the GPU program cache grows
with every process).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

SET = {"path": None}     # set.json of make_set.py; its "file" entries are relative to its folder
BAD_RE = re.compile(r"Validation error|Shape mismatch")


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def load_cases():
  return {c["name"]: c for c in json.load(open(SET["path"]))["cases"]}


def case_images(case, mode, tiles):
  """-> list of (path, sha256) for the case in this mode (one picture, or the tiles of a tiled mode)."""
  if tiles:
    man = json.load(open(tiles))
    d = os.path.dirname(os.path.abspath(tiles))
    it = next(x for x in man["items"] if x["name"] == case["name"])
    return [(os.path.join(d, f["file"]), f["sha256"]) for f in it["files"]]
  f = case["files"][mode]
  return [(os.path.join(os.path.dirname(os.path.abspath(SET["path"])), f["file"]), f["sha256"])]


def backend(lm, name, threads=None):
  if name == "cpu":
    return lm.interfaces.Backend.CPU(thread_count=threads) if threads else lm.interfaces.Backend.CPU()
  return lm.interfaces.Backend.GPU()


def stream(conv, message, cap):
  """-> (text, first-chunk s, total s, error or None); an exception mid-reply keeps the text streamed so far."""
  text, t_first, err = "", None, None
  t0 = time.time()
  try:
    for chunk in conv.send_message_async(message, max_output_tokens=cap):
      if t_first is None:
        t_first = time.time()
      content = chunk.get("content") if isinstance(chunk, dict) else getattr(chunk, "content", None)
      if isinstance(content, list):
        for c in content:
          if isinstance(c, dict) and c.get("type", "text") == "text":
            text += c.get("text", "")
      elif isinstance(content, str):
        text += content
  except Exception as e:  # noqa: BLE001
    err = repr(e)[:800]
  return text, (t_first - t0) if t_first else None, time.time() - t0, err


def bench(conv):
  try:
    bi = conv.get_benchmark_info()
  except Exception as e:  # noqa: BLE001
    return {"error": repr(e)[:200]}
  out = {}
  for k in ("init_time_in_second", "time_to_first_token_in_second", "last_prefill_token_count",
            "last_prefill_tokens_per_second", "last_decode_token_count", "last_decode_tokens_per_second"):
    v = getattr(bi, k, None)
    out[k] = v() if callable(v) else v
  return out


def make_engine(lm, a, n_images):
  cache = a.cache_dir or ":nocache"
  kw = dict(backend=backend(lm, a.backend, a.threads), vision_backend=backend(lm, a.vision_backend),
            max_num_images=max(1, n_images), enable_benchmark=True, cache_dir=cache)
  if a.max_num_tokens:
    kw["max_num_tokens"] = a.max_num_tokens
  return lm.Engine(a.bundle, **kw)


def cmd_one(a):
  import litert_lm as lm
  case = load_cases()[a.name]
  imgs = case_images(case, a.mode, a.tiles)
  for p, s in imgs:
    assert sha256(p) == s, p
  cap = a.max_output_tokens or case["max_new_tokens"]
  content = [{"type": "image", "path": p} for p, _ in imgs] + [{"type": "text", "text": case["prompt"]}]
  t0 = time.time()
  engine = make_engine(lm, a, len(imgs))
  kw = {}
  if a.top_k:
    kw["sampler_config"] = lm.interfaces.SamplerConfig(top_k=a.top_k)
  conv = engine.create_conversation(**kw)
  t1 = time.time()
  text, first, gen_s, err = stream(conv, {"role": "user", "content": content}, cap)
  bi = bench(conv)
  n_dec = bi.get("last_decode_token_count")
  stop = "eos" if (isinstance(n_dec, int) and n_dec < cap) else "max_new_tokens"
  if err:
    stop = "error"
  row = {"name": a.name, "mode": a.mode, "label": a.label, "tag": a.label, "error": err, "backend": a.backend,
         "vision_backend": a.vision_backend, "threads": a.threads, "runtime": getattr(lm, "__version__", "?"),
         "bundle": os.path.abspath(a.bundle), "n_images": len(imgs), "images": [os.path.basename(p) for p, _ in imgs],
         "max_new_tokens": cap, "text": text, "text_raw": text, "stop": stop, "n_gen": n_dec,
         "engine_s": round(t1 - t0, 2), "first_chunk_s": round(first, 3) if first else None,
         "gen_s": round(gen_s, 2), "seconds": round(time.time() - t0, 1), "bench": bi, "pos1d": False,
         "cache_dir": a.cache_dir or ":nocache"}
  print("ROWJSON " + json.dumps(row, ensure_ascii=False), flush=True)
  return 0


def cmd_conv(a):
  import litert_lm as lm
  case = load_cases()[a.name]
  imgs = case_images(case, a.mode, a.tiles)
  for p, s in imgs:
    assert sha256(p) == s, p
  cap = a.max_output_tokens or case["max_new_tokens"]
  t0 = time.time()
  engine = make_engine(lm, a, len(imgs))
  conv = engine.create_conversation()
  turns = []
  msgs = [[{"type": "image", "path": p} for p, _ in imgs] + [{"type": "text", "text": case["prompt"]}]]
  msgs += [[{"type": "text", "text": f}] for f in a.follow]
  for i, content in enumerate(msgs):
    rec = {"turn": i + 1, "user": content[-1]["text"], "images": sum(1 for c in content if c["type"] == "image")}
    try:
      text, first, gen_s, err = stream(conv, {"role": "user", "content": content}, cap)
      bi = bench(conv)
      n_dec = bi.get("last_decode_token_count")
      rec.update({"text": text, "n_gen": n_dec, "stop": "error" if err else ("eos" if (isinstance(n_dec, int)
                  and n_dec < cap) else "max_new_tokens"), "gen_s": round(gen_s, 2), "bench": bi, "error": err})
    except Exception as e:  # noqa: BLE001
      rec.update({"error": repr(e)[:600]})
    turns.append(rec)
    if rec.get("error"):
      break
  row = {"name": a.name, "mode": a.mode, "label": a.label, "backend": a.backend, "vision_backend": a.vision_backend,
         "threads": a.threads, "runtime": getattr(lm, "__version__", "?"), "bundle": os.path.abspath(a.bundle),
         "turns": turns, "seconds": round(time.time() - t0, 1), "cache_dir": a.cache_dir or ":nocache"}
  print("ROWJSON " + json.dumps(row, ensure_ascii=False), flush=True)
  return 0


def child_cmd(a, sub, name, extra=()):
  cmd = [sys.executable, "-I", os.path.abspath(__file__), sub, a.bundle, "--label", a.label, "--backend", a.backend,
         "--vision-backend", a.vision_backend, "--name", name, "--mode", a.mode]
  cmd += ["--set", os.path.abspath(SET["path"])]
  for k in ("threads", "tiles", "cache_dir", "max_output_tokens", "max_num_tokens", "top_k"):
    v = getattr(a, k)
    if v:
      cmd += ["--" + k.replace("_", "-"), str(v)]
  return cmd + list(extra)


def run_child(a, cmd, log_base):
  """-> (row or None, rc, n_bad, seconds); stderr in log_base.stderr.log (void logs renamed *.invalid.N)."""
  for attempt in range(a.retries + 1):
    t0 = time.time()
    try:
      env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")
      # RT_TIME=1: /usr/bin/time -l around the child = its peak memory footprint and max RSS in the row (macOS bytes)
      run = (["/usr/bin/time", "-l"] + cmd) if os.environ.get("RT_TIME") == "1" else cmd
      pr = subprocess.run(run, capture_output=True, text=True, timeout=a.timeout, stdin=subprocess.DEVNULL, env=env)
      out, err, rc = pr.stdout, pr.stderr, pr.returncode
    except subprocess.TimeoutExpired as e:
      out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
      err = (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")) + "\nTIMEOUT"
      rc = -9
    sec = round(time.time() - t0, 1)
    log = log_base + ".stderr.log"
    open(log, "w").write(err)
    n_bad = len(BAD_RE.findall(err))
    row = None
    for line in out.splitlines():
      if line.startswith("ROWJSON "):
        row = json.loads(line[8:])
    if row is not None and os.environ.get("RT_TIME") == "1":
      fp = re.findall(r"(\d+)\s+peak memory footprint", err)
      rs = re.findall(r"(\d+)\s+maximum resident set size", err)
      row["peak_footprint_b"] = int(fp[-1]) if fp else None
      row["max_rss_b"] = int(rs[-1]) if rs else None
    why = (f"{n_bad} validation / shape lines in stderr" if n_bad else
           "no row (the child died)" if row is None else
           f"error {row.get('error', '')[:120]}" if row.get("stop") == "error" else None)
    if why and attempt < a.retries:
      os.replace(log, f"{log_base}.stderr.invalid.{attempt + 1}.log")
      if row is not None:
        open(f"{log_base}.row.invalid.{attempt + 1}.json", "w").write(json.dumps(row, ensure_ascii=False))
      print(f"   void: {why} -> rerun in a new process", flush=True)
      continue
    return row, rc, n_bad, sec
  return row, rc, n_bad, sec


def cmd_suite(a):
  cases = load_cases()
  only = a.only.split(",") if a.only else None
  logdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "logs", a.label)
  os.makedirs(logdir, exist_ok=True)
  done = set()
  if os.path.exists(a.out):
    done = {json.loads(l)["name"] for l in open(a.out) if l.strip()}
  bsha = sha256(a.bundle)
  names = only or list(cases)
  if a.tiles:
    tiled = {x["name"] for x in json.load(open(a.tiles))["items"]}
    names = [n for n in names if n in tiled]
  else:
    names = [n for n in names if a.mode in cases[n]["files"]]
  for name in names:
    if name in done:
      continue
    row, rc, n_bad, sec = run_child(a, child_cmd(a, "one", name), os.path.join(logdir, f"{name}_{a.mode}"))
    if row is None:
      row = {"name": name, "mode": a.mode, "label": a.label, "backend": a.backend, "vision_backend": a.vision_backend,
             "threads": a.threads, "text": "", "text_raw": "", "stop": "error", "n_gen": None, "seconds": sec,
             "pos1d": False}
    row.update({"rc": rc, "stderr_bad_lines": n_bad, "bundle_sha256": bsha, "wall_s": sec})
    with open(a.out, "a") as fh:
      fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name:18s} {a.mode} rc {rc} bad {n_bad} {sec:7.1f}s stop {row.get('stop')} n {row.get('n_gen')} | "
          f"{row.get('text', '')[:70]!r}", flush=True)
  print("SUITE_DONE", a.label, a.out, flush=True)
  return 0


def cmd_conv_suite(a):
  logdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "logs", a.label)
  os.makedirs(logdir, exist_ok=True)
  extra = []
  for f in a.follow:
    extra += ["--follow", f]
  row, rc, n_bad, sec = run_child(a, child_cmd(a, "conv1", a.name, extra), os.path.join(logdir, f"conv_{a.name}"))
  row = row or {"name": a.name, "label": a.label, "turns": [], "error": "no row"}
  row.update({"rc": rc, "stderr_bad_lines": n_bad, "wall_s": sec, "bundle_sha256": sha256(a.bundle)})
  with open(a.out, "a") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
  for t in row.get("turns", []):
    print(f"turn {t['turn']} stop {t.get('stop')} n {t.get('n_gen')} err {(t.get('error') or '')[:120]} | "
          f"{(t.get('text') or '')[:100]!r}", flush=True)
  print("CONV_DONE", a.label, rc, n_bad, flush=True)
  return 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=("suite", "one", "conv", "conv1"))
  ap.add_argument("bundle")
  ap.add_argument("--set", required=True, help="set.json of make_set.py")
  ap.add_argument("--label", default="rt")
  ap.add_argument("--backend", default="cpu")
  ap.add_argument("--vision-backend", default="cpu")
  ap.add_argument("--threads", type=int, default=4)   # CPU thread_count, always explicit and recorded in the row
  ap.add_argument("--mode", default="p1344x768")
  ap.add_argument("--tiles", default=None)
  ap.add_argument("--name", default=None)
  ap.add_argument("--only", default=None)
  ap.add_argument("--out", default=None)
  ap.add_argument("--cache-dir", default=None)
  ap.add_argument("--max-output-tokens", type=int, default=None)
  ap.add_argument("--max-num-tokens", type=int, default=None)
  ap.add_argument("--top-k", type=int, default=None)
  ap.add_argument("--follow", action="append", default=[])
  ap.add_argument("--timeout", type=int, default=1800)
  ap.add_argument("--retries", type=int, default=1)
  a = ap.parse_args()
  a.bundle = os.path.abspath(a.bundle)
  SET["path"] = os.path.abspath(a.set)
  if a.cmd == "one":
    return cmd_one(a)
  if a.cmd == "conv1":
    return cmd_conv(a)
  if a.cmd == "conv":
    return cmd_conv_suite(a)
  return cmd_suite(a)


if __name__ == "__main__":
  sys.exit(main())
