#!/usr/bin/env python3
"""Gate runner of the Unlimited-OCR-2 fast_vlm bundle on the set of build_set.py: every picture through the LiteRT-LM
Python API, ONE fresh process per picture, stdin /dev/null.

  suite  python rt_uocr.py suite BUNDLE --set SET.json [--ref-rows ROWS_R.jsonl] --label L --backend cpu|gpu
             --vision-backend cpu|gpu --arm a1024|s1024|w1024 --out ROWS.jsonl [--threads 4] [--only a,b]
             [--cache-dir DIR | default :nocache] [--max-num-tokens N] [--ngram 35,128 | 0] [--split]
  one    (the per-picture child; the suite starts it)

The message = one user turn [image x n_pages (the arm's PNG of each page), text "Multi page parsing."] (the bundle's
template writes the fixed instruction either way); max_output_tokens = max(512, min(6000, 2 x R's generated tokens))
from --ref-rows (the reference rows of uocr_oracle.py), 6,000 without them, unless --max-output-tokens; greedy = the
bundle's sampler unless --top-k; generation through send_message_async (a blocking send ends at 10 minutes).
--ngram N,W passes the runtime's NoRepeatNgramConfig(N, W) to the reply (0 = off). --split runs a multi-page picture
one page per process (each run's prompt holds one <PAGE>; rows tag "page<i>", pages [i], cap = the one-page cap of
the whole document's R); score_uocr.py scores them as one row.
gen_ids = Engine.tokenize of the returned text (the runtime's tokenizer, no BOS): the Python API does not return the
generated ids, so prefix lengths against R are re-tokenized (the <PAGE> count is exact: one added token).

Rows: image, arm, label, backend, vision_backend, threads, text, stop ("eos" when the decode count is below the
cap, else "max_new_tokens"; "error" on an exception), finish ("stop" = eos), n_gen, num_pages_out (the "<PAGE>"
count in the text), engine / first-chunk / generation seconds, the runtime's BenchmarkInfo. Each child's stderr is
kept; a child whose stderr holds `Validation error` or `Shape mismatch` lines is void (its log renamed *.invalid.N)
and the picture is run again in a new process (up to --retries); so is a child that died without a row or whose
reply ended in an exception. The runtime caches go to --cache-dir or nowhere (:nocache). RT_TIME=1 runs each child
under /usr/bin/time -l and records its peak memory footprint and maximum resident set size (macOS).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PROMPT = "Multi page parsing."
BAD_RE = re.compile(r"Validation error|Shape mismatch")


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def load_images(a):
  return {im["name"]: im for im in json.load(open(a.set))["images"]}


def r_ngen(a):
  out = {}
  if a.ref_rows and os.path.exists(a.ref_rows):
    for line in open(a.ref_rows):
      if line.strip():
        r = json.loads(line)
        out[r["image"]] = r["n_gen"]
  return out


def image_pages(a, im):
  """-> list of (path, sha256) of the arm's PNG of each page (paths in set.json are relative to its directory)."""
  base = os.path.dirname(os.path.abspath(a.set))
  return [(os.path.join(base, p["variants"][a.arm]["file"]), p["variants"][a.arm]["sha256"]) for p in im["pages"]]


def cap_for(name, a):
  if a.max_output_tokens:
    return a.max_output_tokens
  n = r_ngen(a).get(name)
  return max(512, min(6000, 2 * n)) if n else 6000


def backend(lm, name, threads=None):
  if name == "cpu":
    return lm.interfaces.Backend.CPU(thread_count=threads) if threads else lm.interfaces.Backend.CPU()
  return lm.interfaces.Backend.GPU()


def stream(conv, message, cap, ngram_cfg=None):
  """-> (text, first-chunk s, total s, error or None); an exception mid-reply keeps the text streamed so far."""
  text, t_first, err = "", None, None
  t0 = time.time()
  kw = {"no_repeat_ngram_config": ngram_cfg} if ngram_cfg is not None else {}
  try:
    for chunk in conv.send_message_async(message, max_output_tokens=cap, **kw):
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
  im = load_images(a)[a.name]
  pages = image_pages(a, im)
  if a.page is not None:
    pages = [pages[a.page]]
  for p, s in pages:
    assert sha256(p) == s, p
  cap = cap_for(a.name, a)
  content = [{"type": "image", "path": p} for p, _ in pages] + [{"type": "text", "text": PROMPT}]
  t0 = time.time()
  engine = make_engine(lm, a, len(pages))
  kw = {}
  if a.top_k:
    kw["sampler_config"] = lm.interfaces.SamplerConfig(top_k=a.top_k)
  conv = engine.create_conversation(**kw)
  t1 = time.time()
  ngram_cfg = None
  if a.ngram and a.ngram != "0":
    n, w = (int(x) for x in a.ngram.split(","))
    ngram_cfg = lm.NoRepeatNgramConfig(no_repeat_ngram_size=n, window_size=w)
  text, first, gen_s, err = stream(conv, {"role": "user", "content": content}, cap, ngram_cfg)
  bi = bench(conv)
  try:
    gen_ids = [int(x) for x in engine.tokenize(text)] if text else []
  except Exception as e:  # noqa: BLE001
    gen_ids, err = [], err or f"tokenize: {e!r}"[:200]
  n_dec = bi.get("last_decode_token_count")
  stop = "eos" if (isinstance(n_dec, int) and n_dec < cap) else "max_new_tokens"
  if err:
    stop = "error"
  row = {"image": a.name, "name": a.name, "arm": a.arm, "label": a.label, "error": err, "backend": a.backend,
         "vision_backend": a.vision_backend, "threads": a.threads, "runtime": getattr(lm, "__version__", "?"),
         "bundle": os.path.abspath(a.bundle), "n_pages": len(pages), "images": [os.path.basename(p) for p, _ in pages],
         "max_new_tokens": cap, "text": text, "stop": stop, "finish": "stop" if stop == "eos" else stop,
         "n_gen": n_dec, "num_pages_out": text.count("<PAGE>"), "engine_s": round(t1 - t0, 2),
         "gen_ids": gen_ids, "gen_ids_from": "retokenized", "ngram": a.ngram or "0",
         "tag": f"page{a.page + 1}" if a.page is not None else None,
         "pages": [a.page + 1] if a.page is not None else list(range(1, len(pages) + 1)),
         "first_chunk_s": round(first, 3) if first else None, "gen_s": round(gen_s, 2),
         "seconds": round(time.time() - t0, 1), "bench": bi, "cache_dir": a.cache_dir or ":nocache"}
  print("ROWJSON " + json.dumps(row, ensure_ascii=False), flush=True)
  return 0


def child_cmd(a, sub, name, extra=()):
  cmd = [sys.executable, "-I", os.path.abspath(__file__), sub, a.bundle, "--label", a.label, "--backend", a.backend,
         "--vision-backend", a.vision_backend, "--name", name, "--arm", a.arm]
  for k in ("set", "ref_rows", "threads", "cache_dir", "max_output_tokens", "max_num_tokens", "top_k", "ngram"):
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
  images = load_images(a)
  if a.cache_dir:
    # the runtime refuses a cache dir that does not exist (INVALID_ARGUMENT); children get the absolute path
    a.cache_dir = os.path.abspath(a.cache_dir)
    os.makedirs(a.cache_dir, exist_ok=True)
  only = a.only.split(",") if a.only else None
  logdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "logs", a.label)
  os.makedirs(logdir, exist_ok=True)
  done = set()
  if os.path.exists(a.out):
    done = {(json.loads(l)["image"], json.loads(l).get("tag")) for l in open(a.out) if l.strip()}
  bsha = sha256(a.bundle)
  jobs = []
  for name in (only or list(images)):
    npg = len(images[name]["pages"])
    jobs += [(name, i) for i in range(npg)] if (a.split and npg > 1) else [(name, None)]
  for name, page in jobs:
    tag = f"page{page + 1}" if page is not None else None
    if (name, tag) in done:
      continue
    extra = ["--page", str(page)] if page is not None else []
    ts = time.time()
    row, rc, n_bad, sec = run_child(a, child_cmd(a, "one", name, extra),
                                    os.path.join(logdir, f"{name}_{a.arm}" + (f"_{tag}" if tag else "")))
    if row is None:
      row = {"image": name, "name": name, "arm": a.arm, "label": a.label, "backend": a.backend, "tag": tag,
             "pages": [page + 1] if page is not None else None, "vision_backend": a.vision_backend,
             "threads": a.threads, "text": "", "stop": "error", "finish": "error", "n_gen": None, "gen_ids": [],
             "seconds": sec}
    # t_start / t_end (epoch s) of the child
    row.update({"rc": rc, "stderr_bad_lines": n_bad, "bundle_sha256": bsha, "wall_s": sec,
                "t_start": round(ts, 1), "t_end": round(time.time(), 1)})
    with open(a.out, "a") as fh:
      fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name:18s} {tag or '':6s} {a.arm} rc {rc} bad {n_bad} {sec:7.1f}s stop {row.get('stop')} "
          f"n {row.get('n_gen')} | {row.get('text', '')[:70]!r}", flush=True)
  print("SUITE_DONE", a.label, a.out, flush=True)
  return 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=("suite", "one"))
  ap.add_argument("bundle")
  ap.add_argument("--label", default="rt")
  ap.add_argument("--backend", default="cpu")
  ap.add_argument("--vision-backend", default="cpu")
  ap.add_argument("--set", required=True, help="set.json of build_set.py")
  ap.add_argument("--ref-rows", default=None, help="R's rows (uocr_oracle.py) for the caps")
  ap.add_argument("--threads", type=int, default=4)   # CPU thread_count, always explicit and recorded
  ap.add_argument("--arm", default="s1024", choices=("s1024", "w1024", "a1024"))
  ap.add_argument("--name", default=None)
  ap.add_argument("--only", default=None)
  ap.add_argument("--out", default=None)
  ap.add_argument("--cache-dir", default=None)
  ap.add_argument("--max-output-tokens", type=int, default=None)
  ap.add_argument("--max-num-tokens", type=int, default=None)
  ap.add_argument("--top-k", type=int, default=None)
  ap.add_argument("--timeout", type=int, default=1800)
  ap.add_argument("--retries", type=int, default=1)
  ap.add_argument("--ngram", default=None, help="N,W for the runtime's NoRepeatNgramConfig (0 = off)")
  ap.add_argument("--split", action="store_true", help="multi-page pictures one page per run")
  ap.add_argument("--page", type=int, default=None, help="(child) the 0-based page to run alone")
  a = ap.parse_args()
  a.bundle = os.path.abspath(a.bundle)
  a.set = os.path.abspath(a.set)
  a.ref_rows = os.path.abspath(a.ref_rows) if a.ref_rows else None
  if a.cmd == "one":
    return cmd_one(a)
  return cmd_suite(a)


if __name__ == "__main__":
  sys.exit(main())
