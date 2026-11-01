#!/usr/bin/env python3
"""Run a .litertlm VLM on the plan's placement requests the way an app sends them, with the LiteRT-LM Python package.

  python tools/demo_scene_check/run_litertlm.py <model.litertlm> --plan DIR/plan/lfm3.json --label int4_gpu_on \
      --backend gpu --think on --ctx 4096 --prompt-tokens-from DIR/raw/lfm3_mps.jsonl --out DIR/litertlm
  python tools/demo_scene_check/run_litertlm.py <model.litertlm> --plan ... --label int4_gpu_off --backend gpu \
      --think off --max-new 256 --out DIR/litertlm

One user message = [image part (the plan's S512 picture = what the app hands the vision encoder), text part (the same
request as the transformers runs)], extra_context enable_thinking per --think, greedy (no sampler in the file; the
engine default is top-k 1). One fresh process per case, cache_dir ":nocache", stdin /dev/null; a case whose stderr
carries DEADLINE_EXCEEDED is marked void and runs again on the next call.
Output budget: --max-new N, or --ctx C with --prompt-tokens-from <transformers run of the same plan>: max_output_tokens
= C - that run's input token count for the same case (a thought that does not fit ends with an empty answer, scored as
no answer; that is what the app would show).
Output <out>/<label>.json, written after every case (a re-run skips the cases it has): per case the answer
(content), the thought channel, prefill / decode token counts from the engine's benchmark info, the return code, peak
RSS and the counts of error lines in stderr. The benchmark info goes through the package's C bindings
(litert_lm._ffi); a version without them leaves the token counts out and the run goes on.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHILD = r'''
import json, sys, time, ctypes, resource
import litert_lm
from litert_lm import interfaces
spec = json.loads(sys.argv[1])
B = {"cpu": interfaces.Backend.CPU(), "gpu": interfaces.Backend.GPU()}
out = {}
t0 = time.time()
eng = litert_lm.Engine(spec["bundle"], backend=B[spec["backend"]], vision_backend=B[spec["backend"]],
                       cache_dir=":nocache", enable_benchmark=True)
out["load_s"] = round(time.time() - t0, 2)
msg = {"role": "user", "content": [{"type": "image", "path": spec["image"]}, {"type": "text", "text": spec["text"]}]}
c0 = eng.create_conversation(extra_context={"enable_thinking": spec["think"] == "on"})
try:
  out["render"] = c0.render_message_to_string(msg)
except Exception as e:
  out["render_error"] = f"{type(e).__name__}: {e}"[:600]
c0.close()
conv = eng.create_conversation(extra_context={"enable_thinking": spec["think"] == "on"}, max_output_tokens=spec["max_new"])
t1 = time.time()
reply = conv.send_message(msg)
out["gen_s"] = round(time.time() - t1, 2)
raw = dict(reply)
out["content"] = "".join(c.get("text", "") for c in (raw.get("content") or []) if isinstance(c, dict) and c.get("type") == "text")
out["channels"] = dict(getattr(reply, "channels", {}) or {})
out["raw"] = raw
out["token_count"] = int(conv.token_count)
try:
  from litert_lm._ffi import create_checked, get_checked
  lib = conv._lib
  info = create_checked(lib, "litert_lm_conversation_get_benchmark_info", conv._ptr)
  try:
    turns = {}
    for kind in ("prefill", "decode"):
      n = get_checked(lib, f"litert_lm_benchmark_info_get_num_{kind}_turns", ctypes.c_size_t, info)
      turns[kind] = [{"tokens": get_checked(lib, f"litert_lm_benchmark_info_get_{kind}_token_count_at", ctypes.c_size_t, info, i),
                      "tokens_per_s": round(get_checked(lib, f"litert_lm_benchmark_info_get_{kind}_tokens_per_sec_at", ctypes.c_double, info, i), 2)}
                     for i in range(n)]
    out["turns"] = turns
  finally:
    lib.litert_lm_benchmark_info_delete(info)
except Exception as e:
  out["turns_error"] = f"{type(e).__name__}: {e}"[:400]
out["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
print("RESULT " + json.dumps(out), flush=True)
'''


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("bundle")
  ap.add_argument("--plan", required=True)
  ap.add_argument("--label", required=True)
  ap.add_argument("--backend", choices=("cpu", "gpu"), required=True)
  ap.add_argument("--think", choices=("off", "on"), default="off")
  ap.add_argument("--arm", default="S512")
  ap.add_argument("--max-new", type=int, default=None)
  ap.add_argument("--ctx", type=int, default=None)
  ap.add_argument("--prompt-tokens-from", default=None)
  ap.add_argument("--only", default="")
  ap.add_argument("--out", required=True)
  ap.add_argument("--timeout", type=int, default=3600)
  a = ap.parse_args()
  if (a.max_new is None) == (a.ctx is None):
    raise SystemExit("give --max-new N, or --ctx C with --prompt-tokens-from")
  plan = json.loads(Path(a.plan).read_text())
  fx_dir = Path(plan["fixtures"]).parent
  cases = [c for c in plan["cases"] if c["arm"] == a.arm and c["think"] == a.think and c["q_type"] == "place"]
  if a.only:
    want = set(a.only.split(","))
    cases = [c for c in cases if c["key"] in want or c["id"] in want]
  hf_in = {}
  if a.ctx is not None:
    if not a.prompt_tokens_from:
      raise SystemExit("--ctx needs --prompt-tokens-from")
    for line in open(a.prompt_tokens_from):
      if line.strip():
        r = json.loads(line)
        hf_in[r["key"]] = r["n_input_ids"]
    missing = [c["key"] for c in cases if c["key"] not in hf_in]
    if missing:
      raise SystemExit(f"no input token count for {len(missing)} cases, e.g. {missing[:3]}")
  out_dir = Path(a.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"{a.label}.json"
  res = json.loads(path.read_text()) if path.exists() else {
      "label": a.label, "bundle": str(Path(a.bundle).resolve()), "bundle_bytes": os.path.getsize(a.bundle),
      "bundle_sha256": sha256(a.bundle), "backend": a.backend, "think": a.think,
      "max_new": a.max_new if a.max_new is not None else f"{a.ctx} - input tokens", "python": sys.executable,
      "plan": str(Path(a.plan).resolve()), "cases": []}
  done = {c["key"] for c in res["cases"] if "content" in c and not c.get("void")}
  todo = [c for c in cases if c["key"] not in done]
  print(f"{a.label}: {len(todo)} to run, {len(done)} done", flush=True)
  for c in todo:
    img = str((fx_dir / c["image"]).resolve())
    max_new = a.max_new if a.max_new is not None else a.ctx - hf_in[c["key"]]
    spec = {"bundle": str(Path(a.bundle).resolve()), "backend": a.backend, "max_new": max_new, "image": img,
            "text": c["text"], "think": a.think}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-I", "-c", CHILD, json.dumps(spec)], capture_output=True, text=True,
                       timeout=a.timeout, stdin=subprocess.DEVNULL)
    rline = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    rec = {"key": c["key"], "id": c["id"], "q_type": c["q_type"], "image": c["image"], "image_sha256": sha256(img),
           "text": c["text"], "think": a.think, "max_new": max_new, "hf_input_tokens": hf_in.get(c["key"]),
           "rc": p.returncode, "wall_s": round(time.time() - t0, 1), "ended": time.strftime("%Y-%m-%d %H:%M:%S")}
    if rline:
      rec.update(json.loads(rline[len("RESULT "):]))
      pre = [t["tokens"] for t in rec.get("turns", {}).get("prefill", [])]
      dec = [t["tokens"] for t in rec.get("turns", {}).get("decode", [])]
      rec["prefill_tokens"] = sum(pre) if pre else None
      rec["decode_tokens"] = sum(dec) if dec else None
    else:
      rec["error_tail"] = (p.stdout + p.stderr)[-2500:]
    err = p.stderr
    rec["void"] = "DEADLINE_EXCEEDED" in err
    rec["stderr_flags"] = {k: err.count(k) for k in ("Validation error", "Shape mismatch", "not fully delegated",
                                                     "DEADLINE_EXCEEDED", "ERROR", "WARNING")}
    rec["stderr_tail"] = "\n".join(ln for ln in err.splitlines() if "rror" in ln or "WARN" in ln)[-2000:]
    prior = [x for x in res["cases"] if x["key"] == c["key"]]
    if prior:
      res.setdefault("attempts", []).extend(prior)
      res["cases"] = [x for x in res["cases"] if x["key"] != c["key"]]
    res["cases"].append(rec)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[{a.label} {c['key']}] rc={p.returncode} {rec['wall_s']} s load={rec.get('load_s')} gen={rec.get('gen_s')} "
          f"prefill={rec.get('prefill_tokens')} decode={rec.get('decode_tokens')} void={rec['void']} | "
          f"{(rec.get('content') or rec.get('error_tail', ''))[:110]!r}", flush=True)
  print("RUN_LITERTLM_DONE", a.label, path, flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
