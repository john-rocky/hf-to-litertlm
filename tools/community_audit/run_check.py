#!/usr/bin/env python3
"""Run a few LiteRT-LM checks on chosen .litertlm files and set them against the header audit.

Selection (from audit.json, written by audit.py): the LLM repos by the Hub
download count stored there, gated repos and files whose header did not read
left out, at most --top repos, one file per repo (not NPU-constrained, the
smallest); --extra REPO[:FILE] adds files by name. Each file is downloaded at
the audited commit into --dl-dir under its own name (the runtime reads the
format from the extension), its size and sha256 checked against the Hub's LFS
record, and deleted after its checks unless --keep.

Checks, each in a new process (one engine, one conversation), per backend:
  engine     Engine creation, and LiteRT-LM's own ModelInfo read of the file
  text       "What is the capital of France? Answer with one word." -> Paris
  image      (files with a vision section) a photo of two cats -> 2 / two
  tool       one tool-call exchange: the model calls get_weather, the tool returns
             a value the model cannot guess (23.7), the answer carries it
  think_on / think_off  (files that declare a channel or whose template reads
             enable_thinking) a thought is present with thinking on and absent
             with it off
Thinking is switched off for text / image / tool when the template switches it.
On the GPU, a vision file whose engine fails is tried again with the vision
encoder on the CPU (backend gpu_vcpu).

Output: runs.json (audit.py --runs) with each check's status, the first 300
characters of the reply, the error text and the seconds, plus notes where the
run and the header disagree.

Usage:
  python3 tools/community_audit/run_check.py --list
  python3 tools/community_audit/run_check.py --dl-dir /tmp/lc_files \
      --extra litert-community/MiniCPM-V-4 --top 9
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ENDPOINT = "https://huggingface.co"
UA = "hf-to-litertlm-community-audit/1"
# A photo used by reference only (COCO val2017 000000039769: two cats on a sofa).
PHOTO_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
RETRY_MARKS = ("DEADLINE_EXCEEDED", "already has", "timed out")


# ---------------------------------------------------------------- child
def reply_parts(msg):
  # litert_lm.Message: its to_json() is the C API's JSON (role, content, tool_calls, channels).
  if hasattr(msg, "to_json"):
    msg = msg.to_json()
  text, other = [], {}
  content = (msg.get("content", msg.get("contents")) if isinstance(msg, dict) else msg)
  if isinstance(content, str):
    text.append(content)
  elif isinstance(content, list):
    for c in content:
      if isinstance(c, dict) and c.get("type") == "text":
        text.append(c.get("text", ""))
      else:
        other.setdefault("content_parts", []).append(c)
  if isinstance(msg, dict):
    for k, v in msg.items():
      if k not in ("role", "content"):
        other[k] = v
  return "".join(text), other


def _strings(obj, skip=("role", "type", "name")):
  if isinstance(obj, str):
    return [obj] if obj.strip() else []
  if isinstance(obj, dict):
    return [s for k, v in obj.items() if k not in skip for s in _strings(v)]
  if isinstance(obj, list):
    return [s for v in obj for s in _strings(v)]
  return []


def thought_seen(text, other):
  return "<think>" in text or "</think>" in text or bool(
      _strings({k: v for k, v in other.items() if k != "tool_calls"}))


def _props(obj, depth=0):
  d = {}
  for n in dir(obj):
    if n.startswith("_"):
      continue
    try:
      v = getattr(obj, n)
    except Exception as e:  # noqa: BLE001
      d[n] = f"error: {type(e).__name__}: {e}"[:200]
      continue
    if callable(v):
      continue
    if isinstance(v, (str, int, float, bool, type(None))):
      d[n] = v
    elif isinstance(v, (list, tuple)):
      d[n] = [str(x) for x in v][:20]
    elif depth < 1:
      d[n] = _props(v, depth + 1)
    else:
      d[n] = repr(v)[:160]
  return d


# What a text or photo answer must hold: (shown, regex on the lowercased reply).
EXPECT = {"text": ("Paris", r"paris"), "image": ("2 / two", r"\b(2|two)\b")}
# A call written out in the reply instead of a parsed one: an XML or JSON call, or a call expression.
# A mention of the tool's name in prose is not one (the prompt names the tool).
CALL_SHAPE = re.compile(r"<tool_call>|<function=|\"name\"\s*:\s*\"get_weather\"|\bget_weather\s*\(")


def result_note(x, reads_tools):
  """A note on a stored result, from its reply and the header (so it can be derived again from runs.json)."""
  check, status, reply = x.get("check"), x.get("status"), x.get("reply") or ""
  if check == "tool" and status == "fail" and not x.get("tool_calls"):
    if not reads_tools:
      return "no call: the template does not read tools, so the model gets no tool list"
    return "the call arrived as text" if CALL_SHAPE.search(reply) else "no call"
  if check in EXPECT and status == "pass" and not re.search(EXPECT[check][1], reply.lower()):
    # The run behind the 2026-11-02 table kept the first 300 characters of each reply.
    return "passed on the full reply; the expected word is past the stored part"
  return None


def conversation(lm, eng, check, meta, photo):
  ctl, thinks = meta.get("thinking_control"), meta.get("thinks")
  big = (meta.get("size_bytes") or 0) > 8e9
  tc = getattr(lm, "ThinkingConfig", None) or lm.interfaces.ThinkingConfig
  kw, calls = {}, []
  limit = 1024 if thinks and ctl != "switch" else 256
  if check in ("text", "image", "tool") and thinks and ctl == "switch":
    kw["thinking_config"] = tc(enable_thinking=False)
  if check == "text":
    prompt = "What is the capital of France? Answer with one word."
  elif check == "image":
    prompt = {"role": "user", "content": [
        {"type": "image", "path": photo},
        {"type": "text", "text": "How many cats are in this picture? Answer with a number."}]}
  elif check == "tool":
    def get_weather(city: str) -> dict:
      """Get the current weather for a city.

      Args:
        city: The city name, for example "Paris".
      """
      calls.append(city)
      return {"city": city, "temperature_c": 23.7, "condition": "drizzle", "station": "QX-4417"}
    kw["tools"] = [get_weather]
    prompt = ("What is the weather in Paris right now? Use the get_weather tool, "
              "then tell me the temperature.")
  else:
    kw["thinking_config"] = tc(enable_thinking=(check == "think_on"))
    prompt = "What is 17 * 23? Reply with the number."
    limit = (768 if big else 2048) if check == "think_on" else 512
  conv = eng.create_conversation(max_output_tokens=limit, **kw)
  msg = conv.send_message(prompt)
  text, other = reply_parts(msg)
  r = {"reply": text, "other_keys": sorted(other), "max_output_tokens": limit}
  low = text.lower()
  if check in EXPECT:
    r["expect"], pattern = EXPECT[check]
    ok = bool(re.search(pattern, low))
  elif check == "tool":
    r["tool_calls"], r["expect"] = calls, "a get_weather call, and 23.7 in the answer"
    ok = bool(calls) and "23.7" in text
  else:
    seen = thought_seen(text, other)
    r["thought_seen"], r["answer_has_391"] = seen, "391" in text
    if seen:
      r["thought"] = (" ".join(_strings({k: v for k, v in other.items() if k != "tool_calls"})) or text)[:200]
    ok = seen if check == "think_on" else not seen
    r["expect"] = "a thought" if check == "think_on" else "no thought"
  r["status"] = "pass" if ok else "fail"
  return r


def child(a):
  import litert_lm as lm  # noqa: PLC0415 - the child alone loads the runtime
  meta, t0 = json.loads(a.meta), time.time()
  res = {"status": "error"}
  try:
    cpu, gpu = lm.Backend.CPU(), lm.Backend.GPU()
    be = cpu if a.backend == "cpu" else gpu
    kw = {}
    if meta.get("vision"):
      kw.update(vision_backend=gpu if a.backend == "gpu" else cpu, max_num_images=1)
    if a.check == "engine":
      try:
        mi_cls = getattr(lm, "ModelInfo", None)
        if mi_cls is None:
          from litert_lm.model_info import ModelInfo as mi_cls  # noqa: PLC0415
        res["model_info"] = _props(mi_cls(a.path))
      except Exception as e:  # noqa: BLE001
        res["model_info"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    eng = lm.Engine(a.path, backend=be, **kw)
    res["engine_s"] = round(time.time() - t0, 1)
    if a.check == "engine":
      res["status"] = "pass"
    else:
      res.update(conversation(lm, eng, a.check, meta, a.photo))
  except Exception as e:  # noqa: BLE001
    res.update(status="error", error=f"{type(e).__name__}: {e}"[:600])
  res["seconds"] = round(time.time() - t0, 1)
  sys.stdout.write("RESULT " + json.dumps(res, ensure_ascii=False, default=str) + "\n")
  sys.stdout.flush()
  os._exit(0)  # no engine teardown wait


# ---------------------------------------------------------------- driver
def select(audit, top, extra):
  cand = []
  for r in audit["repos"]:
    if r.get("gated"):
      continue
    fs = [f for f in r.get("files", []) if f.get("status") == "ok" and f.get("kind") == "llm"
          and "info:npu" not in f.get("flags", [])]
    if fs:
      # A file without a backend constraint first (the general form), then the smallest.
      cand.append((r, min(fs, key=lambda f: (bool(f.get("backend_constraint")), f["size_bytes"] or 0,
                                              f["file"]))))
  cand.sort(key=lambda x: (-(x[0].get("downloads") or 0), x[0]["repo"]))
  picked = cand[:top]
  for spec in extra:
    repo, _, fname = spec.partition(":")
    r = next((r for r in audit["repos"] if r["repo"] == repo), None)
    fs = [f for f in (r or {}).get("files", []) if f.get("status") == "ok" and (not fname or f["file"] == fname)]
    if fs:
      f = min(fs, key=lambda f: (f["size_bytes"] or 0, f["file"]))
      if all(p[1]["file"] != f["file"] or p[0]["repo"] != repo for p in picked):
        picked.append((r, f))
  return picked


def sha256_file(p):
  h = hashlib.sha256()
  with open(p, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


def download(url, dst, size=None, sha256=None):
  dst = pathlib.Path(dst)
  dst.parent.mkdir(parents=True, exist_ok=True)
  ok = dst.with_name(dst.name + ".sha256")
  if dst.exists() and ok.exists() and (sha256 is None or ok.read_text().strip() == sha256):
    return dst, ok.read_text().strip()
  part = dst.with_name(dst.name + ".part")
  for attempt in range(8):
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have >= size:
      break
    req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Range": f"bytes={have}-"} if have else {})})
    try:
      with urllib.request.urlopen(req, timeout=120) as r:
        mode = "ab" if have and r.status == 206 else "wb"
        with open(part, mode) as out:
          while True:
            b = r.read(1 << 24)
            if not b:
              break
            out.write(b)
      if size is None:
        break
    except Exception as e:  # noqa: BLE001
      print(f"download retry {attempt + 1}: {type(e).__name__}: {e}", file=sys.stderr)
      time.sleep(10)
  got = part.stat().st_size
  if size is not None and got != size:
    raise RuntimeError(f"{dst.name}: {got:,} B downloaded, the Hub lists {size:,}")
  digest = sha256_file(part)
  if sha256 is not None and digest != sha256:
    part.unlink()
    raise RuntimeError(f"{dst.name}: sha256 {digest} is not the Hub's {sha256}")
  part.rename(dst)
  ok.write_text(digest + "\n")
  return dst, digest


def run_child(a, path, backend, check, meta, photo):
  cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--one", "--path", str(path),
         "--backend", backend, "--check", check, "--meta", json.dumps(meta), "--photo", str(photo)]
  try:
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=a.timeout)
  except subprocess.TimeoutExpired:
    return {"status": "error", "error": f"timed out after {a.timeout} s"}
  lines = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
  err = p.stderr.splitlines()
  res = json.loads(lines[-1][7:]) if lines else {
      "status": "error", "error": f"no result (exit {p.returncode}); stderr: " + " / ".join(err[-3:])[:600]}
  res["validation_error_lines"] = sum(1 for x in err if "Validation error" in x)
  if res["status"] != "pass":
    res["stderr_errors"] = [x[:300] for x in err if re.search(
        r"ERROR|UNIMPLEMENTED|INVALID|INTERNAL|NOT_FOUND|FAILED|Unsupported|refus", x)][-4:]
  # The results are published: no local path (the download dir, the venv, the home dir) in them.
  s = json.dumps(res, ensure_ascii=False, default=str)
  for local, name in ((sys.prefix, "<venv>"), (str(pathlib.Path(a.dl_dir).resolve()), "<dl-dir>"),
                      (str(pathlib.Path(a.dl_dir)), "<dl-dir>"), (os.path.expanduser("~"), "~")):
    s = s.replace(local, name)
  return json.loads(s)


def header_view(f):
  lm_ = f.get("llm") or {}
  return {"vision": f.get("vision"), "tools": (lm_.get("template") or {}).get("reads_tools"),
          "channels": [c["name"] for c in lm_.get("channels", [])],
          "thinking_control": (lm_.get("thinking_control") or {}).get("value"),
          "backend_constraint": f.get("backend_constraint"), "model_type": lm_.get("model_type"),
          "sampler": (lm_.get("sampler") or {}).get("type")}


def compare(r, f, results):
  notes, hv = [], header_view(f)
  mine = {(x["backend"], x["check"]): x for x in results if x["repo"] == r["repo"] and x["file"] == f["file"]}
  tag = f"{r['repo']} ({f['file']})"
  for be in ("cpu", "gpu", "gpu_vcpu"):
    e = mine.get((be, "engine"))
    if e and e["status"] != "pass":
      notes.append(f"{tag} {be}: no engine — header: backend constraint "
                   f"{', '.join(hv['backend_constraint'] or []) or 'none'}; run: {(e.get('error') or '')[:220]}")
    for c, says in (("text", "a text answer"), ("image", "vision section" if hv["vision"] else "no vision"),
                    ("tool", f"template reads tools: {hv['tools']}, model type {hv['model_type'] or 'not set'}"),
                    ("think_on", f"channels {hv['channels']}, control {hv['thinking_control']}"),
                    ("think_off", f"channels {hv['channels']}, control {hv['thinking_control']}")):
      x = mine.get((be, c))
      if x and x["status"] not in ("pass", "skip"):
        got = x.get("error") or x.get("note") or x.get("thought") or x.get("reply") or ""
        notes.append(f"{tag} {be} {c}: {x['status']} — header: {says}; run: {got[:200]!r}")
  # LiteRT-LM v0.19.0's ModelInfo carries input_modalities and max_context_tokens (no supports_* flags).
  for be in ("cpu", "gpu"):
    mi = (mine.get((be, "engine")) or {}).get("model_info") or {}
    if not mi:
      continue
    im = mi.get("input_modalities") or {}
    for key in ("vision", "audio"):
      if isinstance(im.get(key), bool) and im[key] != bool(f.get(key)):
        notes.append(f"{tag}: LiteRT-LM ModelInfo input_modalities.{key} = {im[key]}, "
                     f"the header audit reads {bool(f.get(key))}")
    mnt, ctx = (f.get("llm") or {}).get("max_num_tokens"), mi.get("max_context_tokens")
    if mnt and isinstance(ctx, int) and ctx != mnt:
      notes.append(f"{tag}: LiteRT-LM ModelInfo max_context_tokens = {ctx}, the header's max_num_tokens = {mnt}")
    break
  return notes


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--audit", default=str(HERE / "table" / "audit.json"))
  ap.add_argument("--out", default=str(HERE / "table" / "runs.json"))
  ap.add_argument("--top", type=int, default=10)
  ap.add_argument("--extra", nargs="*", default=[])
  ap.add_argument("--dl-dir")
  ap.add_argument("--backends", default="cpu,gpu")
  ap.add_argument("--timeout", type=int, default=1500)
  ap.add_argument("--keep", action="store_true", help="keep the downloaded files")
  ap.add_argument("--resume", action="store_true", help="skip checks already in --out")
  ap.add_argument("--list", action="store_true", help="print the selection and stop")
  ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", ENDPOINT))
  ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
  ap.add_argument("--path", help=argparse.SUPPRESS)
  ap.add_argument("--backend", help=argparse.SUPPRESS)
  ap.add_argument("--check", help=argparse.SUPPRESS)
  ap.add_argument("--meta", help=argparse.SUPPRESS)
  ap.add_argument("--photo", help=argparse.SUPPRESS)
  a = ap.parse_args()
  if a.one:
    child(a)
    return
  audit = json.loads(pathlib.Path(a.audit).read_text())
  picked = select(audit, a.top, a.extra)
  selection = (f"Files: the top {a.top} LLM repos by Hub downloads in audit.json (gated repos and files whose "
               "header did not read left out), one file per repo (not NPU-constrained; a file without a backend "
               "constraint first, then the smallest)"
               + (f", plus {', '.join(a.extra)} by name" if a.extra else "")
               + f". Each check in a new process. Host: {platform.system()} {platform.machine()}.")
  if a.list:
    print(selection)
    for r, f in picked:
      print(f"{r.get('downloads')}\t{r['repo']}\t{f['file']}\t{f['size_bytes']:,} B\t{header_view(f)}")
    return
  if not a.dl_dir:
    sys.exit("--dl-dir is needed to run the checks")
  out_p = pathlib.Path(a.out)
  runs = json.loads(out_p.read_text()) if a.resume and out_p.exists() else {}
  runs.update(runtime=importlib.metadata.version("litert-lm-api"), selection=selection,
              audit_generated=audit.get("generated"), audit_commit_list_sha256=audit.get("commit_list_sha256"))
  runs.setdefault("results", [])
  photo, digest = download(PHOTO_URL, pathlib.Path(a.dl_dir) / "photo" / "000000039769.jpg")
  runs["photo"] = {"url": PHOTO_URL, "sha256": digest}
  done = {(x["repo"], x["file"], x["backend"], x["check"]) for x in runs["results"]}

  def save():
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(runs, indent=1, ensure_ascii=False) + "\n")

  def finish(r, f, hv):
    # Notes come from the stored results and the header, so a --resume pass derives them again.
    for x in runs["results"]:
      if (x["repo"], x["file"]) == (r["repo"], f["file"]):
        x.pop("note", None)
        n = result_note(x, hv["tools"])
        if n:
          x["note"] = n
    runs["notes"] = sorted({n for rr, ff in picked for n in compare(rr, ff, runs["results"])})
    save()
  for r, f in picked:
    hv = header_view(f)
    meta = {"vision": bool(f.get("vision")), "size_bytes": f["size_bytes"],
            "thinks": bool(hv["channels"]) or bool(((f.get("llm") or {}).get("template") or {}).get(
                "reads_enable_thinking")), "thinking_control": hv["thinking_control"]}
    checks = ["engine", "text"] + (["image"] if meta["vision"] else []) + ["tool"] + (
        ["think_on", "think_off"] if meta["thinks"] else [])
    stored = [x for x in runs["results"] if (x["repo"], x["file"]) == (r["repo"], f["file"])]
    wanted = {(be, c) for be in a.backends.split(",") for c in checks}
    if meta["vision"] and any(x["backend"] == "gpu" and x["check"] == "engine" and x["status"] != "pass"
                              for x in stored):
      wanted |= {("gpu_vcpu", c) for c in checks}
    if wanted <= {(x["backend"], x["check"]) for x in stored}:
      finish(r, f, hv)  # every check is stored: no download
      continue
    url = f"{a.endpoint}/{r['repo']}/resolve/{r['sha']}/{urllib.parse.quote(f['file'])}"
    dst = pathlib.Path(a.dl_dir) / r["repo"].replace("/", "__") / f["file"]
    print(f"{r['repo']} {f['file']}: download {f['size_bytes']:,} B", file=sys.stderr, flush=True)
    try:
      path, _ = download(url, dst, f["size_bytes"], f["sha256"])
    except Exception as e:  # noqa: BLE001
      runs["results"].append({"repo": r["repo"], "file": f["file"], "backend": "-", "check": "download",
                              "status": "error", "error": str(e)[:300]})
      save()
      continue
    backends = a.backends.split(",")
    i = 0
    while i < len(backends):
      be = backends[i]
      i += 1
      engine_ok = True
      for c in checks:
        key = (r["repo"], f["file"], be, c)
        if key in done:
          res = next(x for x in runs["results"] if (x["repo"], x["file"], x["backend"], x["check"]) == key)
        elif not engine_ok:
          res = {"status": "skip", "error": "no engine"}
        else:
          res = run_child(a, path, be, c, meta, photo)
          res["attempts"] = 1
          if res["status"] == "error" and any(m in (res.get("error") or "") for m in RETRY_MARKS):
            res = dict(run_child(a, path, be, c, meta, photo), attempts=2)
        if key not in done:
          res.update(repo=r["repo"], file=f["file"], sha256=f["sha256"], backend=be, check=c)
          runs["results"].append(res)
          save()
          print(f"  {be} {c}: {res['status']} {(res.get('error') or res.get('reply') or '')[:100]!r}",
                file=sys.stderr, flush=True)
        if c == "engine" and res["status"] != "pass":
          engine_ok = False
          if be == "gpu" and meta["vision"] and "gpu_vcpu" not in backends:
            backends.append("gpu_vcpu")
    finish(r, f, hv)
    if not a.keep:
      # The runtime writes its caches next to the model (XNNPACK, GPU program and weight caches): about
      # as much again as the file. They go with it.
      for p in path.parent.iterdir():
        if p.name.startswith(f["file"]) and p.is_file():
          p.unlink(missing_ok=True)
  print(out_p)


if __name__ == "__main__":
  main()
