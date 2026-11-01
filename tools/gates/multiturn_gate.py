#!/usr/bin/env python3
"""Multi-turn gate for a .litertlm chat bundle, on the LiteRT-LM python API (written against litert-lm 0.19.0).

One conversation of three turns, greedy: plant a fact, ask a short sum, recall the fact. A turn fails when its
text misses the expected answer, is empty, or carries a template marker (a thought tag, a role marker). One-turn
checks cannot see what this catches: the runtime's prefix contract (a hard error "does not start with the previous
rendered template string", or a silent rewind), state damage that shows only in a later turn, a thinking model
that answers the first question and then ends the third turn empty. Each turn's budget (--max-output-tokens) has
to let a thinking model close its thought: a cap hit inside the thought reads like state damage.

History modes (--modes, comma-separated; default filter,keep):
  filter   filter_channel_content_from_kv_cache=True: the reply's channel content (the thought) leaves the KV cache
           before the next turn.
  keep     filter_channel_content_from_kv_cache=False: the thought stays in the KV cache.
  default  the argument not passed. The runtime then filters: ConversationConfig::Builder starts the setting at true
           (runtime/conversation/conversation.h, LiteRT-LM 0.15.0 to 0.19.0; 0.10.2 to 0.14.0 start it at false; earlier releases have no such setting),
           so `default` repeats `filter`.
The report keeps the conversation's token count after each turn, which tells the settings apart.

--thinking: `template` passes no ThinkingConfig, so `enable_thinking` stays out of the template's context and the
chat template's own default decides (runtime/conversation/conversation.cc, 0.19.0); `on` / `off` pass
ThinkingConfig(enable_thinking=...).

Each run is a fresh process, --processes per mode: the CPU path can answer differently from one process to the next,
so one CPU pass is not a CPU verdict. A run whose blocking send_message reached the runtime's 10-minute deadline
(DEADLINE_EXCEEDED; the clock also runs while the process is paused) is void, not an answer of the model: it runs
again in a new process, up to --void-retries times. The process writes the void result as soon as the call fails,
since LiteRT-LM 0.19.0 can end the process right after a deadline, and a process that ends without a result after
DEADLINE_EXCEEDED reached its stderr counts as void too. A run whose process ends without a result otherwise is a
failure.

  python tools/gates/multiturn_gate.py MODEL.litertlm [--backend cpu|gpu] [--modes filter,keep] [--processes 3]
      [--thinking template|on|off] [--max-output-tokens 3072] [--max-num-tokens 4096] [--cache-dir DIR]
      [--turns TURNS.json] [--leak STR]... [--json-out REPORT.json]

--turns replaces the three turns with a JSON list of [message, regex or null]. The verdict is per backend: run the
gate once for each backend the bundle is meant for. Exit 0 = PASS (every run of every mode passed), 1 = FAIL,
2 = usage error, 3 = UNDECIDED (no run failed, and a run was still void after its retries).
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import time

TURNS = [
    ["My cat is named Alfred and he is orange. Please remember that. Just say OK.", None],
    ["What is 23 + 19? Answer briefly.", r"\b42\b"],
    ["What is the name and color of my cat? Answer briefly.", r"alfred.*orange|orange.*alfred"],
]
# Text that must never reach a reply: thought tags and the role markers of common chat templates.
MARKERS = ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<start_of_turn>", "<end_of_turn>", "<|eot_id|>",
           "<|start_header_id|>", "<|User|>", "<|Bot|>", "<|System|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]
MODES = {"filter": True, "keep": False, "default": None}
VOID_MARK = "DEADLINE_EXCEEDED"
TEST_VOID_ENV = "MULTITURN_GATE_TEST_VOID"  # the gate's own test: attempts 1..N of a run end as if the deadline hit
TEST_ABORT_ENV = "MULTITURN_GATE_TEST_ABORT"  # with TEST_VOID: the process then aborts, as 0.19.0 can after a deadline


def runtime_version():
  for dist in ("litert-lm", "litert-lm-api"):
    try:
      return f"{dist} {importlib.metadata.version(dist)}"
    except importlib.metadata.PackageNotFoundError:
      pass
  return "unknown"


def reply_text(resp):
  parts = resp.get("content") if isinstance(resp, dict) else None
  if isinstance(parts, str):
    return parts.strip()
  if isinstance(parts, list):
    return "".join(c.get("text", "") for c in parts if isinstance(c, dict) and c.get("type") == "text").strip()
  return str(resp).strip()


def thought_chars(resp):
  ch = resp.get("channels") if isinstance(resp, dict) else None
  t = ch.get("thought", "") if isinstance(ch, dict) else ""
  return len(t) if isinstance(t, str) else len(json.dumps(t))


def write_json(path, obj):
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(obj, f, ensure_ascii=False)
  os.replace(tmp, path)


def load_turns(path):
  if not path:
    return TURNS
  turns = json.load(open(path))
  if not (isinstance(turns, list) and turns and all(isinstance(t, list) and len(t) == 2 and isinstance(t[0], str)
                                                    and (t[1] is None or isinstance(t[1], str)) for t in turns)):
    raise ValueError(f"{path}: expected a JSON list of [message, regex or null]")
  for _, pat in turns:
    if pat is not None:
      re.compile(pat)
  return turns


def one_run(a):
  """Child process: one conversation in one mode. Writes its result after every turn, so a process that dies
  later still leaves what it did."""
  res = {"mode": a.mode, "attempt": a.attempt, "status": "ok", "pass": False, "complete": False, "rows": [],
         "runtime": runtime_version()}
  write_json(a.result, res)
  from litert_lm import engine as engine_lib
  from litert_lm import interfaces
  turns = load_turns(a.turns)
  markers = MARKERS + list(a.leak or [])
  backend = {"cpu": interfaces.Backend.CPU(), "gpu": interfaces.Backend.GPU()}[a.backend]
  ekw = {"cache_dir": a.cache_dir} if a.cache_dir else {}
  eng = engine_lib.Engine(a.model, backend=backend, max_num_tokens=a.max_num_tokens, **ekw)
  ckw = {"sampler_config": interfaces.SamplerConfig(top_k=1)}
  if a.thinking != "template":
    ckw["thinking_config"] = interfaces.ThinkingConfig(enable_thinking=a.thinking == "on")
  if MODES[a.mode] is not None:
    ckw["filter_channel_content_from_kv_cache"] = MODES[a.mode]
  conv = eng.create_conversation(**ckw)
  fake_void = int(os.environ.get(TEST_VOID_ENV) or 0) >= a.attempt
  for i, (msg, pat) in enumerate(turns, 1):
    t0 = time.time()
    try:
      if fake_void:
        raise RuntimeError(f"litert_lm_conversation_send_message failed with status {VOID_MARK} (4): "
                           f"set by {TEST_VOID_ENV} for attempt {a.attempt}")
      resp = conv.send_message(msg, max_output_tokens=a.max_output_tokens)
    except Exception as e:  # noqa: BLE001 - a runtime error is an outcome of the run
      err = f"{type(e).__name__}: {e}"
      res["status"] = "void" if VOID_MARK in err else "error"
      res["rows"].append({"turn": i, "error": err[:800], "seconds": round(time.time() - t0, 1)})
      write_json(a.result, res)  # first: after a deadline the generation left running can end the process
      if fake_void and os.environ.get(TEST_ABORT_ENV):
        os.abort()
      break
    text = reply_text(resp)
    leaks = [m for m in markers if m in text]
    expected = pat is None or re.search(pat, text, re.I | re.S) is not None
    res["rows"].append({"turn": i, "ok": bool(text) and not leaks and expected, "expected": expected,
                        "empty": not text, "leaks": leaks, "token_count": conv.token_count,
                        "thought_chars": thought_chars(resp), "seconds": round(time.time() - t0, 1),
                        "text": text[:400]})
    write_json(a.result, res)
  res["complete"] = True
  res["pass"] = res["status"] == "ok" and len(res["rows"]) == len(turns) and all(r["ok"] for r in res["rows"])
  write_json(a.result, res)


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def child_cmd(a, mode, attempt, out):
  cmd = [sys.executable, os.path.abspath(__file__), a.model, "--one", "--mode", mode, "--attempt", str(attempt),
         "--result", out, "--backend", a.backend, "--max-output-tokens", str(a.max_output_tokens),
         "--max-num-tokens", str(a.max_num_tokens), "--thinking", a.thinking]
  for opt, val in (("--cache-dir", a.cache_dir), ("--turns", a.turns)):
    if val:
      cmd += [opt, val]
  for s in a.leak or []:
    cmd += ["--leak", s]
  return cmd


def run_once(a, mode, process, attempt, td):
  out = os.path.join(td, f"{mode}_{process}_{attempt}.json")
  t0 = time.time()
  try:
    pr = subprocess.run(child_cmd(a, mode, attempt, out), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        timeout=a.run_timeout)
    rc, err = pr.returncode, pr.stderr or ""
    tail = err[-1500:]
  except subprocess.TimeoutExpired:
    rc, err = None, ""
    tail = f"killed after --run-timeout {a.run_timeout} s"
  res = json.load(open(out)) if os.path.exists(out) else {"mode": mode, "attempt": attempt, "status": "ok",
                                                         "pass": False, "complete": False, "rows": []}
  if not res.get("complete") and res["status"] == "ok":
    # The process ended inside a turn or before its first one, without a recorded Python exception. The runtime logs
    # a deadline to stderr before the call returns; a crash trace after it can be long, so search all of stderr.
    res["status"] = "void" if VOID_MARK in err else "crash"
    res["pass"] = False
  res.update(process=process, rc=rc, wall_seconds=round(time.time() - t0, 1))
  if res["status"] != "ok" or rc != 0:
    res["stderr_tail"] = tail
  return res


def drive(a, modes):
  turns = load_turns(a.turns)
  report = {"model": os.path.abspath(a.model), "model_bytes": os.path.getsize(a.model), "model_sha256": sha256(a.model),
            "backend": a.backend, "modes": modes, "processes": a.processes, "thinking": a.thinking,
            "max_output_tokens": a.max_output_tokens, "max_num_tokens": a.max_num_tokens,
            "turns": turns, "runs": []}
  undecided = False
  with tempfile.TemporaryDirectory(prefix="multiturn_gate_") as td:
    for mode in modes:
      for p in range(1, a.processes + 1):
        for attempt in range(1, a.void_retries + 2):
          res = run_once(a, mode, p, attempt, td)
          report["runs"].append(res)
          marks = " ".join("ok" if r.get("ok") else ("ERR" if "error" in r else "NG") for r in res["rows"]) or "-"
          counts = "/".join(str(r["token_count"]) for r in res["rows"] if "token_count" in r) or "-"
          state = ("PASS" if res["pass"] else "FAIL") if res["status"] == "ok" else res["status"].upper()
          print(f"{mode} process {p} attempt {attempt}: {state} turns [{marks}] token_count {counts} "
                f"{res['wall_seconds']} s", flush=True)
          if res["status"] != "void":
            break
        else:
          undecided = True
  counted = [r for r in report["runs"] if r["status"] != "void"]
  report["summary"] = {m: {"pass": sum(r["pass"] for r in counted if r["mode"] == m),
                           "fail": sum(not r["pass"] for r in counted if r["mode"] == m),
                           "void": sum(r["status"] == "void" for r in report["runs"] if r["mode"] == m)}
                       for m in modes}
  report["verdict"] = "FAIL" if any(not r["pass"] for r in counted) else ("UNDECIDED" if undecided else "PASS")
  report["runtime"] = sorted({r["runtime"] for r in report["runs"] if r.get("runtime")})
  print(f"multi-turn verdict={report['verdict']} backend={a.backend} " +
        " ".join(f"{m}={s['pass']}/{s['pass'] + s['fail']}" + (f"(+{s['void']} void)" if s["void"] else "")
                 for m, s in report["summary"].items()), flush=True)
  if a.json_out:
    with open(a.json_out, "w") as f:
      json.dump(report, f, indent=1, ensure_ascii=False)
  return {"PASS": 0, "FAIL": 1, "UNDECIDED": 3}[report["verdict"]]


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("model")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--modes", default="filter,keep", help="comma-separated: filter, keep, default (default filter,keep)")
  ap.add_argument("--processes", type=int, default=3, help="fresh processes per mode (default 3)")
  ap.add_argument("--thinking", choices=["template", "on", "off"], default="template",
                  help="no ThinkingConfig (the chat template decides), or enable_thinking on / off")
  ap.add_argument("--max-output-tokens", type=int, default=3072, help="per turn (default 3072)")
  ap.add_argument("--max-num-tokens", type=int, default=4096, help="the engine's context (default 4096)")
  ap.add_argument("--cache-dir", help="the engine's cache dir, created if missing (one process at a time per dir)")
  ap.add_argument("--turns", help="JSON list of [message, regex or null] replacing the built-in three turns")
  ap.add_argument("--leak", action="append", help="one more marker a reply must not contain (repeatable)")
  ap.add_argument("--void-retries", type=int, default=2, help="new processes for a void run (default 2)")
  ap.add_argument("--run-timeout", type=int, default=3600, help="seconds before a run's process is killed")
  ap.add_argument("--json-out")
  ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)  # child process
  ap.add_argument("--mode", choices=list(MODES), help=argparse.SUPPRESS)
  ap.add_argument("--attempt", type=int, default=1, help=argparse.SUPPRESS)
  ap.add_argument("--result", help=argparse.SUPPRESS)
  a = ap.parse_args()
  if a.one:
    one_run(a)
    return 0
  modes = [m.strip() for m in a.modes.split(",") if m.strip()]
  if not modes or any(m not in MODES for m in modes):
    ap.error(f"--modes {a.modes!r}: choose from {', '.join(MODES)}")
  if a.processes < 1 or a.void_retries < 0:
    ap.error("--processes must be >= 1 and --void-retries >= 0")
  if not os.path.isfile(a.model):
    ap.error(f"{a.model}: no such file")
  try:
    load_turns(a.turns)
  except (OSError, ValueError, re.error) as e:
    ap.error(f"--turns: {e}")
  if a.cache_dir:  # the engine refuses a cache dir that does not exist (INVALID_ARGUMENT)
    a.cache_dir = os.path.abspath(a.cache_dir)
    os.makedirs(a.cache_dir, exist_ok=True)
  return drive(a, modes)


if __name__ == "__main__":
  sys.exit(main())
