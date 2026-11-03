#!/usr/bin/env python3
"""Read a built Unlimited-OCR-2 .litertlm back through litert-lm's own unpack and check it against the bundle
contract: the tokenizer section unpacks to the repo's tokenizer.json bytes, stop token 1 (EOS) only, start token
0 (BOS, the runtime's HF tokenizer adds none), the decoder section declares prefer_activation_type fp32, the executor
metadata lists the expected state buffers (--kv N K + N V of --kv-type global|local, --linear M linear-attention), the embedded
template is the expected file byte for byte, and (with --image G) the fast_vlm image size is G and the vision
encoder's input is [1, G, G, 3]; optional: --kv-shapes (the K / V decode shapes), --mask-ctx (the mask input's last
dim in every signature), --same-as (every section but --may-differ byte-equal to another bundle).

    python readback.py BUNDLE --unpack-dir DIR --tok-sha SHA --template FILE --kv 12 [--kv-type local] \
        [--linear 0] [--image 1024] [--litert-lm CLI] [--tflite-python PY] --out OUT.json
    (litert-lm-builder for the proto; CLI = a litert-lm with unpack; PY = a Python with ai-edge-litert)

The unpack dir is removed at the end unless --keep. Exit 1 when a check fails.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zlib

from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2

ENTRY_RE = re.compile(r'\{\s*key\s*=\s*"prefer_activation_type"\s*,\s*value_type\s*=\s*"[^"]*"\s*,\s*value\s*=\s*"([^"]*)"')


def sha256_file(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def tflite_io(py, path):
  code = (
      "import json,sys\nfrom ai_edge_litert.interpreter import Interpreter\n"
      "it=Interpreter(model_path=sys.argv[1]); sigs=it.get_signature_list()\n"
      "out={}\n"
      "for k in sigs:\n"
      "  r=it.get_signature_runner(k)\n"
      "  out[k]={'inputs':{n:[list(map(int,d['shape'])),str(d['dtype'].__name__)] for n,d in r.get_input_details().items()},"
      "'outputs':{n:[list(map(int,d['shape'])),str(d['dtype'].__name__)] for n,d in r.get_output_details().items()}}\n"
      "print(json.dumps(out))\n")
  r = subprocess.run([py, "-I", "-c", code, path], capture_output=True, text=True, stdin=subprocess.DEVNULL)
  if r.returncode:
    return {"error": r.stderr[-400:]}
  return json.loads(r.stdout.strip().splitlines()[-1])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("bundle")
  ap.add_argument("--unpack-dir", required=True)
  ap.add_argument("--tok-sha", required=True)
  ap.add_argument("--template", required=True)
  ap.add_argument("--image", type=int, default=None)
  ap.add_argument("--image-hw", default=None, help="H,W for a non-square fast_vlm picture (overrides --image)")
  ap.add_argument("--litert-lm", default=shutil.which("litert-lm") or "litert-lm")
  ap.add_argument("--tflite-python", default=sys.executable)
  ap.add_argument("--venc-act", default=None,
                  help="also check the vision encoder section's prefer_activation_type (a value, or 'none' = undeclared)")
  ap.add_argument("--kv", type=int, required=True, help="number of layers with a K and V state buffer")
  ap.add_argument("--kv-type", default="global", choices=("global", "local"), help="the K / V state buffer type")
  ap.add_argument("--kv-shapes", default=None, help="K;V decode shapes, e.g. 1,11,416,128;1,10,128,416")
  ap.add_argument("--mask-ctx", type=int, default=None, help="the mask input's last dim in every signature")
  ap.add_argument("--same-as", default=None, help="a reference bundle: every section but --may-differ is byte-equal")
  ap.add_argument("--may-differ", default="TFLiteModel:prefill_decode,ExecutorMetadata:")
  ap.add_argument("--dec-act", default="fp32", help="the activation type the prefill_decode section must declare")
  ap.add_argument("--linear", type=int, default=0, help="number of TYPE_LINEAR_ATTENTION state buffers")
  ap.add_argument("--keep", action="store_true")
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  hw = tuple(int(x) for x in a.image_hw.split(",")) if a.image_hw else ((a.image, a.image) if a.image else None)
  checks, res = [], {"bundle": os.path.abspath(a.bundle), "bytes": os.path.getsize(a.bundle),
                     "sha256": sha256_file(a.bundle)}

  def check(name, ok, got):
    checks.append({"check": name, "ok": bool(ok), "got": got})

  ver = subprocess.run([a.litert_lm, "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
  res["litert_lm"] = ver
  if os.path.exists(a.unpack_dir):
    shutil.rmtree(a.unpack_dir)
  u = subprocess.run([a.litert_lm, "unpack", a.bundle, "--output-dir", a.unpack_dir], capture_output=True, text=True,
                     stdin=subprocess.DEVNULL)
  res["unpack_rc"] = u.returncode
  toml_path = os.path.join(a.unpack_dir, "model.toml")
  if not os.path.exists(toml_path):
    check("unpack", False, u.stdout[-300:] + u.stderr[-300:])
    json.dump({**res, "checks": checks}, open(a.out, "w"), indent=1)
    print("READBACK FAIL unpack")
    return 1
  raw_toml = open(toml_path).read()
  toml = tomllib.loads(raw_toml)
  sections = []
  for i, s in enumerate(toml.get("section", [])):
    p = os.path.join(a.unpack_dir, s["data_path"])
    rec = {"i": i, "section_type": s["section_type"], "model_type": s.get("model_type"), "data_path": s["data_path"],
           "bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    pat = s.get("prefer_activation_type")
    for e in s.get("additional_metadata", []) or []:
      if isinstance(e, dict) and e.get("key") == "prefer_activation_type":
        pat = e.get("value")
    rec["prefer_activation_type"] = pat
    sections.append(rec)
  res["sections"] = sections
  res["system_metadata"] = toml.get("system_metadata", {})

  # tokenizer
  tok = [s for s in sections if s["section_type"] == "HF_Tokenizer"]
  check("one HF tokenizer section", len(tok) == 1, len(tok))
  if tok:
    blob = open(os.path.join(a.unpack_dir, tok[0]["data_path"]), "rb").read()
    n = int.from_bytes(blob[:8], "little")
    data = zlib.decompress(blob[8:])
    got = hashlib.sha256(data).hexdigest()
    res["tokenizer"] = {"declared_size": n, "size": len(data), "sha256": got}
    check("tokenizer section = repo tokenizer.json bytes", got == a.tok_sha and n == len(data), got)

  # LLM metadata
  llm = [s for s in sections if s["section_type"] == "LlmMetadata"]
  check("one LlmMetadata section", len(llm) == 1, len(llm))
  if llm:
    md = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(open(os.path.join(a.unpack_dir, llm[0]["data_path"])).read(), md)
    stops = sorted(i for st in md.stop_tokens for i in st.token_ids.ids)
    strs = [st.token_str for st in md.stop_tokens if st.HasField("token_str")]
    res["llm_metadata"] = {"stop_ids": stops, "stop_strs": strs, "has_start_token": md.HasField("start_token"),
                           "max_num_tokens": md.max_num_tokens,
                           "model_type": md.llm_model_type.WhichOneof("model_type"),
                           "fast_vlm": ({"h": md.llm_model_type.fast_vlm.image_tensor_height,
                                         "w": md.llm_model_type.fast_vlm.image_tensor_width}
                                        if md.llm_model_type.HasField("fast_vlm") else None),
                           "jinja_sha256": hashlib.sha256(md.jinja_prompt_template.encode()).hexdigest(),
                           "jinja_bytes": len(md.jinja_prompt_template.encode()),
                           "channels": len(md.channels), "sampler": text_format.MessageToString(md.sampler_params),
                           "user_prefix": md.prompt_templates.user.prefix,
                           "model_prefix": md.prompt_templates.model.prefix}
    res["llm_metadata"]["start_ids"] = list(md.start_token.token_ids.ids)
    check("stop ids = {1}", stops == [1] and not strs, [stops, strs])
    check("start token = [0]", md.HasField("start_token") and list(md.start_token.token_ids.ids) == [0],
          res["llm_metadata"]["start_ids"])
    check("structured prompt_templates present", md.prompt_templates.HasField("user")
          and md.prompt_templates.HasField("model"), text_format.MessageToString(md.prompt_templates))
    want = open(a.template, encoding="utf-8").read()
    check("embedded template = expected file", md.jinja_prompt_template == want,
          res["llm_metadata"]["jinja_sha256"])
    check("no channels declared", len(md.channels) == 0, len(md.channels))
    if hw:
      check(f"fast_vlm image H {hw[0]} x W {hw[1]}", res["llm_metadata"]["fast_vlm"] == {"h": hw[0], "w": hw[1]},
            res["llm_metadata"]["fast_vlm"])

  # executor metadata
  ex = [s for s in sections if s["section_type"] == "ExecutorMetadata"]
  check("one ExecutorMetadata section", len(ex) == 1, len(ex))
  if ex:
    txt = open(os.path.join(a.unpack_dir, ex[0]["data_path"])).read()
    kinds = re.findall(r"type:\s*(TYPE_[A-Z_]+)", txt)
    names = re.findall(r'decode_input_name:\s*"([^"]+)"', txt)
    cnt = {k: kinds.count(k) for k in sorted(set(kinds))}
    res["executor_metadata"] = {"state_buffers": len(names), "types": cnt}
    want_n = 2 * a.kv + a.linear
    kt = a.kv_type.upper()
    check(f"{want_n} state buffers ({a.kv} {a.kv_type} K + {a.kv} {a.kv_type} V + {a.linear} linear-attn)",
          len(names) == want_n and cnt.get(f"TYPE_{kt}_KEY_CACHE") == a.kv
          and cnt.get(f"TYPE_{kt}_VALUE_CACHE") == a.kv and cnt.get("TYPE_LINEAR_ATTENTION", 0) == a.linear
          and len(cnt) == (2 if not a.linear else 3), cnt)
    res["executor_metadata"]["text"] = txt

  # TFLite sections
  tfl = {s["model_type"]: s for s in sections if s["section_type"] == "TFLiteModel"}
  res["tflite_model_types"] = sorted(tfl)
  pd = tfl.get("prefill_decode") or tfl.get("tf_lite_prefill_decode")
  check(f"prefill_decode declares {a.dec_act} activations", pd is not None and pd["prefer_activation_type"] == a.dec_act,
        pd and pd["prefer_activation_type"])
  others = {k: v["prefer_activation_type"] for k, v in tfl.items() if v is not pd}
  res["other_tflite_activation_types"] = others
  if pd:
    io = tflite_io(a.tflite_python, os.path.join(a.unpack_dir, pd["data_path"]))
    res["prefill_decode_signatures"] = sorted(io) if "error" not in io else io
    if "error" not in io:
      res["prefill_decode_io"] = io
      if a.kv_shapes:
        ks, vs = ([int(x) for x in t.split(",")] for t in a.kv_shapes.split(";"))
        got = {sig: sorted({(n.split("_")[2], tuple(v[0])) for n, v in d["inputs"].items() if n.startswith("kv_cache_")})
               for sig, d in io.items()}
        check(f"K {ks} / V {vs} in every signature",
              all(g == [("k", tuple(ks)), ("v", tuple(vs))] for g in got.values()), {k: list(v) for k, v in got.items()})
      if a.mask_ctx:
        mk = {sig: d["inputs"].get("mask", [None])[0] for sig, d in io.items()}
        check(f"mask [1, 1, T, {a.mask_ctx}] in every signature",
              all(m is not None and m[:2] == [1, 1] and m[3] == a.mask_ctx for m in mk.values()), mk)
  if hw:
    venc = tfl.get("vision_encoder") or tfl.get("tf_lite_vision_encoder")
    check("vision encoder section present", venc is not None, sorted(tfl))
    if venc:
      io = tflite_io(a.tflite_python, os.path.join(a.unpack_dir, venc["data_path"]))
      res["vision_encoder_io"] = io
      shapes = [v[0] for sig in io.values() if isinstance(sig, dict) for v in sig["inputs"].values()] \
          if "error" not in io else []
      check(f"vision encoder input [1,{hw[0]},{hw[1]},3]", [1, hw[0], hw[1], 3] in shapes, shapes)
      if a.venc_act:
        want = None if a.venc_act == "none" else a.venc_act
        check(f"vision encoder declares {a.venc_act} activations", venc["prefer_activation_type"] == want,
              venc["prefer_activation_type"])

  if a.same_as:
    ref_dir = a.unpack_dir.rstrip("/") + "_ref"
    if os.path.exists(ref_dir):
      shutil.rmtree(ref_dir)
    subprocess.run([a.litert_lm, "unpack", a.same_as, "--output-dir", ref_dir], capture_output=True, text=True,
                   stdin=subprocess.DEVNULL)
    rt = tomllib.loads(open(os.path.join(ref_dir, "model.toml")).read())
    ref = {}
    for s_ in rt.get("section", []):
      ref[(s_["section_type"], s_.get("model_type") or "")] = sha256_file(os.path.join(ref_dir, s_["data_path"]))
    mine = {(s_["section_type"], s_.get("model_type") or ""): s_["sha256"] for s_ in sections}
    allowed = {tuple(x.split(":", 1)) for x in a.may_differ.split(",") if x}
    diff = sorted(f"{k[0]}:{k[1]}" for k in set(ref) | set(mine) if ref.get(k) != mine.get(k))
    res["same_as"] = {"ref": os.path.abspath(a.same_as), "ref_sha256": sha256_file(a.same_as), "differ": diff,
                      "equal": sorted(f"{k[0]}:{k[1]}" for k in set(ref) & set(mine) if ref[k] == mine[k])}
    check(f"sections equal to {os.path.basename(a.same_as)} except {sorted(a.may_differ.split(','))}",
          set(tuple(x.split(":", 1)) for x in diff) <= allowed and set(ref) == set(mine), diff)
    if not a.keep:
      shutil.rmtree(ref_dir, ignore_errors=True)
  d = subprocess.run([a.litert_lm, "describe", a.bundle], capture_output=True, text=True, stdin=subprocess.DEVNULL)
  res["describe"] = {"rc": d.returncode, "stdout": d.stdout, "stderr_tail": d.stderr[-500:]}
  res["checks"] = checks
  ok = all(c["ok"] for c in checks)
  res["verdict"] = "PASS" if ok else "FAIL"
  json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
  for c in checks:
    print(("ok  " if c["ok"] else "FAIL"), c["check"], "" if c["ok"] else c["got"])
  print("READBACK", res["verdict"], a.bundle, res["bytes"], res["sha256"])
  if not a.keep:
    shutil.rmtree(a.unpack_dir, ignore_errors=True)
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
