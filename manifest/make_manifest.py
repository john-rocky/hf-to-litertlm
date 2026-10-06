#!/usr/bin/env python3
"""Generate litertlm_manifest.json for a Hugging Face .litertlm model repo.

Derived fields (sha256, size, sections, context_length, capabilities, thinking
channel and control) are read from the HF API and from each bundle's header —
two HTTP range requests per file, no weight download. Curated fields (quantization, backends,
recommendations, measured rows, known issues) come from a hand-verified JSON
passed via --curated (see manifest/examples/ for finished manifests). The output is validated against manifest/litertlm_manifest.schema.json.

Usage (needs: pip install litert-lm-builder jsonschema jinja2):
    python manifest/make_manifest.py litert-community/LFM2.5-1.2B-Instruct
    ... --public          # strip every private `evidence` pointer for publication
    ... --local-file F    # parse a local bundle instead of range-reading HF

"""
import argparse
import hashlib
import os
import datetime
import json
import pathlib
import re
import sys
import urllib.request

import jinja2
from jinja2 import meta as jinja2_meta
from jinja2.sandbox import ImmutableSandboxedEnvironment
from litert_lm_builder import litertlm_core  # noqa: E402
from litert_lm_builder import litertlm_header_schema_py_generated as schema  # noqa: E402
from litert_lm_builder.runtime.proto import llm_metadata_pb2  # noqa: E402

HDR_BEGIN = litertlm_core.HEADER_BEGIN_BYTE_OFFSET
HDR_END_LOC = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET

# capabilities.thinking.control (0.1.3): the value written for a bundle that
# declares a thought channel its template neither switches, opens nor closes
# (derived as UNSWITCHED). None leaves `control` out of those manifests.
CONTROL_UNSWITCHED = "model"
UNSWITCHED = "unswitched"
PROBE_TEXT = "manifest probe"
# Model types for which the runtime builds the template of a bundle without a
# jinja template from its affixes (LiteRT-LM GetDefaultJinjaPromptTemplate).
# For any other type it renders a template of its own, or refuses the bundle.
AFFIX_TEMPLATE_TYPES = ("generic_model", "qwen3", "qwen2p5", "gemma4",
                        "fast_vlm", "minicpmv4")
# The template it builds: $0-$2 the user / model / system prefix, $3-$5 their
# suffixes, $6 the model prefix again. The affixes go in as template source.
AFFIX_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message.role == 'user' %}$0"
    "{% elif message.role == 'assistant' %}$1"
    "{% elif message.role == 'system' %}$2"
    "{% endif -%}"
    "{%- for item in message.content %}"
    "{%- if item.type == 'text' %}{{ item.text }}"
    "{% elif item.type == 'image' -%}{{ '<start_of_image>' }}"
    "{%- elif item.type == 'audio' -%}{{ '<start_of_audio>' }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if message.role == 'user' %}$3"
    "{% elif message.role == 'assistant' %}$4"
    "{% elif message.role == 'system' %}$5"
    "{% endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt %}$6{% endif -%}")

def http_json(url):
  with urllib.request.urlopen(url, timeout=120) as r:
    return json.load(r)

def http_range(url, start, end):
  req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
  with urllib.request.urlopen(req, timeout=120) as r:
    return r.read()

def parse_bundle_header(fetch):
  """fetch(start, end) -> bytes. Returns (sections, llm_metadata|None)."""
  head = fetch(0, 4095)
  if head[:8] != litertlm_core.HEADER_MAGIC_BYTES:
    raise ValueError("not a litertlm file (bad magic)")
  hdr_end = int.from_bytes(head[HDR_END_LOC:HDR_END_LOC + 8], "little")
  if hdr_end > len(head):
    head = fetch(0, hdr_end + 64)
  meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[HDR_BEGIN:hdr_end]), 0)

  sections, llm_meta = [], None
  for i in range(meta.SectionMetadata().ObjectsLength()):
    so = meta.SectionMetadata().Objects(i)
    dtype = litertlm_core.any_section_data_type_to_string(so.DataType())
    entry = {"type": dtype, "size_bytes": so.EndOffset() - so.BeginOffset()}
    for j in range(so.ItemsLength()):
      it = so.Items(j)
      key = it.Key().decode() if it.Key() else None
      if key in ("model_type", "backend_constraint"):
        tab = it.Value()
        if tab is not None and it.ValueType() == schema.VData.StringValue:
          sv = schema.StringValue()
          sv.Init(tab.Bytes, tab.Pos)
          raw_s = sv.Value()
          if raw_s:
            entry[key] = raw_s.decode()
    sections.append(entry)
    if dtype == "LlmMetadataProto":
      raw = fetch(so.BeginOffset(), so.EndOffset() - 1)
      llm_meta = llm_metadata_pb2.LlmMetadata()
      llm_meta.ParseFromString(raw)
  return sections, llm_meta

def strip_evidence(obj):
  """Drop every `evidence` key under obj (measured rows, quality rows, ...).
  Evidence points at private logs; --public output must carry none."""
  if isinstance(obj, dict):
    obj.pop("evidence", None)
    for val in obj.values():
      strip_evidence(val)
  elif isinstance(obj, list):
    for item in obj:
      strip_evidence(item)

def derive_capabilities(m):
  caps = {"vision": False, "audio": False}
  which = m.llm_model_type.WhichOneof("model_type") if m.HasField("llm_model_type") else None
  if which:
    sub = getattr(m.llm_model_type, which)
    if which == "fast_vlm":
      # FastVlm carries only image tensor dims — the type itself means vision.
      caps["vision"] = True
    elif which == "generic_model":
      caps["vision"] = bool(getattr(sub, "image_enabled", False))
      caps["audio"] = bool(getattr(sub, "audio_enabled", False))
    else:
      for f, _ in sub.ListFields():
        if f.name == "start_of_image_token":
          caps["vision"] = True
        if f.name == "start_of_audio_token":
          caps["audio"] = True
  if m.channels:
    ch = m.channels[0]
    caps["thinking"] = {"declared": True,
                        "channel": {"start": ch.start, "end": ch.end}}
    # 0.1.1: mirror the full declared channel set, not just the first one —
    # a model declaring e.g. non-default tool-call markers flows through.
    chans = []
    # An older builder proto has no is_reasoning_channel field: the bundle's
    # value would be dropped silently, so warn and omit the key instead of
    # writing a wrong `false`.
    has_reasoning_flag = any(
        f.name == "is_reasoning_channel"
        for f in llm_metadata_pb2.Channel.DESCRIPTOR.fields)
    if not has_reasoning_flag:
      print("WARN: builder proto lacks Channel.is_reasoning_channel — "
            "channels[].is_reasoning omitted; use a newer litert-lm-builder",
            file=sys.stderr)
    for c in m.channels:
      row = {"name": c.channel_name, "start": c.start, "end": c.end}
      if has_reasoning_flag and c.is_reasoning_channel:
        row["is_reasoning"] = True
      chans.append(row)
    caps["channels"] = chans
  else:
    caps["thinking"] = {"declared": False}
  return caps

def template_env():
  """jinja2 with the options and helpers the runtime's template engine
  (minijinja) sets; it stands in for that engine, it is not that engine.
  A bundle's template is untrusted text: sandboxed, nothing mutable."""
  env = ImmutableSandboxedEnvironment(
      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
  def raise_exception(msg):
    raise jinja2.TemplateError(msg)
  env.globals["raise_exception"] = raise_exception
  # A fixed date keeps the render the same from one run to the next.
  env.globals["strftime_now"] = (
      lambda fmt: datetime.date(2026, 1, 1).strftime(fmt))
  env.filters["tojson"] = (
      lambda v, indent=None, ensure_ascii=False, **_:
      json.dumps(v, indent=indent, ensure_ascii=ensure_ascii))
  env.filters["lstrip"] = lambda v, chars=None: str(v).lstrip(chars)
  env.filters["rstrip"] = lambda v, chars=None: str(v).rstrip(chars)
  # The runtime's `none` test is true for an undefined value only.
  env.tests["none"] = lambda v: isinstance(v, jinja2.Undefined)
  return env

def control_from_prompt(prompt, channel):
  """The control a fixed prompt gives, by how it ends. The runtime's own
  reading of the same prompt (LiteRT-LM GetOpenChannelName: the channel's
  exact start marker after its last exact end marker) has to agree, or the
  reply would not be split the way the value says."""
  if channel is None:
    return "never", "no thought channel"
  start, end = channel.start.strip(), channel.end.strip()
  if not start or not end:
    return None, "the thought channel has an empty marker"
  last_start, last_end = prompt.rfind(channel.start), prompt.rfind(channel.end)
  runtime_open = last_start >= 0 and last_start > last_end
  tail = prompt.rstrip()
  if tail.endswith(end):
    control, basis = "never", "prompt closes the thought block"
  elif tail.endswith(start):
    control, basis = "always", "prompt opens the thought block"
  else:
    control, basis = UNSWITCHED, "template leaves the thought block to the model"
  if runtime_open != (control == "always"):
    return None, (f"{basis}, but the runtime reads the channel as "
                  f"{'open' if runtime_open else 'not open'} (exact markers)")
  return control, basis

def control_from_prompts(prompts, reads_switch, channel):
  """The control one render gives: the prompts for enable_thinking unset, on
  and off. A switch counts only when neither state contradicts it."""
  if not reads_switch:
    return control_from_prompt(prompts["unset"], channel)
  if prompts["on"] == prompts["off"]:
    return None, "template reads enable_thinking, prompt unchanged"
  if channel is not None:
    on, on_basis = control_from_prompt(prompts["on"], channel)
    off, off_basis = control_from_prompt(prompts["off"], channel)
    if on is None or on == "never":
      return None, f"enable_thinking on: {on_basis}"
    if off is None or off == "always":
      return None, f"enable_thinking off: {off_basis}"
  return "switch", "enable_thinking changes the prompt"

def control_from_template(source, channel):
  """Render one user turn with a generation prompt, in both content shapes a
  caller can send. A shape the template cannot render is not counted; the
  shapes it renders have to agree."""
  env = template_env()
  try:
    reads_switch = "enable_thinking" in jinja2_meta.find_undeclared_variables(
        env.parse(source))
    template = env.from_string(source)
  except Exception as e:  # noqa: BLE001 — untrusted template, any error
    return None, f"template does not parse ({type(e).__name__}: {e})"

  results, error = set(), None
  for content in (PROBE_TEXT, [{"type": "text", "text": PROBE_TEXT}]):
    try:
      prompts = {
          state: template.render(
              messages=[{"role": "user", "content": content}],
              add_generation_prompt=True, bos_token="", eos_token="", **extra)
          for state, extra in (("unset", {}),
                               ("on", {"enable_thinking": True}),
                               ("off", {"enable_thinking": False}))}
    except Exception as e:  # noqa: BLE001
      error = e
      continue
    results.add(control_from_prompts(prompts, reads_switch, channel))
  if not results:
    return None, f"template does not render ({type(error).__name__}: {error})"
  if len(results) > 1:
    return None, f"content shapes disagree ({sorted(map(str, results))})"
  return results.pop()

def derive_thinking_control(m):
  """(control, basis): how the bundle's own template controls thinking —
  switch, always, never or UNSWITCHED; None when it cannot be derived. Read
  from the template the runtime renders: the jinja template, else the one it
  builds from the affixes."""
  if len(m.channels) > 1:
    # Which of several channels a prompt leaves open is not derived here.
    return None, "more than one channel declared"
  channel = m.channels[0] if m.channels else None
  if m.HasField("jinja_prompt_template"):
    # The runtime drops transformers' generation markers before it parses.
    source = (m.jinja_prompt_template.replace("{% generation %}", "")
              .replace("{% endgeneration %}", ""))
    return control_from_template(source, channel)

  which = (m.llm_model_type.WhichOneof("model_type")
           if m.HasField("llm_model_type") else None)
  if not m.HasField("prompt_templates"):
    return None, "no template in the bundle"
  if which not in AFFIX_TEMPLATE_TYPES:
    return None, f"no jinja template, and none built from affixes for {which}"
  t = m.prompt_templates
  affixes = (t.user.prefix, t.model.prefix, t.system.prefix,
             t.user.suffix, t.model.suffix, t.system.suffix, t.model.prefix)
  control, basis = control_from_template(
      re.sub(r"\$(\d)", lambda n: affixes[int(n.group(1))], AFFIX_TEMPLATE),
      channel)
  return control, basis + " (affixes)"

def model_thinking_control(controls):
  """One value for the manifest from the per-file answers {file: (control,
  basis)}; None leaves the key out. A repo whose files answer differently is
  not described by a model-level field, so stop instead of picking one."""
  if len({control for control, _ in controls.values()}) > 1:
    sys.exit("variants disagree on thinking control: " + "; ".join(
        f"{n} = {c or 'not derived'} ({b})" for n, (c, b) in controls.items()))
  control = next(iter(controls.values()))[0]
  return CONTROL_UNSWITCHED if control == UNSWITCHED else control

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("repo")
  ap.add_argument("--curated", default=None)
  ap.add_argument("--out", default=None)
  ap.add_argument("--public", action="store_true",
                  help="strip private evidence pointers")
  ap.add_argument("--local-file", action="append", default=[],
                  metavar="NAME=PATH",
                  help="use a local copy for bundle NAME instead of HF ranges; "
                       "its sha256 and size are then computed from that file "
                       "(a re-ship whose new bundle is not on the Hub yet)")
  ap.add_argument("--only-curated", action="store_true",
                  help="describe only the files the curated file lists "
                       "(a repo that also hosts third-party bundles)")
  args = ap.parse_args()

  mdir = pathlib.Path(__file__).resolve().parent
  curated_path = pathlib.Path(args.curated) if args.curated else (
      mdir / "curated" / (args.repo.replace("/", "__") + ".json"))
  curated = json.loads(curated_path.read_text()) if curated_path.exists() else {}
  if not curated:
    print(f"WARN: no curated file at {curated_path} — derived-only manifest",
          file=sys.stderr)
  local = dict(kv.split("=", 1) for kv in args.local_file)

  tree = http_json(f"https://huggingface.co/api/models/{args.repo}/tree/main")
  files = {f["path"]: f for f in tree if f["path"].endswith(".litertlm")}
  if not files:
    sys.exit(f"no .litertlm files in {args.repo}")

  curated_variants = {v["file"]: v for v in curated.get("variants", [])}
  unknown = set(curated_variants) - set(files)
  if unknown:
    sys.exit(f"curated variants not in repo: {sorted(unknown)}")
  if args.only_curated:
    files = {n: f for n, f in files.items() if n in curated_variants}

  variants, model_meta, controls = [], None, {}
  for name, f in sorted(files.items()):
    lfs = f.get("lfs") or {}
    v = {"file": name}
    if lfs.get("oid"):
      v["sha256"] = lfs["oid"]
    v["size_bytes"] = lfs.get("size", f.get("size"))

    if name in local:
      p = local[name]
      # identity read out of the local bundle, never typed in
      h = hashlib.sha256()
      with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
          h.update(chunk)
      v["sha256"] = h.hexdigest()
      v["size_bytes"] = os.path.getsize(p)
      def fetch(s, e, _p=p):
        with open(_p, "rb") as fh:
          fh.seek(s)
          return fh.read(e - s + 1)
    else:
      url = f"https://huggingface.co/{args.repo}/resolve/main/{name}"
      def fetch(s, e, _u=url):
        return http_range(_u, s, e)
    sections, llm_meta = parse_bundle_header(fetch)
    v["sections"] = sections
    if llm_meta is not None:
      controls[name] = derive_thinking_control(llm_meta)
      control, basis = controls[name]
      print(f"WARN {name}: thinking control not derived — {basis}"
            if control is None else
            f"{name}: thinking control {control} — {basis}", file=sys.stderr)

    cv = dict(curated_variants.get(name, {}))
    cv.pop("file", None)
    # backend sanity: curated backends must not exceed a bundle constraint
    constraints = {s["backend_constraint"] for s in sections
                   if "backend_constraint" in s}
    if constraints and "backends" in cv:
      allowed = set()
      for c in constraints:
        allowed |= {b.strip() for b in c.split(",")}
      extra = set(cv["backends"]) - allowed
      if extra:
        print(f"WARN {name}: curated backends {sorted(extra)} outside bundle "
              f"backend_constraint {sorted(allowed)}", file=sys.stderr)
    v.update(cv)
    if args.public:
      strip_evidence(v)
    variants.append(v)

    if llm_meta is not None and model_meta is None:
      model_meta = llm_meta

  model = dict(curated.get("model", {}))
  model.setdefault("display_name", args.repo.split("/")[-1])
  if model_meta is not None:
    if model_meta.max_num_tokens:
      model["context_length"] = model_meta.max_num_tokens
    model["capabilities"] = derive_capabilities(model_meta)
    control = model_thinking_control(controls)  # 0.1.3
    if control is not None:
      model["capabilities"]["thinking"]["control"] = control

  manifest = {
      "manifest_schema": "0.1.3",
      "repo": args.repo,
      "generated": datetime.date.today().isoformat(),
      "generator": "make_manifest.py",
      "model": model,
      "variants": variants,
  }

  try:
    import jsonschema
    jsonschema.validate(
        manifest, json.loads((mdir / "litertlm_manifest.schema.json").read_text()))
    print("schema: OK", file=sys.stderr)
  except ImportError:
    print("WARN: jsonschema not installed — skipped validation", file=sys.stderr)

  out = pathlib.Path(args.out) if args.out else (
      mdir / "generated" / (args.repo.replace("/", "__") + "__litertlm_manifest.json"))
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
  print(out)

if __name__ == "__main__":
  main()
