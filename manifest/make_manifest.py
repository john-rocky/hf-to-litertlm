#!/usr/bin/env python3
"""Generate litertlm_manifest.json for a Hugging Face .litertlm model repo.

Derived fields (sha256, size, sections, context_length, capabilities, thinking
channel and control) are read from the HF API and from each bundle's header —
two HTTP range requests per file, no weight download. Curated fields (quantization, backends,
recommendations, measured rows, known issues) come from a hand-verified JSON
passed via --curated (see manifest/examples/ for finished manifests). The output is validated against manifest/litertlm_manifest.schema.json.

The sampler hints in session_defaults (top_k, top_p, temperature) come from the
source model (model.base_model) at its current commit, the first of: its
generation_config.json, the text of its model card, a code example in that card;
session_defaults.source names which (0.1.5). Only values written there are
copied; with none found, or with several values for one key, no sampler key is
written. Sampler keys in the curated file win over the lookup.

The same run writes the model card's `litert_lm:` block (0.1.5): the YAML key an
app reads from the Hub's model list (cardData) without fetching the manifest. It
is a projection of the manifest (SCHEMA.md, "The model card block"); paste it
into the card's YAML front matter.

Usage (needs: pip install litert-lm-builder jsonschema jinja2; pyyaml to read the
block back):
    python manifest/make_manifest.py litert-community/LFM2.5-1.2B-Instruct
    ... --public          # strip every private `evidence` pointer for publication
    ... --local-file F    # parse a local bundle instead of range-reading HF
    ... --no-hub --local-file NAME=PATH [...]   # a repo not on the Hub yet: the files
                          # are the --local-file ones; nothing is requested from the Hub
                          # (no sampler lookup either)
    ... --card-out PATH   # where the card block goes (default: next to the manifest)
    ... --no-sampler-lookup   # keep session_defaults as curated

"""
import argparse
import hashlib
import os
import datetime
import json
import pathlib
import re
import sys
import urllib.error
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

# LiteRT-LM 0.19.0's capability check (litert_lm_loaded_file_create in the C
# API, litert_lm.ModelInfo in Python) refuses a file whose
# tf_lite_vision_encoder section is over 2 GiB (schema/model_info/model_info.cc:
# kMaxVisionSectionSize, `section.size() > max_size`); the engine itself
# still loads the same file.
VISION_ENCODER_LIMIT = 2 * 1024 ** 3

def vision_section_warnings(name, sections):
  """One WARN line per tf_lite_vision_encoder section over the limit (from
  the section sizes in the header; the check reads nothing more)."""
  return [f"WARN {name}: tf_lite_vision_encoder section of {s['size_bytes']:,}"
          f" B is over 2 GiB ({VISION_ENCODER_LIMIT:,} B): LiteRT-LM 0.19.0's"
          " capability check (litert_lm_loaded_file_create in the C API,"
          " litert_lm.ModelInfo in Python) refuses this file; the engine"
          " itself still loads it"
          for s in sections
          if s.get("model_type") == "tf_lite_vision_encoder"
          and s["size_bytes"] > VISION_ENCODER_LIMIT]

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

def merge_variant_io(caps, var_io):
  """The model's vision / audio = true when one of its variants takes that input: the variant's own model type
  declares it and the variant's own file has the encoder section (derive_capabilities per variant). The first
  variant's model type alone decided it before, so a repo whose first file in the curated order is text-only
  lost the vision of its VL file (Qwen3.5-0.8B and Qwen3.5-2B, once the curated order came first on 2026-11-07).
  Thinking and channels stay the first variant's. `var_io` = [(vision, audio)] of the variants with an
  LlmMetadata; empty = `caps` unchanged."""
  if var_io:
    caps["vision"] = any(v for v, _ in var_io)
    caps["audio"] = any(a for _, a in var_io)
  return caps


def derive_capabilities(m, sections=None):
  """Capabilities from the LlmMetadata model type, gated by the bundle's sections when given:
  a text-only bundle packed with a family configuration that declares image / audio tokens
  (LiteRT-LM models/lfm2 since 2026-09-28) has no vision encoder section, so vision stays False.
  `sections` = the section lists of every variant (model_type items); None = the model type alone."""
  caps = {"vision": False, "audio": False}
  which = m.llm_model_type.WhichOneof("model_type") if m.HasField("llm_model_type") else None
  if which:
    sub = getattr(m.llm_model_type, which)
    if which == "fast_vlm":
      # FastVlm carries only image tensor dims — the type itself means vision.
      caps["vision"] = True
    elif which == "lfm2":
      # LiteRT-LM's lfm2 processor uses <|image_start|> / <|image_end|> when the type leaves its
      # image tokens unset (Lfm2DataProcessorConfig), and the LFM2.5-VL files are packed with an
      # empty lfm2 {} — the type itself means vision; the sections below decide for the file.
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
  if sections is not None:
    mt = [str(sec.get("model_type", "")).lower() for sec in sections]
    if not any("vision" in t for t in mt):
      caps["vision"] = False
    if not any("audio" in t for t in mt):
      caps["audio"] = False
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

# session_defaults' sampler hints (0.1.5): where they are looked up, in order,
# and the word session_defaults.source takes for each.
SAMPLER_KEYS = ("top_k", "top_p", "temperature")
SAMPLER_SOURCES = ("generation_config", "readme", "code_example")
# A key name in a card: `temperature`, `top_k` / `top-k` / `topk` / `Top K`.
SAMPLER_NAMES = {
    "temperature": re.compile(r"(?i)(?<![A-Za-z0-9_])temperature(?![A-Za-z0-9_(])"),
    "top_k": re.compile(r"(?i)(?<![A-Za-z0-9_])top[ _-]?k(?![A-Za-z0-9_(])"),
    "top_p": re.compile(r"(?i)(?<![A-Za-z0-9_])top[ _-]?p(?![A-Za-z0-9_(])"),
}
# What may stand between the name and its number (`temperature: 0.1`,
# `temperature=0.8`, `"temperature": 0.7`, `| temperature | 0.7 |`, `temperature of
# 0.7`), and the number. A number that goes on (`0.6-0.8`, `25°C`, `90%`) is not
# a value.
SAMPLER_VALUE = re.compile(
    r"""\s*[`'"*]*\s*(?:[:=|]|is\b|of\b|set\s+to\b|to\b|at\b)?\s*[`'"*]*\s*"""
    r"""(-?\d+(?:\.\d+)?)(?!\d)(?!\.\d)(?!\s*(?:°|%|[-–~]\s*\d))""")

def sampler_value(key, raw):
  """The value as the manifest takes it, or None when it is not a sampler value
  (top_k a whole number from 1 to 1000, top_p above 0 up to 1, temperature 0 to
  2: a card's `"temperature": 22` is a weather example, not a sampler)."""
  if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
    return None
  try:
    num = float(raw)
  except ValueError:
    return None
  if key == "top_k":
    return int(num) if num == int(num) and 1 <= num <= 1000 else None
  if key == "top_p":
    return num if 0 < num <= 1 else None
  return num if 0 <= num <= 2 else None

def sampler_from_config(cfg):
  """The sampler keys a generation_config.json writes, as written."""
  values = {}
  for key in SAMPLER_KEYS:
    if key in cfg:
      value = sampler_value(key, cfg[key])
      if value is not None:
        values[key] = value
  return values

def readme_parts(text):
  """(prose, code): the card's lines outside and inside fenced code blocks, as
  (line number, text); the YAML front matter is neither."""
  lines = text.splitlines()
  start = 0
  if lines and lines[0].strip() == "---":
    for i in range(1, len(lines)):
      if lines[i].strip() == "---":
        start = i + 1
        break
  prose, code, fence = [], [], None
  for n in range(start, len(lines)):
    line = lines[n]
    m = re.match(r"\s*(```+|~~~+)", line)
    if m and (fence is None or m.group(1)[0] == fence[0]):
      fence = None if fence else m.group(1)
      continue
    (code if fence else prose).append((n + 1, line))
  return prose, code

def sampler_mentions(lines):
  """{key: {value: [line numbers]}} for every sampler value the lines write."""
  found = {}
  for n, line in lines:
    for key, name in SAMPLER_NAMES.items():
      for m in name.finditer(line):
        v = SAMPLER_VALUE.match(line, m.end())
        if not v:
          continue
        value = sampler_value(key, v.group(1))
        if value is not None:
          found.setdefault(key, {}).setdefault(value, []).append(n)
  return found

def hub_get(url):
  """(HTTP status, body or None). 401 / 403 (a gated repo) and 404 are answers."""
  try:
    with urllib.request.urlopen(url, timeout=120) as r:
      return r.status, r.read()
  except urllib.error.HTTPError as e:
    return e.code, None

def lookup_sampler(base_model):
  """(values, source, notes) for the source model at its current commit: the
  first of its generation_config.json, its card's text, a code example in its
  card. Several values for one key in that place = nothing (the card says
  more than one thing; a person picks and puts it in the curated file)."""
  status, body = hub_get(f"https://huggingface.co/api/models/{base_model}")
  if status != 200:
    return {}, None, [f"{base_model}: HTTP {status} for the model, no sampler lookup"]
  sha = json.loads(body).get("sha")
  if not sha:
    return {}, None, [f"{base_model}: no commit in the model info, no sampler lookup"]
  at = f"{base_model}@{sha}"
  notes = []
  status, body = hub_get(
      f"https://huggingface.co/{base_model}/resolve/{sha}/generation_config.json")
  if status == 200:
    try:
      values = sampler_from_config(json.loads(body))
    except ValueError:
      values = {}
      notes.append(f"{at}: generation_config.json is not JSON")
    if values:
      return values, "generation_config", notes + [f"{at}: generation_config.json"]
    notes.append(f"{at}: generation_config.json has no sampler key")
  else:
    notes.append(f"{at}: HTTP {status} for generation_config.json")
  status, body = hub_get(f"https://huggingface.co/{base_model}/resolve/{sha}/README.md")
  if status != 200:
    return {}, None, notes + [f"{at}: HTTP {status} for README.md"]
  prose, code = readme_parts(body.decode("utf-8", errors="replace"))
  for source, lines in (("readme", prose), ("code_example", code)):
    found = sampler_mentions(lines)
    if not found:
      continue
    where = "; ".join(f"{k} " + ", ".join(f"{v} (l.{','.join(map(str, ns))})"
                                          for v, ns in vs.items())
                      for k, vs in found.items())
    if any(len(vs) > 1 for vs in found.values()):
      return {}, None, notes + [
          f"{at}: README.md ({source}) gives more than one value for a key — {where}; "
          "no sampler written (put the one to use in the curated session_defaults "
          "with its source)"]
    values = {k: next(iter(vs)) for k, vs in found.items()}
    return values, source, notes + [f"{at}: README.md ({source}) — {where}"]
  return {}, None, notes + [f"{at}: README.md names no sampler value"]

# The model card block (0.1.5): the `litert_lm:` key of the card's YAML front
# matter, which the Hub's model list returns as cardData.litert_lm. A projection
# of the manifest: the same names and nesting (model.* at the top, variants[] as
# they are) for what an app sets before a download; what the Hub's API already
# gives (sha256, size) and the curated prose (reason, notes, measured,
# known_issues, requirements) stay in the manifest.
BLOCK_SESSION_KEYS = ("top_k", "top_p", "temperature", "source", "max_num_tokens",
                      "request_timeout_s", "max_output_tokens_min")
BLOCK_VARIANT_KEYS = ("file", "backends", "default_backend", "recommended",
                      "context_length", "min_runtime_version", "checked")
BLOCK_RECOMMENDED_KEYS = ("platform", "device_class", "soc", "backend")
BLOCK_CHECKED_KEYS = ("device", "os", "backend", "runtime", "top_k", "top_p",
                      "temperature", "runs", "date")

def card_block(manifest):
  """{"litert_lm": {...}} for the manifest."""
  model = manifest["model"]
  block = {"manifest_schema": manifest["manifest_schema"]}
  sd = model.get("session_defaults") or {}
  sd = {k: sd[k] for k in BLOCK_SESSION_KEYS if k in sd}
  if sd:
    block["session_defaults"] = sd
  if "stateful" in model:
    block["stateful"] = model["stateful"]
  caps = model.get("capabilities")
  if caps:
    out = {k: caps[k] for k in ("vision", "audio") if k in caps}
    thinking = caps.get("thinking") or {}
    thinking = {k: thinking[k] for k in ("declared", "control") if k in thinking}
    if thinking:
      out["thinking"] = thinking
    if out:
      block["capabilities"] = out
  variants = []
  for v in manifest["variants"]:
    b = {}
    for k in BLOCK_VARIANT_KEYS:
      if k == "recommended" and k in v:
        b[k] = [{kk: r[kk] for kk in BLOCK_RECOMMENDED_KEYS if kk in r}
                for r in v[k]]
      elif k == "checked":
        # an empty list says no check yet; a reader that lists files only after
        # a check folds such a file
        b[k] = [{kk: r[kk] for kk in BLOCK_CHECKED_KEYS if kk in r}
                for r in v.get(k, [])]
      elif k in v:
        b[k] = v[k]
    variants.append(b)
  block["variants"] = variants
  return {"litert_lm": block}

YAML_PLAIN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
YAML_RESERVED = {"y", "n", "yes", "no", "on", "off", "true", "false", "null"}

def yaml_scalar(v):
  if isinstance(v, bool):
    return "true" if v else "false"
  if isinstance(v, int):
    return str(v)
  if isinstance(v, float):
    s = repr(v)
    if not re.fullmatch(r"-?\d+\.\d+", s):
      raise ValueError(f"card block: no plain YAML number for {v!r}")
    return s
  if isinstance(v, str):
    if YAML_PLAIN.fullmatch(v) and v.lower() not in YAML_RESERVED:
      return v
    # a JSON string is a YAML double-quoted scalar ("0.19.1", "2026-11-08", ...)
    return json.dumps(v, ensure_ascii=False)
  raise TypeError(f"card block: {type(v).__name__} value {v!r}")

def yaml_lines(obj, indent=0):
  pad = " " * indent
  out = []
  for k, v in obj.items():
    if isinstance(v, dict) and not v:
      out.append(f"{pad}{k}: {{}}")
    elif isinstance(v, dict):
      out.append(f"{pad}{k}:")
      out.extend(yaml_lines(v, indent + 2))
    elif isinstance(v, list) and not v:
      out.append(f"{pad}{k}: []")
    elif isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
      out.append(f"{pad}{k}: [" + ", ".join(yaml_scalar(x) for x in v) + "]")
    elif isinstance(v, list) and all(isinstance(x, dict) and x for x in v):
      out.append(f"{pad}{k}:")
      for item in v:
        sub = yaml_lines(item, indent + 4)
        sub[0] = f"{pad}  - " + sub[0][indent + 4:]
        out.extend(sub)
    elif isinstance(v, (list, dict)):
      raise TypeError(f"card block: {k} holds a shape the block does not use")
    else:
      out.append(f"{pad}{k}: {yaml_scalar(v)}")
  return out

def card_block_yaml(block):
  """The block as YAML text, read back with PyYAML when it is installed."""
  text = "\n".join(yaml_lines(block)) + "\n"
  try:
    import yaml
  except ImportError:
    print("WARN: pyyaml not installed — the card block was not read back",
          file=sys.stderr)
    return text
  if yaml.safe_load(text) != block:
    sys.exit("card block: the YAML does not read back as the block")
  return text

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
  ap.add_argument("--no-hub", action="store_true",
                  help="the repo is not on the Hub yet (a dry run before the "
                       "first upload): the file list is the --local-file "
                       "names with the local sizes instead of the Hub tree, "
                       "and every derived field is read from the local files")
  ap.add_argument("--card-out", default=None,
                  help="where to write the model card's litert_lm: block "
                       "(default: next to the manifest, *__card_block.yaml)")
  ap.add_argument("--no-sampler-lookup", action="store_true",
                  help="do not look up the source model's sampler; "
                       "session_defaults stays as curated")
  args = ap.parse_args()

  mdir = pathlib.Path(__file__).resolve().parent
  curated_path = pathlib.Path(args.curated) if args.curated else (
      mdir / "curated" / (args.repo.replace("/", "__") + ".json"))
  curated = json.loads(curated_path.read_text()) if curated_path.exists() else {}
  if not curated:
    print(f"WARN: no curated file at {curated_path} — derived-only manifest",
          file=sys.stderr)
  local = dict(kv.split("=", 1) for kv in args.local_file)

  if args.no_hub:
    if not local:
      sys.exit("--no-hub needs at least one --local-file NAME=PATH")
    tree = [{"path": n, "type": "file", "size": os.path.getsize(p)}
            for n, p in local.items()]
  else:
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

  # the decided order: the curated file's variants as it lists them (its first = the default the reader takes), then
  # any other file of the repo by name. Before 2026-11-07 every manifest went by file name, so a repo whose
  # recommended file sorts later (MNN-Pocket-0.3B: `_int8.litertlm` before `_int8_ekv2048.litertlm`) could not say so.
  order = [n for n in curated_variants if n in files] + sorted(n for n in files if n not in curated_variants)
  variants, model_meta, controls = [], None, {}
  var_io = []   # (vision, audio) of each variant: its own model type gated by its own sections
  for name in order:
    f = files[name]
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
    for line in vision_section_warnings(name, sections):
      print(line, file=sys.stderr)
    if llm_meta is not None:
      vc = derive_capabilities(llm_meta, sections)
      var_io.append((vc["vision"], vc["audio"]))
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
    if llm_meta is not None and llm_meta.max_num_tokens:
      # each variant's own context length, read out of its bundle (two files of one model can differ only in the KV
      # length); a curated value never replaces it
      if "context_length" in cv and cv["context_length"] != llm_meta.max_num_tokens:
        print(f"WARN {name}: curated context_length {cv['context_length']} replaced by the bundle's "
              f"{llm_meta.max_num_tokens}", file=sys.stderr)
      v["context_length"] = llm_meta.max_num_tokens
    if args.public:
      strip_evidence(v)
    variants.append(v)

    if llm_meta is not None and model_meta is None:
      model_meta = llm_meta

  model = dict(curated.get("model", {}))
  model.setdefault("display_name", args.repo.split("/")[-1])
  if model_meta is not None:
    if model_meta.max_num_tokens:
      # the first variant's in the decided order (the default); variants[].context_length has each file's own
      model["context_length"] = model_meta.max_num_tokens
    all_sections = [sec for var in variants for sec in var.get("sections", [])]
    model["capabilities"] = merge_variant_io(derive_capabilities(model_meta, all_sections), var_io)
    control = model_thinking_control(controls)  # 0.1.3
    if control is not None:
      model["capabilities"]["thinking"]["control"] = control

  # 0.1.5: the sampler hints and the word for where they came from
  sd = dict(model.get("session_defaults") or {})
  if any(k in sd for k in SAMPLER_KEYS):
    if sd.get("source") not in SAMPLER_SOURCES:
      print("WARN: curated session_defaults has sampler keys but no source "
            f"(one of {', '.join(SAMPLER_SOURCES)})", file=sys.stderr)
    print("sampler: curated — " + ", ".join(
        f"{k} {sd[k]}" for k in SAMPLER_KEYS if k in sd), file=sys.stderr)
  elif args.no_hub or args.no_sampler_lookup:
    print("sampler: not looked up (--no-hub / --no-sampler-lookup)", file=sys.stderr)
  elif not model.get("base_model"):
    print("WARN: no model.base_model in the curated file — no sampler lookup",
          file=sys.stderr)
  else:
    values, source, notes = lookup_sampler(model["base_model"])
    for line in notes:
      print(f"sampler: {line}", file=sys.stderr)
    if values:
      sd.update(values)
      sd["source"] = source
      model["session_defaults"] = sd
      print("sampler: " + ", ".join(f"{k} {values[k]}" for k in SAMPLER_KEYS
                                    if k in values) + f" from {source}",
            file=sys.stderr)
    else:
      print("sampler: none written (the app keeps its own defaults)",
            file=sys.stderr)

  manifest = {
      "manifest_schema": "0.1.5",
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

  # 0.1.5: the model card's litert_lm: block, from the manifest just written
  block_text = card_block_yaml(card_block(manifest))
  card_out = pathlib.Path(args.card_out) if args.card_out else (
      out.with_name(out.name.replace("__litertlm_manifest.json", "")
                    .removesuffix(".json") + "__card_block.yaml"))
  card_out.parent.mkdir(parents=True, exist_ok=True)
  card_out.write_text(block_text)
  print(card_out)

if __name__ == "__main__":
  main()
