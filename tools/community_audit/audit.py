#!/usr/bin/env python3
"""Audit every .litertlm file of a Hugging Face org from the bundle headers.

For each file the tool reads only the bundle header and its metadata sections
over HTTP ranges (no weight download; the reads manifest/make_manifest.py
makes) and records:
  weights      inline, or external (TFLiteWeights sections)
  executor     the ExecutorMetadata section (state buffers, attention mask,
               NPU flags), or none
  vision / audio sections, the LLM model type, the chat template (does it read
  `tools`, `enable_thinking`), the thinking channels and how the template
  switches them, the sampler, max_num_tokens, min_runtime_version and the
  backend constraint of the main model section
and, when the repo card names a base_model, how the file compares with that
model's config.json, generation_config.json and chat template.

Each file gets flags from RULES (one line each, printed at the top of the
table): `rebuild:` = the LLM file lacks a form this audit asks of every LLM
file; `warn:` = worth a look; `info:` = context. A failed read (repo missing,
gated, HTTP error, a server that ignores Range, a file too small for its
header) becomes a row with the stage and the reason; the run goes on.

Every repo is read at its commit and the table head carries the date and the
sha256 of the repo@commit list, so `--pin <audit.json>` reads the same files
again and gives the same table (only the date line can differ).

Usage (repo root; needs litert-lm-builder, jinja2; anonymous HTTP only):
  python3 tools/community_audit/audit.py --author litert-community --out-dir tools/community_audit/table
  python3 tools/community_audit/audit.py --pin tools/community_audit/table/audit.json --out-dir /tmp/again
  python3 tools/community_audit/audit.py --repos litert-community/MiniCPM-V-4 --out-dir /tmp/one
  python3 tools/community_audit/audit.py --render-only --out-dir tools/community_audit/table \
      --runs tools/community_audit/table/runs.json
  python3 tools/community_audit/audit.py --self-test --out-dir /tmp/selftest   # the failure paths
"""
import argparse
import collections
import concurrent.futures
import datetime
import functools
import hashlib
import http.server
import importlib.util
import json
import os
import pathlib
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TOOL = "tools/community_audit/audit.py"
TOOL_VERSION = 1


def _load_make_manifest():
  # Loaded by path, not through sys.path, so no other make_manifest.py wins.
  spec = importlib.util.spec_from_file_location(
      "community_audit_make_manifest", REPO_ROOT / "manifest" / "make_manifest.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


mm = _load_make_manifest()
schema = mm.schema
litertlm_core = mm.litertlm_core
llm_metadata_pb2 = mm.llm_metadata_pb2
try:
  from litert_lm_builder.runtime.proto import executor_metadata_pb2
except ImportError:  # an older builder: the section is reported, not parsed
  executor_metadata_pb2 = None
from google.protobuf import json_format  # noqa: E402

ENDPOINT = "https://huggingface.co"
UA = {"User-Agent": "hf-to-litertlm-community-audit/1"}
MIN_HEADER = mm.HDR_END_LOC + 8
EXAMPLE = "litert-community/MiniCPM-V-4"

RULES = [
    ("rebuild:inline-weights",
     "LLM file without a TFLiteWeights section: every weight sits inside the TFLite model sections"),
    ("rebuild:no-executor-metadata",
     "LLM file without an ExecutorMetadata section (state buffers, attention mask); LiteRT-LM v0.19.0 "
     "reads the section when it is there and runs without it"),
    ("rebuild:cpu-only",
     "the backend_constraint of the main model section names cpu and not gpu"),
    ("rebuild:tool-calls-as-text",
     "the template reads `tools`, the model type is generic_model and the file has no tool_call_config: "
     "LiteRT-LM v0.19.0's generic processor then has no tool-call fence (generic_data_processor.h "
     "CodeFenceStart), so a call arrives as text"),
    ("rebuild:thinking-without-channel",
     "the template reads enable_thinking, the file declares no channel: with thinking on, the thought "
     "arrives inside the answer"),
    ("warn:think-tag-without-channel",
     "the template writes <think> but does not read enable_thinking, the file declares no channel (an empty "
     "think block of a thinking-off template also counts)"),
    ("warn:header-past-end-offset",
     "the header flatbuffer reads past the header end offset written in the file; the audit read it from "
     "the bytes after that offset"),
    ("warn:sampler-top-k",
     "sampler type TOP_K: LiteRT-LM v0.19.0's sampler factory (runtime/components/sampler_factory.cc) "
     "has cases for TYPE_UNSPECIFIED and TOP_P only"),
    ("warn:sampler-p0", "sampler TOP_P with p = 0: one token kept, greedy"),
    ("warn:vision-over-2gib",
     "a tf_lite_vision_encoder section over 2 GiB: LiteRT-LM v0.19.0's ModelInfo and "
     "litert_lm_loaded_file_create refuse the file (the engine loads it)"),
    ("warn:no-jinja-template", "no jinja template (prompt affixes only): no `tools`, no `enable_thinking`"),
    ("warn:template-does-not-parse", "the jinja template does not parse in jinja2 (the runtime uses minijinja)"),
    ("warn:source-eos-not-a-stop-token",
     "an eos_token_id of the base model's generation_config / config is not among the file's stop token ids"),
    ("warn:source-template-has-tools", "the base model's chat template reads `tools`, the file's does not"),
    ("warn:source-template-thinks",
     "the base model's chat template reads enable_thinking or writes <think>, the file declares no channel"),
    ("warn:source-has-vision", "the base model's config has a vision_config, the file has no vision section"),
    ("info:npu", "the main model section is constrained to npu (an ahead-of-time NPU file)"),
]


class AuditError(Exception):
  def __init__(self, stage, reason):
    super().__init__(f"{stage}: {reason}")
    self.stage, self.reason = stage, reason


# ---------------------------------------------------------------- HTTP
def _open(url, headers=None, timeout=60):
  return urllib.request.urlopen(
      urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=timeout)


def http_get(url, stage, rng=None, gated=False, tries=8):
  """GET url (a byte range when rng=(start, end)); AuditError on a failure.
  HTTP 429 (the Hub's rate limit for anonymous reads) and 5xx are retried,
  after Retry-After when the server sends one."""
  hdr = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng is not None else {}
  for attempt in range(tries):
    retry = attempt + 1 < tries
    try:
      with _open(url, hdr) as r:
        status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
      if e.code in (429, 500, 502, 503, 504) and retry:
        after = (e.headers.get("Retry-After") or "") if e.headers else ""
        wait = float(after) if re.fullmatch(r"\d+(\.\d+)?", after) else (
            (20 if e.code == 429 else 4) * (attempt + 1))
        time.sleep(min(wait, 180))
        continue
      why = f"HTTP {e.code}"
      if e.code in (401, 403):
        # The Hub answers 401, not 404, for a repo that does not exist.
        why += (" (gated repo: an anonymous read is refused)" if gated else
                " (refused: no such repo, or a private one)")
      elif e.code == 404:
        why += " (not found)"
      elif e.code == 416:
        why += " (range not satisfiable)"
      raise AuditError(stage, why) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
      if retry:
        time.sleep(4 * (attempt + 1))
        continue
      raise AuditError(stage, f"network: {type(e).__name__}: {e}") from None
    if rng is not None:
      want = rng[1] - rng[0] + 1
      if status != 206:
        raise AuditError(stage, f"the server ignored the Range request (HTTP {status} with {len(body):,} B)")
      if len(body) > want:
        raise AuditError(stage, f"the range reply holds {len(body):,} B for {want:,} requested")
    return body
  raise AuditError(stage, "no attempt left")


def http_json(url, stage, gated=False):
  body = http_get(url, stage, gated=gated)
  try:
    return json.loads(body)
  except ValueError:
    raise AuditError(stage, f"the reply is not JSON ({len(body):,} B)") from None


def list_author(endpoint, author):
  """Every model repo of `author` with downloads, gated, sha and file names (pages followed)."""
  url = (f"{endpoint}/api/models?author={urllib.parse.quote(author)}&limit=1000"
         "&expand[]=downloads&expand[]=gated&expand[]=sha&expand[]=siblings")
  out = []
  while url:
    for attempt in range(4):
      try:
        with _open(url) as r:
          out.extend(json.load(r))
          link = r.headers.get("Link", "") or ""
        break
      except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        if attempt == 3:
          raise
        time.sleep(4 * (attempt + 1))
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    url = m.group(1) if m else None
  return out


# ---------------------------------------------------------------- header
_VDATA = {v: k for k, v in vars(schema.VData).items() if not k.startswith("_")}


def item_value(it):
  name = _VDATA.get(it.ValueType())
  tab = it.Value()
  if tab is None or name in (None, "NONE"):
    return None
  cls = getattr(schema, name, None)
  if cls is None or not hasattr(cls, "Init"):
    return f"<{name}>"
  obj = cls()
  obj.Init(tab.Bytes, tab.Pos)
  v = obj.Value()
  return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def read_bundle(fetch, size):
  """fetch(start, end) -> bytes. The header, the LlmMetadata and the
  ExecutorMetadata sections, described. AuditError on a short or broken file."""
  if size is not None and size < MIN_HEADER:
    raise AuditError("header read", f"the file is {size:,} B; a bundle header needs at least {MIN_HEADER} B")
  head = fetch(0, 4095)
  if len(head) < MIN_HEADER:
    raise AuditError("header read", f"got {len(head):,} B; a bundle header needs at least {MIN_HEADER} B")
  if head[:8] != litertlm_core.HEADER_MAGIC_BYTES:
    raise AuditError("header read", "not a LiteRT-LM bundle (bad magic bytes)")
  hdr_end = int.from_bytes(head[mm.HDR_END_LOC:mm.HDR_END_LOC + 8], "little")
  if hdr_end <= mm.HDR_BEGIN or (size is not None and hdr_end > size):
    raise AuditError("header read", f"the header end offset {hdr_end:,} lies outside the file "
                     f"({size if size is not None else '?':,} B)")
  if hdr_end > len(head):
    head = fetch(0, hdr_end + 63)
    if len(head) < hdr_end:
      raise AuditError("header read", f"got {len(head):,} B of a {hdr_end:,} B header")
  try:
    return _read_sections(fetch, size, bytearray(head[mm.HDR_BEGIN:hdr_end]))
  except AuditError:
    raise
  except Exception:  # noqa: BLE001 - the flatbuffer reads past hdr_end: retry on the next bytes
    more = fetch(0, hdr_end + 4095)
    d = _read_sections(fetch, size, bytearray(more[mm.HDR_BEGIN:]))
    d["header_past_end_offset"] = True
    return d


def _read_sections(fetch, size, buf):
  meta = schema.LiteRTLMMetaData.GetRootAs(buf, 0)
  sections, llm, execmd = [], None, None
  for i in range(meta.SectionMetadata().ObjectsLength()):
    so = meta.SectionMetadata().Objects(i)
    try:
      dtype = litertlm_core.any_section_data_type_to_string(so.DataType())
    except ValueError:
      dtype = f"unknown({so.DataType()})"
    items = {}
    for j in range(so.ItemsLength()):
      it = so.Items(j)
      items[it.Key().decode() if it.Key() else f"#{j}"] = item_value(it)
    nbytes = so.EndOffset() - so.BeginOffset()
    sec = {"type": dtype, "size_bytes": nbytes}
    if items:
      sec["items"] = items
    sections.append(sec)
    if size is not None and so.EndOffset() > size:
      raise AuditError("header read", f"section {i} ({dtype}) ends at {so.EndOffset():,}, "
                       f"past the file's end ({size:,} B)")
    if dtype in ("LlmMetadataProto", "ExecutorMetadataProto"):
      raw = fetch(so.BeginOffset(), so.EndOffset() - 1)
      if len(raw) != nbytes:
        raise AuditError("metadata read", f"got {len(raw):,} B of the {dtype} section's {nbytes:,}")
      if dtype == "LlmMetadataProto":
        llm = llm_metadata_pb2.LlmMetadata()
        llm.ParseFromString(raw)
      elif executor_metadata_pb2 is not None:
        execmd = executor_metadata_pb2.ExecutorMetadata()
        execmd.ParseFromString(raw)
      else:
        execmd = "unparsed"
  return describe(sections, llm, execmd)


def _msg(m):
  return json_format.MessageToDict(m, preserving_proto_field_name=True)


def describe_executor(e):
  if e is None:
    return None
  if e == "unparsed":
    return {"present": True, "parsed": False, "summary": "present (builder without the proto)"}
  d = _msg(e)
  lem, parts = d.get("llm_executor_metadata", {}), []
  if "max_history_size" in lem:
    parts.append(f"max_history={lem['max_history_size']}")
  if lem.get("state_buffers"):
    c = collections.Counter(str(sb.get("type", "TYPE_UNSPECIFIED")).replace("TYPE_", "").lower()
                            for sb in lem["state_buffers"])
    parts.append("state:" + ",".join(f"{k}x{v}" for k, v in sorted(c.items())))
  ams = lem.get("attention_mask_settings") or {}
  if ams:
    parts.append("mask:" + ",".join(
        f"{k.replace('attention_mask_type', 'type')}={str(v).replace('ATTENTION_MASK_TYPE_', '').lower()}"
        for k, v in sorted(ams.items())))
  npu = d.get("llm_npu_executor_metadata") or {}
  if npu:
    parts.append("npu:" + ",".join(k for k, v in sorted(npu.items()) if v))
  return {"present": True, "parsed": True, "summary": " ".join(parts) or "empty", "value": d}


def _ints(obj):
  if isinstance(obj, bool):
    return set()
  if isinstance(obj, int):
    return {obj}
  if isinstance(obj, str) and obj.lstrip("-").isdigit():
    return {int(obj)}
  if isinstance(obj, dict):
    return set().union(*[_ints(v) for v in obj.values()]) if obj else set()
  if isinstance(obj, list):
    return set().union(*[_ints(v) for v in obj]) if obj else set()
  return set()


def describe_llm(m, sections):
  set_fields = {f.name for f, _ in m.ListFields()}
  which = m.llm_model_type.WhichOneof("model_type") if m.HasField("llm_model_type") else None
  d = {"model_type": which}
  if which:
    d["model_type_fields"] = _msg(getattr(m.llm_model_type, which))
  d["max_num_tokens"] = m.max_num_tokens or None
  d["min_runtime_version"] = getattr(m, "min_runtime_version", "") or None
  for f in ("supports_thinking", "supports_function_calling"):
    d[f] = getattr(m, f) if f in set_fields else None
  d["tool_call_config"] = (_msg(m.tool_call_config) or {"set": True}) if (
      "tool_call_config" in set_fields) else None
  d["sampler"] = _msg(m.sampler_params) if m.HasField("sampler_params") else None
  d["start_token"] = _msg(m.start_token) if m.HasField("start_token") else None
  d["stop_tokens"] = [_msg(t) for t in m.stop_tokens]
  d["channels"] = [dict({"name": c.channel_name, "start": c.start, "end": c.end},
                        **({"is_reasoning": True} if getattr(c, "is_reasoning_channel", False) else {}))
                   for c in m.channels]
  tpl = m.jinja_prompt_template if m.HasField("jinja_prompt_template") else None
  if tpl is not None:
    src = tpl.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    t = {"form": "jinja", "bytes": len(tpl.encode()), "sha256": hashlib.sha256(tpl.encode()).hexdigest()}
    try:
      names = mm.jinja2_meta.find_undeclared_variables(mm.template_env().parse(src))
      t.update(parses=True, reads_tools="tools" in names, reads_enable_thinking="enable_thinking" in names)
    except Exception as e:  # noqa: BLE001 - untrusted template text
      t.update(parses=False, error=f"{type(e).__name__}: {e}"[:200])
    d["template"] = t
  elif m.HasField("prompt_templates"):
    d["template"] = {"form": "affixes", "reads_tools": False, "reads_enable_thinking": False}
  else:
    d["template"] = {"form": "none"}
  d["think_marker_in_template"] = bool(tpl and "<think>" in tpl)
  ctl, basis = mm.derive_thinking_control(m)
  d["thinking_control"] = {"value": ctl, "basis": basis}
  d["capabilities"] = mm.derive_capabilities(m, sections)
  return d


def describe(sections, llm, execmd):
  types = [s["type"] for s in sections]
  d = {"sections": sections}
  # The kind = the task metadata section (LlmMetadataProto -> llm,
  # EmbeddingMetadataProto -> embedding, ...); ExecutorMetadata is not a task.
  task = [t for t in types if t.endswith("MetadataProto") and t != "ExecutorMetadataProto"]
  d["kind"] = ("llm" if "LlmMetadataProto" in task else
               task[0][:-len("MetadataProto")].lower() if task else "other")
  mtype = lambda s: str((s.get("items") or {}).get("model_type", ""))  # noqa: E731
  wsecs = [s for s in sections if s["type"] == "TFLiteWeights"]
  d["weights"] = {"form": "external" if wsecs else "inline", "external_sections": len(wsecs),
                  "external_bytes": sum(s["size_bytes"] for s in wsecs),
                  "external_for": sorted({mtype(s) or "?" for s in wsecs})}
  d["vision"] = any("vision" in mtype(s).lower() for s in sections)
  d["audio"] = any("audio" in mtype(s).lower() for s in sections)
  models = [s for s in sections if s["type"] == "TFLiteModel"]
  main = [s for s in models if mtype(s) == "tf_lite_prefill_decode"] or models
  cons = set()
  for s in main:
    c = (s.get("items") or {}).get("backend_constraint")
    if c:
      cons |= {x.strip().lower() for x in str(c).split(",") if x.strip()}
  d["backend_constraint"] = sorted(cons)
  d["executor_metadata"] = describe_executor(execmd)
  if llm is not None:
    d["llm"] = describe_llm(llm, sections)
  return d


# ---------------------------------------------------------------- flags
def apply_file_flags(f):
  fl = []
  if f.get("status") == "ok":
    cons = set(f["backend_constraint"])
    npu = bool(cons) and cons <= {"npu"}
    if npu:
      fl.append("info:npu")
    if f["kind"] == "llm":
      lm, t = f["llm"], f["llm"]["template"]
      if not npu:
        if f["weights"]["form"] == "inline":
          fl.append("rebuild:inline-weights")
        if not f["executor_metadata"]:
          fl.append("rebuild:no-executor-metadata")
        # gpu, gpu_artisan, ... all name a GPU executor.
        if cons and not any(c.startswith("gpu") for c in cons):
          fl.append("rebuild:cpu-only")
      if (t.get("reads_tools") and lm["model_type"] == "generic_model"
          and not lm.get("tool_call_config")):
        fl.append("rebuild:tool-calls-as-text")
      if not lm["channels"] and t.get("reads_enable_thinking"):
        fl.append("rebuild:thinking-without-channel")
      elif not lm["channels"] and lm["think_marker_in_template"]:
        fl.append("warn:think-tag-without-channel")
      s = lm.get("sampler") or {}
      if s.get("type") == "TOP_K":
        fl.append("warn:sampler-top-k")
      if s.get("type") == "TOP_P" and not s.get("p"):
        fl.append("warn:sampler-p0")
      if t.get("form") in ("affixes", "none"):
        fl.append("warn:no-jinja-template")
      if t.get("parses") is False:
        fl.append("warn:template-does-not-parse")
    if f.get("header_past_end_offset"):
      fl.append("warn:header-past-end-offset")
    if mm.vision_section_warnings(f["file"], f["sections"]):
      fl.append("warn:vision-over-2gib")
  f["flags"] = fl


def apply_source_flags(row):
  src = row.get("source") or {}
  if src.get("status") != "ok":
    return
  eos = set(src.get("eos_token_ids") or [])
  for f in row["files"]:
    if f.get("status") != "ok" or f.get("kind") != "llm":
      continue
    lm, add = f["llm"], []
    stop_ids = set().union(*[_ints(t) for t in lm["stop_tokens"]]) if lm["stop_tokens"] else set()
    if eos and stop_ids and not eos <= stop_ids:
      add.append("warn:source-eos-not-a-stop-token")
      f["source_eos_missing"] = sorted(eos - stop_ids)
    if src.get("template_reads_tools") and not lm["template"].get("reads_tools"):
      add.append("warn:source-template-has-tools")
    if src.get("template_thinks") and not lm["channels"]:
      add.append("warn:source-template-thinks")
    if src.get("has_vision_config") and not f["vision"]:
      add.append("warn:source-has-vision")
    f["flags"] += add


# ---------------------------------------------------------------- repos
def repo_info(endpoint, repo, rev, stage="repo info"):
  return http_json(f"{endpoint}/api/models/{repo}/revision/{urllib.parse.quote(rev, safe='')}?blobs=true", stage)


def audit_file(endpoint, repo, sha, sib, gated):
  name = sib["rfilename"]
  lfs = sib.get("lfs") or {}
  f = {"file": name, "size_bytes": lfs.get("size", sib.get("size")), "sha256": lfs.get("sha256")}
  url = f"{endpoint}/{repo}/resolve/{sha}/{urllib.parse.quote(name)}"
  size = f["size_bytes"]

  def fetch(s, e):
    if size:
      e = min(e, size - 1)
    return http_get(url, "header read", (s, e), gated=gated)
  try:
    f.update(read_bundle(fetch, size))
    f["status"] = "ok"
  except AuditError as e:
    f.update(status="error", stage=e.stage, reason=e.reason)
  except Exception as e:  # noqa: BLE001 - a header the schema cannot read
    f.update(status="error", stage="header parse", reason=f"{type(e).__name__}: {e}"[:300])
  apply_file_flags(f)
  return f


def audit_source(endpoint, base, pinned_sha=None):
  src = {"repo": base}
  try:
    info = repo_info(endpoint, base, pinned_sha or "main", "source info")
    sha, gated = info.get("sha"), bool(info.get("gated"))
    src["sha"] = sha
    names = {s["rfilename"] for s in info.get("siblings", [])}

    def get(name):
      if name not in names:
        return None
      return http_get(f"{endpoint}/{base}/resolve/{sha}/{urllib.parse.quote(name)}",
                      "source file", gated=gated).decode("utf-8", "replace")
    cfg = json.loads(get("config.json") or "{}")
    gen = json.loads(get("generation_config.json") or "{}")
    tok = json.loads(get("tokenizer_config.json") or "{}")
    tpl = tok.get("chat_template")
    if isinstance(tpl, list):
      tpl = "\n".join(t.get("template", "") for t in tpl if isinstance(t, dict))
    if not tpl:
      tpl = get("chat_template.jinja")
    if not tpl:
      ct = get("chat_template.json")
      tpl = json.loads(ct).get("chat_template") if ct else None
  except AuditError as e:
    src.update(status="error", stage=e.stage, reason=e.reason)
    return src
  except (ValueError, AttributeError) as e:
    src.update(status="error", stage="source file", reason=f"{type(e).__name__}: {e}"[:200])
    return src
  eos = set()
  for c in (gen, cfg, cfg.get("text_config") or {}):
    v = c.get("eos_token_id") if isinstance(c, dict) else None
    eos |= {v} if isinstance(v, int) else {x for x in v if isinstance(x, int)} if isinstance(v, list) else set()
  tc = cfg.get("text_config") or {}
  src.update(status="ok", eos_token_ids=sorted(eos),
             max_position_embeddings=cfg.get("max_position_embeddings") or tc.get("max_position_embeddings"),
             has_chat_template=bool(tpl),
             template_reads_tools=bool(tpl and re.search(r"\btools\b", tpl)),
             template_thinks=bool(tpl and ("enable_thinking" in tpl or "<think>" in tpl)),
             has_vision_config="vision_config" in cfg)
  return src


def audit_repo(endpoint, repo, pinned=None, listing=None, source=True):
  pinned, listing = pinned or {}, listing or {}
  row = {"repo": repo}
  try:
    info = repo_info(endpoint, repo, pinned.get("sha") or "main")
  except AuditError as e:
    row.update(status="error", stage=e.stage, reason=e.reason, files=[])
    return row
  row["sha"] = info.get("sha")
  row["gated"] = info.get("gated", False)
  row["downloads"] = pinned.get("downloads", listing.get("downloads", info.get("downloads")))
  base = (info.get("cardData") or {}).get("base_model")
  row["base_model"] = (base[0] if base else None) if isinstance(base, list) else base
  sib = sorted((s for s in info.get("siblings", []) if s["rfilename"].endswith(".litertlm")),
               key=lambda s: s["rfilename"])
  if not sib:
    row.update(status="error", stage="files", reason="no .litertlm file in the repo", files=[])
    return row
  row["files"] = [audit_file(endpoint, repo, row["sha"], s, bool(row["gated"])) for s in sib]
  oks = [f["status"] == "ok" for f in row["files"]]
  row["status"] = "ok" if all(oks) else "partial" if any(oks) else "error"
  if source and row["base_model"] and row["base_model"] != repo:
    row["source"] = audit_source(endpoint, row["base_model"], pinned.get("source_sha"))
    apply_source_flags(row)
  return row


def transient_errors(rows):
  pat = re.compile(r"HTTP (429|5\d\d)|network:")
  objs = rows + [f for r in rows for f in r.get("files", [])] + [r["source"] for r in rows if r.get("source")]
  return sum(1 for o in objs if o.get("status") == "error" and pat.search(o.get("reason") or ""))


def commit_digest(rows):
  lines = sorted(f"{r['repo']}@{r.get('sha') or '-'}" for r in rows)
  return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ---------------------------------------------------------------- render
def _cell(s):
  return str(s).replace("|", "\\|").replace("\n", " ")


def _gb(n):
  return f"{n / 1e9:.2f} GB" if n else "0"


def short_flags(fl):
  return " ".join(x.split(":", 1)[1] for x in fl if not x.startswith("info:"))


def file_cells(f):
  if f["status"] != "ok":
    return None
  w = f["weights"]
  weights = (f"external ({w['external_sections']} sec, {_gb(w['external_bytes'])})"
             if w["form"] == "external" else "inline")
  ex = (f["executor_metadata"] or {}).get("summary", "—") if f["executor_metadata"] else "—"
  va = ("V" if f["vision"] else "") + ("A" if f["audio"] else "")
  lm = f.get("llm")
  if not lm:
    return [f["kind"], weights, ex, va or "—", "—", "—", "—", "—", "—", short_flags(f["flags"]) or "—"]
  t = lm["template"]
  tpl = t["form"] + ("+tools" if t.get("reads_tools") else "") + (" (no parse)" if t.get("parses") is False else "")
  ch = ",".join(c["name"] for c in lm["channels"])
  thinking = f"{ch}/{lm['thinking_control']['value'] or '?'}" if ch else (
      "in template" if (t.get("reads_enable_thinking") or lm["think_marker_in_template"]) else "—")
  s = lm.get("sampler") or {}
  sampler = " ".join(x for x in (s.get("type", ""), f"k{s['k']}" if "k" in s else "",
                                 f"p{s['p']}" if "p" in s else "",
                                 f"T{s['temperature']}" if "temperature" in s else "") if x) or "—"
  return [f["kind"], weights, ex, va or "—", lm["model_type"] or "—", tpl, thinking, sampler,
          lm["max_num_tokens"] or "—", short_flags(f["flags"]) or "—"]


def render_example(r, runs=None):
  out = [f"## Worked example: {r['repo']}", ""]
  if r.get("status") == "error" and not r.get("files"):
    return out + [f"Not read: {r.get('stage')}: {r.get('reason')}", ""]
  out.append(f"Commit `{r.get('sha')}`; base_model: {r.get('base_model') or '—'}.")
  for f in r["files"]:
    out += ["", f"### {f['file']}", ""]
    if f["status"] != "ok":
      out.append(f"Not read: {f['stage']}: {f['reason']}")
      continue
    out.append(f"- size {f['size_bytes']:,} B, sha256 `{f['sha256']}`")
    out.append("- sections: " + "; ".join(
        f"{s['type']}" + (f" {s['items'].get('model_type')}" if s.get("items", {}).get("model_type") else "")
        + (f" [backend_constraint {s['items']['backend_constraint']}]"
           if s.get("items", {}).get("backend_constraint") else "") + f" {_gb(s['size_bytes'])}"
        for s in f["sections"]))
    out.append(f"- weights: {f['weights']['form']}; executor metadata: "
               f"{(f['executor_metadata'] or {}).get('summary', 'none') if f['executor_metadata'] else 'none'}")
    lm = f.get("llm")
    if lm:
      out.append(f"- model type {lm['model_type']} {json.dumps(lm.get('model_type_fields', {}))}; "
                 f"max_num_tokens {lm['max_num_tokens']}; min_runtime_version {lm['min_runtime_version'] or '—'}")
      out.append(f"- template: {json.dumps({k: v for k, v in lm['template'].items() if k != 'sha256'})}; "
                 f"channels {json.dumps(lm['channels'])}; thinking control "
                 f"{lm['thinking_control']['value']} ({lm['thinking_control']['basis']})")
      out.append(f"- sampler {json.dumps(lm['sampler'])}; stop tokens {json.dumps(lm['stop_tokens'])}")
    out.append(f"- flags: {', '.join(f['flags']) or 'none'}")
    by = collections.OrderedDict()
    for x in (runs or {}).get("results", []):
      if (x["repo"], x["file"]) == (r["repo"], f["file"]):
        by.setdefault(x["backend"], []).append(x)
    for be, xs in by.items():
      parts = [f"{x['check']} {x['status']}" + (f" ({x.get('note') or x.get('error')})"
                                                 if x["status"] != "pass" and (x.get("note") or x.get("error")) else "")
               for x in xs if x["status"] != "skip"]
      skipped = sum(1 for x in xs if x["status"] == "skip")
      out.append(f"- run, {be} (Mac, LiteRT-LM {runs.get('runtime', '?')}): " + ", ".join(parts)
                 + (f"; {skipped} checks skipped (no engine)" if skipped else ""))
  src = r.get("source")
  if src:
    out += ["", f"Base model {src['repo']}: " + (
        json.dumps({k: v for k, v in src.items() if k not in ('repo',)}, ensure_ascii=False))]
  return out + [""]


def render_runs(runs):
  out = [f"## Run check (Mac, LiteRT-LM {runs.get('runtime', '?')})", "",
         runs.get("selection", ""), ""]
  checks = ["text", "image", "tool", "think_on", "think_off"]
  out.append("| repo | file | backend | engine | " + " | ".join(checks) + " |")
  out.append("|---|---|---|---|" + "---|" * len(checks))
  by = collections.OrderedDict()
  for r in runs.get("results", []):
    by.setdefault((r["repo"], r["file"], r["backend"]), {})[r["check"]] = r
  for (repo, file, backend), cs in by.items():
    eng = cs.get("engine", {}).get("status", "—")
    cells = [cs[c]["status"] if c in cs else "—" for c in checks]
    out.append(f"| {_cell(repo)} | {_cell(file)} | {backend} | {eng} | " + " | ".join(cells) + " |")
  notes = runs.get("notes") or []
  out += ["", "Header against run: each engine or check that did not pass, with what the header says about it, "
          "and each LiteRT-LM ModelInfo field (input_modalities vision and audio, max_context_tokens) that differs "
          "from the header" + (":" if notes else ": none."), ""] + [f"- {_cell(n)}" for n in notes]
  return out + [""]


def render(data, runs=None, example=EXAMPLE):
  rows = data["repos"]
  files = [(r, f) for r in rows for f in r.get("files", [])]
  ok = [(r, f) for r, f in files if f["status"] == "ok"]
  llm = [(r, f) for r, f in ok if f["kind"] == "llm"]
  out = [f"# {data['author']} .litertlm audit", "",
         f"Generated {data['generated']} by `{TOOL}` (version {TOOL_VERSION}) from {data['endpoint']}, "
         "anonymous reads of the bundle headers only (HTTP ranges, no weights).",
         f"Repos {len(rows)} (each read at the commit in its row; sha256 of the sorted repo@commit list "
         f"`{data['commit_list_sha256']}`), files {len(files)}, headers read {len(ok)}, "
         f"failed {len(files) - len(ok)}, repos not read {sum(1 for r in rows if not r.get('files'))}, "
         f"transient read errors {transient_errors(rows)} (HTTP 429 / 5xx / network: a table with any is "
         "not complete; read again with --pin).",
         "Downloads = the Hub's count when the repo list was read.", "", "## Rules", ""]
  out += [f"- `{k}` — {v}" for k, v in RULES] + [""]
  cnt = collections.Counter(x for _, f in ok for x in f["flags"])
  kinds = collections.Counter(f["kind"] for _, f in ok)
  out += ["## Summary", "", "| count | of |", "|---:|---|"]
  out += [f"| {v} | {k} files |" for k, v in sorted(kinds.items())]
  out += [f"| {sum(1 for _, f in llm if f['weights']['form'] == 'external')} | LLM files with external weights |",
          f"| {sum(1 for _, f in llm if f['executor_metadata'])} | LLM files with ExecutorMetadata |",
          f"| {sum(1 for _, f in llm if f['vision'])} | LLM files with a vision section |",
          f"| {sum(1 for _, f in llm if f['audio'])} | LLM files with an audio section |",
          f"| {sum(1 for _, f in llm if f['llm']['template'].get('reads_tools'))} | LLM files whose template reads `tools` |",
          f"| {sum(1 for _, f in llm if f['llm']['channels'])} | LLM files that declare a channel |"]
  out += [f"| {v} | `{k}` |" for k, v in sorted(cnt.items())] + [""]
  if example:
    ex = [r for r in rows if r["repo"] == example]
    if ex:
      out += render_example(ex[0], runs)
  if runs:
    out += render_runs(runs)
  out += ["## Rebuild list (LLM files with a `rebuild:` flag)", "",
          "| repo | file | why |", "|---|---|---|"]
  for r, f in sorted(llm, key=lambda x: (-(x[0].get("downloads") or 0), x[0]["repo"], x[1]["file"])):
    why = [x.split(":", 1)[1] for x in f["flags"] if x.startswith("rebuild:")]
    if why:
      out.append(f"| {_cell(r['repo'])} | {_cell(f['file'])} | {', '.join(why)} |")
  out += ["", "## All files", "",
          "| repo @ commit | downloads | file | kind | weights | executor metadata | V/A | model type | template "
          "| thinking | sampler | max tokens | flags |",
          "|---|---:|---|---|---|---|---|---|---|---|---|---:|---|"]
  for r, f in sorted(files, key=lambda x: (-(x[0].get("downloads") or 0), x[0]["repo"], x[1]["file"])):
    c = file_cells(f) or ["not read", f"{f.get('stage')}: {f.get('reason')}"] + ["—"] * 8
    out.append(f"| {_cell(r['repo'])} @ {(r.get('sha') or '-')[:8]} | {r.get('downloads') or 0} | "
               f"{_cell(f['file'])} | " + " | ".join(_cell(x) for x in c) + " |")
  bad = [(r, f) for r, f in files if f["status"] != "ok"] + [(r, None) for r in rows if not r.get("files")]
  out += ["", "## Failed reads", "", "| repo | file | stage | reason |", "|---|---|---|---|"]
  for r, f in sorted(bad, key=lambda x: (x[0]["repo"], x[1]["file"] if x[1] else "")):
    src = f or r
    out.append(f"| {_cell(r['repo'])} | {_cell(f['file']) if f else '—'} | {src.get('stage')} | {_cell(src.get('reason'))} |")
  return "\n".join(out) + "\n"


# ---------------------------------------------------------------- self-test
def self_test(endpoint, gated_repo, small_repo, out_dir):
  """The failure paths: a missing repo, a gated repo, a server that ignores
  Range, a file too small for its header. Each must give a row, not a crash."""
  res = []
  r = audit_repo(endpoint, "litert-community/no-such-repo-community-audit", source=False)
  res.append(("missing repo", r["status"] == "error" and r.get("stage") == "repo info"
              and re.search(r"HTTP (401|404)", r.get("reason", "")) is not None,
              f"{r.get('stage')}: {r.get('reason')}"))
  r = audit_repo(endpoint, gated_repo, source=False)
  reasons = sorted({f.get("reason", "") for f in r.get("files", [])})
  res.append(("gated repo", bool(r.get("files")) and r["status"] == "error"
              and all("gated" in x for x in reasons), f"{len(r.get('files', []))} files: {reasons}"))
  with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    sha = "0" * 40
    (root / "api/models/fake/norange/revision").mkdir(parents=True)
    (root / "api/models/fake/norange/revision/main").write_text(json.dumps(
        {"sha": sha, "siblings": [{"rfilename": "model.litertlm", "size": 8192}]}))
    (root / f"fake/norange/resolve/{sha}").mkdir(parents=True)
    (root / f"fake/norange/resolve/{sha}/model.litertlm").write_bytes(
        litertlm_core.HEADER_MAGIC_BYTES + bytes(8184))

    class Quiet(http.server.SimpleHTTPRequestHandler):
      def log_message(self, *a):
        pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=tmp))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
      r = audit_repo(f"http://127.0.0.1:{srv.server_port}", "fake/norange", source=False)
    finally:
      srv.shutdown()
    f = (r.get("files") or [{}])[0]
    res.append(("server ignores Range", f.get("status") == "error" and "ignored the Range" in f.get("reason", ""),
                f"{f.get('stage')}: {f.get('reason')}"))
    info = repo_info(endpoint, small_repo, "main")
    sib = sorted(s for s in (x["rfilename"] for x in info["siblings"]) if s.endswith(".litertlm"))[0]
    head = http_get(f"{endpoint}/{small_repo}/resolve/{info['sha']}/{urllib.parse.quote(sib)}",
                    "header read", (0, 4095))
    for cut in (24, 4096):
      p = root / f"cut{cut}.litertlm"
      p.write_bytes(head[:cut])
      data = p.read_bytes()
      g = {"file": p.name, "size_bytes": len(data)}
      try:
        g.update(read_bundle(lambda s, e, _d=data: _d[s:e + 1], len(data)))
        g["status"] = "ok"
      except AuditError as e:
        g.update(status="error", stage=e.stage, reason=e.reason)
      res.append((f"file cut to {cut} B ({small_repo} {sib})", g["status"] == "error",
                  f"{g.get('stage')}: {g.get('reason')}"))
  lines = [f"{'PASS' if ok else 'FAIL'} {name} -> {why}" for name, ok, why in res]
  print("\n".join(lines))
  if out_dir:
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(out_dir) / "selftest.txt").write_text(
        f"{TOOL} --self-test, {datetime.date.today().isoformat()}\n" + "\n".join(lines) + "\n")
  return all(ok for _, ok, _ in res)


# ---------------------------------------------------------------- main
def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--author", default="litert-community")
  ap.add_argument("--repos", nargs="*", help="audit these repos instead of the author's list")
  ap.add_argument("--pin", help="a previous audit.json: the same repos at the same commits")
  ap.add_argument("--out-dir", default=str(HERE / "table"))
  ap.add_argument("--runs", help="run-check results (run_check.py) to add to the table")
  ap.add_argument("--example", default=EXAMPLE, help="repo shown in full at the top ('' = none)")
  ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", ENDPOINT))
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--no-source", action="store_true", help="skip the base-model comparison")
  ap.add_argument("--render-only", action="store_true", help="re-render audit.md from audit.json")
  ap.add_argument("--self-test", action="store_true")
  ap.add_argument("--gated-repo", default="litert-community/Gemma3-1B-IT")
  ap.add_argument("--small-repo", default="litert-community/SmolLM2-360M-Instruct")
  a = ap.parse_args()
  out = pathlib.Path(a.out_dir)
  if a.self_test:
    sys.exit(0 if self_test(a.endpoint, a.gated_repo, a.small_repo, a.out_dir) else 1)
  runs = json.loads(pathlib.Path(a.runs).read_text()) if a.runs else None
  if a.render_only:
    data = json.loads((out / "audit.json").read_text())
    (out / "audit.md").write_text(render(data, runs, a.example))
    print(out / "audit.md")
    return
  pins, listing = {}, {}
  if a.pin:
    old = json.loads(pathlib.Path(a.pin).read_text())
    author = old["author"]
    pins = {r["repo"]: {"sha": r.get("sha"), "downloads": r.get("downloads"),
                        "source_sha": (r.get("source") or {}).get("sha")} for r in old["repos"]}
    repos = sorted(pins)
  elif a.repos:
    author, repos = a.author, sorted(a.repos)
  else:
    author = a.author
    listed = list_author(a.endpoint, author)
    listing = {m["id"]: m for m in listed
               if any(s["rfilename"].endswith(".litertlm") for s in m.get("siblings", []))}
    repos = sorted(listing)
    print(f"{len(listed)} repos listed, {len(repos)} hold a .litertlm file", file=sys.stderr)
  with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
    futs = {ex.submit(audit_repo, a.endpoint, rp, pins.get(rp), listing.get(rp), not a.no_source): rp
            for rp in repos}
    rows = []
    for n, fu in enumerate(concurrent.futures.as_completed(futs), 1):
      rows.append(fu.result())
      if n % 20 == 0:
        print(f"{n}/{len(repos)} repos", file=sys.stderr)
  rows.sort(key=lambda r: r["repo"])
  data = {"tool": TOOL, "tool_version": TOOL_VERSION, "generated": datetime.date.today().isoformat(),
          "endpoint": a.endpoint, "author": author, "commit_list_sha256": commit_digest(rows),
          "repos": rows}
  out.mkdir(parents=True, exist_ok=True)
  (out / "audit.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
  (out / "audit.md").write_text(render(data, runs, a.example))
  print(out / "audit.md")


if __name__ == "__main__":
  main()
