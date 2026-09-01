#!/usr/bin/env python3
"""Generate litertlm_manifest.json for a Hugging Face .litertlm model repo.

Derived fields (sha256, size, sections, context_length, capabilities, thinking
channel) are read from the HF API and from each bundle's header — two HTTP range
requests per file, no weight download. Curated fields (quantization, backends,
recommendations, measured rows, known issues) come from a hand-verified JSON
passed via --curated (see manifest/examples/ for finished manifests). The output is validated against manifest/litertlm_manifest.schema.json.

Usage (needs: pip install litert-lm-builder jsonschema):
    python manifest/make_manifest.py litert-community/LFM2.5-1.2B-Instruct
    ... --public          # strip every private `evidence` pointer for publication
    ... --local-file F    # parse a local bundle instead of range-reading HF

"""
import argparse
import datetime
import json
import pathlib
import sys
import urllib.request

from litert_lm_builder import litertlm_core  # noqa: E402
from litert_lm_builder import litertlm_header_schema_py_generated as schema  # noqa: E402
from litert_lm_builder.runtime.proto import llm_metadata_pb2  # noqa: E402

HDR_BEGIN = litertlm_core.HEADER_BEGIN_BYTE_OFFSET
HDR_END_LOC = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET

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

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("repo")
  ap.add_argument("--curated", default=None)
  ap.add_argument("--out", default=None)
  ap.add_argument("--public", action="store_true",
                  help="strip private evidence pointers")
  ap.add_argument("--local-file", action="append", default=[],
                  metavar="NAME=PATH",
                  help="use a local copy for bundle NAME instead of HF ranges")
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

  variants, model_meta = [], None
  for name, f in sorted(files.items()):
    lfs = f.get("lfs") or {}
    v = {"file": name}
    if lfs.get("oid"):
      v["sha256"] = lfs["oid"]
    v["size_bytes"] = lfs.get("size", f.get("size"))

    if name in local:
      p = local[name]
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

  manifest = {
      "manifest_schema": "0.1.2",
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
