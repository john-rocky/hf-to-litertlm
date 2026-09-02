#!/usr/bin/env python3
"""litertlm_manifest.json -> Play for On-device AI pack.

Turns the deployment manifest a `.litertlm` model repo ships (manifest/SCHEMA.md)
into the pieces Google Play needs to deliver the right file to the right phone:

  * an AI pack module (`<pack>/build.gradle`, `src/main/assets/model#group_<g>/...`)
  * a device-targeting XML (one Play device group per manifest recommendation)
  * the app/settings/gradle.properties snippets, or a complete host project
  * a `litertlm_pack.json` sidecar in every group so the app knows which file,
    which backend and which session defaults it received

Every device group is built only from what the manifest can prove: a variant's
Android `recommended[]` entry, the devices in its `measured[]` rows (mapped to Play
brand/device/SoC codes by device_classes.json) and its `requirements`. Policy knobs
(RAM margin, SoC extrapolation, pack cap) live in device_classes.json, documented.

Play facts this tool depends on (developer.android.com/google/play/on-device-ai,
"Last updated 2026-08-14"): plugin `com.android.ai-pack` (AGP >= 8.8); device
targeting needs AGP >= 8.10 and `android.experimental.enableDeviceTargetingConfigApi=true`;
one pack <= 1.5 GB compressed; `#group_<name>` asset-directory suffix, stripped at
build time; un-suffixed assets go to EVERY device; the first matching group in the
XML wins; up to 5 selectors per group; bundletool >= 1.18.0 for local testing.

Usage:
  make_ai_pack.py plan     <manifest.json | org/name> [...]   # what would be packed, and why not
  make_ai_pack.py generate <manifest.json | org/name> --out DIR [--host-app] [--delivery fast-follow]
  make_ai_pack.py fetch    --out DIR                          # download the variant files (sha256-verified)
  make_ai_pack.py from-adb [--serial S]                       # print a device registry entry for an attached phone

No third-party dependencies.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATES = HERE / "templates" / "host_app"
HF = "https://huggingface.co"
GENERATOR = "make_ai_pack.py"
PACK_SCHEMA = "0.1.0"

# Toolchain pins for the generated host project (verified against Google's maven
# metadata and the AGP release notes on 2026-09-02; bump deliberately).
PINS = {
    "agp": "8.13.2",            # com.android.ai-pack ships inside AGP; 8.13.x needs Gradle 8.13
    "gradle": "8.13",
    "kotlin": "2.2.21",
    "litertlm_android": "0.16.1",
    "ai_delivery": "0.2.0-beta01",
    "activity_ktx": "1.13.0",
    "fragment_ktx": "1.8.9",        # lintVital: ActivityResult APIs need Fragment >= 1.3.0 on the classpath
    "core_ktx": "1.18.0",           # 1.19.0 needs compileSdk 37 + AGP 9.1
    "coroutines": "1.11.0",
    "compile_sdk": "36",
    "min_sdk": "24",            # litertlm-android AAR declares minSdkVersion 24
    "target_sdk": "36",
    "bundletool": "1.18.3",
}


# ----------------------------------------------------------------------------- io

def log(msg):
  print(msg, file=sys.stderr)


def load_json(path):
  return json.loads(pathlib.Path(path).read_text())


def load_manifest(src):
  """Local path, or `org/name` (fetched from the HF repo root)."""
  p = pathlib.Path(src)
  if p.exists():
    return load_json(p), str(p)
  if re.fullmatch(r"[^/\s]+/[^/\s]+", src):
    url = f"{HF}/{src}/resolve/main/litertlm_manifest.json"
    try:
      with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r), url
    except urllib.error.HTTPError as e:
      # 401 hides existence (gated/private/nonexistent); 404 = public repo, no manifest.
      sys.exit(f"{src}: HTTP {e.code} fetching {url}"
               + (" (no litertlm_manifest.json in this public repo)" if e.code == 404 else ""))
  sys.exit(f"not a file and not an org/name repo id: {src}")


def load_registry(path):
  reg = load_json(path or (HERE / "device_classes.json"))
  for d in reg["devices"]:
    d["_match"] = [re.compile(m) for m in d["match"]]
  return reg


# ------------------------------------------------------------------------ naming

def sanitize(name, prefix="ai_"):
  """Play names: start with a letter, then letters/digits/underscores."""
  s = name.replace("+", "plus").lower()
  s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
  s = re.sub(r"_+", "_", s)
  if not s or not s[0].isalpha():
    s = prefix + s
  return s


def pack_name_for(repo):
  return sanitize(repo.split("/")[-1])


# ------------------------------------------------------------------------ planning

def match_device(reg, device_str):
  for d in reg["devices"]:
    if any(m.search(device_str) for m in d["_match"]):
      return d
  return None


def verified_devices(variant, backend, reg):
  """Registry entries for every measured row of this variant on this backend."""
  seen, out = set(), []
  for row in variant.get("measured", []):
    if row.get("backend") != backend:
      continue
    d = match_device(reg, row.get("device", ""))
    if d and d["id"] not in seen:
      seen.add(d["id"])
      out.append((d, row))
  return out


def build_selectors(variant, rec, cls_def, reg, policy, evidence):
  """Return a list of Play device selectors (OR-ed) for one recommendation.

  Each selector is a dict with any of: ram_min_bytes, device_ids[], socs[].
  Properties inside one selector are AND-ed by Play, so RAM, device ids and SoCs
  each get their own selector (= OR).
  """
  backend = rec["backend"]
  devs = verified_devices(variant, backend, reg)
  sels = []

  ram_floor = None
  use_floor = cls_def.get("ram_floor", True) if cls_def else True
  if use_floor and devs:
    nominal = min(d["ram_nominal_bytes"] for d, _ in devs)
    ram_floor = int(nominal * (1 - policy["ram_margin"]))
    small = min(devs, key=lambda x: x[0]["ram_nominal_bytes"])[0]
    evidence.append(f"RAM floor {ram_floor:,} B = {nominal/1e9:g} GB nominal of the smallest "
                    f"verified device ({small['id']}) x (1 - {policy['ram_margin']})")
  elif use_floor and cls_def and cls_def.get("ram_nominal_bytes"):
    # The curated device_class is itself evidence; without a measured row on this
    # backend the class's nominal RAM (registry policy) stands in for a device.
    nominal = cls_def["ram_nominal_bytes"]
    ram_floor = int(nominal * (1 - policy["ram_margin"]))
    evidence.append(f"RAM floor {ram_floor:,} B = {nominal/1e9:g} GB nominal of device_class "
                    f"'{rec.get('device_class')}' (registry) x (1 - {policy['ram_margin']}); the manifest has "
                    f"no measured Android row for this file on {backend}")
  peak = (variant.get("requirements") or {}).get("peak_ram_mb")
  if use_floor and peak:
    need = int(peak * 1e6 * policy["peak_ram_headroom"])
    if need > (ram_floor or 0):
      ram_floor = need
      evidence.append(f"RAM floor raised to {need:,} B = requirements.peak_ram_mb {peak} x "
                      f"{policy['peak_ram_headroom']} headroom")
  if ram_floor:
    sels.append({"ram_min_bytes": ram_floor})

  ids = [(d["brand"], d["device"], d["id"]) for d, _ in devs if d.get("brand") and d.get("device")]
  if ids:
    sels.append({"device_ids": [{"brand": b, "device": dv} for b, dv, _ in ids]})
    evidence.append("verified device ids: " + ", ".join(f"{b}/{dv} ({i})" for b, dv, i in ids))

  if policy["extrapolate_soc"]:
    socs, seen = [], set()
    for d, _ in devs:
      if d.get("soc_manufacturer") and d.get("soc_model"):
        key = (d["soc_manufacturer"], d["soc_model"])
        if key not in seen:
          seen.add(key)
          socs.append({"manufacturer": key[0], "model": key[1]})
    if socs:
      sels.append({"socs": socs})
      evidence.append("SoC extrapolation from verified devices: "
                      + ", ".join(f"{s['manufacturer']} {s['model']}" for s in socs))
    missing = [d["id"] for d, _ in devs if not (d.get("soc_manufacturer") and d.get("soc_model"))]
    if missing:
      evidence.append("SoC strings unknown for: " + ", ".join(missing)
                      + " (run `from-adb` with the phone attached and update device_classes.json)")

  return sels[:5]


def plan_manifest(manifest, reg, opts):
  policy = dict(reg["policy"])
  if opts.max_pack_bytes:
    policy["max_pack_bytes"] = opts.max_pack_bytes
  if opts.floor_default:
    policy["floor_default"] = opts.floor_default
  classes = reg["classes"]
  repo = manifest["repo"]
  warnings, excluded, eligible = [], [], []

  for v in manifest["variants"]:
    recs = [r for r in v.get("recommended", []) if r.get("platform") == "android"]
    if not recs:
      if opts.allow_unrecommended and not v.get("recommended") and "cpu" in v["backends"]:
        recs = [{"platform": "android", "backend": v.get("default_backend") or v["backends"][0],
                 "reason": "SYNTHETIC: manifest carries no Android recommendation "
                           "(--allow-unrecommended); not Android-verified"}]
        warnings.append(f"{v['file']}: packed without an Android recommendation (--allow-unrecommended)")
      else:
        excluded.append({"file": v["file"], "reason": "no Android recommendation in the manifest "
                         "(`recommended[]` has no platform=android entry)"})
        continue
    size = v.get("size_bytes") or 0
    if size > policy["max_pack_bytes"]:
      excluded.append({"file": v["file"], "size_bytes": size,
                       "reason": f"{size/1e9:.2f} GB exceeds the Play AI pack cap "
                                 f"({policy['max_pack_bytes']/1e9:g} GB compressed; weights do not compress)"})
      continue
    for r in recs:
      if r["backend"] not in v["backends"]:
        excluded.append({"file": v["file"], "reason": f"recommendation names backend {r['backend']} "
                         f"that is not in the variant's verified backends {v['backends']}"})
        continue
      eligible.append((v, r))

  classed = [(v, r) for v, r in eligible if r.get("device_class")]
  classless = [(v, r) for v, r in eligible if not r.get("device_class")]

  groups = []
  by_class = {}
  for v, r in classed:
    by_class.setdefault(r["device_class"], []).append((v, r))
  for cls, pairs in by_class.items():
    if len(pairs) > 1:
      # One Play group carries one file. Order: --prefer-backend, then the file with a
      # measured Android row on its recommended backend, then the smaller file
      # (the reference readers' tie-break).
      def rank(p):
        v, r = p
        return (0 if opts.prefer_backend and r["backend"] == opts.prefer_backend else 1,
                0 if verified_devices(v, r["backend"], reg) else 1,
                v["size_bytes"])
      ordered = sorted(pairs, key=rank)
      keep = ordered[0]
      why = ("--prefer-backend" if rank(keep)[0] == 0 else
             "it has a measured Android row on its backend" if rank(keep)[1] == 0 else
             "it is the smaller file (reader tie-break)")
      for p in ordered[1:]:
        excluded.append({"file": p[0]["file"], "reason": f"device_class '{cls}' is also claimed by "
                         f"{keep[0]['file']} [{keep[1]['backend']}], kept because {why}"})
      pairs = [keep]
    v, r = pairs[0]
    cdef = classes.get(cls)
    if cdef is None:
      warnings.append(f"device_class '{cls}' is not in device_classes.json; treated as ram_floor=true, "
                      "lowest priority")
      cdef = {"priority": 50, "ram_floor": True}
    evidence = []
    sels = build_selectors(v, r, cdef, reg, policy, evidence)
    if not sels:
      excluded.append({"file": v["file"], "reason": f"device_class '{cls}': no verified Android device "
                       "row on backend {0} and no peak_ram_mb, so no selector can be derived".format(r["backend"])})
      continue
    groups.append({"name": sanitize(cls), "device_class": cls, "priority": cdef["priority"],
                   "file": v["file"], "backend": r["backend"], "size_bytes": v["size_bytes"],
                   "reason": r.get("reason"), "selectors": sels, "evidence": evidence,
                   "_variant": v, "_rec": r})

  default = None
  if classless:
    v, r = min(classless, key=lambda p: p[0]["size_bytes"])
    for ov, orr in classless:
      if ov is not v:
        excluded.append({"file": ov["file"], "reason": f"second class-less Android recommendation; "
                         f"{v['file']} is smaller (reader tie-break)"})
    evidence = []
    sels = build_selectors(v, r, None, reg, policy, evidence)
    floor_sels = [s for s in sels if "ram_min_bytes" in s]
    if groups or (policy["floor_default"] == "auto" and floor_sels):
      if not sels:
        sels = [{"ram_min_bytes": 1}]
        evidence.append("catch-all: no verified Android device row, RAM floor 1 byte matches every phone")
      groups.append({"name": "baseline", "device_class": None, "priority": 99,
                     "file": v["file"], "backend": r["backend"], "size_bytes": v["size_bytes"],
                     "reason": r.get("reason"), "selectors": sels, "evidence": evidence,
                     "_variant": v, "_rec": r})
      if not groups[:-1]:
        warnings.append("class-less recommendation became group 'baseline' with a derived RAM floor "
                        "(policy floor_default=auto); phones below it get an empty pack")
    else:
      default = {"file": v["file"], "backend": r["backend"], "size_bytes": v["size_bytes"],
                 "reason": r.get("reason"), "_variant": v, "_rec": r}

  groups.sort(key=lambda g: g["priority"])
  if groups:
    # a group after 'baseline' could never be reached
    names = [g["name"] for g in groups]
    if "baseline" in names and names[-1] != "baseline":
      warnings.append("a device group is ordered after 'baseline' and can never match")
    total = sum(g["size_bytes"] for g in groups) + (default["size_bytes"] if default else 0)
    if total > 4e9:
      warnings.append(f"all variants together are {total/1e9:.2f} GB; each device only receives its "
                      "own group, but keep the 4 GB cumulative app-size rule in mind for the base APK")

  return {"repo": repo, "manifest_schema": manifest.get("manifest_schema"),
          "manifest_generated": manifest.get("generated"),
          "pack_name": opts.pack_name or pack_name_for(repo),
          "delivery": opts.delivery, "policy": {k: v for k, v in policy.items() if not k.startswith("_")},
          "groups": groups, "default": default, "excluded": excluded, "warnings": warnings}


def strip_private(plan):
  return json.loads(json.dumps(plan, default=lambda o: None, ensure_ascii=False),
                    object_hook=lambda d: {k: v for k, v in d.items() if not k.startswith("_")})


def print_plan(plan):
  print(f"# {plan['repo']}  (manifest {plan['manifest_schema']}, {plan['manifest_generated']})")
  print(f"pack: {plan['pack_name']}  delivery: {plan['delivery']}")
  if plan["groups"]:
    print("device groups (Play serves the first match):")
    for g in plan["groups"]:
      sel = []
      for s in g["selectors"]:
        if "ram_min_bytes" in s:
          sel.append(f"RAM>={s['ram_min_bytes']/1e9:.2f}GB")
        if "device_ids" in s:
          sel.append("device in {" + ",".join(f"{d['brand']}/{d['device']}" for d in s["device_ids"]) + "}")
        if "socs" in s:
          sel.append("SoC in {" + ",".join(f"{x['manufacturer']} {x['model']}" for x in s["socs"]) + "}")
      print(f"  {g['name']:<18} {g['file']}  [{g['backend']}]  {g['size_bytes']/1e9:.2f} GB")
      print(f"  {'':<18} matches: " + "  OR  ".join(sel))
      for e in g["evidence"]:
        print(f"  {'':<18} - {e}")
  if plan["default"]:
    d = plan["default"]
    print(f"default (every device): {d['file']}  [{d['backend']}]  {d['size_bytes']/1e9:.2f} GB")
  elif plan["groups"]:
    print("default (every device): empty pack (litertlm_pack_index.json only)")
  else:
    print("NOTHING TO PACK")
  for x in plan["excluded"]:
    print(f"  excluded: {x['file']} - {x['reason']}")
  for w in plan["warnings"]:
    print(f"  WARNING: {w}")


# ------------------------------------------------------------------------ writing

def render(template_name, subs):
  text = (TEMPLATES / template_name).read_text()
  for k, v in subs.items():
    text = text.replace("{{" + k + "}}", str(v))
  leftover = re.findall(r"\{\{[A-Z_]+\}\}", text)
  if leftover:
    sys.exit(f"template {template_name}: unfilled {leftover}")
  return text


def write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text)
  log(f"  wrote {path}")


def selector_xml(sel, indent="      "):
  attrs = ""
  if "ram_min_bytes" in sel:
    attrs += f' ram-min-bytes="{sel["ram_min_bytes"]}"'
  if "ram_max_bytes" in sel:
    attrs += f' ram-max-bytes="{sel["ram_max_bytes"]}"'
  inner = []
  for d in sel.get("device_ids", []):
    inner.append(f'{indent}  <config:included-device-id brand="{d["brand"]}" device="{d["device"]}"/>')
  for s in sel.get("socs", []):
    inner.append(f'{indent}  <config:system-on-chip manufacturer="{s["manufacturer"]}" model="{s["model"]}"/>')
  if inner:
    return f"{indent}<config:device-selector{attrs}>\n" + "\n".join(inner) + f"\n{indent}</config:device-selector>"
  return f"{indent}<config:device-selector{attrs}/>"


def device_targeting_xml(plan):
  lines = ['<?xml version="1.0" encoding="utf-8"?>',
           f'<!-- generated by {GENERATOR} from {plan["repo"]} litertlm_manifest.json '
           f'({plan["manifest_schema"]}, {plan["manifest_generated"]}) on {datetime.date.today()}.',
           '     Group order = priority: Play serves the first group a device matches.',
           '     Selectors inside a group are OR; properties inside a selector are AND. -->',
           '<config:device-targeting-config',
           '    xmlns:config="http://schemas.android.com/apk/config">']
  for g in plan["groups"]:
    lines.append("")
    lines.append(f'  <!-- {g["file"]} on {g["backend"]}: {g.get("reason") or "(no reason given)"} -->')
    for e in g["evidence"]:
      lines.append(f"  <!-- {e} -->")
    lines.append(f'  <config:device-group name="{g["name"]}">')
    for s in g["selectors"]:
      lines.append(selector_xml(s, "    "))
    lines.append("  </config:device-group>")
  lines.append("")
  lines.append("</config:device-targeting-config>")
  return "\n".join(lines) + "\n"


def variant_capabilities(v, model):
  """model.capabilities is derived from the first bundle in the repo; a text-only
  variant next to a VL build must not inherit vision=true. Sections are per file."""
  caps = dict((model.get("capabilities") or {}))
  types = {s.get("model_type") for s in v.get("sections", [])}
  if types:
    caps["vision"] = "tf_lite_vision_encoder" in types
    caps["audio"] = "tf_lite_audio_encoder" in types
  return caps


def sidecar(plan, manifest, entry, group_name):
  v = entry["_variant"]
  model = dict(manifest.get("model", {}))
  if "capabilities" in model:
    model["capabilities"] = variant_capabilities(v, model)
  return {
      "pack_schema": PACK_SCHEMA,
      "generator": GENERATOR,
      "generated": datetime.date.today().isoformat(),
      "source": {"repo": plan["repo"], "manifest_schema": plan["manifest_schema"],
                 "manifest_generated": plan["manifest_generated"]},
      "group": group_name,
      "file": v["file"],
      "sha256": v.get("sha256"),
      "size_bytes": v.get("size_bytes"),
      "backend": entry["backend"],
      "quantization": v.get("quantization"),
      "min_runtime_version": v.get("min_runtime_version"),
      "recommendation_reason": entry.get("reason"),
      "model": {k: model.get(k) for k in ("display_name", "base_model", "license", "context_length",
                                           "capabilities", "session_defaults") if k in model},
      "platform_notes": (v.get("requirements") or {}).get("platform_notes", []),
      "known_issues": v.get("known_issues", []),
  }


def generate(manifest, plan, out, opts):
  out = pathlib.Path(out)
  pack = plan["pack_name"]
  assets = out / pack / "src" / "main" / "assets"
  targets = []  # (dir, entry, group_name)
  for g in plan["groups"]:
    targets.append((assets / f"model#group_{g['name']}", g, g["name"]))
  if plan["default"]:
    targets.append((assets / "model", plan["default"], "default"))
  if not targets:
    sys.exit("nothing to pack (see `plan`)")

  # pack module
  write(out / pack / "build.gradle", render("pack.build.gradle", {"PACK_NAME": pack, "DELIVERY": plan["delivery"]}))
  for d, entry, gname in targets:
    write(d / "litertlm_pack.json", json.dumps(sidecar(plan, manifest, entry, gname), indent=2, ensure_ascii=False) + "\n")
  index = {"pack_schema": PACK_SCHEMA, "generator": GENERATOR, "pack": pack,
           "source": {"repo": plan["repo"], "manifest_schema": plan["manifest_schema"],
                      "manifest_generated": plan["manifest_generated"]},
           "groups": [{"name": g["name"], "file": g["file"], "backend": g["backend"],
                       "size_bytes": g["size_bytes"], "selectors": g["selectors"]} for g in plan["groups"]],
           "default": ({"file": plan["default"]["file"], "backend": plan["default"]["backend"]}
                       if plan["default"] else None)}
  # Everyone receives un-suffixed directories. bundletool refuses an un-suffixed
  # `model/` next to `model#group_*/` ("must have exactly one device group"), so the
  # index lives in its own directory. A stale index from an earlier layout is removed.
  stale = assets / "model" / "litertlm_pack_index.json"
  if stale.exists() and not plan["default"]:
    stale.unlink()
    if not any(stale.parent.iterdir()):
      stale.parent.rmdir()
  write(assets / "index" / "litertlm_pack_index.json", json.dumps(index, indent=2, ensure_ascii=False) + "\n")
  # fetch list: what `fetch` downloads and where
  fetch_list = [{"repo": plan["repo"], "file": e["file"], "sha256": e["_variant"].get("sha256"),
                 "size_bytes": e["size_bytes"], "dest": str((d / e["file"]).relative_to(out))}
                for d, e, _ in targets]
  write(out / "litertlm_fetch.json", json.dumps(fetch_list, indent=2) + "\n")

  has_groups = bool(plan["groups"])
  if has_groups:
    write(out / "app" / "device_targeting_config.xml", device_targeting_xml(plan))

  subs = dict(PINS)
  subs.update({"PACK_NAME": pack, "APPLICATION_ID": opts.application_id,
               "PACKAGE_PATH": opts.application_id.replace(".", "/"),
               "REPO": plan["repo"], "DISPLAY_NAME": manifest.get("model", {}).get("display_name", plan["repo"]),
               "DELIVERY": plan["delivery"],
               "DEVICE_TARGETING_BLOCK": (
                   "    bundle {\n"
                   "        deviceTargetingConfig = file('device_targeting_config.xml')\n"
                   "        deviceGroup {\n"
                   "            enableSplit = true   // one split per #group_ directory\n"
                   "            defaultGroup = \"other\"\n"
                   "        }\n"
                   "    }\n") if has_groups else "",
               "DEVICE_TARGETING_PROPERTY": (
                   "android.experimental.enableDeviceTargetingConfigApi=true\n") if has_groups else "",
               "GROUP_NAMES": ",".join(g["name"] for g in plan["groups"]) or "(none)",
               "PROMPT": opts.prompt.replace('"', '\\"')})
  subs = {k.upper(): v for k, v in subs.items()}

  snippets = out / "snippets"
  write(snippets / "settings.gradle.snippet", f"include ':{pack}'\n")
  write(snippets / "app.build.gradle.snippet",
        f"android {{\n    assetPacks = [\":{pack}\"]\n{subs['DEVICE_TARGETING_BLOCK']}}}\n")
  write(snippets / "gradle.properties.snippet", subs["DEVICE_TARGETING_PROPERTY"] or "# no device targeting for this pack\n")

  if opts.host_app:
    write(out / "settings.gradle", render("settings.gradle", subs))
    write(out / "build.gradle", render("root.build.gradle", subs))
    write(out / "gradle.properties", render("gradle.properties", subs))
    write(out / "gradle" / "wrapper" / "gradle-wrapper.properties", render("gradle-wrapper.properties", subs))
    write(out / "app" / "build.gradle", render("app.build.gradle", subs))
    write(out / "app" / "src" / "main" / "AndroidManifest.xml", render("AndroidManifest.xml", subs))
    write(out / "app" / "src" / "main" / "java" / subs["PACKAGE_PATH"] / "MainActivity.kt",
          render("MainActivity.kt", subs))
    write(out / "app" / "src" / "main" / "res" / "layout" / "activity_main.xml", render("activity_main.xml", subs))
    write(out / "app" / "src" / "main" / "res" / "values" / "strings.xml", render("strings.xml", subs))
    write(out / ".gitignore", "build/\n.gradle/\nlocal.properties\n*.litertlm\n*.aab\n*.apks\n*.jks\n")
  write(out / "PACK_README.md", render("PACK_README.md", subs))
  write(out / "plan.json", json.dumps(strip_private(plan), indent=2, ensure_ascii=False) + "\n")
  return out


# ------------------------------------------------------------------------ fetch

def sha256_of(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def fetch(out, local, symlink):
  out = pathlib.Path(out)
  items = load_json(out / "litertlm_fetch.json")
  for it in items:
    dest = out / it["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = local.get(it["file"])
    if src:
      src = pathlib.Path(src).resolve()
      if dest.exists() or dest.is_symlink():
        dest.unlink()
      if symlink:
        dest.symlink_to(src)
      else:
        shutil.copyfile(src, dest)
      log(f"  {'linked' if symlink else 'copied'} {src} -> {dest}")
    else:
      url = f"{HF}/{it['repo']}/resolve/main/{it['file']}"
      log(f"  downloading {url}")
      # curl -C - resumes a multi-GB pull; urllib does not.
      subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "-o", str(dest), url], check=True)
    if it.get("sha256"):
      got = sha256_of(dest)
      if got != it["sha256"]:
        sys.exit(f"sha256 mismatch for {dest}: manifest {it['sha256']} != file {got}")
      log(f"  sha256 OK {dest.name}")
    size = dest.stat().st_size
    if it.get("size_bytes") and size != it["size_bytes"]:
      sys.exit(f"size mismatch for {dest}: manifest {it['size_bytes']} != file {size}")


# ------------------------------------------------------------------------ from-adb

def from_adb(serial):
  base = ["adb"] + (["-s", serial] if serial else [])
  def prop(name):
    return subprocess.run(base + ["shell", "getprop", name], capture_output=True, text=True).stdout.strip()
  mem = subprocess.run(base + ["shell", "grep", "MemTotal", "/proc/meminfo"], capture_output=True, text=True).stdout
  kb = int(re.search(r"(\d+)", mem).group(1)) if mem else None
  entry = {
      "id": sanitize(prop("ro.product.model") or "device", prefix="dev_").replace("_", "-"),
      "match": [re.escape(prop("ro.product.model"))],
      "brand": prop("ro.product.brand"),
      "device": prop("ro.product.device"),
      "soc_manufacturer": prop("ro.soc.manufacturer") or None,
      "soc_model": prop("ro.soc.model") or None,
      "ram_nominal_bytes": None,
      "_mem_total_kb": kb,
      "verified": f"adb getprop on the device, {datetime.date.today()}: ro.product.brand/device, ro.soc.*, "
                  f"MemTotal {kb} kB, Android {prop('ro.build.version.release')} (SDK {prop('ro.build.version.sdk')})",
  }
  print(json.dumps(entry, indent=2))
  log("Fill ram_nominal_bytes with the marketed RAM (e.g. 8000000000) and add a `match` pattern for the "
      "device string your manifests use in measured[].device.")


# ------------------------------------------------------------------------ main

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)

  def common(p):
    p.add_argument("manifests", nargs="+", help="litertlm_manifest.json path(s) or org/name repo id(s)")
    p.add_argument("--registry", help="device_classes.json (default: next to this script)")
    p.add_argument("--pack-name", help="override the AI pack name (letters/digits/underscore)")
    p.add_argument("--delivery", default="fast-follow", choices=["install-time", "fast-follow", "on-demand"])
    p.add_argument("--max-pack-bytes", type=int, help="override the 1.5 GB cap (policy)")
    p.add_argument("--floor-default", choices=["auto", "none"], help="override policy.floor_default")
    p.add_argument("--prefer-backend", choices=["cpu", "gpu", "npu"],
                   help="tie-break when two variants share a device_class")
    p.add_argument("--allow-unrecommended", action="store_true",
                   help="pack a CPU variant even when the manifest has no Android recommendation (flagged)")

  p = sub.add_parser("plan", help="print what would be packed and why the rest is not")
  common(p)
  p.add_argument("--json", action="store_true")

  p = sub.add_parser("generate", help="write the pack module (+ host project)")
  common(p)
  p.add_argument("--out", required=True)
  p.add_argument("--host-app", action="store_true", help="also write a complete minimal Gradle host project")
  p.add_argument("--application-id", default="com.example.litertlm.aipack")
  p.add_argument("--prompt", default="In one sentence, what is on-device inference?",
                 help="prompt the host app sends after loading the model")

  p = sub.add_parser("fetch", help="download the variant files listed in <out>/litertlm_fetch.json")
  p.add_argument("--out", required=True)
  p.add_argument("--local-file", action="append", default=[], metavar="NAME=PATH",
                 help="use a local copy for bundle NAME instead of downloading")
  p.add_argument("--symlink", action="store_true", help="symlink local files instead of copying")

  p = sub.add_parser("from-adb", help="print a device registry entry for an attached phone")
  p.add_argument("--serial")

  args = ap.parse_args()
  if args.cmd == "from-adb":
    return from_adb(args.serial)
  if args.cmd == "fetch":
    return fetch(args.out, dict(kv.split("=", 1) for kv in args.local_file), args.symlink)

  reg = load_registry(args.registry)
  plans = []
  for src in args.manifests:
    manifest, where = load_manifest(src)
    plan = plan_manifest(manifest, reg, args)
    plan["_manifest"] = manifest
    plan["_where"] = where
    plans.append(plan)

  if args.cmd == "plan":
    if args.json:
      print(json.dumps([strip_private(p) for p in plans], indent=2, ensure_ascii=False))
    else:
      for p in plans:
        print_plan(p)
        print()
    return

  if len(plans) != 1:
    sys.exit("generate takes exactly one manifest")
  plan = plans[0]
  print_plan(plan)
  out = generate(plan["_manifest"], plan, args.out, args)
  log(f"\nnext: {sys.argv[0]} fetch --out {out}   # then see {out}/PACK_README.md")


if __name__ == "__main__":
  main()
