#!/usr/bin/env python3
"""Round 2: check a built WeVisOne-VL-2B bundle against its build record, through the runtime's own tools
(litert-lm 0.18.1 `describe` and `unpack`): every TFLite section's sha256 = the input file, the tokenizer section =
the Hub's tokenizer.json bytes (sha256), the LlmMetadata (fast_vlm 512 x 512, max_num_tokens, the two stop tokens, no
start token, the jinja byte-equal to the build's), the ExecutorMetadata state buffers, the per-section
prefer_activation_type. Writes r2/bundle_check/<name>.json + .describe.txt; removes the unpack dir.

    ~/venvs/lt0100dev/bin/python -I tools/wv_bundle_check.py <name>
"""
import hashlib
import os
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
RUN = os.environ.get("LITERT_LM", "litert-lm")
TOKENIZER_SHA256 = "3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e"


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(name):
    bundle = K / "out" / "r2" / "bundle" / f"{name}.litertlm"
    build = json.loads((K / "out" / "r2" / "bundle" / f"{name}_noexec.build.json").read_text())
    out_dir = K / "r2" / "bundle_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = subprocess.run([str(RUN), "describe", str(bundle)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    (out_dir / f"{name}.describe.txt").write_text(d.stdout + "\n--- stderr\n" + d.stderr)
    up = K / "out" / "r2" / "tmp" / f"unpack_{name}"
    shutil.rmtree(up, ignore_errors=True)
    u = subprocess.run([str(RUN), "unpack", str(bundle), "--output-dir", str(up)], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    files = sorted(p.name for p in up.iterdir()) if up.exists() else []
    toml = (up / "model.toml").read_text() if (up / "model.toml").exists() else ""
    blocks = [b for b in toml.split("[[section]]")[1:]]
    sections = []
    for b in blocks:
        kv = dict(re.findall(r'^\s*(\w+)\s*=\s*"?([^"\n]*)"?', b, re.M))
        p = up / kv.get("data_path", "")
        sections.append({**kv, "sha256": sha256_file(p) if p.is_file() else None,
                         "bytes": p.stat().st_size if p.is_file() else None})
    want = {v["sha256"]: k for k, v in build["inputs"].items()}
    tfl = [s for s in sections if s.get("data_path", "").endswith(".tflite")]
    rec = {"bundle": str(bundle.relative_to(K)), "bytes": bundle.stat().st_size, "sha256": sha256_file(bundle),
           "describe_rc": d.returncode, "unpack_rc": u.returncode, "unpack_stderr_tail": u.stderr[-800:],
           "files": files, "sections": sections,
           "tflite_sections_match_inputs": sorted(want.get(s["sha256"], "MISSING") for s in tfl),
           "tflite_all_match": len(tfl) == 4 and all(s["sha256"] in want for s in tfl)}
    tok = [s for s in sections if "okenizer" in s.get("section_type", "")]
    rec["tokenizer_sections"] = []
    for s in tok:
        raw = (up / s["data_path"]).read_bytes()
        t = {"type": s.get("section_type"), "section_sha256": s["sha256"]}
        if s["data_path"].endswith(".zlib"):
            # builder layout: 8-byte little-endian uncompressed size + one zlib stream
            import zlib
            body = zlib.decompress(raw[8:])
            t["size_prefix_ok"] = int.from_bytes(raw[:8], "little") == len(body)
        else:
            body = raw
        t["sha256"] = hashlib.sha256(body).hexdigest()
        t["bytes"] = len(body)
        t["equals_hub"] = t["sha256"] == TOKENIZER_SHA256
        rec["tokenizer_sections"].append(t)
    md_txt = next((up / f for f in files if f.startswith("LlmMetadata") and f.endswith(".pbtext")), None)
    md = md_txt.read_text() if md_txt else ""
    jinja_m = re.search(r'jinja_prompt_template: "((?:[^"\\]|\\.)*)"', md)
    jinja = jinja_m.group(1).encode().decode("unicode_escape") if jinja_m else None
    rec["llm_metadata"] = {"max_num_tokens": re.findall(r"max_num_tokens: (\d+)", md),
                           "stop_ids": [int(x) for x in re.findall(r"ids: (\d+)", md)],
                           "has_start_token": "start_token" in md, "fast_vlm": "fast_vlm" in md,
                           "image_hw": re.findall(r"image_tensor_(?:height|width): (\d+)", md),
                           "jinja_equals_build": jinja == build["jinja"]}
    ex = next((up / f for f in files if f.startswith("ExecutorMetadata")), None)
    ex_txt = ex.read_text() if ex else ""
    rec["executor_metadata"] = {"present": bool(ex), "state_buffers": ex_txt.count("state_buffers"),
                                "key": ex_txt.count("TYPE_GLOBAL_KEY_CACHE"),
                                "value": ex_txt.count("TYPE_GLOBAL_VALUE_CACHE"),
                                "max_seq": sorted(set(re.findall(r"maximum_sequence_length: (\d+)", ex_txt)))}
    rec["ok"] = bool(rec["tflite_all_match"] and rec["tokenizer_sections"] and
                     all(t["equals_hub"] for t in rec["tokenizer_sections"]) and
                     rec["llm_metadata"]["jinja_equals_build"] and rec["llm_metadata"]["stop_ids"] == [151645, 151643]
                     and not rec["llm_metadata"]["has_start_token"] and rec["executor_metadata"]["state_buffers"] == 56)
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
    shutil.rmtree(up, ignore_errors=True)
    print("BUNDLE_CHECK", name, "OK" if rec["ok"] else "NG", json.dumps({k: rec[k] for k in (
        "tflite_all_match", "tokenizer_sections", "llm_metadata", "executor_metadata")}), flush=True)
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
