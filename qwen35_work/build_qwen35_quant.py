#!/usr/bin/env python3
"""build_qwen35_quant.py int8|int4 FLOAT.litertlm OUTDIR NAME - the quantized file from a float Qwen3.5 export
(export_qwen35_float.py): steps 2-4 of convert_qwen35_hybrid.py (int8) or the blockwise-32 min-max int4 of make_int4_9b.py (int4),
then the fp32 GPU activation declaration, with two differences from those scripts:
  * int4 runs ../falcon_h1_work/quantize_clamped.py (= ../minicpm5_work/quantize_minicpm5.py with every blockwise int4 scale floored
    at 2^-14, so XNNPACK prepares every int4 FULLY_CONNECTED on the CPU); int8 runs quantize_minicpm5.py as the rail does;
  * stops: both 248046 (<|im_end|>) and 248044 (<|endoftext|>) end up in the LlmMetadata, whatever the export wrote (a
    generation_config.json that names both makes the export write both; convert_qwen35_hybrid.py asserts a 248044 block and adds
    only 248046). The export's own order is kept and a missing id is appended as its own stop_tokens block.
Recipes: int8 = wi8fc (channelwise int8 FC + embedding weights, float activations); int4 = wi4b32_wi8 minmax (blockwise-32 int4 FCs,
int8 channelwise lm_head + embedding); both with QWEN35_FLOAT_FC_REGEX (default 'Linear_in_proj_(a|b);' = the GatedDeltaNet
decay / gate projections float, 48 FCs in a 4B). Then: the rail's simple ChatML template (chat_template_simple.jinja; a later
metadata step replaces it), the ExecutorMetadata section (../scripts/add_executor_metadata.py with its default CLI, or
EXEC_META_LITERT_LM when set), prefer_activation_type fp32 (../scripts/set_activation_type.py, read back by the tool).
`litert-lm` on PATH = the CLI that unpacks / packs (litert-lm 0.18.0). Output OUTDIR/NAME_int8.litertlm or
NAME_mixed_int4.litertlm; OUTDIR/build_<mode>.json records each step. Intermediates are removed once the next step has read them;
TMPDIR should be a directory of this run only (quantize_minicpm5.py leaves an mc5q_* directory there, removed here)."""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "qwen35_work")
STOPS = (248046, 248044)


def drop(*paths):
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
        else:
            continue
        print("removed", p, flush=True)


def sh(cmd, rec, name):
    t0 = time.time()
    print("### CMD:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    rec["steps"][name] = {"s": round(time.time() - t0, 1), "cmd": cmd}


def ensure_stops(pbtext):
    blocks = list(re.finditer(r"stop_tokens \{\n(?:  .*\n)*?\}\n", pbtext))
    have = []
    for b in blocks:
        have += [int(x) for x in re.findall(r"ids: (\d+)", b.group(0))]
    added = [s for s in STOPS if s not in have]
    if added:
        at = blocks[-1].end() if blocks else None
        new = "".join("stop_tokens {\n  token_ids {\n    ids: %d\n  }\n}\n" % s for s in added)
        if at is None:
            raise SystemExit("no stop_tokens block in the export's LlmMetadata")
        pbtext = pbtext[:at] + new + pbtext[at:]
    return pbtext, have, added


def main():
    mode, fp, outdir, name = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]), sys.argv[4]
    assert mode in ("int8", "int4"), mode
    os.makedirs(outdir, exist_ok=True)
    rec = {"mode": mode, "float": fp, "start": time.strftime("%F %T %Z"), "steps": {},
           "float_fc_regex": os.environ.get("QWEN35_FLOAT_FC_REGEX", "Linear_in_proj_(a|b);")}
    regexes = [r for r in rec["float_fc_regex"].split(";;") if r]
    q = f"{outdir}/model_{mode}.litertlm"
    tmpl = f"{outdir}/model_{mode}_tmpl.litertlm"
    meta = f"{outdir}/model_{mode}_meta.litertlm"
    final = os.path.join(outdir, f"{name}_int8.litertlm" if mode == "int8" else f"{name}_mixed_int4.litertlm")
    if mode == "int8":
        quant = [sys.executable, os.path.join(ROOT, "minicpm5_work", "quantize_minicpm5.py"), "apply", fp, q, "--recipe", "wi8fc"]
    else:
        quant = [sys.executable, os.path.join(ROOT, "falcon_h1_work", "quantize_clamped.py"), "apply", fp, q,
                 "--recipe", "wi4b32_wi8", "--algo", "minmax"]
    for rx in regexes:
        quant += ["--float-fc-regex", rx]
    sh(quant, rec, "quantize")
    drop(*glob.glob(os.path.join(tempfile.gettempdir(), "mc5q_*")))
    unpack = f"{outdir}/{mode}_unpacked"
    sh(["litert-lm", "unpack", q, "--output-dir", unpack, "--allow-overwrite"], rec, "unpack")
    template = open(os.path.join(Q, "chat_template_simple.jinja")).read()
    esc = template.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    pp = os.path.join(unpack, "LlmMetadataProto.pbtext")
    pbtext = open(pp).read()
    new = re.sub(r'jinja_prompt_template: ".*"', lambda m: 'jinja_prompt_template: "' + esc + '"', pbtext, count=1)
    assert new != pbtext, "jinja_prompt_template not found in pbtext"
    new, had, added = ensure_stops(new)
    rec["stops_from_export"], rec["stops_added"] = had, added
    print("stops from the export:", had, "added:", added, flush=True)
    open(pp, "w").write(new)
    sh(["litert-lm", "pack", unpack, "--output", tmpl, "--allow-overwrite"], rec, "pack")
    drop(q, unpack)
    exec_meta = [sys.executable, os.path.join(ROOT, "scripts", "add_executor_metadata.py"), tmpl, meta]
    if os.environ.get("EXEC_META_LITERT_LM"):
        exec_meta += ["--litert-lm", os.environ["EXEC_META_LITERT_LM"]]
    sh(exec_meta, rec, "executor_metadata")
    drop(tmpl)
    sh([sys.executable, os.path.join(ROOT, "scripts", "set_activation_type.py"), meta, final, "--type", "fp32"], rec, "fp32act")
    drop(meta)
    h = hashlib.sha256()
    with open(final, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    rec.update(final=final, bytes=os.path.getsize(final), sha256=h.hexdigest(), end=time.strftime("%F %T %Z"))
    json.dump(rec, open(os.path.join(outdir, f"build_{mode}.json"), "w"), indent=1)
    print("DONE:", final, rec["bytes"], rec["sha256"], flush=True)


if __name__ == "__main__":
    main()
