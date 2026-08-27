#!/usr/bin/env python3
"""Build a Mixed INT4 Qwen3.5-4B .litertlm from the float export.

Steps 2-4 of convert_qwen35_hybrid.py with the int4 recipe swapped in, plus the
fp32-activation declaration this family's GPU executor needs:

  float model.litertlm
    -> wi4b{32,128}_wi8 (int4 blockwise linears, int8 embedder + lm_head,
       minicpm5_work/quantize_minicpm5.py)
    -> simple ChatML template + <|im_end|> (248046) stop declaration
    -> ExecutorMetadata section (state-buffer binding, litert-lm >= 0.15)
    -> prefer_activation_type fp32 (scripts/set_activation_type.py)

    PATH=<venv-with-litert-lm-0.15>/bin:$PATH python3 make_int4_4b.py \
        <float model.litertlm> <outdir> <b32|b128>

The vocab table is ONE buffer shared by embedder and lm_head; if the head
regex in quantize_minicpm5.py ever misses, ai-edge-quantizer duplicates the
248320-row table once per signature (7x ~630 MB here). After building, ALWAYS
run `quantize_minicpm5.py inspect` and check the file size (~2.4 GB expected).
"""
import os
import re
import subprocess
import sys

fp = os.path.abspath(sys.argv[1])
outdir = os.path.abspath(sys.argv[2])
block = sys.argv[3]
# Optional 4th arg: int4 algorithm (minmax | octav). OCTAV clips outliers;
# on MiniCPM5 it recovered most of the min-max int4 GSM8K gap — model-dependent,
# so it is an A/B cell here, not a default.
algo = sys.argv[4] if len(sys.argv) > 4 else "minmax"
# "int8" builds the wi8fc control on the same fresh export — the A/B baseline
# for GSM8K (compares against the shipped int8 recipe without re-downloading it)
# and a rail-sanity control (must gate 8/8 like the shipped file).
assert block in ("b32", "b128", "int8"), block
assert algo in ("minmax", "octav"), algo
recipe = {"b32": "wi4b32_wi8", "b128": "wi4b128_wi8", "int8": "wi8fc"}[block]
name = {"b32": "Qwen3.5-4B_mixed_int4_b32", "b128": "Qwen3.5-4B_mixed_int4_b128",
        "int8": "Qwen3.5-4B_int8ctl"}[block]
if algo != "minmax":
    name += "_" + algo
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(outdir, exist_ok=True)

wi4 = os.path.join(outdir, f"model_{recipe}_{algo}.litertlm")
tmpl = os.path.join(outdir, f"model_{recipe}_{algo}_tmpl.litertlm")
meta = os.path.join(outdir, f"model_{recipe}_{algo}_meta.litertlm")
final = os.path.join(outdir, name + ".litertlm")

# 1. Post-hoc int4 blockwise on linears, int8 on embedder + lm_head.
subprocess.run([sys.executable,
                os.path.join(here, "..", "minicpm5_work", "quantize_minicpm5.py"),
                "apply", fp, wi4, "--recipe", recipe, "--algo", algo], check=True)

# 2. Simple ChatML template + declare <|im_end|> (248046) as a stop token
#    alongside config.json's <|endoftext|> (248044) — same as the int8 ship.
unpack_dir = os.path.join(outdir, f"{recipe}_unpacked")
subprocess.run(["litert-lm", "unpack", wi4, "--output-dir", unpack_dir,
                "--allow-overwrite"], check=True)
template = open(os.path.join(here, "chat_template_simple.jinja")).read()
esc = template.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
pbtext_path = os.path.join(unpack_dir, "LlmMetadataProto.pbtext")
pbtext = open(pbtext_path).read()
pbtext_new = re.sub(r'jinja_prompt_template: ".*"',
                    lambda m: 'jinja_prompt_template: "' + esc + '"',
                    pbtext, count=1)
assert pbtext_new != pbtext, "jinja_prompt_template not found in pbtext"
if "ids: 248046" not in pbtext_new:
    eos_block = "stop_tokens {\n  token_ids {\n    ids: 248044\n  }\n}\n"
    assert eos_block in pbtext_new, "expected <|endoftext|> stop_tokens block"
    pbtext_new = pbtext_new.replace(
        eos_block,
        eos_block + "stop_tokens {\n  token_ids {\n    ids: 248046\n  }\n}\n",
        1)
open(pbtext_path, "w").write(pbtext_new)
subprocess.run(["litert-lm", "pack", unpack_dir,
                "--output", tmpl, "--allow-overwrite"], check=True)

# 3. ExecutorMetadata section (litert-lm >= 0.15 binds the 48 state buffers).
meta_script = next(p for p in (
    os.path.join(here, "..", "scripts", "add_executor_metadata.py"),
    os.path.join(here, "..", "lfm_work", "add_executor_metadata.py"),
) if os.path.exists(p))
subprocess.run([sys.executable, meta_script, tmpl, meta], check=True)

# 4. fp32 activations (fp16 overflows on this family's real weights).
subprocess.run([sys.executable,
                os.path.join(here, "..", "scripts", "set_activation_type.py"),
                meta, final, "--type", "fp32"], check=True)

for p in (wi4, tmpl, meta):
    os.remove(p)
print("DONE:", final)
