#!/usr/bin/env python3
"""Finalize an exported MTP .litertlm: int8 + template + stops + ExecutorMetadata.

Steps (bundle-preserving unpack/pack via the litert-lm CLI, so the EMBEDDER
section survives — the minicpm5 rebuild path assumes a single tflite):
  1. unpack the float bundle;
  2. wi8fc-quantize BOTH tflites in place (prefill_decode linears+embedding,
     embedder table — decode and verify must read the SAME quantized table for
     greedy equivalence, which holding both in one bundle guarantees);
  3. replace the chat template with the minijinja-safe simple ChatML template
     (stock Qwen3.5 template violates the incremental-render prefix contract —
     same surgery as convert_qwen35_hybrid.py) and declare <|im_end|> (248046)
     as a stop token alongside <|endoftext|> (248044);
  4. pack, then append the ExecutorMetadata section (add_executor_metadata.py —
     classifies kv_cache_{lc,lr}_* and kv_cache_h_mtp as TYPE_LINEAR_ATTENTION,
     kv_cache_{k,v}_* incl. _mtp as global KV).

Usage:
  ~/venvs/lt094dev/bin/python3 mtp_work/finalize_mtp_bundle.py \
      mtp_work/out_08b/model.litertlm mtp_work/out_08b/Qwen3.5-0.8B_mtp_int8.litertlm
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CLI = os.path.expanduser("~/venvs/lt0160run/bin/litert-lm")
sys.path.insert(0, os.path.join(REPO, "minicpm5_work"))


def quantize_tflite(path, kind="wi8fc"):
    from quantize_minicpm5 import build_recipe
    from ai_edge_quantizer import quantizer

    qt = quantizer.Quantizer(path, build_recipe(kind))
    assert not qt.need_calibration
    res = qt.quantize()
    tmp = path + ".quant"
    res.export_model(tmp)
    print(f"  {os.path.basename(path)}: "
          f"{os.path.getsize(path)/1e6:.0f} -> {os.path.getsize(tmp)/1e6:.0f} MB")
    shutil.move(tmp, path)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    skip_quant = os.environ.get("MTP_SKIP_QUANT", "0") == "1"
    with tempfile.TemporaryDirectory(prefix="mtpfin_") as td:
        unpack = os.path.join(td, "unpack")
        subprocess.run([CLI, "unpack", src, "--output-dir", unpack], check=True,
                       capture_output=True, text=True)
        tflites = sorted(f for f in os.listdir(unpack) if f.endswith(".tflite"))
        print("sections:", sorted(os.listdir(unpack)))
        assert len(tflites) == 2, f"expected prefill_decode + embedder: {tflites}"
        if not skip_quant:
            for f in tflites:
                quantize_tflite(os.path.join(unpack, f))

        # template + stop tokens (== convert_qwen35_hybrid.py steps 3)
        template = open(os.path.join(
            REPO, "qwen35_work", "chat_template_simple.jinja")).read()
        esc = (template.replace("\\", "\\\\").replace('"', '\\"')
               .replace("\n", "\\n"))
        pbtext_path = os.path.join(unpack, "LlmMetadataProto.pbtext")
        pbtext = open(pbtext_path).read()
        pbtext_new = re.sub(r'jinja_prompt_template: ".*"',
                            lambda m: 'jinja_prompt_template: "' + esc + '"',
                            pbtext, count=1)
        assert pbtext_new != pbtext, "jinja_prompt_template not found in pbtext"
        if "ids: 248046" not in pbtext_new:
            eos_block = "stop_tokens {\n  token_ids {\n    ids: 248044\n  }\n}\n"
            assert eos_block in pbtext_new, "expected <|endoftext|> stop block"
            pbtext_new = pbtext_new.replace(
                eos_block,
                eos_block
                + "stop_tokens {\n  token_ids {\n    ids: 248046\n  }\n}\n",
                1)
        open(pbtext_path, "w").write(pbtext_new)

        packed = os.path.join(td, "packed.litertlm")
        toml = os.path.join(unpack, "model.toml")
        subprocess.run([CLI, "pack", toml, "--output", packed], check=True)

        meta_script = os.path.join(REPO, "scripts", "add_executor_metadata.py")
        if os.path.exists(dst):
            os.remove(dst)
        subprocess.run([sys.executable, meta_script, packed, dst,
                        "--litert-lm", CLI, "--python", sys.executable],
                       check=True)
    print("DONE:", dst, f"{os.path.getsize(dst)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
