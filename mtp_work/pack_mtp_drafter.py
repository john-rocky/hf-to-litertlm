#!/usr/bin/env python3
"""Pack the MTP drafter tflite into a P1 bundle as a tf_lite_mtp_drafter section.

The LiteRT-LM runtime loads the drafter from its own bundle section
(ModelType::kTfLiteMtpDrafter, string "tf_lite_mtp_drafter") and takes
signature 0; litert-lm-builder >= 0.16.0 knows the section type, so the CLI
unpack -> toml append -> pack round-trip is enough. Sanity checks: the drafter
must declare exactly the contract inputs, and its logits vocab must equal the
base verify signature's (the runtime sizes both greedy samplers from the
VERIFIER logits and samples the drafter buffer with that vocab).

Usage:
  ~/venvs/lt094dev/bin/python3 mtp_work/pack_mtp_drafter.py \
      mtp_work/out_08b/Qwen3.5-0.8B_mtp_int8.litertlm \
      mtp_work/out_08b/drafter_int8.tflite \
      mtp_work/out_08b/Qwen3.5-0.8B_mtp_drafter_int8.litertlm
"""
import os
import shutil
import subprocess
import sys
import tempfile

CLI = os.path.expanduser("~/venvs/lt0160run/bin/litert-lm")


def sig_dims(tflite_path, sig_key, tensor, kind="outputs"):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tflite_path)
    sigs = it.get_signature_list()
    assert sig_key in sigs, f"{sig_key} not in {sorted(sigs)}"
    runner = it.get_signature_runner(sig_key)
    details = (runner.get_output_details() if kind == "outputs"
               else runner.get_input_details())
    return [int(x) for x in details[tensor]["shape"]]


def main():
    src, drafter, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    with tempfile.TemporaryDirectory(prefix="mtpp2_") as td:
        unpack = os.path.join(td, "unpack")
        subprocess.run([CLI, "unpack", src, "--output-dir", unpack],
                       check=True, capture_output=True, text=True)
        base = [f for f in os.listdir(unpack) if "prefill_decode" in f]
        assert len(base) == 1, base

        v_logits = sig_dims(os.path.join(unpack, base[0]), "verify", "logits")
        d_logits = sig_dims(drafter, "draft", "logits")
        assert v_logits[-1] == d_logits[-1], (
            f"vocab mismatch: verify {v_logits} vs drafter {d_logits}")
        d_proj = sig_dims(drafter, "draft", "projected_activations")
        print(f"verify logits {v_logits}, drafter logits {d_logits}, "
              f"projected_activations {d_proj}")

        shutil.copy(drafter, os.path.join(unpack, "drafter.tflite"))
        with open(os.path.join(unpack, "model.toml"), "a") as f:
            f.write("\n[[section]]\n"
                    'model_type = "mtp_drafter"\n'
                    'section_type = "TFLiteModel"\n'
                    'data_path = "drafter.tflite"\n')
        if os.path.exists(dst):
            os.remove(dst)
        subprocess.run([CLI, "pack", os.path.join(unpack, "model.toml"),
                        "--output", dst], check=True)

    with tempfile.TemporaryDirectory(prefix="mtpv_") as td:
        u = os.path.join(td, "u")
        subprocess.run([CLI, "unpack", dst, "--output-dir", u],
                       check=True, capture_output=True, text=True)
        names = sorted(os.listdir(u))
        assert any("mtp_drafter" in n for n in names), names
        print("packed sections:", names)
    print("DONE:", dst, f"{os.path.getsize(dst)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
