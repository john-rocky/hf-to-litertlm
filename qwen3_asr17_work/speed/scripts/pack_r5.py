#!/usr/bin/env python3
"""Round 5 of the Qwen3-ASR-1.7B speed ladder: pack a form from an unpacked arm + an existing encoder tflite with
export_bundle_r3.pack() unchanged (metadata = build_metadata(): generic_model audio block + the litert jinja; litert-lm
0.17.1 pack; unpack check of every section). Only the module's two roots move: the pack dirs to out/r5/unpack/<name>
(+ <name>_check, removed after the check) and the record to speed/r5/pack_<name>.json.

export_bundle_r3.py has no --enc_dir: its --stages pack reads out/r3/enc/<enc_name>/enc__<recipe>.tflite and writes its
record under speed/r3/. This wrapper takes the encoder path itself and first asserts its sha256 (--expect_enc_sha256 =
F3's encoder section in speed/r3/forms.json), so the form's encoder is F3's encoder byte for byte.

  cd qwen3_asr17_work && PYTHONPATH=out/pyoverlay_builder017 ~/venvs/ltmain0918/bin/python speed/scripts/pack_r5.py \
      --src out/r5/unpack/F6_C_src --enc out/r3/enc/enc15/enc__fp16.tflite --name F6_win15_pb128_256 --kv 1024 \
      --expect_enc_sha256 3f6394ff85db33f3c2b6560c1fefb626d3b8e2610b21667ed4f92aa462fa1385 \
      --form_out speed/forms/F6_win15_pb128_256/model.litertlm
"""
import argparse
import json
import os
import sys
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import export_bundle_r3 as eb  # noqa: E402

eb.OUT = os.path.join(eb.HERE, "out", "r5")
eb.R3 = os.path.join(eb.K, "r5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unpacked arm (litert-lm unpack) whose LM / embedder / metadata are kept")
    ap.add_argument("--enc", required=True, help="encoder tflite that becomes the audio_encoder_hw section")
    ap.add_argument("--expect_enc_sha256", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kv", type=int, required=True)
    ap.add_argument("--prompt", choices=["official", "litert"], default="litert")
    ap.add_argument("--form_out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    got = eb.sha256(args.enc)
    print("ENC_SHA256", args.enc, os.path.getsize(args.enc), got, f"{time.time() - t0:.1f} s", flush=True)
    if got != args.expect_enc_sha256:
        print("STOP: encoder sha256 differs from --expect_enc_sha256", args.expect_enc_sha256, flush=True)
        sys.exit(3)
    os.makedirs(eb.R3, exist_ok=True)
    print("R5_PACK_ARGS", json.dumps(vars(args)), flush=True)
    eb.pack(args.src, args.enc, args.name, args.kv, args.prompt, args.form_out)


if __name__ == "__main__":
    main()
