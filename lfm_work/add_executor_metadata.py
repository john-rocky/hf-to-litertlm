#!/usr/bin/env python3
"""Retrofit an ExecutorMetadata section onto an existing .litertlm (weights untouched).

litert-lm 0.15.0 changed how the runtime binds per-layer state buffers: for
state-carrying models (hybrid conv/attention like LFM2.5) the executor now reads
a dedicated `ExecutorMetadataProto` section (written by newer exporters together
with litert-lm-builder >= 0.15). Files exported before that lack the section and
fail at inference on 0.15 with

  NOT_FOUND: The given map is missing some output TensorBuffers

while plain attention-only models keep working through the legacy path. Adding
the section makes the SAME file run on both litert-lm 0.14 and 0.15 — weights
and graph stay byte-identical, so previous quality numbers carry over.

The state buffers are reconstructed from the tflite signatures, using the
exporter's naming convention:

  kv_cache_c_N  conv state (ShortConv / mamba conv)   -> TYPE_LINEAR_ATTENTION
  kv_cache_s_N  ssm recurrent state                   -> TYPE_LINEAR_ATTENTION
  kv_cache_k_N  attention K cache                     -> TYPE_GLOBAL_KEY_CACHE
  kv_cache_v_N  attention V cache                     -> TYPE_GLOBAL_VALUE_CACHE

K/V sequence_axis = index of the max-size dim; maximum_sequence_length = that dim.

Usage:
  python add_executor_metadata.py <in.litertlm> <out.litertlm> \
      [--litert-lm <litert-lm CLI>] [--python <python with ai_edge_litert>]

Requires: litert-lm >= 0.15 (for the `unpack`/`pack` CLI and the proto schema)
and ai-edge-litert (to read the tflite signatures).
"""
import argparse
import os
import subprocess
import sys
import tempfile


def read_state_buffers(py, tflite_path):
    code = f"""
import json
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path={tflite_path!r})
sigs = it.get_signature_list()
key = 'decode' if 'decode' in sigs else sorted(sigs)[0]
runner = it.get_signature_runner(key)
shapes = {{n: [int(x) for x in i['shape']] for n, i in runner.get_input_details().items()}}
names = sorted(n for n in sigs[key]['inputs'] if n.startswith('kv_cache_'))
print(json.dumps([[n, shapes[n]] for n in names]))
"""
    out = subprocess.run([py, "-c", code], check=True, capture_output=True, text=True)
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


def build_pbtext(buffers):
    blocks = []
    for name, shape in buffers:
        kind = name.split("_")[2]
        # c = conv, s = ssm recurrent (mc/mr: mamba-style variants some exporters emit)
        if kind in ("c", "s", "mc", "mr", "lc", "lr"):
            typ, extra = "TYPE_LINEAR_ATTENTION", ""
        elif kind in ("k", "v"):
            typ = "TYPE_GLOBAL_KEY_CACHE" if kind == "k" else "TYPE_GLOBAL_VALUE_CACHE"
            axis = shape.index(max(shape))
            extra = (f"\n    sequence_axis: {axis}"
                     f"\n    maximum_sequence_length: {max(shape)}")
        else:
            raise SystemExit(f"unknown state tensor kind for {name}")
        blocks.append(
            "  state_buffers {\n"
            f'    prefill_input_name: "{name}"\n'
            f'    prefill_output_name: "{name}"\n'
            f'    decode_input_name: "{name}"\n'
            f'    decode_output_name: "{name}"\n'
            f"    type: {typ}{extra}\n"
            "  }"
        )
    return ("llm_executor_metadata {\n  max_history_size: 0\n"
            + "\n".join(blocks) + "\n}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--litert-lm", default="litert-lm",
                    help="litert-lm CLI (needs the 0.15+ pack/unpack commands)")
    ap.add_argument("--python", default=sys.executable,
                    help="python with ai_edge_litert installed")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        unpack = os.path.join(td, "unpack")
        subprocess.run([args.litert_lm, "unpack", args.src, "--output-dir", unpack],
                       check=True, capture_output=True, text=True)
        toml_path = os.path.join(unpack, "model.toml")
        toml = open(toml_path).read()
        if "ExecutorMetadata" in toml:
            raise SystemExit("file already has an ExecutorMetadata section")
        tflite = [f for f in os.listdir(unpack) if f.endswith(".tflite")]
        if len(tflite) != 1:
            raise SystemExit(f"expected exactly one tflite section, got {tflite}")

        buffers = read_state_buffers(args.python, os.path.join(unpack, tflite[0]))
        if not buffers:
            raise SystemExit("no kv_cache_* state inputs found — nothing to do")
        with open(os.path.join(unpack, "ExecutorMetadataProto.pbtext"), "w") as f:
            f.write(build_pbtext(buffers))

        marker = 'data_path = "LlmMetadataProto.pbtext"\n'
        if marker not in toml:
            raise SystemExit("model.toml has no LlmMetadata section to anchor on")
        toml = toml.replace(marker, marker +
                            '\n[[section]]\nsection_type = "ExecutorMetadata"\n'
                            'data_path = "ExecutorMetadataProto.pbtext"\n')
        open(toml_path, "w").write(toml)

        # `litert-lm pack` exits 0 without writing when the output file already
        # exists — remove it first so a stale artifact can never masquerade as
        # the fresh repair.
        if os.path.exists(args.dst):
            os.remove(args.dst)
        subprocess.run([args.litert_lm, "pack", toml_path, "--output", args.dst],
                       check=True)
    n = len(buffers)
    print(f"OK: {args.dst} ({n} state buffers: "
          f"{sum(1 for b in buffers if b[0].split('_')[2] in ('c', 's', 'mc', 'mr', 'lc', 'lr'))} linear-attn, "
          f"{sum(1 for b in buffers if b[0].split('_')[2] in ('k', 'v'))} kv)")


if __name__ == "__main__":
    main()
