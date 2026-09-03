#!/usr/bin/env python3
"""Retrofit an ExecutorMetadata section onto an existing .litertlm (weights untouched).

Why: litert-lm 0.15.0's compiled-model executor binds per-layer state buffers via a
new ExecutorMetadataProto section (written by litert-torch main + litert-lm-builder
>= 0.15.0). Files exported before that (all shipped LFM2.5, all June hybrid probes)
lack the section and die at inference with
  NOT_FOUND: The given map is missing some output TensorBuffers
    (llm_litert_compiled_model_executor.cc:708)
on 0.15.0, while dense (pure kv k/v) models keep working via the legacy
GetKVCacheRootNames path. Adding the section makes the SAME file run on both 0.14.0
and 0.15.0 (verified 2026-08-04: LFM2.5-1.2B int4 8Q 7/8 == pre-repair baseline, and
it is what first made Granite-4.0-h-350m (Mamba2 hybrid) EXECUTE on a released
runtime — state names are bound as opaque TYPE_LINEAR_ATTENTION buffers).

State classification, from exporter naming conventions (litert-torch export_hf):
  kv_cache_c_N  conv state (ShortConv / mamba conv)   -> TYPE_LINEAR_ATTENTION
  kv_cache_s_N  ssm recurrent state (mamba2)          -> TYPE_LINEAR_ATTENTION
  kv_cache_k_N  attention K cache                     -> TYPE_GLOBAL_KEY_CACHE
  kv_cache_v_N  attention V cache                     -> TYPE_GLOBAL_VALUE_CACHE
K/V sequence_axis = index of the max-size dim (K: axis 2, V: axis 3 in current
exports); maximum_sequence_length = that dim.

Usage:
  add_executor_metadata.py <in.litertlm> <out.litertlm> [--litert-lm <cli>] [--python <py>]

--litert-lm: litert-lm CLI with pack/unpack (default: ~/venvs/lt0150run/bin/litert-lm)
--python:    python with ai_edge_litert installed, used to read tflite signatures
             (default: this interpreter)
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
        # c = conv (ShortConv/mamba), s = ssm recurrent, mc/mr = mamba conv/recurrent,
        # lc/lr = linear-attention (gated-delta-net) conv/recurrent,
        # h = MTP hidden ring (kv_cache_h_mtp; opaque position-ringed buffer)
        # (naming from the private C16 model_ext pytree prefixes)
        # kv_cache_{k,v}_mtp (the MTP drafter's teacher-forced KV) parse as
        # kind k/v and classify as global KV automatically.
        if kind in ("c", "s", "mc", "mr", "lc", "lr", "h"):
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
    ap.add_argument("--litert-lm",
                    default=os.path.expanduser("~/venvs/lt0150run/bin/litert-lm"))
    ap.add_argument("--python", default=sys.executable)
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
        if not tflite:
            raise SystemExit("no tflite section found")
        if len(tflite) > 1:
            # VLM bundles carry embedder/vision tflites alongside prefill_decode;
            # the state buffers live in the one with kv_cache_* inputs.
            with_state = []
            for f in tflite:
                try:
                    if read_state_buffers(args.python, os.path.join(unpack, f)):
                        with_state.append(f)
                except subprocess.CalledProcessError:
                    pass
            if len(with_state) != 1:
                raise SystemExit(
                    f"expected exactly one tflite with kv_cache_* inputs, got "
                    f"{with_state} out of {tflite}")
            tflite = with_state

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
          f"{sum(1 for b in buffers if b[0].split('_')[2] in ('c', 's', 'mc', 'mr', 'lc', 'lr', 'h'))} linear-attn, "
          f"{sum(1 for b in buffers if b[0].split('_')[2] in ('k', 'v'))} kv)")


if __name__ == "__main__":
    main()
