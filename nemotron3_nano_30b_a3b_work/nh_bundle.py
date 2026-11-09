#!/usr/bin/env python3
"""nh_bundle.py --tflite MODEL.tflite --out OUT.litertlm --tokenizer tokenizer.json [--jinja T.jinja | --pbtext M.pbtext]
                [--act fp32|fp16|none] [--max-tokens N] [--thinking]

Packs the NemotronH writer's prefill/decode .tflite into a .litertlm: LlmMetadata (from --pbtext verbatim, or built here
in the form of the shipped litert-community/Nemotron-3-Nano-4B bundle, read over HTTP 2026-11-09 20:2x: generic model
type, no start token (add_bos_token false; the template opens with <|im_start|>), stop tokens = ids 2 and 11 and the
string "<|im_end|>\n", the channel "thought" from "<think>\n" to "\n</think>", max_num_tokens = the graph's cache length
unless --max-tokens, the template = --jinja (default template/nemotron3nano_history_guard.jinja = the 4B's embedded
template hf-to-litertlm templates/nemotron3nano_history.jinja f335297c... with its one messages[0] read guarded
(`messages and messages[0]...`; sha256 checked); --jinja none = no template), supports_thinking
only with --thinking (the 4B does not set it)), the HF tokenizer section = the repo's tokenizer.json bytes (sha256 checked against
the pinned Hub LFS sha256), an ExecutorMetadata section that declares every kv_cache_* state of the graph (mc / mr =
TYPE_LINEAR_ATTENTION, k / v = global key / value cache with the sequence axis and length read from the graph; the
form hf-to-litertlm scripts/add_executor_metadata.py writes for the Nemotron-H bundles), and the TFLiteModel section
with prefer_activation_type (fp32 by default: relu² written as the moe op's GELU overflows fp16, README).
Writes OUT.build.json (paths, sha256, the metadata text without the jinja, the state list).
Run with a python that has litert_lm_builder and ai_edge_litert (builder 0.19.0 here).
"""
import argparse
import hashlib
import json
import os
import sys

TOK_SHA = "c6021eb6847e682f89aa52d5eb6e8c7d902a23acfc8137e25211cf84828f1592"  # Hub LFS sha256 at bf77c317
JINJA_SHA = "f335297cdef3f420ed210cbf35a56dc9f974ff8926bb2c029c1a5a5f9571d281"  # the 4B bundle's embedded template
JINJA_GUARD_SHA = "0774fb2e2c6cc34c31d5217b9a07d30bb2bfeb9ae5a5511481ca5872a40d1a2b"  # + the messages[0] guard


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(64 << 20), b""):
            h.update(c)
    return h.hexdigest()


def state_buffers(tflite):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tflite)
    sigs = it.get_signature_list()
    r = it.get_signature_runner("decode")
    shapes = {n: [int(x) for x in i["shape"]] for n, i in r.get_input_details().items()}
    names = sorted(n for n in sigs["decode"]["inputs"] if n.startswith("kv_cache_"))
    return [(n, shapes[n]) for n in names], sorted(sigs)


def executor_pbtext(buffers):
    blocks = []
    for name, shape in buffers:
        kind = name.split("_")[2]
        if kind in ("mc", "mr"):
            typ, extra = "TYPE_LINEAR_ATTENTION", ""
        elif kind in ("k", "v"):
            typ = "TYPE_GLOBAL_KEY_CACHE" if kind == "k" else "TYPE_GLOBAL_VALUE_CACHE"
            axis = 2 if kind == "k" else 3
            extra = f"\n    sequence_axis: {axis}\n    maximum_sequence_length: {shape[axis]}"
        else:
            raise SystemExit(f"unknown state kind {name}")
        blocks.append("  state_buffers {\n"
                      f'    prefill_input_name: "{name}"\n    prefill_output_name: "{name}"\n'
                      f'    decode_input_name: "{name}"\n    decode_output_name: "{name}"\n'
                      f"    type: {typ}{extra}\n  }}")
    return "llm_executor_metadata {\n  max_history_size: 0\n" + "\n".join(blocks) + "\n}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--jinja", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "template", "nemotron3nano_history_guard.jinja"))
    ap.add_argument("--pbtext")
    ap.add_argument("--act", default="fp32")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--no-tok-check", action="store_true")
    a = ap.parse_args()
    import litert_lm_builder as lb
    from google.protobuf import text_format
    from litert_lm_builder.runtime.proto import llm_metadata_pb2

    tok_sha = sha256(os.path.realpath(a.tokenizer))
    if not a.no_tok_check:
        assert tok_sha == TOK_SHA, (tok_sha, TOK_SHA)
    buffers, sigs = state_buffers(a.tflite)
    cache = max([s[2] for n, s in buffers if n.startswith("kv_cache_k_")] or [0])
    md = llm_metadata_pb2.LlmMetadata()
    if a.pbtext:
        text_format.Parse(open(a.pbtext).read(), md)
    else:
        md.stop_tokens.add().token_ids.ids.append(2)
        md.stop_tokens.add().token_ids.ids.append(11)
        md.stop_tokens.add().token_str = "<|im_end|>\n"
        md.max_num_tokens = cache
        md.llm_model_type.generic_model.SetInParent()
        ch = md.channels.add()
        ch.channel_name, ch.start, ch.end = "thought", "<think>\n", "\n</think>"
        if a.jinja and a.jinja != "none":
            jt = open(a.jinja).read()
            want = {"nemotron3nano_history.jinja": JINJA_SHA, "nemotron3nano_history_guard.jinja": JINJA_GUARD_SHA}
            if os.path.basename(a.jinja) in want:
                assert hashlib.sha256(jt.encode()).hexdigest() == want[os.path.basename(a.jinja)], a.jinja
            md.jinja_prompt_template = jt
        if a.thinking:
            md.supports_thinking = True
    if a.max_tokens:
        md.max_num_tokens = a.max_tokens
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    md_path = out + ".llm_metadata.pb"
    open(md_path, "wb").write(md.SerializeToString())
    ex_path = out + ".executor_metadata.pbtext"
    open(ex_path, "w").write(executor_pbtext(buffers))
    shown = llm_metadata_pb2.LlmMetadata()
    shown.CopyFrom(md)
    shown.ClearField("jinja_prompt_template")
    md_text = text_format.MessageToString(shown) + f"jinja_prompt_template: <{len(md.jinja_prompt_template)} chars>\n"

    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_llm_metadata(md_path)
    b.add_executor_metadata(ex_path)
    b.add_hf_tokenizer(a.tokenizer)
    b.add_tflite_model(a.tflite, "prefill_decode", prefer_activation_type=(None if a.act == "none" else a.act))
    if os.path.exists(out):
        os.remove(out)
    with open(out, "wb") as f:
        b.build(f)
    rec = {"bundle": out, "bytes": os.path.getsize(out), "tflite": os.path.abspath(a.tflite),
           "tflite_sha256": sha256(a.tflite), "tokenizer": a.tokenizer, "tokenizer_sha256": tok_sha,
           "signatures": sigs, "states": buffers, "cache": cache, "prefer_activation_type": a.act,
           "llm_metadata_text": md_text,
           "jinja_sha256": hashlib.sha256(md.jinja_prompt_template.encode()).hexdigest(),
           "bundle_sha256": sha256(out)}
    json.dump(rec, open(out + ".build.json", "w"), indent=1)
    print("BUNDLE_DONE", out, rec["bytes"], "B", rec["bundle_sha256"][:16], "states", len(buffers), "cache", cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
