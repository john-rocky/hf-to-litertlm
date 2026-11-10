#!/usr/bin/env python3
"""make_pbtext.py [--out template/gptoss_litertlm.pbtext]: the LlmMetadata of the gpt-oss bundles, written through the
builder's own proto (text_format), so every field name and enum is checked.

- jinja_prompt_template = template/gptoss_litertlm.jinja (make_ship_template.py)
- stop tokens = generation_config.json's eos ids: 200002 <|return|>, 199999 <|endoftext|>, 200012 <|call|>
- no start token (the harmony prompt opens with <|start|>system; the template adds no BOS and neither may the runtime,
  or the prompt starts with a stray BOS)
- sampler = transformers' sampling defaults for this generation_config (do_sample, no temperature / top_p / top_k
  given): TOP_P, k 50, p 1.0, temperature 1.0
- llm_model_type generic_model (no gpt-oss type in LiteRT-LM dab2e708 or main 02d7e8d80c)
- max_num_tokens 4096 (the graph's KV cache)
- channels (LiteRT-LM keeps special tokens in the decoded text, support/tokenizer/
  tokenizer.h 76-77, and cuts channels by text, conversation/channel_util.cc 39-89), so the reply's content is the
  final message's text alone:
    analysis              <|channel|>analysis<|message|> ... <|end|>      (reasoning: reasoning_content, KV filter)
    final_header          <|start|>assistant<|channel|>final ... <|message|>
    commentary_header     <|start|>assistant<|channel|>commentary ... <|message|>  (tool call: recipient after channel)
    tool_header           <|start|>assistant to= ... <|message|>                    (tool call: recipient first)
    final_header_bare     <|channel|>final ... <|message|>                          (a final with no analysis before)
  No START is the head of another START, so the streaming parser never commits on a shorter marker
  (conversation/internal_callback_util.cc 170-210).
- min_runtime_version 0.19.0 (to be confirmed by a run on that release before an upload)
- supports_thinking left unset: in dab2e708 it is read only by the Python describe / model_info, and the template has
  no thinking switch (gpt-oss always writes an analysis message), so no app should offer a toggle.
"""
import argparse
import os

K = os.path.dirname(os.path.abspath(__file__))
CHANNELS = [
    ("analysis", "<|channel|>analysis<|message|>", "<|end|>", True),
    ("final_header", "<|start|>assistant<|channel|>final", "<|message|>", False),
    ("commentary_header", "<|start|>assistant<|channel|>commentary", "<|message|>", False),
    ("tool_header", "<|start|>assistant to=", "<|message|>", False),
    ("final_header_bare", "<|channel|>final", "<|message|>", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(K, "gptoss_litertlm.pbtext"))
    ap.add_argument("--template", default=os.path.join(K, "gptoss_litertlm.jinja"))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--min-runtime", default="0.19.0")
    a = ap.parse_args()
    from google.protobuf import text_format
    from litert_lm_builder.runtime.proto import llm_metadata_pb2

    starts = [c[1] for c in CHANNELS]
    for i, s in enumerate(starts):
        for j, t in enumerate(starts):
            assert i == j or not t.startswith(s), f"channel start {s!r} is the head of {t!r}"
    md = llm_metadata_pb2.LlmMetadata()
    for tid in (200002, 199999, 200012):
        md.stop_tokens.add().token_ids.ids.append(tid)
    sp = md.sampler_params
    sp.type = sp.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_name["TOP_P"].number
    sp.k, sp.p, sp.temperature = 50, 1.0, 1.0
    md.max_num_tokens = a.max_tokens
    md.llm_model_type.generic_model.SetInParent()
    md.jinja_prompt_template = open(a.template, encoding="utf-8").read()
    for name, start, end, reasoning in CHANNELS:
        c = md.channels.add()
        c.channel_name, c.start, c.end = name, start, end
        if reasoning:
            c.is_reasoning_channel = True
    md.min_runtime_version = a.min_runtime
    txt = ("# proto-file: runtime/proto/llm_metadata.proto\n# proto-message: LlmMetadata\n"
           "# Written by template/make_pbtext.py. Do not edit by hand.\n" + text_format.MessageToString(md))
    open(a.out, "w", encoding="utf-8").write(txt)
    back = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(txt, back)
    assert back == md
    print("wrote", a.out, len(txt), "B;", len(md.channels), "channels; stops", [s.token_ids.ids[0] for s in md.stop_tokens])


if __name__ == "__main__":
    main()
