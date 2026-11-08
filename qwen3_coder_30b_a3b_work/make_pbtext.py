#!/usr/bin/env python3
"""make_pbtext.py — writes template/qwen3_coder_litertlm.pbtext, the LlmMetadata of the Qwen3-Coder-30B-A3B-Instruct
files, from:
  - the chat template template/qwen3_coder_litertlm.jinja (jinja_prompt_template);
  - tool_call_config copied field for field from LiteRT-LM's own models/qwen3_5 configuration at LiteRT-LM dab2e708
    (template/qwen3_5_official_dab2e708.pbtext: fence <tool_call> / </tool_call>, a regex for <function=...> and one
    for <parameter=...>, the XML call form of Qwen3-Coder's chat template); LiteRT-LM accepts a tool_call_config
    only with llm_model_type generic_model (runtime/conversation/model_data_processor/model_data_processor_factory.cc
    at dab2e708);
  - stop tokens = the source's generation_config.json eos_token_id [151645, 151643] = <|im_end|>, <|endoftext|>;
  - sampler = the source's generation_config.json (top_k 20, top_p 0.8, temperature 0.7; its repetition_penalty 1.05
    has no LlmMetadata field);
  - max_num_tokens 4096 (make_bundle.py --max-tokens rewrites it per file), supports_function_calling true,
    min_runtime_version "0.19.0" (the floor the qwen3_5 configuration gives tool_call_config).
Needs litert-lm-builder 0.19.0 or newer (its proto has tool_call_config). The output is parsed back and compared field
by field. Usage: python make_pbtext.py [OUT] (default template/qwen3_coder_litertlm.pbtext next to this script).
"""
import hashlib
import os
import sys

from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2

HERE = os.path.dirname(os.path.abspath(__file__))
jinja_path = os.path.join(HERE, "template/qwen3_coder_litertlm.jinja")
official = os.path.join(HERE, "template/qwen3_5_official_dab2e708.pbtext")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "template/qwen3_coder_litertlm.pbtext")

off = llm_metadata_pb2.LlmMetadata()
text_format.Parse(open(official).read(), off)

md = llm_metadata_pb2.LlmMetadata()
md.stop_tokens.add().token_str = "<|im_end|>"
md.stop_tokens.add().token_str = "<|endoftext|>"
md.sampler_params.type = md.sampler_params.TOP_P
md.sampler_params.k = 20
md.sampler_params.p = 0.8
md.sampler_params.temperature = 0.7
md.max_num_tokens = 4096
md.llm_model_type.generic_model.SetInParent()
md.supports_function_calling = True
md.tool_call_config.CopyFrom(off.tool_call_config)
md.min_runtime_version = "0.19.0"
md.jinja_prompt_template = open(jinja_path, encoding="utf-8").read()

head = ("# proto-file: runtime/proto/llm_metadata.proto\n# proto-message: LlmMetadataProto\n"
        "# Written by make_pbtext.py. Do not edit by hand.\n")
open(out, "w", encoding="utf-8").write(head + text_format.MessageToString(md, as_utf8=True))

back = llm_metadata_pb2.LlmMetadata()
text_format.Parse(open(out, encoding="utf-8").read(), back)
assert back == md
assert back.jinja_prompt_template == open(jinja_path, encoding="utf-8").read()
assert back.tool_call_config == off.tool_call_config
print("PBTEXT_DONE", out, hashlib.sha256(open(out, "rb").read()).hexdigest())
print("  serialized LlmMetadata sha256", hashlib.sha256(back.SerializeToString()).hexdigest())
print("  jinja sha256", hashlib.sha256(back.jinja_prompt_template.encode()).hexdigest())
print("  tool_call_config = the qwen3_5 configuration's:", back.tool_call_config == off.tool_call_config)
