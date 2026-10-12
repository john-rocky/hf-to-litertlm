#!/usr/bin/env python3
"""Build the 'workaround' LlmMetadataProto.pbtext for one unpacked bundle (the claim under test: llm_model_type
qwen3 + a JSON tool-call instruction makes the runtime parse the model's tool calls).

Start from the bundle's own pbtext (stop tokens, max_num_tokens untouched); change exactly three fields:
  llm_model_type  generic_model {} -> qwen3 {}
  jinja_prompt_template -> LiteRT-LM models/qwen3/chat_template.jinja (JSON within <tool_call> instruction)
  channels -> the thought channel of models/qwen3/LlmMetadataProto.pbtext (so thinking text is separable)
Run with: uv run --no-project --with litert-lm==0.18.0 python -I make_workaround_pbtext.py IN.pbtext TEMPLATE.jinja OUT.pbtext
"""
import sys
from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2

src, tpl, out = sys.argv[1:4]
msg = llm_metadata_pb2.LlmMetadata()
text_format.Parse(open(src, encoding="utf-8").read(), msg)
assert msg.llm_model_type.HasField("generic_model"), msg.llm_model_type
msg.llm_model_type.Clear()
msg.llm_model_type.qwen3.SetInParent()
msg.jinja_prompt_template = open(tpl, encoding="utf-8").read()
del msg.channels[:]
ch = msg.channels.add()
ch.channel_name = "thought"
ch.start = "<think>\n"
ch.end = "\n</think>"
open(out, "w", encoding="utf-8").write(text_format.MessageToString(msg, as_utf8=True))
print("wrote", out, "model_type:", msg.llm_model_type.WhichOneof("model_type"), "template bytes:", len(msg.jinja_prompt_template),
      "stop_tokens:", len(msg.stop_tokens), "max_num_tokens:", msg.max_num_tokens)
