#!/usr/bin/env python3
"""Write LlmMetadataProto.pbtext for granite4_2 with jinja_prompt_template = chat_template.jinja (byte-exact)."""
import os
from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2

here = os.path.dirname(os.path.abspath(__file__))
tmpl = open(os.path.join(here, "chat_template.jinja")).read()
md = llm_metadata_pb2.LlmMetadata()
md.jinja_prompt_template = tmpl
jinja_line = text_format.MessageToString(md, as_utf8=True).rstrip("\n")
assert jinja_line.startswith("jinja_prompt_template: ")
body = f'''# proto-file: runtime/proto/llm_metadata.proto
# proto-message: LlmMetadataProto

stop_tokens {{
  token_str: "<|im_end|>"
}}
# generation_config.json of the vendor checkpoint: temperature 1.0, top_p 0.95 (no top_k).
sampler_params {{
  type: TOP_P
  k: 40
  p: 0.95
  temperature: 1.0
}}
max_num_tokens: 4096
llm_model_type {{
  generic_model {{
  }}
}}
# Granite 4.2 is a thinking model: the generation prompt pre-opens the thought
# channel (`<think>\\n`) unless enable_thinking is false, in which case the template
# emits the closed pair `<think></think>` and the model answers directly.
supports_thinking: true
{jinja_line}
channels {{
  channel_name: "thought"
  start: "<think>"
  end: "</think>"
}}
'''
open(os.path.join(here, "LlmMetadataProto.pbtext"), "w").write(body)
# round-trip check: parsing the pbtext must give back the template byte-exact
md2 = llm_metadata_pb2.LlmMetadata()
text_format.Parse(body, md2)
assert md2.jinja_prompt_template == tmpl, "pbtext round-trip mismatch"
print("LlmMetadataProto.pbtext written; jinja round-trips byte-exact;", len(tmpl), "chars")
