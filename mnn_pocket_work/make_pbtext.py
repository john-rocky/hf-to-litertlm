#!/usr/bin/env python3
"""Write the LlmMetadata pbtext of MNN-Pocket-0.3B_int8.litertlm (passed to export_hf --litert_lm_llm_metadata_override).

  python mnn_pocket_work/make_pbtext.py --template templates/mnn_pocket_0_3b.jinja --out <LlmMetadataProto.pbtext>

Fields: no start_token (the source has no BOS: add_bos_token false, bos_token null); stop tokens as ids, <|im_end|> 65512
then <|endoftext|> 65510 (the order of generation_config.json's eos_token_id); the sampler of the source card's usage
example (top_k 20, top_p 0.8, temperature 0.8) as TOP_P; max_num_tokens 4096 (= the exported KV cache); model type
qwen2p5; the chat template text verbatim from --template; no channels (the model has no thinking block); no pad_token.
"""
import argparse

from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2

ap = argparse.ArgumentParser()
ap.add_argument("--template", required=True)
ap.add_argument("--model-type", default="qwen2p5", choices=["qwen2p5", "qwen3", "generic_model"])
ap.add_argument("--max-num-tokens", type=int, default=4096)
ap.add_argument("--out", required=True)
args = ap.parse_args()

md = llm_metadata_pb2.LlmMetadata()
for tid in (65512, 65510):
  md.stop_tokens.add().token_ids.ids.append(tid)
sp = md.sampler_params
sp.type = sp.Type.Value("TOP_P")
sp.k = 20
sp.p = 0.8
sp.temperature = 0.8
md.max_num_tokens = args.max_num_tokens
getattr(md.llm_model_type, args.model_type).SetInParent()
with open(args.template, encoding="utf-8") as f:
  md.jinja_prompt_template = f.read()
txt = ("# proto-file: runtime/proto/llm_metadata.proto\n# proto-message: LlmMetadataProto\n\n"
       + text_format.MessageToString(md, as_utf8=True))
# round trip: the written text parses back to the same message
back = llm_metadata_pb2.LlmMetadata()
text_format.Parse(txt, back)
assert back == md, "pbtext round trip differs"
with open(args.out, "w", encoding="utf-8") as f:
  f.write(txt)
print(f"wrote {args.out}: type={args.model_type} stops=[65512, 65510] max_num_tokens={args.max_num_tokens} "
      f"template={len(md.jinja_prompt_template)} chars")
