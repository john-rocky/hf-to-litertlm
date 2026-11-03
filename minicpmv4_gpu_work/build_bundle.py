#!/usr/bin/env python3
"""build_bundle.py --decode DEC.tflite --emb EMB.tflite --venc VENC.tflite --tok tokenizer.json --out OUT.litertlm
                   [--max-tokens 4096] [--dec-act fp32|fp16] [--venc-act fp32|fp16]

Assembles the MiniCPM-V-4 .litertlm for LiteRT-LM's `minicpmv4` type (the runtime slices the image and calls the fused
vision encoder once per slice; minicpmv_data_processor.cc v0.19.0):
  EMBEDDER        token ids -> [.., 2560]
  PREFILL_DECODE  embeddings + 32 K / V caches -> logits (KV 4096)
  VISION_ENCODER  images [1,1088,588] + positions_xy + vit_positions -> features [1,64,2560] (v4_vision.py)
  LlmMetadata     minicpmv4; start <s>; stops 2 (</s>) and 73440 (<|im_end|>) = the vendor's terminators and
                  config eos; the published file's prompt_templates; jinja = chat_template_v4.jinja (the vendor's
                  ChatML template, with a message's items joined by "\n" as model.chat joins them; an image item
                  renders as <image_soft_token>, which the runtime replaces with the slice layout)
No backend constraint on any section. The tokenizer section is the repo's tokenizer.json verbatim. The ExecutorMetadata section and the external weights come after (externalize_repo.sh
--executor-metadata auto)."""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--venc", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--dec-act")
    ap.add_argument("--venc-act")
    a = ap.parse_args()
    import litert_lm_builder as lb
    from litert_lm_builder.runtime.proto import llm_metadata_pb2, llm_model_type_pb2
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(a.tok)
    ids = {t: tk.token_to_id(t) for t in ("<s>", "</s>", "<|im_end|>", "<|im_start|>", "<image>", "</image>",
                                           "<image_id>", "</image_id>", "<slice>", "</slice>", "<unk>")}
    assert ids["<s>"] == 1 and ids["</s>"] == 2 and ids["<|im_end|>"] == 73440, ids
    assert all(v is not None for v in ids.values()), ids
    assert tk.token_to_id("<image_soft_token>") is None
    jinja = open(os.path.join(HERE, "chat_template_v4.jinja")).read()
    md = llm_metadata_pb2.LlmMetadata()
    md.start_token.token_str = "<s>"
    md.stop_tokens.add().token_ids.ids.append(2)
    md.stop_tokens.add().token_ids.ids.append(73440)
    md.prompt_templates.user.prefix = "<|im_start|>user\n"
    md.prompt_templates.user.suffix = "<|im_end|>\n"
    md.prompt_templates.model.prefix = "<|im_start|>assistant\n"
    md.prompt_templates.model.suffix = "<|im_end|>\n"
    md.prompt_templates.system.prefix = "<|im_start|>system\n"
    md.prompt_templates.system.suffix = "<|im_end|>\n"
    md.max_num_tokens = a.max_tokens
    md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(minicpmv4=llm_model_type_pb2.MiniCpmV4()))
    md.jinja_prompt_template = jinja
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    md_path = a.out + ".llm_metadata.pb"
    with open(md_path, "wb") as f:
        f.write(md.SerializeToString())
    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_llm_metadata(md_path)
    b.add_hf_tokenizer(a.tok)
    b.add_tflite_model(a.decode, "prefill_decode", prefer_activation_type=a.dec_act or None)
    b.add_tflite_model(a.emb, "embedder")
    b.add_tflite_model(a.venc, "vision_encoder", prefer_activation_type=a.venc_act or None)
    with open(a.out, "wb") as f:
        b.build(f)
    rec = {"out": a.out, "bytes": os.path.getsize(a.out), "decode": a.decode, "emb": a.emb, "venc": a.venc,
           "dec_act": a.dec_act, "venc_act": a.venc_act, "max_tokens": a.max_tokens,
           "jinja_sha256": hashlib.sha256(jinja.encode()).hexdigest()}
    print("BUNDLE_DONE", json.dumps(rec))
    json.dump(rec, open(a.out + ".build.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
