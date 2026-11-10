#!/usr/bin/env python3
# Copied from wevisone_work/tools/build_wv_bundle.py (WeVisOne-VL-2B): the 4B snapshot (tools/wv_dims.py), default
# output out/bundle, the template desk check (--check-template) left out; the template, the metadata and the build
# are unchanged (the 4B snapshot's tokenizer.json is the same file as the 2B's).
"""Assemble the WeVisOne-VL-4B fast_vlm `.litertlm` (shape = decider2bv_work/scripts/build_bundle_g256.py,
template form = minicpmv47_work/build_mv47_bundle.py's dual jinja).

  EMBEDDER       : token_ids [1, 1] -> [1, 1, 10244]
  PREFILL_DECODE : embeddings [1, T, 10244] + input_pos + mask + 72 KV states (-> logits)   prefer_activation_type = arg
  VISION_ENCODER : image NHWC [1, 512, 512, 3] in [0, 1] -> [1, 1024, 4096]
  VISION_ADAPTER : [1, 1024, 4096] -> [1, 256, 10244]
  HF tokenizer   : the pinned snapshot's tokenizer.json, bytes as on the Hub (sha256 3bb17242...fc5e)
  LlmMetadata    : fast_vlm 512 x 512, max_num_tokens 2048 (= the export's KV cache), NO start token (bos_token null),
                   stop = <|im_end|> 151645 and <|endoftext|> 151643, STRUCTURED prompt_templates present and empty
                   (jinja-only metadata gives a null conversation on the fast_vlm path), the jinja below.

Template contract: the system turn is always "You are a helpful assistant." (a passed system message is
not rendered: the derived image positions start at P = 15); the image of the FIRST user turn, its first image item,
renders first as <|vision_start|><image_soft_token><|vision_end|> (the runtime splits on <image_soft_token> and
injects the adapter's 256 rows); any other image item is not rendered, and the runtime then refuses the request
("Provided more images than expected in the prompt", multimodal_processor_helper.cc v0.18.1) instead of placing it
at wrong positions; text items follow in order, concatenated (the checkpoint template's form); user content may be
a string or a list (0.18 content parts). Assistant turns close at <|im_end|> and the next turn opens with "\\n"
(the KV holds <|im_end|> as the pending token, not the "\\n").

    PY_EXPORT -I tools/build_wv_bundle.py --build --name NAME --decoder ... --embedder ... --encoder ... \
        --adapter ... --act fp32      (litert_lm_builder 0.18.1) -> <out-dir>/NAME_noexec.litertlm + record
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wv_dims", Path(__file__).resolve().parent / "wv_dims.py")
DIMS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DIMS)
SNAP = DIMS.SNAP
TOKENIZER = SNAP / "tokenizer.json"
TOKENIZER_SHA256 = DIMS.TOKENIZER_SHA256
IMAGE_SIZE = 512
MAX_TOKENS = 2048
STOP_IDS = [151645, 151643]
SOFT = "<image_soft_token>"
N_IMAGE = 256
IMAGE_PAD = 151655
SYSTEM = "You are a helpful assistant."


def _lit(s):
    assert "'" not in s and "\\" not in s, s
    return "{{- '" + s.replace("\n", "\\n") + "' -}}"


TEXT_PARTS = ("{%- if message['content'] is string -%}{{- message['content'] -}}"
              "{%- else -%}{%- for item in message['content'] -%}"
              "{%- if item['type'] == 'text' -%}{{- item['text'] -}}{%- endif -%}"
              "{%- endfor -%}{%- endif -%}")

JINJA = (
    "{%- set ns = namespace(nl=false, first_user=true) -%}"
    + _lit("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n") +
    "{%- for message in messages -%}"
    "{%- set role = message['role'] -%}"
    "{%- if role == 'user' or role == 'assistant' or role == 'model' -%}"
    "{%- if ns.nl -%}" + _lit("\n") + "{%- set ns.nl = false -%}{%- endif -%}"
    "{%- if role == 'user' -%}"
    + _lit("<|im_start|>user\n") +
    "{%- if ns.first_user and message['content'] is not string -%}"
    "{%- set img = namespace(done=false) -%}"
    "{%- for item in message['content'] -%}"
    "{%- if item['type'] == 'image' and not img.done -%}"
    + _lit("<|vision_start|>" + SOFT + "<|vision_end|>") +
    "{%- set img.done = true -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    + TEXT_PARTS + _lit("<|im_end|>\n") +
    "{%- set ns.first_user = false -%}"
    "{%- else -%}"
    + _lit("<|im_start|>assistant\n") + TEXT_PARTS + _lit("<|im_end|>") +
    "{%- set ns.nl = true -%}"
    "{%- endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{%- if ns.nl -%}" + _lit("\n") + "{%- endif -%}"
    + _lit("<|im_start|>assistant\n") +
    "{%- endif -%}"
)


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# ------------------------------------------------------------------------------------------------- build
def metadata(image_size=IMAGE_SIZE):
    from litert_lm_builder.runtime.proto import llm_metadata_pb2, llm_model_type_pb2
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOKENIZER))
    assert [tk.id_to_token(i) for i in STOP_IDS] == ["<|im_end|>", "<|endoftext|>"]
    for tok, want in (("<|vision_start|>", 151652), ("<|vision_end|>", 151653), ("<|image_pad|>", IMAGE_PAD)):
        assert tk.token_to_id(tok) == want, (tok, tk.token_to_id(tok))
    assert tk.token_to_id(SOFT) is None
    md = llm_metadata_pb2.LlmMetadata()
    md.max_num_tokens = MAX_TOKENS
    for affix in (md.prompt_templates.user, md.prompt_templates.model, md.prompt_templates.system):
        affix.SetInParent()
    md.jinja_prompt_template = JINJA
    md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
    md.llm_model_type.fast_vlm.image_tensor_height = image_size
    md.llm_model_type.fast_vlm.image_tensor_width = image_size
    for sid in STOP_IDS:
        md.stop_tokens.add().token_ids.ids.append(sid)
    assert not md.HasField("start_token")
    return md


def build(a):
    import importlib.metadata as imd
    import litert_lm_builder as lb
    from google.protobuf import text_format
    assert sha256_file(TOKENIZER) == TOKENIZER_SHA256
    out_dir = K / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.name}_noexec.litertlm"
    assert not out_path.exists(), f"refusing to overwrite {out_path}"
    md = metadata(a.image_size)
    md_path = out_dir / f"{a.name}.llm_metadata.pb"
    md_path.write_bytes(md.SerializeToString())
    (out_dir / f"{a.name}.llm_metadata.pbtext").write_text(text_format.MessageToString(md))
    (out_dir / f"{a.name}.jinja").write_text(JINJA)
    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_llm_metadata(str(md_path))
    b.add_hf_tokenizer(str(TOKENIZER))
    sections = [("EMBEDDER", a.embedder, None), ("PREFILL_DECODE", a.decoder, a.act or None),
                ("VISION_ENCODER", a.encoder, None), ("VISION_ADAPTER", a.adapter, None)]
    inputs = {}
    for kind, rel, act in sections:
        p = K / rel
        inputs[kind] = {"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p), "prefer_activation_type": act}
        b.add_tflite_model(str(p), getattr(lb.TfLiteModelType, kind), prefer_activation_type=act)
    with open(out_path, "wb") as f:
        b.build(f)
    rec = {"status": "DONE", "output": str(out_path.relative_to(K)), "bytes": out_path.stat().st_size,
           "sha256": sha256_file(out_path), "builder": imd.version("litert-lm-builder"), "inputs": inputs,
           "tokenizer": {"path": str(TOKENIZER), "sha256": TOKENIZER_SHA256},
           "metadata_pbtext": text_format.MessageToString(md), "jinja": JINJA}
    (out_dir / f"{a.name}_noexec.build.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("BUNDLE_BUILT", rec["output"], rec["bytes"], rec["sha256"], flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--out-dir", default="out/bundle")
    ap.add_argument("--decoder")
    ap.add_argument("--embedder")
    ap.add_argument("--encoder")
    ap.add_argument("--adapter")
    ap.add_argument("--act", default="", help="PREFILL_DECODE prefer_activation_type ('' = not declared)")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="fast_vlm image tensor side (512 or 768)")
    ap.add_argument("--print-jinja", action="store_true")
    a = ap.parse_args()
    if a.print_jinja:
        print(JINJA)
        return 0
    if a.build:
        return build(a)
    ap.error("one of --build / --print-jinja")


if __name__ == "__main__":
    sys.exit(main())
