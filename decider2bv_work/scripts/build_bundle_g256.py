"""Round 3: assemble the fp16 fast_vlm `.litertlm` for decider-2b-vision at G = 256 (litert-lm-builder 0.17.1,
out/venv-readout). Shape = qwen35vl_work/build_qwen35vl_bundle.py; prompt = decider_work's identity template with
one added branch for the image part.

  EMBEDDER       : out/fp16_r3/embedder_fp16.tflite         token_ids [1,1] -> [1,1,2048]
  PREFILL_DECODE : out/fp16_r3/decoder_fp16.tflite          embeddings + input_pos + mask + 48 states (-> logits)
                   prefer_activation_type = fp32
  VISION_ENCODER : out/fp16_r3/vision_encoder_fp16.tflite   image NHWC [1,256,256,3] in [0,1] -> [1,256,1024]
  VISION_ADAPTER : out/fp16_r3/vision_adapter_fp16.tflite   [1,256,1024] -> [1,64,2048]
  HF tokenizer   : the pinned snapshot's tokenizer.json
  LlmMetadata    : fast_vlm image 256 x 256, max_num_tokens 4096, NO start token, stop = [248044] only,
                   jinja = identity (no role prefix/suffix, nothing on add_generation_prompt; text parts verbatim;
                   image part -> <|vision_start|><image_soft_token><|vision_end|>, the runtime splits on
                   <image_soft_token> and injects the adapter's 64 embeddings there), plus STRUCTURED prompt_templates
                   present with empty user/model/system affixes (memory vlm-fastvlm-ride: jinja-only metadata gave a
                   null conversation).

Writes out/bundle/<name>_noexec.litertlm; ExecutorMetadata is appended afterwards by
../scripts/add_executor_metadata.py (run with TMPDIR under out/, see the round-3 notes).

    out/venv-readout/bin/python -B scripts/build_bundle_g256.py [--name decider-2b-vision_fp16]

Round 4: --decoder / --embedder / --vision-encoder / --vision-adapter / --out-dir point the same metadata at another
weight form (defaults = the round-3 fp16 files and out/bundle); everything else is unchanged.
"""
import argparse
import hashlib
import json
import os
import sys

import litert_lm_builder as litertlm_builder
from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, '../decider_work/scripts'))
from identity_template import IDENTITY                               # noqa: E402  (read-only import, python -B)

IMAGE_SIZE = 256
MAX_TOKENS = 4096
STOP_IDS = [248044]
IMG_RENDER = '<|vision_start|><image_soft_token><|vision_end|>'
TEXT_BRANCH = "{{ part.text }}{%- endif -%}"
IDENTITY_VLM = IDENTITY.replace(TEXT_BRANCH, "{{ part.text }}{%- elif part.type == 'image' -%}{{ '" + IMG_RENDER + "' }}{%- endif -%}")
assert IDENTITY.count(TEXT_BRANCH) == 1 and IDENTITY_VLM != IDENTITY
SNAPSHOT_TOKENIZER = 'out/src/decider-2b-vision/tokenizer.json'
SECTIONS = [('EMBEDDER', 'out/fp16_r3/embedder_fp16.tflite', None),
            ('PREFILL_DECODE', 'out/fp16_r3/decoder_fp16.tflite', 'fp32'),
            ('VISION_ENCODER', 'out/fp16_r3/vision_encoder_fp16.tflite', None),
            ('VISION_ADAPTER', 'out/fp16_r3/vision_adapter_fp16.tflite', None)]


def sha256_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metadata():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(ROOT, SNAPSHOT_TOKENIZER))
    assert tk.id_to_token(STOP_IDS[0]) == '<|endoftext|>', tk.id_to_token(STOP_IDS[0])
    for tok, want in (('<|vision_start|>', 248053), ('<|vision_end|>', 248054), ('<|image_pad|>', 248056)):
        assert tk.token_to_id(tok) == want, (tok, tk.token_to_id(tok))
    assert tk.token_to_id('<image_soft_token>') is None             # a runtime marker, never a vocabulary token
    md = llm_metadata_pb2.LlmMetadata()
    md.max_num_tokens = MAX_TOKENS
    for affix in (md.prompt_templates.user, md.prompt_templates.model, md.prompt_templates.system):
        affix.SetInParent()                                          # present, empty prefix / suffix
    md.jinja_prompt_template = IDENTITY_VLM
    md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
    md.llm_model_type.fast_vlm.image_tensor_height = IMAGE_SIZE
    md.llm_model_type.fast_vlm.image_tensor_width = IMAGE_SIZE
    for sid in STOP_IDS:
        md.stop_tokens.add().token_ids.ids.append(sid)
    assert not md.HasField('start_token')
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='decider-2b-vision_fp16')
    ap.add_argument('--out-dir', default='out/bundle')
    for kind, rel, _ in SECTIONS:
        ap.add_argument('--' + kind.lower().replace('prefill_decode', 'decoder').replace('_', '-'), default=rel)
    args = ap.parse_args()
    paths = {'EMBEDDER': args.embedder, 'PREFILL_DECODE': args.decoder, 'VISION_ENCODER': args.vision_encoder,
             'VISION_ADAPTER': args.vision_adapter}
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.name}_noexec.litertlm')
    assert not os.path.exists(out_path), f'refusing to overwrite {out_path}'
    md = metadata()
    md_path = os.path.join(out_dir, f'{args.name}.llm_metadata.pb')
    with open(md_path, 'wb') as f:
        f.write(md.SerializeToString())
    with open(os.path.join(out_dir, f'{args.name}.llm_metadata.pbtext'), 'w') as f:
        f.write(text_format.MessageToString(md))
    with open(os.path.join(out_dir, f'{args.name}.jinja'), 'w') as f:
        f.write(IDENTITY_VLM)

    b = litertlm_builder.LitertLmFileBuilder()
    b.add_system_metadata(litertlm_builder.Metadata(key='Authors', value='', dtype=litertlm_builder.DType.STRING))
    b.add_llm_metadata(md_path)
    b.add_hf_tokenizer(os.path.join(ROOT, SNAPSHOT_TOKENIZER))
    inputs = {}
    for kind, _, act in SECTIONS:
        rel = paths[kind]
        p = os.path.join(ROOT, rel)
        inputs[kind] = dict(path=rel, bytes=os.path.getsize(p), sha256=sha256_file(p), prefer_activation_type=act)
        b.add_tflite_model(p, getattr(litertlm_builder.TfLiteModelType, kind), prefer_activation_type=act)
    with open(out_path, 'wb') as f:
        b.build(f)
    rec = dict(status='DONE', output=os.path.relpath(out_path, ROOT), bytes=os.path.getsize(out_path), sha256=sha256_file(out_path),
               builder=__import__('importlib.metadata').metadata.version('litert-lm-builder'), inputs=inputs,
               tokenizer=dict(path=SNAPSHOT_TOKENIZER, sha256=sha256_file(os.path.join(ROOT, SNAPSHOT_TOKENIZER))),
               metadata_pbtext=text_format.MessageToString(md), jinja=IDENTITY_VLM,
               jinja_from='decider_work/scripts/identity_template.py IDENTITY + one image branch',
               identity_template_sha256=sha256_file(os.path.join(ROOT, '../decider_work/scripts/identity_template.py')))
    with open(os.path.join(out_dir, f'{args.name}_noexec.build.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    print('BUNDLE_BUILT', rec['output'], rec['bytes'], rec['sha256'], flush=True)


if __name__ == '__main__':
    main()
