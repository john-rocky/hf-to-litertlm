#!/usr/bin/env python3
"""build_bundle.py --hf HF_DIR --lm LM_DIR --audio AUDIO.tflite --vision-encoder ENC.tflite --vision-adapter ADP.tflite
--template T.jinja --out OUT.litertlm - LFM3-Omni-2B's .litertlm file (litert-lm-builder 0.18.1). Sections, in order:
  LlmMetadata       max_num_tokens 4096 (the KV cache), start token 124894 <|startoftext|> (the template renders an empty
                    bos_token and the runtime adds the start token once), stop tokens 124900 <|im_end|> and 125014
                    <|text_end|> (generation_config.json's eos_token_id), the user / model / system prompt templates
                    present and empty, the jinja template T.jinja as it is, and llm_model_type generic_model with
                    image_enabled and audio_enabled: delimiter_regex (<|image|>|<|audio|>), image_token_regex <|image|>,
                    image_tensor_height / width 512 (the runtime stretches every picture to 512 x 512), audio_token_regex
                    <|audio|>, 16 kHz mono PCM in frames of 160 samples handed to the audio section as they are
                    (skip_mel_spectrogram_extraction: the section computes the spectrogram), no end tokens added (the
                    template writes <|vision_start|> / <|vision_end|> itself). In LiteRT-LM 0.18.1 generic_model is the
                    only model type whose metadata declares audio input settings.
  ExecutorMetadata  the builder litert-torch's export runs (lfm3_ext's Lfm3Extension, emit_hidden), on the text_config
                    and a 4,096-token cache: the state buffers by layer type
  HF tokenizer      HF_DIR/tokenizer.json, byte for byte (checked after the build on the unpacked section)
  PREFILL_DECODE    LM_DIR/model.tflite      EMBEDDER          LM_DIR/embedder.tflite
  AUDIO_ENCODER_HW  AUDIO.tflite             VISION_ENCODER    ENC.tflite        VISION_ADAPTER    ADP.tflite
No activation type is declared for any section. lfm3_work must be on PYTHONPATH. After the build every section is read
back: each TFLite section must have its input file's sha256, the tokenizer section must unpack to tokenizer.json.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import zlib

R = pathlib.Path(__file__).resolve().parents[1]
CACHE = 4096
PREFILL = [1024, 256, 64, 16, 4, 1]
BOS, IM_END, TEXT_END = 124894, 124900, 125014
MEDIA_RX = "(<\\|image\\|>|<\\|audio\\|>)"
G = 512
TOKENIZER_SHA256 = "41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44"


def sha256_file(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def load_by_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def build_llm_metadata(hf, template):
  from litert_lm_builder.runtime.proto import llm_metadata_pb2, llm_model_type_pb2  # pylint: disable=g-import-not-at-top
  from tokenizers import Tokenizer  # pylint: disable=g-import-not-at-top
  tk = Tokenizer.from_file(str(hf / "tokenizer.json"))
  for tok, want in (("<|startoftext|>", BOS), ("<|im_end|>", IM_END), ("<|text_end|>", TEXT_END),
                    ("<|audio|>", 125031), ("<|vision_start|>", 125009), ("<|vision_end|>", 125010),
                    ("<|image|>", 125030)):
    assert tk.token_to_id(tok) == want, (tok, tk.token_to_id(tok))
  gen = json.load(open(hf / "generation_config.json"))
  assert gen["eos_token_id"] == [IM_END, TEXT_END], gen["eos_token_id"]
  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = CACHE
  md.start_token.token_ids.ids.append(BOS)
  for t in (IM_END, TEXT_END):
    md.stop_tokens.add().token_ids.ids.append(t)
  for affix in (md.prompt_templates.user, md.prompt_templates.model, md.prompt_templates.system):
    affix.SetInParent()
  md.jinja_prompt_template = template.read_text()
  gm = llm_model_type_pb2.GenericModel(
      audio_enabled=True, delimiter_regex=MEDIA_RX, audio_token_regex="<\\|audio\\|>",
      skip_mel_spectrogram_extraction=True, audio_sample_rate_hz=16000, audio_num_channels=1,
      audio_frame_length=160, audio_hop_length=160, audio_input_scale=1.0, add_audio_end=False)
  gm.image_enabled = True
  gm.image_token_regex = "<\\|image\\|>"
  gm.image_tensor_height = G
  gm.image_tensor_width = G
  gm.add_image_end = False
  md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(generic_model=gm))
  return md


def build_executor_metadata(hf):
  from litert_torch.generative.export_hf.core import export_lib  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.core import exportable_module_config  # pylint: disable=g-import-not-at-top
  from litert_torch.generative.export_hf.model_ext import metadata_builder  # pylint: disable=g-import-not-at-top
  from lfm3_ext import extension, loader  # pylint: disable=g-import-not-at-top
  config = loader.load_lfm3_config(str(hf), json.load(open(hf / "config.json"))["text_config"])
  artifacts = export_lib.SourceModelArtifacts(model=None, model_config=config, text_model_config=config,
                                              tokenizer=None)
  export_config = exportable_module_config.ExportableModuleConfig(
      model=str(hf), cache_length=CACHE, prefill_lengths=PREFILL, externalize_embedder=True,
      single_token_embedder=True, quantization_recipe="none", use_rope_composite=False, bundle_litert_lm=True,
      model_extension=extension.Lfm3Extension(emit_hidden=True))
  builder = metadata_builder.get_executor_metadata_builder(config, export_config)
  return builder(artifacts, export_config, None)


def main():
  ap = argparse.ArgumentParser()
  for k in ("--hf", "--lm", "--audio", "--vision-encoder", "--vision-adapter", "--template", "--out"):
    ap.add_argument(k, required=True)
  a = ap.parse_args()
  import litert_lm_builder as lb  # pylint: disable=g-import-not-at-top
  hf, lm, out = pathlib.Path(a.hf).resolve(), pathlib.Path(a.lm).resolve(), pathlib.Path(a.out).resolve()
  template = pathlib.Path(a.template).resolve()
  assert sha256_file(hf / "tokenizer.json") == TOKENIZER_SHA256, "tokenizer.json is not the source repository's file"
  kinds = [("PREFILL_DECODE", lm / "model.tflite"), ("EMBEDDER", lm / "embedder.tflite"),
           ("AUDIO_ENCODER_HW", pathlib.Path(a.audio).resolve()),
           ("VISION_ENCODER", pathlib.Path(a.vision_encoder).resolve()),
           ("VISION_ADAPTER", pathlib.Path(a.vision_adapter).resolve())]
  inputs = {}
  for kind, p in kinds:
    assert p.exists(), p
    inputs[kind] = sha256_file(p)
  assert not out.exists(), f"refusing to overwrite {out}"
  out.parent.mkdir(parents=True, exist_ok=True)
  md_path = out.with_name(out.stem + ".llm_metadata.pb")
  md_path.write_bytes(build_llm_metadata(hf, template).SerializeToString())
  em_path = out.with_name(out.stem + ".executor_metadata.pb")
  em_path.write_bytes(build_executor_metadata(hf).SerializeToString())
  b = lb.LitertLmFileBuilder()
  b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
  b.add_llm_metadata(str(md_path))
  b.add_executor_metadata(str(em_path))
  b.add_hf_tokenizer(str(hf / "tokenizer.json"))
  for kind, p in kinds:
    b.add_tflite_model(str(p), getattr(lb.TfLiteModelType, kind), prefer_activation_type=None)
  part = out.with_name(out.name + ".part")
  with open(part, "wb") as f:
    b.build(f)
  part.rename(out)
  # read back
  sec = load_by_path("bundle_sections", R / "nemotron_bolt_work" / "sections.py")
  rows = sec.sections(str(out))
  tfl = [r for r in rows if r[1] == "TFLiteModel"]
  ok = sorted(r[4] for r in tfl) == sorted(inputs.values()) and len(tfl) == len(inputs)
  tok = [r for r in rows if r[1] == "HF_Tokenizer_Zlib"]
  with open(out, "rb") as f:
    import struct  # pylint: disable=g-import-not-at-top
    from litert_lm_builder import litertlm_header_schema_py_generated as H  # pylint: disable=g-import-not-at-top
    header_end = struct.unpack("<Q", f.read(32)[24:32])[0]
    f.seek(0)
    objs = H.LiteRTLMMetaData.GetRootAs(f.read(header_end), 32).SectionMetadata()
    data = None
    for i in range(objs.ObjectsLength()):
      o = objs.Objects(i)
      if i == tok[0][0]:
        f.seek(o.BeginOffset())
        data = f.read(o.EndOffset() - o.BeginOffset())
  raw = zlib.decompress(data[8:])
  tok_ok = len(raw) == int.from_bytes(data[:8], "little") and hashlib.sha256(raw).hexdigest() == TOKENIZER_SHA256
  for i, t, m, n, h in rows:
    print(f"  {i} {t:<24} {m:<28} {n:>14,} B  sha256 {h}", flush=True)
  print(f"{out} {out.stat().st_size} B; TFLite sections = their input files: {'OK' if ok else 'BAD'}; tokenizer "
        f"section unpacks to tokenizer.json: {'OK' if tok_ok else 'BAD'}", flush=True)
  return 0 if ok and tok_ok else 1


if __name__ == "__main__":
  sys.exit(main())
