#!/usr/bin/env python3
"""build_bundle.py --hf HF --export FLOAT_DIR --lm DIR --audio AUDIO.tflite --template JINJA --out FILE

Packs the LFM3-Audio-2B `.litertlm` file with litert-lm-builder (0.18.1), sections in this order:
  LlmMetadata       written here: max_num_tokens from the export's own llm_metadata.pb (2,048), start token 124894
                    (<|startoftext|>: the template renders bos_token "" and the runtime prepends the start token), stop
                    tokens 124900 <|im_end|> and 125014 <|text_end|> (generation_config.json eos_token_id), the
                    structured prompt templates present and empty, the chat template JINJA as is, the export's
                    channels (thought: <think> ... </think>) and its supports_* / min_runtime_version fields when set,
                    and llm_model_type generic_model with audio input: 16 kHz mono PCM in frames of 160 samples given to
                    the audio section as is (no mel spectrogram in the runtime), `<|audio|>` as the placeholder. In
                    LiteRT-LM 0.18.1 only generic_model carries audio input.
  ExecutorMetadata  the export's executor_metadata.pb (the decoder's state buffers), bytes as is
  HF tokenizer      HF/tokenizer.json, the source repository's bytes
  PREFILL_DECODE    DIR/decoder.tflite (embeddings in; logits and the final-norm `hidden` out)
  EMBEDDER          DIR/embedder.tflite (one token per call)
  AUDIO_ENCODER_HW  AUDIO.tflite (16 kHz PCM [1, 3000, 160] -> features [1, 375, 2048] + uint8 mask [1, 375])
Each section is read back after the build: the TFLite sections equal their inputs, the tokenizer section's JSON equals
HF/tokenizer.json, the LlmMetadata's template equals JINJA.
"""
import argparse
import hashlib
import json
import os
import struct
import zlib

BOS, IM_END, TEXT_END = 124894, 124900, 125014
AUDIO_RX = "(<\\|audio\\|>)"


def sha256_file(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def build_metadata(hf, stock_md, template_text):
  from litert_lm_builder.runtime.proto import llm_metadata_pb2, llm_model_type_pb2  # pylint: disable=g-import-not-at-top
  from tokenizers import Tokenizer  # pylint: disable=g-import-not-at-top
  tk = Tokenizer.from_file(os.path.join(hf, "tokenizer.json"))
  for tok, want in (("<|startoftext|>", BOS), ("<|im_end|>", IM_END), ("<|text_end|>", TEXT_END),
                    ("<|audio|>", 125031), ("<|audio_start|>", 125012), ("<|mixed_start|>", 125015)):
    assert tk.token_to_id(tok) == want, (tok, tk.token_to_id(tok))
  gen = json.load(open(os.path.join(hf, "generation_config.json")))
  assert gen["eos_token_id"] == [IM_END, TEXT_END], gen["eos_token_id"]
  md = llm_metadata_pb2.LlmMetadata()
  md.max_num_tokens = stock_md.max_num_tokens
  md.start_token.token_ids.ids.append(BOS)
  for t in (IM_END, TEXT_END):
    md.stop_tokens.add().token_ids.ids.append(t)
  for affix in (md.prompt_templates.user, md.prompt_templates.model, md.prompt_templates.system):
    affix.SetInParent()
  md.jinja_prompt_template = template_text
  for ch in stock_md.channels:
    md.channels.add().CopyFrom(ch)
  for f in ("supports_thinking", "supports_function_calling", "min_runtime_version"):
    if stock_md.HasField(f):
      setattr(md, f, getattr(stock_md, f))
  gm = llm_model_type_pb2.GenericModel(
      audio_enabled=True, delimiter_regex=AUDIO_RX, audio_token_regex="<\\|audio\\|>",
      skip_mel_spectrogram_extraction=True, audio_sample_rate_hz=16000, audio_num_channels=1,
      audio_frame_length=160, audio_hop_length=160, audio_input_scale=1.0, add_audio_end=False)
  md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(generic_model=gm))
  return md


def sections(path):
  """[(type name, begin, end)] from the file header (litert-lm-builder's flatbuffer schema)."""
  from litert_lm_builder import litertlm_header_schema_py_generated as H  # pylint: disable=g-import-not-at-top
  types = {v: k for k, v in vars(H.AnySectionDataType).items() if isinstance(v, int)}
  with open(path, "rb") as f:
    header_end = struct.unpack("<Q", f.read(32)[24:32])[0]
    f.seek(0)
    header = f.read(header_end)
  objs = H.LiteRTLMMetaData.GetRootAs(header, 32).SectionMetadata()
  return [(types.get(objs.Objects(i).DataType()), objs.Objects(i).BeginOffset(), objs.Objects(i).EndOffset())
          for i in range(objs.ObjectsLength())]


def read(path, begin, end):
  with open(path, "rb") as f:
    f.seek(begin)
    return f.read(end - begin)


def main():
  ap = argparse.ArgumentParser()
  for k in ("--hf", "--export", "--lm", "--audio", "--template", "--out"):
    ap.add_argument(k, required=True)
  a = ap.parse_args()
  import litert_lm_builder as lb  # pylint: disable=g-import-not-at-top
  from litert_lm_builder.runtime.proto import llm_metadata_pb2  # pylint: disable=g-import-not-at-top
  stock = llm_metadata_pb2.LlmMetadata()
  stock.ParseFromString(open(os.path.join(a.export, "llm_metadata.pb"), "rb").read())
  template_text = open(a.template).read()
  md = build_metadata(a.hf, stock, template_text)
  md_path = os.path.splitext(a.out)[0] + ".llm_metadata.pb"
  with open(md_path, "wb") as f:
    f.write(md.SerializeToString())
  exec_md = os.path.join(a.export, "executor_metadata.pb")
  tokenizer = os.path.join(a.hf, "tokenizer.json")
  inputs = {"PREFILL_DECODE": os.path.join(a.lm, "decoder.tflite"), "EMBEDDER": os.path.join(a.lm, "embedder.tflite"),
            "AUDIO_ENCODER_HW": a.audio}
  assert not os.path.exists(a.out), f"refusing to overwrite {a.out}"
  b = lb.LitertLmFileBuilder()
  b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
  b.add_llm_metadata(md_path)
  b.add_executor_metadata(exec_md)
  b.add_hf_tokenizer(tokenizer)
  for kind in ("PREFILL_DECODE", "EMBEDDER", "AUDIO_ENCODER_HW"):
    b.add_tflite_model(inputs[kind], getattr(lb.TfLiteModelType, kind))
  part = a.out + ".part"
  with open(part, "wb") as f:
    b.build(f)
  os.rename(part, a.out)
  os.remove(md_path)
  secs = sections(a.out)
  want = {"PREFILL_DECODE": sha256_file(inputs["PREFILL_DECODE"]), "EMBEDDER": sha256_file(inputs["EMBEDDER"]),
          "AUDIO_ENCODER_HW": sha256_file(inputs["AUDIO_ENCODER_HW"])}
  tfl = [s for s in secs if s[0] == "TFLiteModel"]
  ok = len(secs) == 6 and len(tfl) == 3
  for kind, (_, begin, end) in zip(("PREFILL_DECODE", "EMBEDDER", "AUDIO_ENCODER_HW"), tfl):
    got = hashlib.sha256(read(a.out, begin, end)).hexdigest()
    ok &= got == want[kind]
    print(f"{kind:17s} {end - begin:>13,} B sha256 {got} {'= input' if got == want[kind] else '!= input'}")
  for name, begin, end in secs:
    data = read(a.out, begin, end)
    if name == "HF_Tokenizer_Zlib":
      raw = zlib.decompress(data[8:])  # 8-byte little-endian size, then the zlib stream
      same = len(raw) == int.from_bytes(data[:8], "little") and hashlib.sha256(raw).hexdigest() == sha256_file(tokenizer)
      ok &= same
      print(f"HF tokenizer      JSON {len(raw):,} B {'= tokenizer.json' if same else '!= tokenizer.json'}")
    if name == "LlmMetadataProto":
      m = llm_metadata_pb2.LlmMetadata()
      m.ParseFromString(data)
      same = m.jinja_prompt_template == template_text
      ok &= same
      print(f"LlmMetadata       {len(data):,} B sha256 {hashlib.sha256(data).hexdigest()} "
            f"template {'= ' if same else '!= '}{os.path.basename(a.template)}")
  print(("PACKED " if ok else "PACK CHECK FAILED ") + f"{a.out} {os.path.getsize(a.out):,} B")
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
