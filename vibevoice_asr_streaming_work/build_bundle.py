#!/usr/bin/env python3
"""Pack VibeVoice-ASR-Streaming-1.5B into one .litertlm for LiteRT-LM's GENERIC audio path,
as a per-chunk MULTI-TURN conversation (the vendor's streaming protocol, see FINDINGS.md):

  render = <bare prompt> ( <|object_ref_start|> {26 audio embeddings} <|object_ref_end|>  <text>  <|text_chunk_end|> )*

  * no ChatML: every prompt_templates prefix/suffix is empty except model.suffix = <|text_chunk_end|>
  * one user turn = ONE encoder window (26 frames = 83 200 samples @ 24 kHz); the app advances
    22 frames per turn and keeps the 4-frame lookahead (the runtime's no-adapter path has stride ==
    window, so the overlap cannot live in the bundle)
  * stop tokens = <|text_chunk_end|> (151665), <|endoftext|> (151643)
  * the sampled stop token never enters the KV; the next turn's render appends model.suffix
    (= the vendor's forced <|text_chunk_end|>)

Sections: llm_metadata, HF tokenizer, EMBEDDER, PREFILL_DECODE, AUDIO_ENCODER_HW.
Usage (env): DEC=<unpack dir> ENC=<audio_encoder tflite> TOK=<tokenizer.json> OUT=<dir>
             [CACHE=2048] [OUT_NAME=...] [EMB=tf_lite_embedder] [DECODE=tf_lite_prefill_decode]
"""
import json
import os
import re

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

PROMPT = ("You are a helpful assistant that transcribes audio input into text output. "
          "Please transcribe the following audios streamingly with these keys: speaker, content\n")
AUDIO_PLACEHOLDER = "<|box_start|>"
BOA, EOA, TCE = "<|object_ref_start|>", "<|object_ref_end|>", "<|text_chunk_end|>"

# system turn (if any) replaces the vendor prompt verbatim (hotwords go there, as the vendor
# appends "and extra info: ..." to the prompt); a user turn renders its audio item as the
# placeholder and any text item verbatim; an assistant turn renders content + <|text_chunk_end|>.
JINJA = (
    "{%- if messages[0].role != 'system' %}{{ '" + PROMPT.replace("\n", "\\n") + "' }}{% endif -%}"
    "{%- for message in messages -%}"
    "{%- if message.role == 'system' -%}"
    "{%- if message.content is string %}{{ message.content }}{% else %}"
    "{%- for item in message.content %}{% if item.type == 'text' %}{{ item.text }}{% endif %}{% endfor %}{% endif -%}"
    "{%- elif message.role == 'user' -%}"
    "{%- if message.content is string %}{{ message.content }}{% else %}"
    "{%- for item in message.content %}{% if item.type == 'audio' %}{{ '" + AUDIO_PLACEHOLDER + "' }}"
    "{% elif item.type == 'text' %}{{ item.text }}{% endif %}{% endfor %}{% endif -%}"
    "{%- elif message.role == 'model' or message.role == 'assistant' -%}"
    "{%- if message.content is string %}{{ message.content }}{% else %}"
    "{%- for item in message.content %}{% if item.type == 'text' %}{{ item.text }}{% endif %}{% endfor %}{% endif -%}"
    "{{ '" + TCE + "' }}"
    "{%- endif -%}{%- endfor -%}"
)


def find_tflite(d, kw):
    cands = [f for f in os.listdir(d) if f.endswith(".tflite")]
    for f in cands:
        if kw in f.lower():
            return os.path.join(d, f)
    raise FileNotFoundError(f"no tflite matching {kw} in {d}: {cands}")


def main():
    dec, enc, tok_path = os.environ["DEC"], os.environ["ENC"], os.environ["TOK"]
    out = os.environ.get("OUT", os.path.join(os.path.dirname(enc), "bundle"))
    os.makedirs(out, exist_ok=True)
    embedder = find_tflite(dec, os.environ.get("EMB", "tf_lite_embedder"))
    prefill_decode = find_tflite(dec, os.environ.get("DECODE", "tf_lite_prefill_decode"))

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_path)
    ids = {t: tk.token_to_id(t) for t in ("<|endoftext|>", BOA, EOA, AUDIO_PLACEHOLDER, TCE)}
    assert ids == {"<|endoftext|>": 151643, BOA: 151646, EOA: 151647, AUDIO_PLACEHOLDER: 151648, TCE: 151665}, ids
    # The markers must tokenize as single specials (the runtime tokenizes start_of_audio_token /
    # audio_suffix / the suffix as text).
    for t in (BOA, EOA, TCE):
        assert tk.encode(t, add_special_tokens=False).ids == [ids[t]], (t, tk.encode(t).ids)
    print("special ids:", ids)

    from ai_edge_litert.interpreter import Interpreter
    sig = Interpreter(model_path=enc).get_signature_runner()
    inp, outp = sig.get_input_details(), sig.get_output_details()
    assert list(inp) == ["audio"] and list(outp) == ["features"], (inp, outp)
    t_frames, frame_len = int(inp["audio"]["shape"][1]), int(inp["audio"]["shape"][2])
    assert frame_len == 3200 and int(outp["features"]["shape"][1]) == t_frames, (inp, outp)
    print(f"encoder window: {t_frames} frames = {t_frames*frame_len/24000:.3f}s")

    md = llm_metadata_pb2.LlmMetadata()
    md.max_num_tokens = int(os.environ.get("CACHE", "2048"))
    md.prompt_templates.user.prefix = ""
    md.prompt_templates.user.suffix = ""
    md.prompt_templates.model.prefix = ""
    md.prompt_templates.model.suffix = TCE
    md.prompt_templates.system.prefix = ""
    md.prompt_templates.system.suffix = ""
    md.jinja_prompt_template = JINJA
    g = llm_model_type_pb2.GenericModel()
    g.audio_enabled = True
    g.delimiter_regex = r"(" + re.escape(AUDIO_PLACEHOLDER) + r")"
    g.audio_token_regex = re.escape(AUDIO_PLACEHOLDER)
    g.start_of_audio_token.token_str = BOA
    g.audio_suffix = EOA
    g.add_audio_end = False
    g.skip_mel_spectrogram_extraction = True
    g.audio_sample_rate_hz = 24000
    g.audio_num_channels = 1
    g.audio_frame_length = frame_len
    g.audio_hop_length = frame_len
    g.audio_input_scale = 1.0
    md.llm_model_type.generic_model.CopyFrom(g)
    md.stop_tokens.add().token_ids.ids.append(ids[TCE])
    md.stop_tokens.add().token_ids.ids.append(ids["<|endoftext|>"])
    md_path = os.path.join(out, "llm_metadata.pb")
    with open(md_path, "wb") as f:
        f.write(md.SerializeToString())

    b = litertlm_builder.LitertLmFileBuilder()
    b.add_system_metadata(litertlm_builder.Metadata(key="Authors", value="", dtype=litertlm_builder.DType.STRING))
    b.add_llm_metadata(md_path)
    b.add_hf_tokenizer(tok_path)
    b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
    b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE)
    b.add_tflite_model(enc, litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW)
    out_path = os.path.join(out, os.environ.get("OUT_NAME", "VibeVoice-ASR-Streaming-1.5B.litertlm"))
    with open(out_path, "wb") as f:
        b.build(f)
    print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")
    print(json.dumps({"embedder": embedder, "prefill_decode": prefill_decode, "audio_encoder": enc,
                      "window_frames": t_frames, "stop": [ids[TCE], ids["<|endoftext|>"]]}))


if __name__ == "__main__":
    main()
