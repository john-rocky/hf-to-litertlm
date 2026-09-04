#!/usr/bin/env python3
"""Pack VibeVoice-ASR-BitNet into one .litertlm for LiteRT-LM's GENERIC audio path:

  sections: llm_metadata (GenericModel + audio config), HF tokenizer, EMBEDDER (int8),
            PREFILL_DECODE (int4 blockwise-128 ternary LM), AUDIO_ENCODER_HW (single
            signature `audio`[1,T,3200] -> `features`[1,T,1536]).

Runtime contract this relies on (LiteRT-LM v0.16.x, verified in source):
  * GenericModel.audio_enabled + skip_mel_spectrogram_extraction: the miniaudio
    preprocessor decodes the file, resamples to audio_sample_rate_hz (24 kHz), and frames
    raw PCM into [1, n_frames, frame_length] with hop_length — no mel, no input_scale.
  * The rendered prompt is split by delimiter_regex; a part matching audio_token_regex
    becomes: InputText(text_before + audio_prefix + start_of_audio_token) → InputAudio →
    InputText(audio_suffix).  So the placeholder <|box_start|> expands to
    <|object_ref_start|> {N audio embeddings} <|object_ref_end|>, which is VibeASR.cpp's
    layout (speech_start / speech_pad*N / speech_end).
  * Without an audio adapter the executor treats the encoder as "streaming" with
    window == T frames, buffering disabled by default → short clips are zero-padded,
    valid token count = ceil(valid_frames / shrink_factor) with shrink_factor == 1.

Usage (env): DEC=<export dir with prefill_decode_*.tflite + embedder_*.tflite>
             ENC=<audio_encoder tflite>  TOK=<tokenizer.json>  OUT=<dir>  [CACHE=2048]
             [DEFAULT_INSTR="Please transcribe it."]  [OUT_NAME=...]
"""
import json
import os
import re

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

SYSTEM_PROMPT = ("You are a helpful assistant that transcribes audio input into text output "
                 "in JSON format.")
AUDIO_PLACEHOLDER = "<|box_start|>"
BOA, EOA = "<|object_ref_start|>", "<|object_ref_end|>"
U_PRE, U_SUF = "<|im_start|>user\n", "<|im_end|>\n"
M_PRE, M_SUF = "<|im_start|>assistant\n", "<|im_end|>\n"
S_PRE, S_SUF = "<|im_start|>system\n", "<|im_end|>\n"
DEFAULT_INSTR = os.environ.get("DEFAULT_INSTR", "Please transcribe it.")

# String and list content both handled.  The vendor prompt always carries the system
# turn, so it is emitted when the conversation does not start with one.  An audio item
# renders as the placeholder; the newline that VibeASR.cpp puts between <|speech_end|>
# and the instruction is emitted by the template so the app's text item is the bare
# instruction ("This is a 5.86 seconds audio, please transcribe it.").  A user turn with
# audio but no text gets DEFAULT_INSTR.
JINJA = (
    "{%- if messages[0].role != 'system' %}" + S_PRE + SYSTEM_PROMPT + S_SUF + "{% endif -%}"
    "{%- for message in messages -%}"
    "{%- if message.content is string -%}"
    "{%- if message.role == 'user' %}" + U_PRE + "{{ message.content }}" + U_SUF + "{% endif -%}"
    "{%- if message.role == 'model' or message.role == 'assistant' %}" + M_PRE + "{{ message.content }}" + M_SUF + "{% endif -%}"
    "{%- if message.role == 'system' %}" + S_PRE + "{{ message.content }}" + S_SUF + "{% endif -%}"
    "{%- else -%}"
    "{%- if message.role == 'user' %}" + U_PRE +
    "{% elif message.role == 'model' or message.role == 'assistant' %}" + M_PRE +
    "{% elif message.role == 'system' %}" + S_PRE + "{% endif -%}"
    "{%- set ns = namespace(has_audio=false, has_text=false) -%}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'audio' -%}{{ '" + AUDIO_PLACEHOLDER + "\\n' }}{%- set ns.has_audio = true -%}"
    "{%- elif item.type == 'text' -%}{{ item.text }}{%- set ns.has_text = true -%}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if ns.has_audio and not ns.has_text %}" + DEFAULT_INSTR + "{% endif -%}"
    "{%- if message.role == 'user' %}" + U_SUF +
    "{% elif message.role == 'model' or message.role == 'assistant' %}" + M_SUF +
    "{% elif message.role == 'system' %}" + S_SUF + "{% endif -%}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt %}" + M_PRE + "{% endif -%}"
)


def find_tflite(d, *keywords):
    cands = [f for f in os.listdir(d) if f.endswith(".tflite")]
    for kw in keywords:
        for f in cands:
            if kw in f.lower():
                return os.path.join(d, f)
    raise FileNotFoundError(f"no tflite matching {keywords} in {d}: {cands}")


def main():
    dec = os.environ["DEC"]
    enc = os.environ["ENC"]
    tok_path = os.environ["TOK"]
    out = os.environ.get("OUT", os.path.join(os.path.dirname(enc), "..", "bundle"))
    os.makedirs(out, exist_ok=True)
    embedder = find_tflite(dec, os.environ.get("EMB", "embedder_wi8"))
    prefill_decode = find_tflite(dec, os.environ.get("DECODE", "prefill_decode_"))
    assert os.path.exists(enc), enc

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_path)
    ids = {t: tk.token_to_id(t) for t in ("<|endoftext|>", "<|im_end|>", "<|im_start|>", BOA, EOA, AUDIO_PLACEHOLDER)}
    assert all(v is not None for v in ids.values()), ids
    assert ids[BOA] == 151646 and ids[EOA] == 151647 and ids[AUDIO_PLACEHOLDER] == 151648, ids
    print("special ids:", ids)

    # Encoder window from the tflite itself (never typed in).
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=enc)
    sig = it.get_signature_runner()
    inp = sig.get_input_details()
    outp = sig.get_output_details()
    assert list(inp) == ["audio"] and list(outp) == ["features"], (inp, outp)
    t_frames, frame_len = int(inp["audio"]["shape"][1]), int(inp["audio"]["shape"][2])
    assert frame_len == 3200 and int(outp["features"]["shape"][1]) == t_frames, (inp, outp)
    print(f"encoder window: {t_frames} frames = {t_frames*frame_len/24000:.1f}s")

    md = llm_metadata_pb2.LlmMetadata()
    md.max_num_tokens = int(os.environ.get("CACHE", "2048"))
    md.prompt_templates.user.prefix = U_PRE
    md.prompt_templates.user.suffix = U_SUF
    md.prompt_templates.model.prefix = M_PRE
    md.prompt_templates.model.suffix = M_SUF
    md.prompt_templates.system.prefix = S_PRE
    md.prompt_templates.system.suffix = S_SUF
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
    md.stop_tokens.add().token_ids.ids.append(ids["<|im_end|>"])
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
    out_path = os.path.join(out, os.environ.get("OUT_NAME", "VibeVoice-ASR-BitNet.litertlm"))
    with open(out_path, "wb") as f:
        b.build(f)
    print("BUNDLE_DONE", out_path, round(os.path.getsize(out_path) / 1e6, 1), "MB")
    print(json.dumps({"embedder": embedder, "prefill_decode": prefill_decode, "audio_encoder": enc,
                      "window_frames": t_frames, "stop": [ids["<|im_end|>"], ids["<|endoftext|>"]],
                      "default_instr": DEFAULT_INSTR}))


if __name__ == "__main__":
    main()
