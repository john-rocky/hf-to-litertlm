#!/usr/bin/env python3
"""Pack Fun-ASR-Nano-2512 into one .litertlm for LiteRT-LM's GENERIC audio path (released runtime, no change):

  sections: llm_metadata (GenericModel + audio config + dual-form jinja), HF tokenizer (out/lm_native/tokenizer.json),
            EMBEDDER + PREFILL_DECODE (unpacked from the export_simple_template.py LM export),
            AUDIO_ENCODER_HW (single signature `encode`: `audio` f32 [1,504,960] -> `features` f32 [1,63,1024] +
            `mask` uint8 [1,63]).

Runtime contract this relies on (LiteRT-LM v0.17.1 source, read for this lane):
  * multimodal_processor_helper.cc: the rendered prompt is split by delimiter_regex; a part matching
    audio_token_regex becomes InputText(text_before + audio_prefix + boa_token) -> InputAudio -> [InputAudioEnd] ->
    [InputText(audio_suffix)]. With no start/end-of-audio token and no prefix/suffix (Fun-ASR-Nano has none around
    the audio embeddings) the text around <|AUDIO|> is tokenized as two strings, which is funasr's own layout
    (prefix 18 ids + audio + suffix 5 ids, asserted below against common.PREFIX_IDS / SUFFIX_IDS).
  * audio_executor_utils.cc: no adapter section => streaming encoder, window = input dim[-2] = 504 frames,
    shrink = 504 / 63 = 8; audio_litert_compiled_model_executor.cc: input buffers cleared per chunk (zero pad),
    valid tokens = last non-zero index + 1 of the output `mask`; longer clips are cut into 504-frame chunks with
    no overlap and their embeddings concatenated.
  * generic_data_processor.cc: a template that renders string content keeps requires_typed_content false, so a
    single-text-item message reaches the template as a string and anything with audio as a list of parts.

Usage (~/venvs/lt094dev/bin/python, litert-lm-builder 0.16.1):
  python build_bundle.py --enc out/audio_encoder/audio_encoder_504f_fp32.tflite --lm out/lm_int8/unpack \
      --out-name Fun-ASR-Nano-2512_fp32enc.litertlm
Ship bundle (round 3): fp16 encoder + house int8 LM, fp32 activations declared for the GPU text decoder:
  python build_bundle.py --enc out/audio_encoder/audio_encoder_504f_fp16.tflite --lm out/lm_int8/unpack \
      --out-name Fun-ASR-Nano-2512.litertlm --prefer-act fp32
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

PH = "<|AUDIO|>"
S_PRE, U_PRE, M_PRE, SUF = "<|im_start|>system\n", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n"
DEFAULT_SYSTEM = "You are a helpful assistant."
DEFAULT_TEXT = C.PROMPT_ITN  # 语音转写： (funasr default: itn=True, no hotwords, no language)
FRAME_LEN, WIN_FRAMES, OUT_TOKENS, FEAT_DIM = 960, 504, 63, 1024

# String and list content both handled (minijinja in 0.17.1 hands a single text item over as a string and anything
# with audio as a list). The system turn is emitted when the conversation does not start with one. In a user turn
# the text items come first (concatenated, in order) and then one placeholder per audio item, whatever the item
# order; a user turn with audio and no text item gets the funasr default instruction.
JINJA = (
    "{%- if messages[0].role != 'system' %}" + S_PRE + DEFAULT_SYSTEM + SUF + "{% endif -%}"
    "{%- for message in messages -%}"
    "{%- if message.content is string -%}"
    "{%- if message.role == 'user' %}" + U_PRE + "{{ message.content }}" + SUF + "{% endif -%}"
    "{%- if message.role == 'model' or message.role == 'assistant' %}" + M_PRE + "{{ message.content }}" + SUF + "{% endif -%}"
    "{%- if message.role == 'system' %}" + S_PRE + "{{ message.content }}" + SUF + "{% endif -%}"
    "{%- else -%}"
    "{%- set ns = namespace(has_audio=false, has_text=false) -%}"
    "{%- for item in message.content -%}"
    "{%- if item.type == 'audio' -%}{%- set ns.has_audio = true -%}"
    "{%- elif item.type == 'text' -%}{%- set ns.has_text = true -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if message.role == 'user' %}" + U_PRE +
    "{% elif message.role == 'model' or message.role == 'assistant' %}" + M_PRE +
    "{% elif message.role == 'system' %}" + S_PRE + "{% endif -%}"
    "{%- for item in message.content -%}{%- if item.type == 'text' -%}{{ item.text }}{%- endif -%}{%- endfor -%}"
    "{%- if message.role == 'user' -%}"
    "{%- if ns.has_audio and not ns.has_text %}" + DEFAULT_TEXT + "{% endif -%}"
    "{%- for item in message.content -%}{%- if item.type == 'audio' -%}" + PH + "{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{%- if message.role == 'user' or message.role == 'model' or message.role == 'assistant' or message.role == 'system' %}"
    + SUF + "{% endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt %}" + M_PRE + "{% endif -%}"
)

EXPECTED_AUDIO_ONLY = S_PRE + DEFAULT_SYSTEM + SUF + U_PRE + DEFAULT_TEXT + PH + SUF + M_PRE


def render_checks():
    """Render the template with Python jinja2 on the shapes the runtime can hand it; asserted strings."""
    import jinja2
    env = jinja2.Environment()
    t = env.from_string(JINJA)
    A = {"type": "audio", "path": "/x.wav"}

    def R(msgs, gen=True):
        return t.render(messages=msgs, add_generation_prompt=gen)

    cases = {
        "audio_only": ([{"role": "user", "content": [A]}], EXPECTED_AUDIO_ONLY),
        "text_then_audio": ([{"role": "user", "content": [{"type": "text", "text": "T"}, A]}],
                            S_PRE + DEFAULT_SYSTEM + SUF + U_PRE + "T" + PH + SUF + M_PRE),
        "audio_then_text": ([{"role": "user", "content": [A, {"type": "text", "text": "T"}]}],
                            S_PRE + DEFAULT_SYSTEM + SUF + U_PRE + "T" + PH + SUF + M_PRE),
        "two_texts_two_audio": ([{"role": "user", "content": [{"type": "text", "text": "a"}, A, {"type": "text", "text": "b"}, A]}],
                                S_PRE + DEFAULT_SYSTEM + SUF + U_PRE + "ab" + PH + PH + SUF + M_PRE),
        "string_user": ([{"role": "user", "content": "hello"}], S_PRE + DEFAULT_SYSTEM + SUF + U_PRE + "hello" + SUF + M_PRE),
        "system_given": ([{"role": "system", "content": "S"}, {"role": "user", "content": [A]}],
                         S_PRE + "S" + SUF + U_PRE + DEFAULT_TEXT + PH + SUF + M_PRE),
        "system_list": ([{"role": "system", "content": [{"type": "text", "text": "S"}]}, {"role": "user", "content": [A]}],
                        S_PRE + "S" + SUF + U_PRE + DEFAULT_TEXT + PH + SUF + M_PRE),
        "history": ([{"role": "user", "content": [A]}, {"role": "model", "content": [{"type": "text", "text": "R"}]},
                     {"role": "user", "content": "more"}],
                    S_PRE + DEFAULT_SYSTEM + SUF + U_PRE + DEFAULT_TEXT + PH + SUF + M_PRE + "R" + SUF + U_PRE + "more" + SUF + M_PRE),
        "history_assistant_string": ([{"role": "user", "content": [A]}, {"role": "assistant", "content": "R"}], None),
    }
    out = {}
    for name, (msgs, want) in cases.items():
        got = R(msgs)
        if want is not None:
            assert got == want, (name, got, want)
        out[name] = got
    # the string form of an audio-free turn must equal the list form (the runtime flattens a single text item)
    assert R([{"role": "user", "content": "q"}]) == R([{"role": "user", "content": [{"type": "text", "text": "q"}]}])
    # prefix contract: rendering [m1] + gen prompt is a prefix of rendering [m1, reply, m2]
    m1 = {"role": "user", "content": [A]}
    base = R([m1])
    longer = R([m1, {"role": "model", "content": "R"}, {"role": "user", "content": "x"}])
    assert longer.startswith(base), (base, longer)
    return out


def find_one(d, pattern):
    c = [f for f in os.listdir(d) if re.search(pattern, f) and f.endswith(".tflite")]
    assert len(c) == 1, (d, pattern, c)
    return os.path.join(d, c[0])


def sig_io_of(path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    sigs = it.get_signature_list()
    res = {}
    for k in sigs:
        r = it.get_signature_runner(k)
        res[k] = {"inputs": {n: {"dtype": __import__("numpy").dtype(v["dtype"]).name, "shape": [int(s) for s in v["shape"]]}
                             for n, v in r.get_input_details().items()},
                  "outputs": {n: {"dtype": __import__("numpy").dtype(v["dtype"]).name, "shape": [int(s) for s in v["shape"]]}
                              for n, v in r.get_output_details().items()}}
    return res


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--lm", required=True, help="unpacked LM export dir (tf_lite_embedder / tf_lite_prefill_decode)")
    ap.add_argument("--tok", default=os.path.join(C.OUT, "lm_native", "tokenizer.json"))
    ap.add_argument("--out-dir", default=os.path.join(C.OUT, "bundle"))
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--cache", type=int, default=2048)
    ap.add_argument("--prefer-act", default=None, choices=["fp16", "fp32", "fp32_fp16"],
                    help="prefer_activation_type section metadata on PREFILL_DECODE (the runtime's text-decoder default "
                         "is fp16 on GPU; an engine-level activation_data_type still overrides it)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rep = {"args": vars(args)}

    renders = render_checks()
    rep["jinja"] = JINJA
    rep["jinja_renders_python"] = renders
    print("jinja render checks: OK", flush=True)

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tok)
    ids = {t: tk.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", PH)}
    assert ids["<|im_end|>"] == 151645 and ids["<|endoftext|>"] == 151643 and ids["<|im_start|>"] == 151644, ids
    rep["token_ids"] = ids  # <|AUDIO|> is None: not in tokenizer.json, consumed by the delimiter regex before tokenizing
    pre, suf = EXPECTED_AUDIO_ONLY.split(PH)
    pre_ids = tk.encode(pre, add_special_tokens=False).ids
    suf_ids = tk.encode(suf, add_special_tokens=False).ids
    assert pre_ids == C.PREFIX_IDS and suf_ids == C.SUFFIX_IDS, (pre_ids, suf_ids)
    rep["prefix_ids_equal_funasr"] = True
    rep["suffix_ids_equal_funasr"] = True
    print("special ids:", ids, "| prefix/suffix ids == funasr prompt ids", flush=True)

    enc_io = sig_io_of(args.enc)
    assert list(enc_io) == ["encode"], enc_io
    e = enc_io["encode"]
    assert e == {"inputs": {"audio": {"dtype": "float32", "shape": [1, WIN_FRAMES, FRAME_LEN]}},
                 "outputs": {"features": {"dtype": "float32", "shape": [1, OUT_TOKENS, FEAT_DIM]},
                             "mask": {"dtype": "uint8", "shape": [1, OUT_TOKENS]}}}, e
    t_frames, frame_len = e["inputs"]["audio"]["shape"][1], e["inputs"]["audio"]["shape"][2]
    assert frame_len == 960 and t_frames == 504
    rep["encoder_io"] = enc_io
    print(f"encoder io OK: window {t_frames} x {frame_len} = {t_frames * frame_len / 16000:.2f} s", flush=True)

    embedder = find_one(args.lm, r"embedder")
    prefill_decode = find_one(args.lm, r"prefill_decode")
    lm_io = {os.path.basename(p): sig_io_of(p) for p in (embedder, prefill_decode)}
    rep["lm_signatures"] = {k: sorted(v) for k, v in lm_io.items()}
    print("lm signatures:", rep["lm_signatures"], flush=True)

    md = llm_metadata_pb2.LlmMetadata()
    md.max_num_tokens = args.cache
    md.prompt_templates.user.prefix = U_PRE
    md.prompt_templates.user.suffix = SUF
    md.prompt_templates.model.prefix = M_PRE
    md.prompt_templates.model.suffix = SUF
    md.prompt_templates.system.prefix = S_PRE
    md.prompt_templates.system.suffix = SUF
    md.jinja_prompt_template = JINJA
    g = llm_model_type_pb2.GenericModel()
    g.audio_enabled = True
    g.delimiter_regex = r"(" + re.escape(PH) + r")"
    g.audio_token_regex = re.escape(PH)
    g.add_audio_end = False
    g.skip_mel_spectrogram_extraction = True
    g.audio_sample_rate_hz = 16000
    g.audio_num_channels = 1
    g.audio_frame_length = frame_len
    g.audio_hop_length = frame_len
    g.audio_input_scale = 1.0
    assert g.delimiter_regex == r"(<\|AUDIO\|>)" and g.audio_token_regex == r"<\|AUDIO\|>", (g.delimiter_regex, g.audio_token_regex)
    assert not g.HasField("start_of_audio_token") and not g.HasField("end_of_audio_token")
    assert g.audio_prefix == "" and g.audio_suffix == ""
    md.llm_model_type.generic_model.CopyFrom(g)
    md.stop_tokens.add().token_ids.ids.append(ids["<|im_end|>"])
    md.stop_tokens.add().token_ids.ids.append(ids["<|endoftext|>"])
    stem = os.path.splitext(args.out_name)[0]
    md_path = os.path.join(args.out_dir, f"{stem}.llm_metadata.pb")
    with open(md_path, "wb") as f:
        f.write(md.SerializeToString())
    from google.protobuf import text_format
    rep["llm_metadata_text"] = text_format.MessageToString(md).replace(JINJA, "<JINJA>")

    b = litertlm_builder.LitertLmFileBuilder()
    b.add_system_metadata(litertlm_builder.Metadata(key="Authors", value="", dtype=litertlm_builder.DType.STRING))
    b.add_llm_metadata(md_path)
    b.add_hf_tokenizer(args.tok)
    b.add_tflite_model(embedder, litertlm_builder.TfLiteModelType.EMBEDDER)
    b.add_tflite_model(prefill_decode, litertlm_builder.TfLiteModelType.PREFILL_DECODE,
                       prefer_activation_type=args.prefer_act)
    b.add_tflite_model(args.enc, litertlm_builder.TfLiteModelType.AUDIO_ENCODER_HW)
    out_path = os.path.join(args.out_dir, args.out_name)
    with open(out_path, "wb") as f:
        b.build(f)
    from importlib.metadata import version
    rep["litert_lm_builder"] = version("litert-lm-builder")
    rep["prefill_decode_prefer_activation_type"] = args.prefer_act
    rep["bundle"] = os.path.relpath(out_path, HERE)
    rep["bundle_bytes"] = os.path.getsize(out_path)
    rep["bundle_sha256"] = sha256(out_path)
    rep["inputs"] = {os.path.relpath(p, HERE): {"bytes": os.path.getsize(p), "sha256": sha256(p)}
                     for p in (args.enc, embedder, prefill_decode, args.tok)}
    print("BUNDLE_DONE", out_path, f"{rep['bundle_bytes']:,} B", rep["bundle_sha256"], flush=True)

    peek = os.path.expanduser("~/venvs/lt0171run/bin/litert-lm-peek")
    if os.path.exists(peek):
        p = subprocess.run([peek, "--litertlm_file", out_path], capture_output=True, text=True)
        txt = p.stdout + p.stderr
        rep["peek_rc"] = p.returncode
        print("---- litert-lm-peek ----\n" + txt[:20000], flush=True)
    with open(os.path.join(HERE, f"build_bundle_{stem}.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
