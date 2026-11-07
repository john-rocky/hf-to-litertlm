#!/usr/bin/env python3
"""Pack granite-speech-5.0-2b into one .litertlm for LiteRT-LM's generic audio path (released runtime 0.19.1, no
runtime change), the shape of funasr_nano_work/build_bundle.py.

  sections: LlmMetadata (GenericModel + audio fields + a chat template that takes content parts), ExecutorMetadata
            (the KV state buffers, from the LM export), HF tokenizer (the repo's tokenizer.json bytes), EMBEDDER +
            PREFILL_DECODE (the LM export), AUDIO_ENCODER_HW (export_encoder.py: single signature `encode`, `audio` f32
            [1,3600,160] -> `features` f32 [1,360,2048] + `mask` uint8 [1,360]; with --win-s 24 an encoder of
            export_encoder6.py: [1,2400,160] -> [1,240,2048] + [1,240]).

The runtime contract (LiteRT-LM v0.19.1 source):
  * runtime/conversation/model_data_processor/multimodal_processor_helper.cc: the rendered prompt is split by
    delimiter_regex; a part matching audio_token_regex becomes InputText(text before + audio_prefix + boa_token) ->
    InputAudio -> [InputAudioEnd] -> [InputText(audio_suffix)]; the rest of the prompt is one more InputText. There
    is no boa / eoa / prefix / suffix here, so the text around <|audio|> is tokenized as two strings; asserted below
    to give the transformers processor's ids.
  * model_data_processor_factory_common.cc: frame / hop / input_scale default to 512 / 160 / 32768 when unset, so all
    three are set.
  * support/preprocessor/audio_preprocessor_miniaudio.cc + audio_preprocessor_utils.cc: skip_mel frames the miniaudio
    f32 PCM (int16 / 32768) into [T, 160], the partial last frame zero padded.
  * audio_executor_utils.cc / audio_litert_compiled_model_executor.cc: without an adapter section the encoder is
    streaming, window = input dims[-2] (3600 frames for 36 s), valid tokens = last non-zero entry of `mask` + 1,
    input buffers cleared for each chunk of that many frames, chunks concatenated without overlap.
  * runtime/engine/engine_settings.cc: a start token only if the metadata has one (none here: the source's
    tokenizer_config has add_bos_token false), stop tokens from the metadata.

Checks before packing: the template renders the repo's chat_template.jinja on string content and the prompt on
content parts (jinja2, and minijinja when installed, both with trim_blocks / lstrip_blocks as the runtime sets them);
the transformers processor's ids for one clip equal the runtime's pieces tokenized one by one. After packing: every
section unpacked again, the tokenizer section = the repo's tokenizer.json bytes, the three TFLite sections = the input
files.

  python build_bundle.py --src <snapshot dir> --enc <encoder.tflite> --lm <dir with the LM export's files> \
      --out <name>.litertlm [--cache 4096] [--win-s 36]
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PH = C.AUDIO_TOKEN  # <|audio|>
U_PRE, U_SUF = "USER: ", "\n ASSISTANT:"

# The repo template (chat_template.jinja):
#   {% for message in messages %}{% if message['role'] == 'user' %}USER: {{ message['content'] }}\n ASSISTANT:
#   {% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}
# extended to typed content parts (LiteRT-LM hands a message with audio over as a list of parts): parts render in the
# order given, audio -> <|audio|>, text -> its text. 'model' is accepted as the assistant role (the runtime's history
# role); system turns render nothing, as in the repo template. No messages[0] access (an empty message list renders
# ''), no Python methods (the runtime's minijinja has no pycompat). The runtime's minijinja sets trim_blocks and
# lstrip_blocks (LiteRT-LM v0.19.1 runtime/components/rust/minijinja_template.rs), as transformers'
# apply_chat_template does: a newline right after a block tag is dropped. The repo template's newline follows `}}`;
# here it follows `{% endif %}`, so an empty expression `{{ "" }}` stands between them (without it the runtime renders
# "... capitalization. ASSISTANT:").
JINJA = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}" + U_PRE +
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for item in message['content'] %}"
    "{% if item['type'] == 'audio' %}" + PH + "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}{% endif %}{{ \"\" }}" + U_SUF +
    "{% elif message['role'] == 'assistant' or message['role'] == 'model' %}"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}{% endif %}"
    "{% endif %}{% endfor %}"
)


def render_checks(src):
    """The template equals the repo template on string content, and the audio-part form gives C.PROMPT."""
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)  # = transformers apply_chat_template and the runtime
    ours = env.from_string(JINJA)
    repo = env.from_string((Path(src) / "chat_template.jinja").read_text())
    A = {"type": "audio", "path": "/x.wav"}
    T = {"type": "text", "text": C.USER_TEXT}
    out = {}
    string_cases = {
        "user": [{"role": "user", "content": PH + C.USER_TEXT}],
        "user_plain": [{"role": "user", "content": "hello"}],
        "history": [{"role": "user", "content": PH + C.USER_TEXT}, {"role": "assistant", "content": "Hi."},
                    {"role": "user", "content": "translate it to French."}],
        "system_first": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}],
        "empty": [],
    }
    for name, msgs in string_cases.items():
        for gen in (True, False):
            a, b = ours.render(messages=msgs, add_generation_prompt=gen), repo.render(messages=msgs,
                                                                                    add_generation_prompt=gen)
            assert a == b, (name, gen, a, b)
        out["string_" + name] = ours.render(messages=msgs, add_generation_prompt=True)
    parts_cases = {
        "audio_text": ([{"role": "user", "content": [A, T]}], C.PROMPT),
        "text_audio": ([{"role": "user", "content": [T, A]}], U_PRE + C.USER_TEXT + PH + U_SUF),
        "audio_only": ([{"role": "user", "content": [A]}], U_PRE + PH + U_SUF),
        "text_only_list": ([{"role": "user", "content": [{"type": "text", "text": "q"}]}], U_PRE + "q" + U_SUF),
        "history_model_parts": ([{"role": "user", "content": [A, T]},
                                 {"role": "model", "content": [{"type": "text", "text": "Hi."}]},
                                 {"role": "user", "content": "more"}],
                                C.PROMPT + "Hi." + U_PRE + "more" + U_SUF),
    }
    for name, (msgs, want) in parts_cases.items():
        got = ours.render(messages=msgs, add_generation_prompt=True)
        assert got == want, (name, got, want)
        out["parts_" + name] = got
    # a string == a one-text-part list (the runtime may hand either)
    assert ours.render(messages=[{"role": "user", "content": "q"}]) == ours.render(
        messages=[{"role": "user", "content": [{"type": "text", "text": "q"}]}])
    # prefix contract: render(history + message, gen) starts with render(history, gen=False)
    h = [{"role": "user", "content": [A, T]}]
    base = ours.render(messages=h, add_generation_prompt=False)
    longer = ours.render(messages=h + [{"role": "model", "content": "Hi."}, {"role": "user", "content": "x"}],
                         add_generation_prompt=True)
    assert longer.startswith(base), (base, longer)
    assert not re.search(r"\.(get|split|strip|lstrip|rstrip|startswith|endswith|items|keys|values)\(", JINJA)
    assert "messages[0]" not in JINJA
    try:  # minijinja (the runtime's template engine) on the same shapes, when installed
        import minijinja
        mj = minijinja.Environment(templates={"t": JINJA})
        mj.trim_blocks, mj.lstrip_blocks = True, True
        for name, (msgs, want) in parts_cases.items():
            got = mj.render_template("t", messages=msgs, add_generation_prompt=True)
            assert got == want, ("minijinja", name, got, want)
        assert mj.render_template("t", messages=[], add_generation_prompt=False) == ""
        out["minijinja"] = "equal on all content-parts cases, empty messages -> ''"
    except ImportError:
        out["minijinja"] = "minijinja not installed (not checked)"
    return out


def id_checks(src, wav):
    """The transformers processor's ids for one real clip vs the runtime's pieces tokenized one by one with the repo's
    tokenizer.json."""
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoProcessor
    src = Path(src)
    tk = Tokenizer.from_file(str(src / "tokenizer.json"))
    proc = AutoProcessor.from_pretrained(str(src))
    x = torch.from_numpy(C.wav_float(wav)).unsqueeze(0)
    hf_ids = proc(C.PROMPT, x, device="cpu", return_tensors="pt")["input_ids"][0].tolist()
    n = C.num_audio_tokens(x.shape[-1])
    pre, suf = C.PROMPT.split(PH)
    pre_ids = tk.encode(pre, add_special_tokens=False).ids
    suf_ids = tk.encode(suf, add_special_tokens=False).ids
    runtime_ids = pre_ids + [C.AUDIO_ID] * n + suf_ids
    assert hf_ids == runtime_ids, (hf_ids[:12], runtime_ids[:12], len(hf_ids), len(runtime_ids))
    special = {t: tk.token_to_id(t) for t in ("<|end_of_text|>", "<|pad|>", PH, "<|unk|>")}
    assert special["<|end_of_text|>"] == C.EOS_ID and special[PH] == C.AUDIO_ID and special["<|pad|>"] == C.PAD_ID
    gen_cfg = json.load(open(src / "generation_config.json"))
    tok_cfg = json.load(open(src / "tokenizer_config.json"))
    assert gen_cfg["eos_token_id"] == C.EOS_ID and tok_cfg["add_bos_token"] is False
    return {"clip": Path(wav).name, "audio_tokens": n, "hf_ids_len": len(hf_ids), "prefix_ids": pre_ids,
            "suffix_ids": suf_ids, "hf_ids_equal_runtime_pieces": True, "first_id_is_bos": hf_ids[0] == C.EOS_ID,
            "special_ids": special, "generation_config_eos": gen_cfg["eos_token_id"],
            "tokenizer_config_add_bos_token": tok_cfg["add_bos_token"]}


def sig_io(path):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path))
    res = {}
    for k in it.get_signature_list():
        r = it.get_signature_runner(k)
        f = lambda d: {n: {"dtype": np.dtype(v["dtype"]).name, "shape": [int(s) for s in v["shape"]]} for n, v in d.items()}
        res[k] = {"inputs": f(r.get_input_details()), "outputs": f(r.get_output_details())}
    return res


def find_one(d, pattern):
    c = [p for p in Path(d).iterdir() if re.search(pattern, p.name) and p.suffix == ".tflite"]
    assert len(c) == 1, (d, pattern, c)
    return c[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="snapshot of ibm-granite/granite-speech-5.0-2b")
    ap.add_argument("--enc", required=True, help="audio encoder .tflite (export_encoder.py)")
    ap.add_argument("--lm", required=True, help="dir with the LM export's embedder / prefill_decode .tflite and "
                                                "executor_metadata.pb")
    ap.add_argument("--out", required=True, help="output .litertlm")
    ap.add_argument("--cache", type=int, default=4096)
    ap.add_argument("--win-s", type=int, default=C.WIN_S, help="the encoder's window in seconds (its signature is "
                                                                "checked against it)")
    ap.add_argument("--id-check-wav", default="", help="clip for the processor id check (default: the repo's "
                                                       "multilingual_sample.wav)")
    ap.add_argument("--prefer-act", default=None, choices=["fp16", "fp32", "fp32_fp16"],
                    help="prefer_activation_type on PREFILL_DECODE only (the published file declares fp32 on every "
                         "TFLite section afterwards with scripts/set_activation_type.py)")
    args = ap.parse_args()
    import litert_lm_builder as lb
    from google.protobuf import text_format
    from litert_lm_builder.runtime.proto import llm_metadata_pb2, llm_model_type_pb2
    src = Path(args.src)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = {"args": vars(args)}

    rep["jinja"] = JINJA
    rep["jinja_renders"] = render_checks(src)
    print("jinja render checks: OK", flush=True)
    rep["ids"] = id_checks(src, args.id_check_wav or str(src / "multilingual_sample.wav"))
    print("ids: HF processor == runtime pieces;", {k: rep["ids"][k] for k in ("prefix_ids", "suffix_ids",
                                                                             "audio_tokens")}, flush=True)

    enc = Path(args.enc)
    e = sig_io(enc)
    wf = args.win_s * C.SR // C.HOP  # runtime frames per window; 10 frames per audio token
    want = {"encode": {"inputs": {"audio": {"dtype": "float32", "shape": [1, wf, C.HOP]}},
                       "outputs": {"features": {"dtype": "float32", "shape": [1, wf // 10, 2048]},
                                   "mask": {"dtype": "uint8", "shape": [1, wf // 10]}}}}
    assert e == want, e
    rep["encoder_io"] = e
    embedder = find_one(args.lm, r"embedder")
    prefill_decode = find_one(args.lm, r"prefill_decode")
    rep["lm_signatures"] = {p.name: sorted(sig_io(p)) for p in (embedder, prefill_decode)}
    print("lm signatures:", rep["lm_signatures"], flush=True)

    md = llm_metadata_pb2.LlmMetadata()
    md.max_num_tokens = args.cache
    md.prompt_templates.user.prefix = U_PRE
    md.prompt_templates.user.suffix = U_SUF
    md.prompt_templates.model.prefix = ""
    md.prompt_templates.model.suffix = ""
    md.jinja_prompt_template = JINJA
    g = llm_model_type_pb2.GenericModel()
    g.audio_enabled = True
    g.delimiter_regex = "(" + re.escape(PH) + ")"
    g.audio_token_regex = re.escape(PH)
    g.add_audio_end = False
    g.skip_mel_spectrogram_extraction = True
    g.audio_sample_rate_hz = C.SR
    g.audio_num_channels = 1
    g.audio_frame_length = C.HOP
    g.audio_hop_length = C.HOP
    g.audio_input_scale = 1.0
    assert g.delimiter_regex == r"(<\|audio\|>)" and g.audio_token_regex == r"<\|audio\|>", g
    assert not g.HasField("start_of_audio_token") and not g.HasField("end_of_audio_token")
    md.llm_model_type.generic_model.CopyFrom(g)
    md.stop_tokens.add().token_ids.ids.append(C.EOS_ID)
    md.pad_token.token_str = "<|pad|>"  # as the litert-torch 0.10.0 export's own LlmMetadata
    assert not md.HasField("start_token") and not md.HasField("sampler_params")
    md_path = out.with_name(out.stem + ".llm_metadata.pb")
    md_path.write_bytes(md.SerializeToString())
    rep["llm_metadata_text"] = re.sub(r'jinja_prompt_template: ".*"\n', "jinja_prompt_template: <JINJA>\n",
                                      text_format.MessageToString(md))

    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_llm_metadata(str(md_path))
    em = Path(args.lm) / "executor_metadata.pb"
    assert em.exists(), em
    b.add_executor_metadata(str(em))
    rep["executor_metadata"] = {"file": em.name, "sha256": C.sha256(em)}
    b.add_hf_tokenizer(str(src / "tokenizer.json"))
    b.add_tflite_model(str(embedder), "embedder")
    b.add_tflite_model(str(prefill_decode), "prefill_decode", prefer_activation_type=args.prefer_act)
    b.add_tflite_model(str(enc), "audio_encoder_hw")
    with open(out, "wb") as f:
        b.build(f)
    from importlib.metadata import version
    rep["litert_lm_builder"] = version("litert-lm-builder")
    rep["bundle"] = out.name
    rep["bundle_bytes"] = out.stat().st_size
    rep["bundle_sha256"] = C.sha256(out)
    rep["inputs"] = {p.name: {"bytes": p.stat().st_size, "sha256": C.sha256(p)}
                     for p in (enc, embedder, prefill_decode, src / "tokenizer.json")}
    print("BUNDLE", out, f"{rep['bundle_bytes']:,} B", rep["bundle_sha256"], flush=True)

    # read back: unpack every section; the tokenizer section = the repo's tokenizer.json bytes, the TFLite sections =
    # the input files
    un = out.with_name(out.stem + ".unpack")
    shutil.rmtree(un, ignore_errors=True)
    lb.unpack(str(out), str(un))
    files = {p.name: {"bytes": p.stat().st_size, "sha256": C.sha256(p)} for p in sorted(un.iterdir()) if p.is_file()}
    rep["unpacked"] = files
    tok_sha = C.sha256(src / "tokenizer.json")
    tok_hit = []
    for p in un.iterdir():
        if p.suffix == ".tflite" or not p.is_file():
            continue
        data = p.read_bytes()
        cands = {"raw": data}
        try:  # HF_Tokenizer_Zlib section = 8-byte little-endian size + zlib stream
            cands["zlib"] = zlib.decompress(data[8:])
        except zlib.error:
            pass
        for how, raw in cands.items():
            if hashlib.sha256(raw).hexdigest() == tok_sha:
                tok_hit.append(f"{p.name} ({how})")
    rep["tokenizer_section_equals_repo_bytes"] = tok_hit
    assert tok_hit, files
    in_shas = {v["sha256"] for k, v in rep["inputs"].items() if k.endswith(".tflite")}
    rep["tflite_sections_equal_inputs"] = sorted(n for n, v in files.items() if v["sha256"] in in_shas)
    assert len(rep["tflite_sections_equal_inputs"]) == 3, rep["tflite_sections_equal_inputs"]
    shutil.rmtree(un)
    print("tokenizer section = the repo's tokenizer.json:", tok_hit, "| TFLite sections = the inputs:",
          rep["tflite_sections_equal_inputs"], flush=True)
    out.with_name(out.stem + ".build.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("BUNDLE_DONE", flush=True)


if __name__ == "__main__":
    main()
