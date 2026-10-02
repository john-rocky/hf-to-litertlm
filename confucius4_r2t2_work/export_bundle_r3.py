#!/usr/bin/env python3
# Third-party code, Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0); details in THIRD_PARTY_NOTICES.md.
# - AudioEncoderR3.encoder_body: the body of Qwen3AsrEncoder.forward from litert-torch @731ef0a,
#   litert_torch/generative/export_hf/model_ext/qwen3/qwen3_asr.py l.106-134 (Copyright 2026 The LiteRT Torch Authors;
#   https://github.com/google-ai-edge/litert-torch), without the prompt-embedding concat at its end.
"""Export of recipe (C) in the shape LiteRT-LM's generic audio path runs (released litert-lm-api 0.17.1,
Engine -> create_conversation -> send_message with an audio item). Contract read from v0.17.1: GENERIC_CONTRACT.md.

What changes against the (C) bundle that quant_sections.py packs (out/export/c4r2_30s_C_lmi8, the source of every
other section):
  audio_encoder_hw  re-exported. Signature 'encode', input `audio` f32 [1, T, 160] = raw 16 kHz PCM in [-1, 1),
                    framed by the runtime with skip_mel_spectrogram_extraction and frame = hop = 160 samples
                    (T = 3000 for 30 s, 500 for 5 s; the runtime zero-fills the window after the clip). In the graph:
                    the Qwen3ASRFeatureExtractor log-mel of that window (torch.stft center/reflect, periodic Hann 400,
                    hop 160, |X|^2, the extractor's own Slaney filterbank, log10(max(., 1e-10)), max(x, max - 8),
                    (x + 4) / 4 = Qwen3ASRFeatureExtractor on the wav zero-padded to the window),
                    then litert-torch's Qwen3AsrEncoder body (conv stack, encoder_window.py's 104-token block-diagonal
                    attention, ln_post, projector) WITHOUT the prompt-id prefix/postfix concat. Outputs `features`
                    f32 [1, N, 2048] (N = 390 / 65 audio tokens) and `mask` uint8 [1, N] = 1 for every row (the
                    runtime takes valid rows = last non-zero mask index + 1, so all N rows are prefilled, as the
                    (C) gate runs did). Quantized like (C): fp16 weights (FLOAT_CASTING,
                    fp32 compute), quant_sections.recipe_of('fp16').
  LlmMetadata       llm_model_type generic_model { audio_enabled, placeholder <|audio_pad|>, skip mel, frame = hop
                    = 160, input_scale 1 } + a jinja that renders the vendor chat_template string (empty system turn
                    + user audio + generation prompt); user text items are appended after '<|im_start|>assistant\n'
                    (the vendor's force_language / streaming prefix position). Stop tokens / max_num_tokens / pad as (C).
  unchanged         ExecutorMetadata, HF tokenizer, prefill_decode (LM dynamic_wi8_afp32), embedder (int8) - the
                    (C) bundle's own bytes (checked by sha256 after pack/unpack).

  PYTHONPATH=out/pyoverlay_builder017 ~/venvs/ltmain0918/bin/python export_bundle_r3.py --window 30 \
      --src out/unpack/r3_src_C --name c4r3_30s_C
  PYTHONPATH=out/pyoverlay_builder017 ~/venvs/ltmain0918/bin/python export_bundle_r3.py --window 5 \
      --src out/unpack/r3_src_5s --name c4r3_5s_C
Stages: encoder (eager parity vs the HF mel and the prompt-baked Qwen3AsrEncoder + fp32 tflite), quant (fp16), tflite
(IO + fp32 / fp16
tflite vs eager on the parity clips), pack (metadata + litert-lm pack + unpack check). The fp32 tflite is deleted after the fp16 one exists (--keep_fp32 to keep it).
"""
import argparse
import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
import common  # noqa: E402
import encoder_window  # noqa: E402

SR = 16000
FRAME = 160  # samples per runtime frame = mel hop
N_FFT = 400
LITERT_LM = os.path.expanduser("~/venvs/lt0171run/bin/litert-lm")
ENC_SECTION = "Section5_TFLiteModel_tf_lite_audio_encoder_hw.tflite"

# The vendor chat_template.json renders [system(context), user(audio)] + generation prompt as
# '<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n'
# '<|im_start|>assistant\n'. This template renders the same string message by message (so the render of a shorter
# history is always a prefix of a longer one, and an empty history renders ''), and puts the text items of the last
# user message after the generation prompt (vendor force_language 'language X<asr_text>' / streaming prefix).
# Shipped template: the litert-torch Qwen3AsrProcessor._PROMPT string, the prompt the (C) bundle baked into its
# encoder output (chosen 2026-10-02: with the original repository's prompt this bundle scored FLEURS zh / ja CER
# 10.75 / 9.33 % against 6.40 / 6.21 % with this one, same 155 clips): '<|im_start|>user<|audio_start|><|audio_pad|><|audio_end|>
# <|im_end|><|im_start|>assistant\n' (no system turn, no newline after the role names or <|im_end|>). Rendered message
# by message (an empty history renders ''); system messages render nothing; the text items of the last user message
# go after the generation prompt (forced language 'language X<asr_text>' / streaming prefix), as in JINJA_VENDOR.
JINJA_LITERT = (
    "{%- set ns = namespace(prefix='') -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}"
    "{{- '<|im_start|>user' -}}"
    "{%- set ns.prefix = '' -%}"
    "{%- if m.content is string -%}{%- set ns.prefix = m.content -%}{%- else -%}"
    "{%- for c in m.content -%}"
    "{%- if c.type == 'audio' -%}{{- '<|audio_start|><|audio_pad|><|audio_end|>' -}}"
    "{%- elif c.type == 'text' -%}{%- set ns.prefix = ns.prefix + c.text -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>' -}}"
    "{%- elif m.role == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if m.content is string -%}{{- m.content -}}{%- else -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' + ns.prefix -}}{%- endif -%}"
)

JINJA_VENDOR = (
    "{%- set ns = namespace(has_system=false, prefix='') -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}"
    "{%- set ns.has_system = true -%}"
    "{{- '<|im_start|>system\\n' -}}"
    "{%- if m.content is string -%}{{- m.content -}}{%- else -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- elif m.role == 'user' -%}"
    "{%- if not ns.has_system -%}{{- '<|im_start|>system\\n<|im_end|>\\n' -}}{%- set ns.has_system = true -%}"
    "{%- endif -%}"
    "{{- '<|im_start|>user\\n' -}}"
    "{%- set ns.prefix = '' -%}"
    "{%- if m.content is string -%}{%- set ns.prefix = m.content -%}{%- else -%}"
    "{%- for c in m.content -%}"
    "{%- if c.type == 'audio' -%}{{- '<|audio_start|><|audio_pad|><|audio_end|>' -}}"
    "{%- elif c.type == 'text' -%}{%- set ns.prefix = ns.prefix + c.text -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- elif m.role == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if m.content is string -%}{{- m.content -}}{%- else -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' + ns.prefix -}}{%- endif -%}"
)


JINJAS = {"litert": JINJA_LITERT, "vendor": JINJA_VENDOR}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class HfLogMel(torch.nn.Module):
    """Qwen3ASRFeatureExtractor._torch_extract_fbank_features for ONE window of n_samples, as graph ops.

    torch.stft(center=True, pad_mode='reflect', window=hann_window(400) periodic, hop 160) is written as a reflect pad
    (two flipped 200-sample slices) + one strided conv1d whose 402 output channels are the windowed DFT basis
    (201 cos rows, 201 -sin rows); the last STFT frame is dropped like the extractor's stft[..., :-1]."""

    def __init__(self, mel_filters):
        super().__init__()
        n = np.arange(N_FFT, dtype=np.float64)
        k = np.arange(N_FFT // 2 + 1, dtype=np.float64)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / N_FFT)  # torch.hann_window(400), periodic=True
        ang = 2.0 * np.pi * np.outer(k, n) / N_FFT
        basis = np.concatenate([window * np.cos(ang), -window * np.sin(ang)], 0).astype(np.float32)
        self.register_buffer("basis", torch.from_numpy(np.ascontiguousarray(basis)).unsqueeze(1))  # [402, 1, 400]
        mel_t = np.ascontiguousarray(np.asarray(mel_filters, dtype=np.float32).T)  # extractor: mel_filters.T @ |X|^2
        self.register_buffer("mel_t", torch.from_numpy(mel_t))  # [128, 201]
        self.bins = N_FFT // 2 + 1

    def forward(self, x):  # x [1, n_samples]
        half = N_FFT // 2
        left = torch.flip(x[:, 1:half + 1], dims=[1])
        right = torch.flip(x[:, -half - 1:-1], dims=[1])
        xp = torch.cat([left, x, right], dim=1)
        spec = F.conv1d(xp.unsqueeze(1), self.basis, stride=FRAME)  # [1, 402, n/160 + 1]
        spec = spec[:, :, :-1]
        re, im = spec[:, :self.bins, :], spec[:, self.bins:, :]
        power = re * re + im * im  # [1, 201, frames]
        mel = torch.matmul(self.mel_t, power)  # [1, 128, frames]
        log_spec = torch.log10(torch.clamp(mel, min=1e-10))
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0


class AudioEncoderR3(torch.nn.Module):
    """`audio` [1, T, 160] -> {'features': [1, N, 2048], 'mask': uint8 [1, N]}: HfLogMel + Qwen3AsrEncoder body."""

    def __init__(self, qwen3asr_encoder, mel_filters):
        super().__init__()
        self.logmel = HfLogMel(mel_filters)
        self._encoder = qwen3asr_encoder._encoder
        self._projector = qwen3asr_encoder._projector

    def encoder_body(self, input_features):
        """litert-torch Qwen3AsrEncoder.forward (model_ext/qwen3/qwen3_asr.py l.106-134) minus the prompt concat."""
        batch_size, num_mel_bins, padded_feature_length = input_features.shape
        chunk_len = self._encoder.n_window * 2
        num_chunks = padded_feature_length // chunk_len
        chunked = (
            input_features.view(batch_size, num_mel_bins, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
        )
        conv_out = F.gelu(self._encoder.conv2d1(chunked))
        conv_out = F.gelu(self._encoder.conv2d2(conv_out))
        conv_out = F.gelu(self._encoder.conv2d3(conv_out))
        b, c, f, t = conv_out.size()
        conv_out = self._encoder.conv_out(conv_out.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        conv_out += self._encoder.positional_embedding.positional_embedding[:t]
        hidden_states = conv_out.view(b * t, -1)
        cu_seqlens = torch.arange(0, b + 1).int() * t
        for layer in self._encoder.layers:
            hidden_states = layer(hidden_states, cu_seqlens)[0]
        hidden_states = self._encoder.ln_post(hidden_states)
        hidden_states = self._projector.linear_1(hidden_states)
        hidden_states = self._projector.act(hidden_states)
        hidden_states = self._projector.linear_2(hidden_states)
        return hidden_states.view(batch_size, num_chunks * t, -1)

    def forward(self, audio):
        x = audio.reshape(1, -1)
        feats = self.encoder_body(self.logmel(x))
        # Valid-row mask for the runtime (GetValidCount): 1 for every finite row, built from the features so the
        # output is a computed tensor and not a constant buffer.
        mask = (feats[:, :, 0] > -3.0e38).to(torch.uint8)
        return {"features": feats, "mask": mask}


def load_model():
    from litert_torch.generative.export_hf.model_ext import exportables as model_ext_exportables
    encoder_window.install()
    model_cls = model_ext_exportables.get_speech_model_cls("qwen3_asr")
    t0 = time.time()
    asr = model_cls(common.HF_LAYOUT, override_transformers=True)  # export_lib.py l.232, same call
    print(f"loaded {common.HF_LAYOUT} in {time.time() - t0:.1f} s", flush=True)
    return asr


def parity(asr, wrapper, window, clips, out_json, npz_dir):
    """Eager torch: in-graph mel vs Qwen3ASRFeatureExtractor on the zero-padded window, and the wrapper's features vs
    the prompt-baked encoder path (Qwen3AsrEncoder on the HF mel, prompt rows stripped)."""
    import transformers
    proc = transformers.AutoProcessor.from_pretrained(common.HF_LAYOUT)
    n = window * SR
    rows = []
    os.makedirs(npz_dir, exist_ok=True)
    with torch.no_grad():
        for c in clips:
            wav = common.read_wav(c["path"])[:n]
            padded = np.concatenate([wav, np.zeros(n - len(wav), np.float32)])
            hf_mel = proc(text=common.LITERT_TORCH_PROMPT, audio=padded, return_tensors="pt")["input_features"]
            g_mel = wrapper.logmel(torch.from_numpy(padded)[None])
            ref = asr.get_encoder()(hf_mel)[0][:, 3:-5, :]  # prompt-baked encoder output minus 3 prefix / 5 postfix rows
            out = wrapper(torch.from_numpy(padded).reshape(1, -1, FRAME))
            d_mel = (g_mel - hf_mel).abs()
            d_f = (out["features"] - ref).abs()
            cos = F.cosine_similarity(out["features"][0], ref[0], dim=-1)
            np.savez(os.path.join(npz_dir, c["name"] + ".npz"), audio=padded.reshape(1, -1, FRAME),
                     features=out["features"].numpy(), mask=out["mask"].numpy(), ref=ref.numpy())
            rows.append({"clip": c["name"], "audio_seconds": round(len(wav) / SR, 3),
                         "mel_shape": list(g_mel.shape), "mel_max_abs": float(d_mel.max()),
                         "mel_mean_abs": float(d_mel.mean()),
                         "features_shape": list(out["features"].shape), "features_max_abs": float(d_f.max()),
                         "features_cos_min": float(cos.min()), "mask_sum": int(out["mask"].sum())})
            print(json.dumps(rows[-1]), flush=True)
    json.dump({"window_s": window, "rows": rows}, open(out_json, "w"), indent=1)
    return rows


def export_encoder(wrapper, window, path):
    from litert_torch._convert import interface as converter_utils
    from litert_torch.generative.export_hf.core import export_lib
    from litert_torch.generative.export_hf.core.mu import mu_pass_lib
    t0 = time.time()
    converter = converter_utils.Converter()
    converter.add_signature("encode", wrapper.eval(),
                            sample_kwargs={"audio": torch.zeros(1, window * SR // FRAME, FRAME)})
    with export_lib.patch_builtin_tuple_for_export():
        lrt_model = converter.convert(lightweight_conversion=False, strict_export=False)  # export_lib.py l.757-760
    lrt_model = mu_pass_lib.update_model(lrt_model)
    lrt_model.export(path)
    print(f"EXPORT_ENCODER_DONE {path} {os.path.getsize(path):,} B in {time.time() - t0:.1f} s", flush=True)


def tflite_check(paths, npz_dir, out_json):
    """Signature IO of each .tflite + its outputs (ai_edge_litert, XNNPACK, 8 threads) vs the eager wrapper (fp32)
    and vs the prompt-baked encoder path (ref = Qwen3AsrEncoder on the HF mel, prompt rows stripped) on the parity
    clips."""
    from ai_edge_litert.interpreter import Interpreter
    doc = {}
    for tag, path in paths.items():
        if not os.path.exists(path):
            continue
        it = Interpreter(model_path=path, num_threads=8)
        run = it.get_signature_runner("encode")
        io = {"signatures": list(it.get_signature_list()),
              "inputs": {k: {"dtype": v["dtype"].__name__, "shape": [int(x) for x in v["shape"]]}
                         for k, v in run.get_input_details().items()},
              "outputs": {k: {"dtype": v["dtype"].__name__, "shape": [int(x) for x in v["shape"]]}
                          for k, v in run.get_output_details().items()}}
        rows = []
        for f in sorted(os.listdir(npz_dir)):
            z = np.load(os.path.join(npz_dir, f))
            t0 = time.time()
            o = run(audio=z["audio"].astype(np.float32))
            dt = time.time() - t0
            feats, mask = o["features"], o["mask"]

            def cmp(a, b):
                a, b = a[0].astype(np.float64), b[0].astype(np.float64)
                cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
                return {"max_abs": float(np.abs(a - b).max()), "mean_abs": float(np.abs(a - b).mean()),
                        "cos_min": float(cos.min())}
            rows.append({"clip": f[:-4], "seconds": round(dt, 3), "mask_dtype": str(mask.dtype),
                         "valid_rows": int(np.nonzero(mask[0])[0][-1] + 1) if mask.any() else 0,
                         "vs_eager_wrapper": cmp(feats, z["features"]), "vs_round2_encoder": cmp(feats, z["ref"])})
            print(tag, json.dumps(rows[-1]), flush=True)
        doc[tag] = {"path": path, "bytes": os.path.getsize(path), "io": io, "rows": rows}
        print(tag, json.dumps(io), flush=True)
        del run, it
    json.dump(doc, open(out_json, "w"), indent=1)
    return doc


def build_metadata(src_pbtext, kv, prompt):
    """(C)'s LlmMetadata with llm_model_type qwen3 -> generic_model (audio) + jinja; everything else kept."""
    txt = open(src_pbtext).read()
    assert "llm_model_type {\n  qwen3 {\n  }\n}\n" in txt, txt
    assert f"max_num_tokens: {kv}\n" in txt, (kv, txt)
    generic = ("llm_model_type {\n  generic_model {\n    audio_enabled: true\n"
               "    delimiter_regex: \"(<\\\\|audio_pad\\\\|>)\"\n    audio_token_regex: \"<\\\\|audio_pad\\\\|>\"\n"
               "    skip_mel_spectrogram_extraction: true\n    audio_sample_rate_hz: 16000\n"
               "    audio_num_channels: 1\n    audio_frame_length: 160\n    audio_hop_length: 160\n"
               "    audio_input_scale: 1.0\n    add_audio_end: false\n  }\n}\n")
    txt = txt.replace("llm_model_type {\n  qwen3 {\n  }\n}\n", generic)
    return txt + "jinja_prompt_template: " + json.dumps(JINJAS[prompt]) + "\n"


def pack(src, enc_fp16, name, kv, prompt):
    dst = os.path.join(OUT, "unpack", name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in os.listdir(src):
        if f in ("LlmMetadataProto.pbtext", "arm.json") or f.endswith(".json"):
            continue
        s = os.path.realpath(os.path.join(src, f))
        if f == ENC_SECTION:
            s = os.path.realpath(enc_fp16)
        os.symlink(s, os.path.join(dst, f))
    with open(os.path.join(dst, "LlmMetadataProto.pbtext"), "w") as f:
        f.write(build_metadata(os.path.join(src, "LlmMetadataProto.pbtext"), kv, prompt))
    out = os.path.join(OUT, "export", name, "model.litertlm")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    subprocess.run([LITERT_LM, "pack", dst, "--output", out, "--allow-overwrite"], check=True)
    pack_s = time.time() - t0
    chk = os.path.join(OUT, "unpack", name + "_check")
    if os.path.exists(chk):
        shutil.rmtree(chk)
    subprocess.run([LITERT_LM, "unpack", out, "--output-dir", chk], check=True)
    res = {"bundle": out, "bytes": os.path.getsize(out), "sha256": sha256(out), "pack_seconds": round(pack_s, 1),
           "sections": {}}
    for f in sorted(os.listdir(dst)):
        p = os.path.join(chk, f)
        if not os.path.exists(p):
            continue
        res["sections"][f] = {"bytes": os.path.getsize(p), "sha256": sha256(p),
                              "equal_to_input": filecmp.cmp(os.path.join(dst, f), p, shallow=False),
                              "equal_to_C_source": (filecmp.cmp(os.path.join(src, f), p, shallow=False)
                                                    if os.path.exists(os.path.join(src, f)) else None)}
    res["metadata"] = open(os.path.join(chk, "LlmMetadataProto.pbtext")).read()
    shutil.rmtree(chk)
    with open(os.path.join(OUT, f"{name}_pack.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "metadata"}, indent=1), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, choices=[5, 30], required=True)
    ap.add_argument("--src", required=True, help="unpacked (C) bundle of the same window (litert-lm unpack)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--stages", default="encoder,quant,tflite,pack")
    ap.add_argument("--keep_fp32", action="store_true")
    ap.add_argument("--prompt", choices=["litert", "vendor"], default="litert",
                    help="jinja baked into LlmMetadata (litert = the shipped template; vendor = the original repository's "
                         "chat_template)")
    ap.add_argument("--enc_dir", default="", help="reuse the fp16 encoder of another r3 build (sections/r3_<name>)")
    args = ap.parse_args()
    kv = 1024 if args.window == 30 else 512
    d = args.enc_dir or os.path.join(OUT, "sections", "r3_" + args.name)
    os.makedirs(d, exist_ok=True)
    enc32 = os.path.join(d, "enc__none.tflite")
    enc16 = os.path.join(d, "enc__fp16.tflite")
    stages = args.stages.split(",")
    npz_dir = os.path.join(OUT, "r3_parity", args.name)
    if "encoder" in stages:
        asr = load_model()
        import transformers
        fe = transformers.AutoProcessor.from_pretrained(common.HF_LAYOUT).feature_extractor
        wrapper = AudioEncoderR3(asr.get_encoder(), fe.mel_filters).eval()
        clips = [c for c in common.load_clips() if c["config"] is None]  # the 5 example clips
        clips += [c for c in common.load_clips() if c["config"] == "cmn_hans_cn"][:1]
        parity(asr, wrapper, args.window, clips, os.path.join(OUT, f"r3_{args.name}_parity_eager.json"), npz_dir)
        export_encoder(wrapper, args.window, enc32)
    if "quant" in stages:
        import quant_sections
        from ai_edge_quantizer import quantizer as quantizer_lib
        t0 = time.time()
        qt = quantizer_lib.Quantizer(enc32)
        rec = quant_sections.recipe_of("fp16")
        json.dump(rec, open(enc16[:-7] + ".recipe.json", "w"), indent=1, default=str)
        qt.load_quantization_recipe(rec)
        qt.quantize().export_model(enc16 + ".part", overwrite=True)
        os.replace(enc16 + ".part", enc16)
        print(f"QUANT_DONE fp16 {os.path.getsize(enc32):,} -> {os.path.getsize(enc16):,} B in "
              f"{time.time() - t0:.1f} s", flush=True)
    if "tflite" in stages:
        tflite_check({"fp32": enc32, "fp16": enc16}, npz_dir, os.path.join(OUT, f"r3_{args.name}_parity_tflite.json"))
    if "pack" in stages:
        pack(args.src, enc16, args.name, kv, args.prompt)
        if not args.keep_fp32 and os.path.exists(enc32) and os.path.exists(enc16):
            print("removing fp32 encoder", enc32, os.path.getsize(enc32), flush=True)
            os.remove(enc32)


if __name__ == "__main__":
    main()
