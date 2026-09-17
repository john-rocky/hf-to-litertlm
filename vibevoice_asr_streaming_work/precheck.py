#!/usr/bin/env python3
"""30-minute precheck for microsoft/VibeVoice-ASR-Streaming-1.5B on the LiteRT-LM generic audio path.

Stages (all Mac, no device):
  load   : legacy `VibeVoiceForASRStreamingTraining` checkpoint (bf16, same key layout as the
           BitNet one) -> transformers-native VibeVoiceAsrForConditionalGeneration (fp32, strict),
           VAE decoder dropped, NO ternarization (dense LM).  Saves lm_native/ for export_simple_template.
  enc    : trace the acoustic+semantic encoder + projector as one tflite with the streaming window
           (chunk_frames + lookahead_frames = 26 frames = 3.47 s), no -25 dBFS normaliser
           (preprocessor_config.json: normalize_audio=false), parity vs eager.
  eager  : the vendor streaming protocol (GitHub modeling_vibevoice_asr.streaming_generate, greedy,
           mean latents) on N fixtures -> WER; plus the "one-shot" variant (all windows inside a single
           <|object_ref_start|> ... <|object_ref_end|> pair, then one generation) = what the current
           single-turn bundle contract would feed the model.

Usage: precheck.py --src <hf dir> --stage load,enc,eager [--n-clips 3]
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
BITNET = os.path.join(os.path.dirname(HERE), "vibevoice_asr_work")
sys.path.insert(0, BITNET)
import common as C  # noqa: E402  (load_wav, WER, fixtures of the BitNet lane)
from load_bitnet import rename  # noqa: E402  (legacy -> native key map, identical layout)

SR, HOP = 24000, 3200
SP_START, SP_END, TCE, EOS = 151646, 151647, 151665, 151643
PROMPT = ("You are a helpful assistant that transcribes audio input into text output. "
          "Please transcribe the following audios streamingly with these keys: speaker, content\n")


def build_native(src):
    from transformers import (Qwen2Config, VibeVoiceAsrConfig, VibeVoiceAsrForConditionalGeneration,
                              VibeVoiceAcousticTokenizerEncoderConfig)
    cfg = json.load(open(os.path.join(src, "config.json")))

    def enc_cfg(legacy):
        depths = [int(x) for x in legacy["encoder_depths"].split("-")]
        ratios = list(reversed(legacy["encoder_ratios"]))
        assert legacy["layernorm"] == "RMSNorm" and legacy["mixer_layer"] == "depthwise_conv"
        assert legacy["causal"] is True and legacy["pad_mode"] == "constant"
        return VibeVoiceAcousticTokenizerEncoderConfig(
            channels=legacy["channels"], hidden_size=legacy["vae_dim"],
            num_filters=legacy["encoder_n_filters"], depths=depths, downsampling_ratios=ratios,
            kernel_size=7, rms_norm_eps=legacy["layernorm_eps"],
            layer_scale_init_value=legacy["layer_scale_init_value"], hidden_act="gelu",
            ffn_expansion=4, vae_std=legacy.get("fix_std", 0.0) or 0.0)

    acoustic, semantic = enc_cfg(cfg["acoustic_tokenizer_config"]), enc_cfg(cfg["semantic_tokenizer_config"])
    dec = dict(cfg["decoder_config"])
    for k in ("dtype", "torch_dtype", "_attn_implementation_autoset"):
        dec.pop(k, None)
    assert dec.pop("model_type") == "qwen2"

    t0 = time.time()
    shards = sorted(f for f in os.listdir(src) if re.match(r"model-\d+-of-\d+\.safetensors$", f))
    sd, dropped, dtypes = {}, 0, set()
    for sh in shards:
        with safe_open(os.path.join(src, sh), framework="pt") as f:
            for k in f.keys():
                nk = rename(k)
                if nk is None:
                    dropped += 1
                    continue
                t = f.get_tensor(k)
                dtypes.add(str(t.dtype))
                sd[nk] = t.float().contiguous()
    print(f"read {len(sd)} tensors (+{dropped} VAE-decoder tensors dropped), src dtypes {dtypes}, {time.time()-t0:.0f}s")
    tied = torch.equal(sd["lm_head.weight"], sd["model.language_model.embed_tokens.weight"])
    print("lm_head == embed_tokens:", tied)
    dec["tie_word_embeddings"] = tied
    text = Qwen2Config(**dec)
    config = VibeVoiceAsrConfig(
        acoustic_tokenizer_encoder_config=acoustic, semantic_tokenizer_encoder_config=semantic,
        text_config=text, audio_token_id=151648, audio_bos_token_id=SP_START,
        audio_eos_token_id=SP_END, acoustic_tokenizer_chunk_size=1440000)
    model = VibeVoiceAsrForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print("strict load: missing", missing, "unexpected", unexpected)
    assert torch.equal(model.lm_head.weight, model.model.language_model.embed_tokens.weight) == tied
    model = model.float().eval()
    model.config.acoustic_tokenizer_encoder_config.vae_std = 0.0  # mean latents (deterministic)
    return model, text, tied


def save_lm(model, text, tied, out_lm):
    from transformers import Qwen2ForCausalLM
    lm = Qwen2ForCausalLM(text)
    lm_sd = {k.replace("model.language_model.", "model."): v for k, v in model.state_dict().items()
             if k.startswith("model.language_model.")}
    if not tied:
        lm_sd["lm_head.weight"] = model.lm_head.weight.data
    missing, unexpected = lm.load_state_dict(lm_sd, strict=False)
    if tied:
        missing = [m for m in missing if m != "lm_head.weight"]
        lm.tie_weights()
    assert not missing and not unexpected, (missing, unexpected)
    lm.generation_config.eos_token_id = [TCE, EOS]
    lm.generation_config.pad_token_id = EOS
    os.makedirs(out_lm, exist_ok=True)
    lm.save_pretrained(out_lm, safe_serialization=True, max_shard_size="5GB")
    print("saved", out_lm)


class StreamingAudioEncoder(torch.nn.Module):
    """[1, T, 3200] raw PCM (no normaliser: normalize_audio=false) -> {'features': [1, T, 1536]}."""

    def __init__(self, model):
        super().__init__()
        self.acoustic = model.model.acoustic_tokenizer_encoder
        self.semantic = model.model.semantic_tokenizer_encoder
        self.proj = model.model.multi_modal_projector

    def forward(self, audio):
        x = audio.reshape(1, 1, -1)
        return {"features": self.proj(self.acoustic(x).latents, self.semantic(x).latents)}


def encode_window(model, wav_window):  # np [n_samples] -> [1, n_frames, 1536]
    with torch.no_grad():
        return model.get_audio_features(input_values=torch.from_numpy(wav_window)[None, None, :]).pooler_output


def stage_enc(model, out_dir, T):
    enc = StreamingAudioEncoder(model).eval()
    wav = C.load_wav(os.path.join(BITNET, "fixtures", "clip02.wav"))
    frames = np.zeros((1, T, HOP), np.float32)
    frames[0] = wav[:T * HOP].reshape(T, HOP)
    audio = torch.from_numpy(frames)
    with torch.no_grad():
        ref = enc(audio)["features"][0].numpy()
        ref_hf = encode_window(model, frames.reshape(-1))[0].numpy()
    print(f"module vs HF get_audio_features: max|diff| {np.abs(ref-ref_hf).max():.3e}")
    import litert_torch
    t0 = time.time()
    edge = litert_torch.signature("encode", enc, sample_kwargs={"audio": audio}).convert()
    path = os.path.join(out_dir, f"audio_encoder_{T}f_fp32.tflite")
    edge.export(path)
    print(f"fp32 export {os.path.getsize(path)/1e6:.1f} MB in {time.time()-t0:.0f}s")
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=8)
    sig = it.get_signature_runner()
    print("signature io:", list(sig.get_input_details()), list(sig.get_output_details()))
    t0 = time.time()
    got = sig(audio=frames)["features"][0]
    dt = time.time() - t0
    err = np.abs(got - ref).max()
    cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
    print(f"tflite vs eager: max|diff| {err:.3e} cos {cos:.5f} ({dt*1000:.0f} ms/invoke, T={T})")
    return {"tflite": path, "max_diff": float(err), "cos": cos, "ms": dt * 1000}


@torch.no_grad()
def lm_step(model, embeds, past):
    out = model(inputs_embeds=embeds, past_key_values=past, use_cache=True, return_dict=True)
    return out.logits[:, -1, :], out.past_key_values


@torch.no_grad()
def run_streaming(model, tok, wav, chunk_frames, la_frames, max_new=256):
    """Vendor protocol (split_then_encode, pad_last_chunk)."""
    emb = model.get_input_embeddings()
    chunk_s, win_s = chunk_frames * HOP, (chunk_frames + la_frames) * HOP
    windows, s = [], 0
    while s < len(wav):
        seg = wav[s:min(s + win_s, len(wav))]
        if len(seg) < win_s:
            seg = np.pad(seg, (0, win_s - len(seg)))
        windows.append(seg.astype(np.float32))
        s += chunk_s
    ids = tok.encode(PROMPT, add_special_tokens=False)
    _, past = lm_step(model, emb(torch.tensor([ids])), None)
    sp_s, sp_e, tce = (emb(torch.tensor([[i]])) for i in (SP_START, SP_END, TCE))
    texts, n_tok = [], 0
    for w in windows:
        feats = encode_window(model, w)
        logits, past = lm_step(model, torch.cat([sp_s, feats, sp_e], 1), past)
        chunk = []
        for _ in range(max_new):
            nid = int(logits.argmax(-1))
            if nid in (TCE, EOS):
                break
            chunk.append(nid)
            logits, past = lm_step(model, emb(torch.tensor([[nid]])), past)
        _, past = lm_step(model, tce, past)
        texts.append(tok.decode(chunk, skip_special_tokens=True))
        n_tok += len(chunk)
    return texts, len(windows), n_tok


@torch.no_grad()
def run_oneshot(model, tok, wav, chunk_frames, la_frames, max_new=256, mode="single-pair"):
    """What a single-turn bundle would feed: prompt, ONE <sp_start> [all features] <sp_end>, then generate.
    mode single-pair  = runtime-style non-overlapping windows of (chunk+la) frames, features concatenated
    mode per-window   = every window bracketed by its own sp_start/sp_end but no text in between."""
    emb = model.get_input_embeddings()
    win_s = (chunk_frames + la_frames) * HOP
    n = math.ceil(len(wav) / win_s)
    wav_p = np.pad(wav, (0, n * win_s - len(wav))).astype(np.float32)
    feats = [encode_window(model, wav_p[i * win_s:(i + 1) * win_s]) for i in range(n)]
    sp_s, sp_e = (emb(torch.tensor([[i]])) for i in (SP_START, SP_END))
    ids = tok.encode(PROMPT, add_special_tokens=False)
    parts = [emb(torch.tensor([ids]))]
    if mode == "single-pair":
        parts += [sp_s] + feats + [sp_e]
    else:
        for f in feats:
            parts += [sp_s, f, sp_e]
    logits, past = lm_step(model, torch.cat(parts, 1), None)
    out = []
    for _ in range(max_new):
        nid = int(logits.argmax(-1))
        out.append(nid)
        if nid == EOS or len(out) > 400:
            break
        logits, past = lm_step(model, emb(torch.tensor([[nid]])), past)
    return tok.decode(out, skip_special_tokens=False)


def stage_eager(model, src, n_clips, chunk_frames, la_frames, out_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src)
    assert tok.convert_tokens_to_ids("<|text_chunk_end|>") == TCE
    meta = C.load_meta()[:n_clips]
    res, err_s, n_s = [], 0, 0
    for m in meta:
        wav = C.load_wav(os.path.join(BITNET, m["file"]))
        t0 = time.time()
        texts, nw, ntok = run_streaming(model, tok, wav, chunk_frames, la_frames)
        dt = time.time() - t0
        hyp = "".join(texts)
        # streaming output carries "speaker, content" keys; strip a leading speaker label for WER
        hyp_plain = re.sub(r"^\s*(speaker\s*\d+|\[?[Ss]peaker[^:\]]*\]?)\s*:\s*", "", hyp)
        e, n = C.wer_counts(C.norm_text(m["text"]), C.norm_text(hyp_plain))
        err_s, n_s = err_s + e, n_s + n
        one = run_oneshot(model, tok, wav, chunk_frames, la_frames)
        r = {"id": m["id"], "sec": round(len(wav) / SR, 2), "windows": nw, "gen_tokens": ntok,
             "streaming_chunks": texts, "streaming_wer": [e, n], "oneshot_single_pair": one,
             "sec_eager": round(dt, 1), "ref": m["text"]}
        print(json.dumps(r, ensure_ascii=False))
        res.append(r)
    print(f"STREAMING WER {err_s}/{n_s} = {100*err_s/max(1,n_s):.2f}% on {len(meta)} clips")
    json.dump(res, open(os.path.join(out_dir, "eager_precheck.json"), "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "hf"))
    ap.add_argument("--stage", default="load,enc,eager")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pre = json.load(open(os.path.join(args.src, "preprocessor_config.json")))
    chunk_frames, la_frames = pre["chunk_frames"], pre["lookahead_frames"]
    assert pre["normalize_audio"] is False and pre["speech_tok_compress_ratio"] == HOP
    print(f"chunk {chunk_frames} + lookahead {la_frames} frames = window {(chunk_frames+la_frames)*HOP/SR:.3f}s, advance {chunk_frames*HOP/SR:.3f}s")
    stages = args.stage.split(",")
    model, text, tied = build_native(args.src)
    if "load" in stages:
        save_lm(model, text, tied, os.path.join(args.out, "lm_native"))
        for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"):
            p = os.path.join(args.src, fn)
            if os.path.exists(p):
                os.system(f"cp '{p}' '{os.path.join(args.out, 'lm_native')}/'")
    if "enc" in stages:
        print(json.dumps(stage_enc(model, args.out, chunk_frames + la_frames)))
    if "eager" in stages:
        stage_eager(model, args.src, args.n_clips, chunk_frames, la_frames, args.out)
    print("PRECHECK_DONE")


if __name__ == "__main__":
    main()
