#!/usr/bin/env python3
"""Eager fp32 oracle for VibeVoice-ASR-BitNet (native class, ternarized LM, deterministic
latents) on the 20 LibriSpeech dev-clean fixtures at 24 kHz.  Reports per-clip transcripts
and corpus WER; --variant probes the prompt layout (see common.build_prompt_ids).

  python eager_gate.py [--variant exact|genprompt|nodur|wrongdur] [--limit N] [--max-new 256]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(C.WORK, "hf_native"))
    ap.add_argument("--variant", default="exact")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--encoder", default="", help="audio-encoder .tflite: take features from it (framed, "
                    "zero-padded window) and run the eager LM on them — isolates encoder quantisation")
    args = ap.parse_args()
    torch.manual_seed(0)

    from transformers import AutoTokenizer, VibeVoiceAsrForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(args.model)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()
    rot = model.model.language_model.rotary_emb
    assert rot.inv_freq.abs().max() > 0, "rotary inv_freq zeroed on load"
    # Deterministic latents: the HF class adds vae_std*randn noise to the acoustic latents
    # even at inference; VibeASR.cpp (the official runtime) uses the mean.  Match the runtime.
    model.config.acoustic_tokenizer_encoder_config.vae_std = 0.0

    enc_sig, enc_T = None, 0
    if args.encoder:
        from ai_edge_litert.interpreter import Interpreter
        enc_it = Interpreter(model_path=args.encoder, num_threads=8)
        enc_sig = enc_it.get_signature_runner()
        enc_T = int(enc_sig.get_input_details()["audio"]["shape"][1])
        print(f"encoder tflite {args.encoder}: window {enc_T} frames")

    def greedy_from_embeds(e, steps):
        """Manual KV-cached greedy loop (the composite class rejects inputs_embeds in generate)."""
        lm, head, emb = model.model.language_model, model.lm_head, model.model.language_model.embed_tokens
        out_ids = []
        with torch.no_grad():
            o = lm(inputs_embeds=e, use_cache=True)
            past, lg = o.past_key_values, head(o.last_hidden_state[:, -1])[0]
            for _ in range(steps):
                nid = int(torch.argmax(lg))
                out_ids.append(nid)
                if nid in (C.IM_END, C.ENDOFTEXT):
                    break
                o = lm(inputs_embeds=emb(torch.tensor([[nid]])), past_key_values=past, use_cache=True)
                past, lg = o.past_key_values, head(o.last_hidden_state[:, -1])[0]
        return out_ids

    meta = C.load_meta()
    if args.limit:
        meta = meta[:args.limit]
    rows, errs, words = [], 0, 0
    for m in meta:
        wav = C.load_wav(os.path.join(C.WORK, m["file"]))
        dur = len(wav) / C.SR
        x = C.normalize_audio(wav)
        x, n_tok = C.pad_to_hop(x)
        ids = C.build_prompt_ids(tok, n_tok, dur, args.variant)
        input_ids = torch.tensor([ids])
        input_values = torch.from_numpy(x)[None, None, :]
        t0 = time.time()
        if enc_sig is not None:
            # Runtime-style framing: RAW (un-normalised) PCM, zero-padded to the window; the
            # encoder graph normalises. Long clips are chunked by window like the executor does.
            raw, _ = C.pad_to_hop(wav)
            feats = []
            for s0 in range(0, n_tok, enc_T):
                frames = np.zeros((1, enc_T, C.HOP), np.float32)
                k = min(enc_T, n_tok - s0)
                frames[0, :k] = raw[s0 * C.HOP:(s0 + k) * C.HOP].reshape(k, C.HOP)
                feats.append(enc_sig(audio=frames)["features"][0, :k])
            feats = torch.from_numpy(np.concatenate(feats, 0))
            e = model.model.language_model.embed_tokens(input_ids).clone()
            e[input_ids == C.SPEECH_PAD] = feats
            gen = greedy_from_embeds(e, args.max_new)
        else:
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, input_values=input_values,
                                     attention_mask=torch.ones_like(input_ids),
                                     max_new_tokens=args.max_new, do_sample=False,
                                     eos_token_id=[C.IM_END, C.ENDOFTEXT], pad_token_id=C.ENDOFTEXT)
            gen = out[0, input_ids.shape[1]:].tolist()
        dt = time.time() - t0
        body = C.strip_header(tok, gen)
        hyp = tok.decode(body, skip_special_tokens=True).strip()
        e, n = C.wer_counts(C.norm_text(m["text"]), C.norm_text(hyp))
        errs += e
        words += n
        rows.append({"id": m["id"], "dur": round(dur, 2), "n_audio_tokens": n_tok,
                     "prompt_len": len(ids), "gen_tokens": len(gen), "raw_gen": tok.decode(gen),
                     "hyp": hyp, "ref": m["text"], "errs": e, "words": n, "sec": round(dt, 1)})
        print(f"[{m['id']}] {dur:5.1f}s ntok={n_tok:3d} gen={len(gen):3d} errs={e}/{n} {dt:5.1f}s | {hyp}")
        sys.stdout.flush()
    print(f"variant={args.variant} corpus WER {errs}/{words} = {100*errs/max(words,1):.2f}%")
    tag = args.tag or args.variant
    json.dump({"variant": args.variant, "model": args.model, "encoder": args.encoder, "wer_errs": errs, "ref_words": words,
               "rows": rows}, open(os.path.join(C.WORK, f"eager_gate_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
