#!/usr/bin/env python3
"""The source model's transcripts of the same clips: transformers GraniteSpeechForConditionalGeneration with the repo's
processor and chat template, the BF16 weights computed in fp32 on the CPU, greedy (do_sample False, num_beams 1,
max_new_tokens 256), the card's punctuated-transcription prompt. Audio = the WAV samples as torchaudio.load(normalize=True)
returns them (int16 / 32768, or the stored float32). Writes one JSONL row per clip: id, text (the new tokens decoded with
skip_special_tokens), the generated ids, the prompt and audio token counts.

  python reference_hf.py --src <snapshot dir> --clips clips/clips.json --out reference_hf.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=C.MAX_NEW)
    args = ap.parse_args()
    import torch
    from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.src)
    model, info = GraniteSpeechForConditionalGeneration.from_pretrained(args.src, dtype=torch.float32,
                                                                        output_loading_info=True)
    assert all(len(v) == 0 for v in info.values()), info
    model = model.eval()
    tok = proc.tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": C.USER_CONTENT}], tokenize=False,
                                     add_generation_prompt=True)
    assert prompt == C.PROMPT, (prompt, C.PROMPT)
    print(f"loaded in {time.time() - t0:.1f} s; prompt {prompt!r}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fo:
        for c in C.load_clips(args.clips):
            wav = torch.from_numpy(C.wav_float(c["abspath"])).unsqueeze(0)
            t1 = time.time()
            inputs = proc(prompt, wav, device="cpu", return_tensors="pt")
            ids_in = inputs["input_ids"]
            n_audio = int((ids_in == C.AUDIO_ID).sum())
            assert n_audio == C.num_audio_tokens(wav.shape[-1]), (n_audio, wav.shape)
            feed = {k: v for k, v in inputs.items() if k != "audio_embed_sizes"}
            with torch.no_grad():
                gen = model.generate(**feed, max_new_tokens=args.max_new, do_sample=False, num_beams=1)
            new = gen[0, ids_in.shape[-1]:].tolist()
            eos = len(new) > 0 and new[-1] == C.EOS_ID
            text = tok.decode(new, skip_special_tokens=True)
            row = {"id": c["id"], "text": text, "gen_ids": new[:-1] if eos else new, "eos": eos,
                   "prompt_tokens": int(ids_in.shape[-1]), "audio_tokens": n_audio, "wall_s": round(time.time() - t1, 2)}
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[{c['id']}] audio tokens {n_audio} | {text}", flush=True)
    print("REFERENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
