#!/usr/bin/env python3
"""transformers 5.14.1 eager reference for Qwen/Qwen3-ASR-1.7B-hf, fp32, CPU, greedy (do_sample=False, num_beams=1,
max_new_tokens=512), unpadded clips. Adapted from confucius4_r2t2_work/eager_hf.py.

Runs in ~/venvs/ltmain0918 (the export venv; nothing is installed into it).

  --prompt official   the checkpoint's own chat_template.jinja as `processor.apply_transcription_request(audio)` renders
                      it (no language hint): '<|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<|audio_start|>
                      <|audio_pad|><|audio_end|><|im_end|>\\n<|im_start|>assistant\\n'. The input_ids of the first clip
                      are asserted equal to apply_transcription_request's own.
  --prompt litert     the litert-torch export prompt (common.LITERT_TORCH_PROMPT, what litert-torch's Qwen3Asr bakes
                      into its encoder output)
  --clips all | crops  155 clips (FLEURS en / zh / ja x 50 + 5 examples) or the 60 five-second crops (make_crops.py)
Every JSONL starts with a {"type": "header"} line naming the stack and the prompt.

  ~/venvs/ltmain0918/bin/python eager_hf.py --prompt official --out out/ref_eager_official.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def official_prompt(proc):
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["official", "litert"], required=True)
    ap.add_argument("--clips", choices=["all", "crops"], default="all")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--model_dir", default=common.MODEL_DIR)
    ap.add_argument("--manifest", default=os.path.join(common.OUT, "crops79999", "manifest.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import transformers
    t0 = time.time()
    model, info = transformers.Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32, output_loading_info=True)
    model.eval()
    proc = transformers.AutoProcessor.from_pretrained(args.model_dir)
    print(json.dumps({"load_seconds": round(time.time() - t0, 2), "model_dir": args.model_dir,
                      "missing": len(info["missing_keys"]), "unexpected": len(info["unexpected_keys"]),
                      "mismatched": len(info.get("mismatched_keys", [])), "torch": torch.__version__,
                      "transformers": transformers.__version__}), flush=True)
    prompt = official_prompt(proc) if args.prompt == "official" else common.LITERT_TORCH_PROMPT
    if args.clips == "all":
        items = [{"name": c["name"], "config": c["config"], "path": c["path"]} for c in common.load_clips()]
    else:
        items = [{"name": c["name"], "config": c["config"], "path": c["path"]}
                 for c in json.load(open(args.manifest))["crops"]]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l).get("clip") for l in open(args.out) if l.strip()}
    checked = args.prompt != "official"
    with open(args.out, "a") as f:
        if not done:
            f.write(json.dumps({"type": "header", "stack": "transformers " + transformers.__version__ + ", torch " +
                                torch.__version__ + ", Qwen3ASRForConditionalGeneration.from_pretrained(" +
                                args.model_dir + ", dtype=float32) + AutoProcessor(text=prompt, audio=wav), CPU",
                                "model": common.REPO + "@" + common.REVISION, "prompt_name": args.prompt,
                                "prompt": prompt, "generate": "do_sample=False, num_beams=1, max_new_tokens=512",
                                "decode": "batch_decode(new ids, skip_special_tokens=True, "
                                          "clean_up_tokenization_spaces=False)",
                                "text_field": "after <asr_text>, stripped; language = 'language X' before the tag",
                                "loading_info": {k: len(v) for k, v in info.items() if not isinstance(v, (str, int))}},
                               ensure_ascii=False) + "\n")
        for it in items:
            if it["name"] in done:
                continue
            wav = common.read_wav(it["path"])
            inputs = proc(text=prompt, audio=wav, return_tensors="pt")
            if not checked:
                ref = proc.apply_transcription_request(audio=wav, return_tensors="pt")
                assert torch.equal(ref["input_ids"], inputs["input_ids"]), "official prompt != apply_transcription_request"
                assert torch.equal(ref["input_features"], inputs["input_features"])
                checked = True
                print("CHECK official prompt input_ids == apply_transcription_request input_ids", flush=True)
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=512)
            dt = time.time() - t1
            n_in = inputs["input_ids"].shape[1]
            gen = out[0, n_in:].tolist()
            raw = proc.batch_decode(out[:, n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            lang, text = common.split_raw(raw)
            row = {"clip": it["name"], "config": it["config"], "audio_seconds": round(len(wav) / 16000, 4),
                   "raw": raw, "text": text, "language": lang, "gen_ids": gen, "n_input_ids": n_in,
                   "seconds": round(dt, 3)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(it["name"], round(dt, 2), lang, text[:80], flush=True)


if __name__ == "__main__":
    main()
