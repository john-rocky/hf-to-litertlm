#!/usr/bin/env python3
"""transformers 5.14.1 eager reference for Confucius4-R2T2 on the converted -hf layout (out/hf_layout),
fp32, CPU, greedy (do_sample=False, num_beams=1, max_new_tokens=512).

Runs in ~/venvs/ltmain0918 (the export venv; nothing is installed into it).

  --mode layout    the 14 layout-check clips with the vendor prompt string (out/vendor_prompt.txt,
                   written by eager_vendor.py --mode layout) passed as text=; saves the same fields as
                   the vendor side to out/layout_cmp/hf/<clip>.npz and the from_pretrained loading info
                   (check b) to out/layout_cmp/hf/loading_info.json.
  --mode full_vendorprompt
                   all 155 clips with the vendor prompt string (out/vendor_prompt.txt = qwen_asr _build_text_prompt(
                   context="", force_language=None): empty system turn + user audio + assistant) -> the 155-clip
                   reference with the original repository's prompt (transformers 5.14.1 uses the same 104-token
                   encoder windows as the vendor's vLLM path) -> out/ref_eager_vendorprompt.jsonl
  --mode full      all 155 clips with the litert-torch export prompt (common.LITERT_TORCH_PROMPT, the
                   input litert-torch Qwen3Asr.run_original_model generates from) -> out/ref_eager_hfprompt.jsonl
  --mode crops     the 60 five-second crops (out/crops/manifest.json) with the litert-torch export prompt
                   -> out/ref_eager_hfprompt_crops.jsonl
Every JSONL starts with a {"type": "header"} line naming the stack and the prompt.
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


def load(threads, model_dir):
    torch.set_num_threads(threads)
    import transformers
    t0 = time.time()
    model, info = transformers.Qwen3ASRForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, output_loading_info=True)
    model.eval()
    proc = transformers.AutoProcessor.from_pretrained(model_dir)
    return model, proc, info, time.time() - t0


def run_one(model, proc, prompt, wav, save=None):
    inputs = proc(text=prompt, audio=wav, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=512,
                             return_dict_in_generate=True, output_logits=True)
    dt = time.time() - t0
    n_in = inputs["input_ids"].shape[1]
    gen = out.sequences[0, n_in:].tolist()
    raw = proc.batch_decode(out.sequences[:, n_in:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)[0]
    if save:
        np.savez(save, input_ids=inputs["input_ids"][0].numpy(),
                 input_features=inputs["input_features"][0].float().numpy(),
                 input_features_mask=inputs["input_features_mask"][0].numpy(),
                 gen=np.array(gen, dtype=np.int64), logits0=out.logits[0][0].float().numpy())
    return raw, gen, n_in, dt, inputs


def mode_layout(model, proc, info):
    out_dir = os.path.join(common.OUT, "layout_cmp", "hf")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "loading_info.json"), "w") as f:
        json.dump({k: sorted(map(str, v)) if not isinstance(v, (str, int)) else v for k, v in info.items()},
                  f, indent=1)
    prompt = open(os.path.join(common.OUT, "vendor_prompt.txt")).read()
    rows = []
    for c in common.layout_check_clips():
        wav = common.read_wav(c["path"])
        raw, gen, n_in, dt, inputs = run_one(model, proc, prompt, wav,
                                             save=os.path.join(out_dir, c["name"] + ".npz"))
        lang, text = common.split_raw(raw)
        rows.append({"clip": c["name"], "raw": raw, "text": text, "language": lang, "n_gen": len(gen),
                     "n_input_ids": n_in, "feat_shape": list(inputs["input_features"][0].shape),
                     "seconds": round(dt, 3)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    with open(os.path.join(out_dir, "rows.json"), "w") as f:
        json.dump({"prompt": prompt, "rows": rows}, f, ensure_ascii=False, indent=1)


def mode_list(model, proc, items, out_path, prompt, prompt_name, model_dir):
    import transformers
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l).get("clip") for l in open(out_path) if l.strip()}
    with open(out_path, "a") as f:
        if not done:
            f.write(json.dumps({"type": "header", "stack": "transformers " + transformers.__version__ + ", torch " +
                                torch.__version__ + ", Qwen3ASRForConditionalGeneration.from_pretrained(" + model_dir +
                                ", dtype=float32) + AutoProcessor(text=prompt, audio=wav), CPU",
                                "prompt_name": prompt_name, "prompt": prompt,
                                "generate": "do_sample=False, num_beams=1, max_new_tokens=512",
                                "decode": "batch_decode(new ids, skip_special_tokens=True, "
                                          "clean_up_tokenization_spaces=False)",
                                "text_field": "after <asr_text>, stripped; language = 'language X' before the tag"},
                               ensure_ascii=False) + "\n")
        for it in items:
            if it["name"] in done:
                continue
            wav = common.read_wav(it["path"])
            raw, gen, n_in, dt, _ = run_one(model, proc, prompt, wav)
            lang, text = common.split_raw(raw)
            row = {"clip": it["name"], "config": it["config"], "audio_seconds": round(len(wav) / 16000, 4),
                   "raw": raw, "text": text, "language": lang, "gen_ids": gen, "n_input_ids": n_in,
                   "seconds": round(dt, 3)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(it["name"], round(dt, 2), lang, text[:80], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["layout", "full_vendorprompt", "full", "crops"], required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--model_dir", default=common.HF_LAYOUT)
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default=os.path.join(common.OUT, "crops", "manifest.json"))
    args = ap.parse_args()
    model, proc, info, load_s = load(args.threads, args.model_dir)
    import transformers
    print(json.dumps({"load_seconds": round(load_s, 2), "model_dir": args.model_dir,
                      "missing": len(info["missing_keys"]), "unexpected": len(info["unexpected_keys"]),
                      "mismatched": len(info.get("mismatched_keys", [])), "torch": torch.__version__,
                      "transformers": transformers.__version__}), flush=True)
    if args.mode == "layout":
        mode_layout(model, proc, info)
    elif args.mode == "full_vendorprompt":
        prompt = open(os.path.join(common.OUT, "vendor_prompt.txt")).read()
        mode_list(model, proc, common.load_clips(),
                  args.out or os.path.join(common.OUT, "ref_eager_vendorprompt.jsonl"), prompt,
                  "vendor _build_text_prompt(context='', force_language=None)", args.model_dir)
    elif args.mode == "full":
        mode_list(model, proc, common.load_clips(),
                  args.out or os.path.join(common.OUT, "ref_eager_hfprompt.jsonl"), common.LITERT_TORCH_PROMPT,
                  "litert-torch Qwen3AsrProcessor._PROMPT", args.model_dir)
    else:
        items = json.load(open(args.manifest))["crops"]
        mode_list(model, proc, items, args.out or os.path.join(common.OUT, "ref_eager_hfprompt_crops.jsonl"),
                  common.LITERT_TORCH_PROMPT, "litert-torch Qwen3AsrProcessor._PROMPT", args.model_dir)


if __name__ == "__main__":
    main()
