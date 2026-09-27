"""Six stress prompts at 512x512 and 256x256 through the all-tflite pipeline,
plus one labelled grid (row per size, 256 shown 2x nearest so both rows share a
cell size) and a JSON of per-stage timings. Graphs load once per size.

    python sweep_sizes.py --model-dir <hub dir> --dit256 <file> --vae256 <file> --out sweep/
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import SEQ, Graph, flowmatch_sigmas, unpatchify  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PROMPTS = [
    ("bonsai",   "a small bonsai tree in a blue ceramic pot on a wooden table, soft daylight"),
    ("text",     "a weathered wooden shop sign that reads OPEN in carved letters, morning light"),
    ("portrait", "a close-up portrait of an elderly fisherman with weathered skin and a grey beard"),
    ("texture",  "an extreme macro photograph of a peacock feather, iridescent barbs in sharp focus"),
    ("gradient", "an empty beach at sunset, smooth clear gradient sky from orange to deep blue"),
    ("clutter",  "a cluttered workshop bench covered in hand tools, screws and wood shavings"),
]

p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--dit256", required=True)
p.add_argument("--vae256", required=True)
p.add_argument("--out", default="sweep")
p.add_argument("--steps", type=int, default=4)
p.add_argument("--threads", type=int, default=os.cpu_count())
p.add_argument("--grid-only", action="store_true", help="rebuild the grid from existing PNGs")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
d = args.model_dir
meta = json.load(open(os.path.join(d, "pipeline_meta.json")))
bn_scale = np.asarray(meta["latent_bn_scale"], np.float32)
bn_shift = np.asarray(meta["latent_bn_shift"], np.float32)
SIZES = {512: (os.path.join(d, meta["files"]["dit"]), os.path.join(d, meta["files"]["vae"])),
         256: (args.dit256, args.vae256)}

if not args.grid_only:
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    te = Graph(os.path.join(d, meta["files"]["textenc"]), args.threads)
    embeds, te_t = {}, []
    for name, prompt in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=SEQ)
        t0 = time.time()
        embeds[name] = te(enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32))
        te_t.append(time.time() - t0)
    del te
    timing = {"textenc_s": float(np.mean(te_t))}
    txt_ids = np.stack([np.zeros(SEQ)] * 3 + [np.arange(SEQ)], -1).astype(np.float32)

    for size, (dit_path, vae_path) in SIZES.items():
        grid = size // 16
        tokens = grid * grid
        hh, ww = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
        img_ids = np.stack([np.zeros_like(hh), hh, ww, np.zeros_like(hh)], -1).reshape(tokens, 4).astype(np.float32)
        sigmas = flowmatch_sigmas(args.steps, tokens)
        dit = Graph(dit_path, args.threads)
        lats, step_t = {}, []
        for i, (name, _) in enumerate(PROMPTS):
            lat = np.random.default_rng(i).standard_normal((1, tokens, 128)).astype(np.float32)
            for k in range(args.steps):
                t0 = time.time()
                v = dit(lat, embeds[name], sigmas[k:k + 1], img_ids, txt_ids)
                step_t.append(time.time() - t0)
                lat = lat + (sigmas[k + 1] - sigmas[k]) * v
            lats[name] = lat
            print(f"{size} {name:9s} DiT {sum(step_t[-args.steps:]):.1f}s", flush=True)
        del dit
        vae = Graph(vae_path, args.threads)
        vae_t = []
        for name, _ in PROMPTS:
            t0 = time.time()
            y = vae(unpatchify(lats[name][0], bn_scale, bn_shift, grid))
            vae_t.append(time.time() - t0)
            rgb = (np.clip(y[0] / 2 + 0.5, 0, 1) * 255).round().astype(np.uint8)
            Image.fromarray(rgb.transpose(1, 2, 0)).save(f"{args.out}/{name}__{size}.png")
        del vae
        timing[str(size)] = {"dit_step_s": float(np.mean(step_t)), "vae_s": float(np.mean(vae_t)),
                             "image_s": float(np.mean(step_t) * args.steps + np.mean(vae_t) + timing["textenc_s"])}
        print(f"{size}: DiT {np.mean(step_t):.2f} s/step, VAE {np.mean(vae_t):.2f} s", flush=True)

    json.dump(timing, open(f"{args.out}/timing_mac.json", "w"), indent=1)

# grid: one row per size, one column per prompt, 512-px cells, labels in a real TTF
cell, pad, top, left = 512, 12, 44, 230
W = left + len(PROMPTS) * (cell + pad)
H = top + 2 * (cell + pad)
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
font = None
for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
    if os.path.exists(f):
        font = ImageFont.truetype(f, 28)
        break
for j, (name, _) in enumerate(PROMPTS):
    draw.text((left + j * (cell + pad) + 8, 8), name, fill="black", font=font)
for r, size in enumerate((512, 256)):
    y0 = top + r * (cell + pad)
    label = f"{size}x{size}" + (" (2x)" if size == 256 else "")
    draw.text((8, y0 + cell // 2 - 14), label, fill="black", font=font)
    for j, (name, _) in enumerate(PROMPTS):
        im = Image.open(f"{args.out}/{name}__{size}.png")
        if size != cell:
            im = im.resize((cell, cell), Image.NEAREST)
        canvas.paste(im, (left + j * (cell + pad), y0))
canvas.save(f"{args.out}/grid_512_vs_256.png")
print(f"grid -> {args.out}/grid_512_vs_256.png; timings -> {args.out}/timing_mac.json")
