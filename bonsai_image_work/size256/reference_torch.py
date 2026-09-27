"""Torch (diffusers) reference image at a given size, sharing the prompt embeds
and the initial noise with generate.py so the two images differ only by the
DiT + VAE conversion.

- prompt embeds: the shipped text-encoder graph (same file generate.py uses),
  so the comparison isolates the DiT and VAE decoder;
- initial noise: numpy default_rng(seed) in packed (1, tokens, 128) order,
  exactly generate.py's draw; it is written to --noise-out so generate.py can
  load it via BONSAI_INIT_LATENTS;
- transformer + VAE: torch fp32 through Flux2KleinPipeline (untouched fp64 rope,
  the pipeline's own schedule, packing, BatchNorm affine and decode).

    python reference_torch.py --hub-dir <dir with textenc + tokenizer> --size 256 \
        --prompt "..." --seed 0 --out ref_torch_256.png --noise-out noise_256.bin
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                       Flux2KleinPipeline, Flux2Transformer2DModel)
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import SEQ, Graph, flowmatch_sigmas  # noqa: E402

SNAP = os.environ.get("SNAP") or sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--prism-ml--bonsai-image-ternary-4B-unpacked/snapshots/*")))[-1]

p = argparse.ArgumentParser()
p.add_argument("--hub-dir", required=True)
p.add_argument("--size", type=int, default=256)
p.add_argument("--prompt", default="a small bonsai tree in a blue ceramic pot")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--textenc", default="textenc_int4.tflite")
p.add_argument("--out", default="ref_torch.png")
p.add_argument("--noise-out", default="noise.bin")
args = p.parse_args()
grid = args.size // 16
tokens = grid * grid

tok = AutoTokenizer.from_pretrained(os.path.join(args.hub_dir, "tokenizer"))
text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                               tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=SEQ)
t0 = time.time()
te = Graph(os.path.join(args.hub_dir, args.textenc))
embeds = te(enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32))
del te
print(f"prompt embeds {embeds.shape} from {args.textenc} in {time.time()-t0:.1f}s", flush=True)

packed = np.random.default_rng(args.seed).standard_normal((1, tokens, 128)).astype(np.float32)
packed.tofile(args.noise_out)
unpacked = torch.from_numpy(packed).permute(0, 2, 1).reshape(1, 128, grid, grid).contiguous()

idx = json.load(open(os.path.join(SNAP, "model_index.json")))
pipe = Flux2KleinPipeline(
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(SNAP, subfolder="scheduler"),
    vae=AutoencoderKLFlux2.from_pretrained(SNAP, subfolder="vae", torch_dtype=torch.float32),
    text_encoder=None, tokenizer=None,
    transformer=Flux2Transformer2DModel.from_pretrained(SNAP, subfolder="transformer",
                                                        torch_dtype=torch.float32),
    is_distilled=bool(idx.get("is_distilled", False)))
print(f"pipeline ready (is_distilled={pipe.config.is_distilled})", flush=True)

step_t = []
def on_step(pipeline, i, t, kw):
    step_t.append(time.time())
    return kw

t0 = time.time()
img = pipe(prompt=None, prompt_embeds=torch.from_numpy(embeds), height=args.size,
           width=args.size, num_inference_steps=args.steps, guidance_scale=1.0,
           latents=unpacked, generator=torch.Generator("cpu").manual_seed(args.seed),
           max_sequence_length=SEQ, callback_on_step_end=on_step).images[0]
wall = time.time() - t0
img.save(args.out)
sig = pipe.scheduler.sigmas.numpy()
mine = flowmatch_sigmas(args.steps, tokens)
print(f"pipeline sigmas {np.round(sig, 4).tolist()}")
print(f"generate.py     {np.round(mine, 4).tolist()}  max|d|={np.abs(sig - mine).max():.2e}")
if len(step_t) > 1:
    print(f"torch DiT step ~{np.diff(step_t).mean():.2f} s (fp32, CPU)")
print(f"generated {args.size}x{args.size} in {wall:.0f}s -> {args.out}; noise -> {args.noise_out}")
