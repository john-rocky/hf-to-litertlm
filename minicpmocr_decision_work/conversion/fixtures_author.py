#!/usr/bin/env python3
"""The author's numbers the graph checks compare against, one npz per record or picture (out/fix/). Needs fetch_inputs.py first.
The author's modules in the reference environment (transformers 5.10.4, torch 2.8.0, score.py's prompt, picture padding and token gates, loaded
by path), float32, CPU, batch 1, the depthwise patch of the reference (fast_dwconv.py).

    $REF -I conversion/fixtures_author.py <work> --pass 1 [--threads 8]

rec_<id>.npz (a record of the list, or ex_<n> = the official case n): ids int32 [T], grids int32 [k, 2], pos3 int32 [3, T] (the
  planes the text model reads), inputs_embeds f32 [T, 2048] (after masked_scatter), cos, sin f32 [T, 64] (the text model's rotary
  output), hidden f32 [T, 2048] (last_hidden_state), logits f32 [2], prob_1.
pic_<id>_<which>.npz (a picture of the list): pixel_values f32 [n, 1536], grid int32 [2], rows f32 [n / 4, 2048] (`model.visual`
  pooler_output), cos, sin f32 [n, 64] and pos f32 [n, 1024] (the author's own tables), hidden_last f32 (the blocks' last rows are
  not kept). A file that exists is skipped."""
import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("--pass", dest="pass_", type=int, default=1)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
WORK = a.work.resolve()
torch.set_num_threads(a.threads)
torch.set_grad_enabled(False)


def by_path(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


score = by_path("author_score", WORK / "author/scripts/minicpm-ocr-jev/eval/score.py")
dw = by_path("fast_dwconv", WORK / "conversion/fast_dwconv.py")
dw.install()
SNAP, BENCH, TMP = WORK / "snapshot", WORK / "bench", WORK / "out/tmp_padded"
FIX = WORK / "out/fix"
FIX.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

from transformers import AutoProcessor, Qwen3_5ForSequenceClassification  # noqa: E402
from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids  # noqa: E402

processor = AutoProcessor.from_pretrained(SNAP)
t0 = time.time()
model = Qwen3_5ForSequenceClassification.from_pretrained(SNAP, dtype=torch.float32).eval()
print(f"loaded in {time.time() - t0:.1f} s", flush=True)
core = model.model
cap = {}
core.language_model.register_forward_pre_hook(lambda m, ar, kw: cap.update(lang_kwargs=kw), with_kwargs=True)
core.language_model.rotary_emb.register_forward_hook(lambda m, i, o: cap.update(rotary=o))
core.language_model.register_forward_hook(lambda m, i, o: cap.update(last=o.last_hidden_state))


def run_record(tag, images, text):
    path = FIX / f"rec_{tag}.npz"
    if path.exists():
        return
    t = time.perf_counter()
    images = [score.pad_extreme_aspect_image(p, TMP) for p in images]
    prompt = score.build_prompt(text)
    score.check_prompt_tail(prompt)
    inputs = processor(images=images, text=[prompt], return_tensors="pt")
    score.assert_pooled_token(inputs)
    logits = model(**inputs).logits
    prob = torch.softmax(logits.float(), dim=-1)[0].tolist()
    kw = cap["lang_kwargs"]
    pos = kw["position_ids"]
    assert pos.shape[1] == 1 and pos.shape[0] == 3, pos.shape
    cos, sin = cap["rotary"]
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=inputs["input_ids"][0].numpy().astype(np.int32), grids=inputs["image_grid_thw"][:, 1:].numpy().astype(np.int32),
             pos3=pos[:, 0].numpy().astype(np.int32), inputs_embeds=kw["inputs_embeds"][0].float().numpy(), cos=cos[0].float().numpy(),
             sin=sin[0].float().numpy(), hidden=cap["last"][0].float().numpy(), logits=logits[0].float().numpy(), prob_1=np.float64(prob[1]))
    tmp.rename(path)
    print(tag, "T", inputs["input_ids"].shape[1], "prob", round(prob[1], 6), f"{time.perf_counter() - t:.1f} s",
          "rss", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2), flush=True)


def run_picture(tag, image):
    path = FIX / f"pic_{tag}.npz"
    if path.exists():
        return
    t = time.perf_counter()
    image = score.pad_extreme_aspect_image(image, TMP)
    out = processor.image_processor(images=[image], return_tensors="pt")
    pv, grid = out["pixel_values"], out["image_grid_thw"]
    v = core.visual
    rows = v(pv.to(torch.float32), grid_thw=grid, return_dict=True).pooler_output
    if isinstance(rows, (tuple, list)):
        assert len(rows) == 1
        rows = rows[0]
    idx, w = get_vision_bilinear_indices_and_weights(grid, num_grid_per_side=v.num_grid_per_side, spatial_merge_size=v.config.spatial_merge_size)
    pos = (v.pos_embed(idx) * w[:, :, None]).sum(0)
    rot = v.rotary_pos_emb(get_vision_position_ids(grid, v.spatial_merge_size)).reshape(pv.shape[0], -1)
    emb = torch.cat((rot, rot), dim=-1)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, pixel_values=pv.float().numpy(), grid=grid[0, 1:].numpy().astype(np.int32), rows=rows.float().numpy(), pos=pos.float().numpy(),
             cos=emb.cos().float().numpy(), sin=emb.sin().float().numpy())
    tmp.rename(path)
    print(tag, "n", pv.shape[0], f"{time.perf_counter() - t:.1f} s", "rss", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2), flush=True)


fx = json.loads((WORK / "conversion/sets/fixtures.json").read_text())
rows_all = {r["item_id"]: r for r in map(json.loads, (BENCH / "data/jevocrbench.jsonl").read_text(encoding="utf-8").splitlines())}
if a.pass_ == 1:
    for n, d in enumerate(sorted((WORK / "author/scripts/minicpm-ocr-jev/examples").iterdir())):
        c = json.loads((d / "case.json").read_text())
        cand = next(x for x in c["candidates"] if x["role"] == c["reference_render_of"])
        run_record(f"ex_{n}", [str(d / c["block_image"]), str(d / c["reference_render"])], cand["text"])
for t in sorted(fx["tower"], key=lambda x: x["n"]):
    if t["pass_"] == a.pass_:
        r = rows_all[t["item_id"]]
        run_picture(f"{t['item_id']}_{t['which']}", str(BENCH / (r["image_path"] if t["which"] == 0 else r["render_image_path"])))
for d in sorted(fx["decide"], key=lambda x: x["T"]):
    if d["pass_"] == a.pass_:
        r = rows_all[d["item_id"]]
        run_record(str(d["item_id"]), [str(BENCH / r["image_path"]), str(BENCH / r["render_image_path"])], str(r["ocr_text"] or ""))
print("DONE pass", a.pass_, "peak_rss_gib", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2), flush=True)
