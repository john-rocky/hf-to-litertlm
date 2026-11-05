#!/usr/bin/env python3
"""The author's score path as the reference: Qwen3_5ForSequenceClassification, transformers 5.10.4, prompt, picture padding
and the token gate from the author's score.py (loaded by path, its functions unchanged), default attention, batch 1, CPU.
One row per record, appended and flushed, so a stopped run resumes. Numbers only: no text and no pixels in the rows.

    $REF -I conversion/ref_run.py <work> --set A --dtype fp32 --fast-dwconv [--arm base] [--max-n 9216]
    $REF -I conversion/ref_run.py <work> --examples --dtype fp32          (the three official cases, plus the control:
                                                                                     score.py's own score_batch on the same input)
Arms (conversion/sets/red_arms.json partners for R1, R4):
    base  [original, render], the record's text. Each row also carries the logits one position before the pooled token (R3).
    R1    the partner's text, this record's two pictures.
    R2    [render, original], the record's text.
    R4    [the partner's original, this record's render], the record's text.
    S448  both pictures stretched to 448 x 448 (Pillow BICUBIC, the aspect is lost: what a fixed-size runtime resize does).
    P448  both pictures padded white to a centred square, then to 448 x 448 (the aspect is kept, the picture gets smaller).

Output: results/ref/<tag>.jsonl, skipped records in results/ref/<tag>.skipped.jsonl; hidden[pooled] in out/ref/<tag>/<id>.npy.
Tag = <set>__<arm>__<dtype>[__dw]. Largest records last (sorted by the distribution's N and T), so a watchdog stop costs least.
"""
import argparse, hashlib, importlib.util, json, os, platform, resource, signal, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("--set", choices=("A", "red", "floor", "device", "ids"), default=None)
ap.add_argument("--ids", default=None, help="comma-separated item ids with --set ids")
ap.add_argument("--only", default=None, help="comma-separated item ids inside the set, same tag: one process per record, so a memory stop costs that record")
ap.add_argument("--examples", action="store_true")
ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
ap.add_argument("--arm", choices=("base", "R1", "R2", "R4", "S448", "P448"), default="base")
ap.add_argument("--fast-dwconv", action="store_true")
ap.add_argument("--max-n", type=int, default=9216, help="records with a picture above this many patches go to the next pass")
ap.add_argument("--max-seconds", type=int, default=1800, help="per record")
ap.add_argument("--threads", type=int, default=4, help="torch threads")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
WORK = a.work.resolve()


def by_path(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


score = by_path("author_score", WORK / "author/scripts/minicpm-ocr-jev/eval/score.py")
SNAP, BENCH, TMP = WORK / "snapshot", WORK / "bench", WORK / "out/tmp_padded"
TMP.mkdir(parents=True, exist_ok=True)
(WORK / "results/ref").mkdir(parents=True, exist_ok=True)
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
if a.fast_dwconv:
    dw = by_path("fast_dwconv", WORK / "conversion/fast_dwconv.py")
    dw.install()

from transformers import AutoProcessor, Qwen3_5ForSequenceClassification  # noqa: E402

processor = AutoProcessor.from_pretrained(SNAP)
t0 = time.time()
model, info = Qwen3_5ForSequenceClassification.from_pretrained(SNAP, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16,
                                                               output_loading_info=True)
model.eval()
assert model.config.num_labels == 2
load_s = time.time() - t0
missing = [k for k in info["missing_keys"]]
unexpected = [k for k in info["unexpected_keys"]]
print(f"loaded in {load_s:.1f} s, missing {missing[:6]}, unexpected {unexpected[:6]}, attn text "
      f"{model.config.get_text_config()._attn_implementation} vision {model.config.vision_config._attn_implementation}, "
      f"dtype {next(model.parameters()).dtype}, threads {torch.get_num_threads()}", flush=True)
assert not missing and not unexpected
capture = {}
model.score.register_forward_hook(lambda m, i, o: capture.__setitem__("hidden", i[0]))


def stretch(path, pad):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    if pad:
        s = max(rgb.size)
        c = Image.new("RGB", (s, s), "white")
        c.paste(rgb, ((s - rgb.size[0]) // 2, (s - rgb.size[1]) // 2))
        rgb = c
    return rgb.resize((448, 448), Image.Resampling.BICUBIC)


def forward(images, text):
    images = [score.pad_extreme_aspect_image(p, TMP) if isinstance(p, str) else p for p in images]
    prompt = score.build_prompt(text)
    score.check_prompt_tail(prompt)
    inputs = processor(images=images, text=[prompt], return_tensors="pt")
    score.assert_pooled_token(inputs)
    logits = model(**inputs).logits
    prob = torch.softmax(logits.float(), dim=-1)[0].tolist()
    hid = capture["hidden"][0]
    pooled = hid.shape[0] - 1
    ids = inputs["input_ids"][0]
    prev = torch.softmax(model.score(hid[pooled - 1:pooled]).float(), dim=-1)[0].tolist()
    prev_logit = model.score(hid[pooled - 1:pooled])[0].float().tolist()
    grid = inputs["image_grid_thw"].tolist()
    return {"prob_1": prob[1], "logits": logits[0].float().tolist(), "prob_1_prev": prev[1], "logits_prev": prev_logit,
            "T": int(ids.shape[0]), "N": [g[1] * g[2] for g in grid], "grids": [g[1:] for g in grid], "pooled_id": int(ids[pooled]),
            "ids_sha": hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:16]}, hid[pooled].float().numpy()


class Stop(Exception):
    pass


def on_alarm(*_):
    raise Stop("watchdog")


signal.signal(signal.SIGALRM, on_alarm)
envrow = {"torch": torch.__version__, "platform": platform.machine(), "threads": torch.get_num_threads(), "dtype": a.dtype,
          "fast_dwconv": a.fast_dwconv, "load_s": round(load_s, 1)}

if a.examples:
    ex = WORK / "author/scripts/minicpm-ocr-jev/examples"
    tag = a.tag or f"examples__{a.dtype}{'__dw' if a.fast_dwconv else ''}"
    outp = WORK / f"results/ref/{tag}.json"
    rows = []
    for d in sorted(ex.iterdir()):
        c = json.loads((d / "case.json").read_text())
        cand = next(x for x in c["candidates"] if x["role"] == c["reference_render_of"])
        t0 = time.perf_counter()
        row, hid = forward([str(d / c["block_image"]), str(d / c["reference_render"])], cand["text"])
        sec = time.perf_counter() - t0
        sample = {"item_id": c["case_id"], "category": c["category"], "label": None, "images": [str(d / c["block_image"]), str(d / c["reference_render"])], "ocr_text": cand["text"]}
        control = score.score_batch(model, processor, [sample], TMP)[0]
        want = c["observed_prob_yes"][c["reference_render_of"]]
        rows.append(dict(row, case=c["case_id"], observed=want, diff=row["prob_1"] - want, p1=abs(row["prob_1"] - want) <= 0.02,
                         control_score_batch=control, control_equal=control == row["prob_1"], seconds=round(sec, 2),
                         hidden_sha=hashlib.sha256(hid.tobytes()).hexdigest()[:16]))
        print(rows[-1], flush=True)
    outp.write_text(json.dumps({"env": envrow, "rows": rows}, indent=1))
    print("P1", "PASS" if all(r["p1"] for r in rows) else "FAIL", [round(r["diff"], 5) for r in rows], "peak_rss_gib",
          round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2))
    sys.exit(0)

rows_all = {r["item_id"]: r for r in map(json.loads, (BENCH / "data/jevocrbench.jsonl").read_text(encoding="utf-8").splitlines())}
dist = {r["item_id"]: r for r in map(json.loads, (WORK / "conversion/sets/record_sizes.jsonl").read_text().splitlines())}
red = json.loads((WORK / "conversion/sets/red_arms.json").read_text())
partner = {int(k): v for k, v in red["partner"].items()}
if a.set == "A":
    ids = json.loads((WORK / "conversion/sets/set_A.json").read_text())["item_ids"]
elif a.set == "red":
    ids = red["item_ids"]
elif a.set == "floor":
    ids = json.loads((WORK / "conversion/sets/floor_ids.json").read_text())["item_ids"]
elif a.set == "device":
    ids = json.loads((WORK / "conversion/sets/set_device.json").read_text())["item_ids"]
else:
    ids = [int(x) for x in a.ids.split(",")]
if a.only:
    keep = {int(x) for x in a.only.split(",")}
    assert keep <= set(ids), f"--only names records outside the set: {sorted(keep - set(ids))}"
    ids = [i for i in ids if i in keep]
tag = a.tag or f"{a.set}__{a.arm}__{a.dtype}{'__dw' if a.fast_dwconv else ''}"
outp, skipp = WORK / f"results/ref/{tag}.jsonl", WORK / f"results/ref/{tag}.skipped.jsonl"
hdir = WORK / f"out/ref/{tag}"
hdir.mkdir(parents=True, exist_ok=True)
done = set()
if outp.exists():
    done = {json.loads(l)["item_id"] for l in outp.read_text().splitlines() if l.strip()}
order = sorted((i for i in ids if i not in done), key=lambda i: (max(dist[i]["N1"], dist[i]["N2"]) ** 2 / 1000 + dist[i]["T"], i))
print(f"tag {tag}: {len(ids)} records, {len(done)} done, env {envrow}", flush=True)
n_skip = 0
t_start = time.time()
with outp.open("a") as out, skipp.open("a") as sk:
    for done_n, i in enumerate(order):
        d, r = dist[i], rows_all[i]
        orig, rend = str(BENCH / r["image_path"]), str(BENCH / r["render_image_path"])
        text = str(r["ocr_text"] or "")
        if a.arm == "R1":
            text = str(rows_all[partner[i]]["ocr_text"] or "")
            images = [orig, rend]
        elif a.arm == "R2":
            images = [rend, orig]
        elif a.arm == "R4":
            images = [str(BENCH / rows_all[partner[i]]["image_path"]), rend]
        elif a.arm in ("S448", "P448"):
            images = [stretch(orig, a.arm == "P448"), stretch(rend, a.arm == "P448")]
        else:
            images = [orig, rend]
        ns = [d["N1"], d["N2"]]
        if a.arm == "R4":
            ns = [dist[partner[i]]["N1"], d["N2"]]
        if a.arm in ("base", "R1", "R2", "R4") and max(ns) > a.max_n:
            n_skip += 1
            sk.write(json.dumps({"item_id": i, "why": f"N {ns} above --max-n {a.max_n}", "time": time.strftime("%H:%M:%S")}) + "\n")
            sk.flush()
            continue
        t0 = time.perf_counter()
        signal.alarm(a.max_seconds)
        try:
            row, hid = forward(images, text)
        except (Stop, RuntimeError, MemoryError, ValueError) as e:
            signal.alarm(0)
            n_skip += 1
            sk.write(json.dumps({"item_id": i, "why": repr(e)[:200], "seconds": round(time.perf_counter() - t0, 1), "time": time.strftime("%H:%M:%S")}) + "\n")
            sk.flush()
            print("SKIP", i, repr(e)[:200], flush=True)
            continue
        signal.alarm(0)
        sec = time.perf_counter() - t0
        np.save(hdir / f"{i}.npy", hid)
        row.update(item_id=i, category=r["category"], label=score.label_to_int(r["label"]), family=d["family"], lang=d["lang"], arm=a.arm,
                   threads=torch.get_num_threads(), seconds=round(sec, 2), peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
                   hidden_sha=hashlib.sha256(hid.tobytes()).hexdigest()[:16], time=time.strftime("%H:%M:%S"))
        out.write(json.dumps(row) + "\n")
        out.flush()
        if done_n % 10 == 0:
            print(done_n, i, "T", row["T"], "N", row["N"], f"{sec:.1f} s", "prob", round(row["prob_1"], 5), "rss", row["peak_rss_gib"], f"total {time.time() - t_start:.0f} s", flush=True)
print("DONE", tag, "rows", len(done) + len(order) - n_skip, "skipped", n_skip, f"{time.time() - t_start:.0f} s", flush=True)
