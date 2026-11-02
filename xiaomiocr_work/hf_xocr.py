#!/usr/bin/env python3
"""Upstream (HF transformers) reference runs of Xiaomi-OCR-0 on the set of make_set.py (SET.json).

  python hf_xocr.py --set SET.json --snap SNAP --out runs.jsonl --jobs name:mode,name:mode,... [--device cpu|mps]
                    [--attn sdpa|eager] [--pos1d] [--fast-dwconv] [--threads N] [--max-new N]
  python hf_xocr.py --set SET.json --snap SNAP --profile-decode name:mode --out prof.json   (one decode step, profiled)
  python hf_xocr.py --set SET.json --snap SNAP --dwconv-proof name:mode --out proof.json   (orig conv1d vs fast_dwconv)

One model load (Qwen3_5ForConditionalGeneration through AutoModelForImageTextToText, fp32), then for each job one greedy
generation: do_sample False, no repetition penalty, eos = [248046, 248044] (the repo has no generation_config.json),
the checkpoint's chat template with enable_thinking=False on one user turn [image(s), text], no system prompt.

Modes (the picture each job feeds; set.json holds every file's sha256, checked before use):
  native   the original picture, EXIF applied (ImageOps.exif_transpose), resized by the processor itself (smart_resize)
  s{G}     the GxG stretch PNG, p{G} the white-padded GxG PNG (the processor must keep it: grid G/16 x G/16 asserted,
           and pixel_values equal to the PNG's own normalisation within --pix-tol)
  p{H}x{W} the white-padded H x W PNG (aspect kept, make_set.py's pad_to); grid H/16 x W/16 asserted the same way
  t{G}     app-side tiles of size G (tiles_xocr.py's rule, recorded in the row): [thumbnail p{G}] + tiles, row-major,
  t{G}n    the same tiles without the thumbnail; every image is one more <|vision_start|><|image_pad|><|vision_end|>
--pos1d replaces get_rope_index with sequential 1-D positions on every M-RoPE channel (the fast_vlm decoder contract);
the row records the largest position id the language model saw in the prefill, so the hook is shown to be in force.
A finished (name, mode, pos1d) row already in --out is skipped (resume). Rows: JSONL, one per job.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time

import numpy as np
import torch
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
EOS = [248046, 248044]
IMAGE_PAD = 248056
SET = {"path": None}     # set.json of make_set.py; its "file" entries are relative to its folder


def peer(name, path):
    """Import one file by its path (a peer folder's module must not shadow anything on sys.path)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fast_dwconv_module():
    return peer("minicpmocr_fast_dwconv", os.path.join(os.path.dirname(HERE), "minicpmocr_work", "fast_dwconv.py"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_set(path=None):
    if path:
        SET["path"] = path
    return {c["name"]: c for c in json.load(open(SET["path"]))["cases"]}


def open_checked(case, mode):
    f = case["files"][mode]
    path = os.path.join(os.path.dirname(os.path.abspath(SET["path"])), f["file"])
    h = sha(path)
    assert h == f["sha256"], (path, h, f["sha256"])
    img = Image.open(path)
    if mode == "native":
        img = ImageOps.exif_transpose(img)
    return img.convert("RGB"), f["file"], h


def own_pixel_values(img, patch=16, merge=2, tps=2):
    """The processor's normalisation and patch order applied to an image it must not resize."""
    x = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5            # H W C
    t = torch.from_numpy(x).permute(2, 0, 1)                              # C H W
    C, H, W = t.shape
    gh, gw = H // patch, W // patch
    p = t.reshape(C, gh // merge, merge, patch, gw // merge, merge, patch).permute(1, 4, 2, 5, 0, 3, 6)
    p = p.unsqueeze(5).expand(-1, -1, -1, -1, -1, tps, -1, -1).reshape(gh * gw, C * tps * patch * patch)
    return p


def build_images(case, mode):
    """-> (list of PIL images, meta). Tiles come from tiles_xocr.tile_images (the rule is in its docstring)."""
    if mode == "native" or mode[0] in "sp":
        img, rel, h = open_checked(case, mode)
        return [img], {"file": rel, "sha256": h, "image_size": list(img.size)}
    if mode[0] == "t":
        tiles_xocr = peer("xiaomiocr_tiles", os.path.join(HERE, "tiles_xocr.py"))
        g = int(mode[1:].rstrip("n"))
        src, rel, h = open_checked(case, "native")
        imgs, meta = tiles_xocr.tile_images(src, g, thumbnail=not mode.endswith("n"))
        meta.update({"file": rel, "sha256": h, "image_size": list(src.size)})
        return imgs, meta
    raise ValueError(mode)


def mode_hw(mode):
    """(H, W) of a fixed-size mode: s{G} / p{G} / t{G}[n] -> (G, G); p{H}x{W} -> (H, W)."""
    body = mode[1:].rstrip("n")
    if "x" in body:
        h, w = body.split("x")
        return int(h), int(w)
    return int(body), int(body)


POS1D = {"on": False}


def install_pos1d(model):
    """Wrap get_rope_index once; POS1D["on"] (set per job) turns the 1-D positions on or off."""
    orig = model.model.get_rope_index

    def seq_rope_index(*a, **kw):
        pos3, delta = orig(*a, **kw)               # [3, B, S]
        if not POS1D["on"]:
            return pos3, delta
        S = pos3.shape[-1]
        seq = torch.arange(S, device=pos3.device).view(1, 1, S).expand_as(pos3)
        return seq.contiguous(), torch.zeros_like(delta)

    model.model.get_rope_index = seq_rope_index


class PosProbe:
    """Records the position ids of the language model's first (prefill) call of each generation."""

    def __init__(self, model):
        self.seen = None
        model.model.language_model.register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, module, args, kwargs):
        if self.seen is None:
            p = kwargs.get("position_ids")
            if p is not None:
                p = p.detach().to("cpu")
                self.seen = {"shape": list(p.shape), "max_per_channel": [int(p[i].max()) for i in range(p.shape[0])],
                             "seq_len": int(p.shape[-1])}
        return None


def render(processor, imgs, prompt, tokenize):
    content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=tokenize, return_dict=tokenize,
                                         return_tensors="pt" if tokenize else None, enable_thinking=False)


def register_qchunk(chunk):
    """sdpa over query chunks (each query row's softmax is unchanged): bounds the vision tower's attention memory on
    devices whose sdpa materialises the score matrix (32,640 patches x 12 heads would be 51 GB in fp32)."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def sdpa_qchunk(module, query, key, value, attention_mask, **kw):
        L = query.shape[2]
        if L <= chunk:
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        outs = [sdpa_attention_forward(module, query[:, :, i:i + chunk], key, value, attention_mask, **kw)[0]
                for i in range(0, L, chunk)]
        return torch.cat(outs, dim=1), None

    AttentionInterface.register("sdpa_qchunk", sdpa_qchunk)


def load(snap, device, attn, vision_qchunk=0):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(snap)
    model = AutoModelForImageTextToText.from_pretrained(snap, dtype=torch.float32, attn_implementation=attn).eval()
    if vision_qchunk:
        assert attn == "sdpa", "the query-chunked vision attention wraps sdpa"
        register_qchunk(vision_qchunk)
        model.model.visual.config._attn_implementation = "sdpa_qchunk"
    model = model.to(device)
    return processor, model


def prep(processor, case, mode, device, pix_tol):
    imgs, meta = build_images(case, mode)
    inputs = render(processor, imgs, case["prompt"], tokenize=True)
    grids = inputs["image_grid_thw"].tolist()
    if mode[0] in "spt" and mode != "native":
        h, w = mode_hw(mode)
        assert all(gr == [1, h // 16, w // 16] for gr in grids), (mode, grids)
        if mode[0] in "sp":
            d = float((inputs["pixel_values"] - own_pixel_values(imgs[0])).abs().max())
            assert d <= pix_tol, (mode, d)
            meta["pixel_maxdiff_vs_png"] = d
    meta["grids"] = grids
    return inputs.to(device), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="set.json of make_set.py")
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--jobs-file", default=None, help="one name:mode[:pos1d] per line (# comments)")
    ap.add_argument("--follow", action="store_true", help="keep reading lines appended to --jobs-file until STOP")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--vision-qchunk", type=int, default=0, help="query chunk of the vision attention (0 = plain)")
    ap.add_argument("--pos1d", action="store_true")
    ap.add_argument("--fast-dwconv", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=0, help="override set.json's cap (smoke / probes only)")
    ap.add_argument("--pix-tol", type=float, default=1e-6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--profile-decode", default=None)
    ap.add_argument("--dwconv-proof", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    fast_dwconv = fast_dwconv_module()
    if args.fast_dwconv and not args.dwconv_proof:
        fast_dwconv.install()
    import transformers
    cases = load_set(args.set)
    t0 = time.time()
    processor, model = load(args.snap, args.device, args.attn, args.vision_qchunk)
    env = {"transformers": transformers.__version__, "torch": torch.__version__, "device": args.device,
           "attn": args.attn, "vision_attn": model.model.visual.config._attn_implementation,
           "vision_qchunk": args.vision_qchunk, "fast_dwconv": bool(args.fast_dwconv), "pos1d_process": bool(args.pos1d),
           "threads": torch.get_num_threads(), "python": platform.python_version(), "load_s": round(time.time() - t0, 1)}
    print("loaded", json.dumps(env), flush=True)
    install_pos1d(model)
    POS1D["on"] = bool(args.pos1d)
    probe = PosProbe(model)

    if args.profile_decode or args.dwconv_proof:
        name, mode = (args.profile_decode or args.dwconv_proof).split(":")
        inputs, meta = prep(processor, cases[name], mode, args.device, args.pix_tol)
        res = {"env": env, "name": name, "mode": mode, "n_input": int(inputs["input_ids"].shape[1])}
        if args.profile_decode:
            from torch.profiler import ProfilerActivity, profile
            with torch.no_grad():
                out = model(**inputs, use_cache=True)
                pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)     # warm decode step
                pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                t1 = time.time()
                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    model(input_ids=nxt, past_key_values=pkv, use_cache=True)
                res["decode_step_s"] = round(time.time() - t1, 3)
            ka = prof.key_averages()
            top = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:12]
            res["top_self_cpu"] = [{"op": e.key, "calls": e.count, "self_cpu_ms": round(e.self_cpu_time_total / 1e3, 1)}
                                   for e in top]
            res["total_self_cpu_ms"] = round(sum(e.self_cpu_time_total for e in ka) / 1e3, 1)
        else:
            def run3(m):
                logits, t = [], time.time()
                with torch.no_grad():
                    out = m(**inputs, use_cache=True)
                    logits.append(out.logits[0, -1].double().cpu())
                    pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                    for _ in range(3):
                        out = m(input_ids=nxt, past_key_values=pkv, use_cache=True)
                        logits.append(out.logits[0, -1].double().cpu())
                        pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                return torch.stack(logits), round(time.time() - t, 2)
            a, ta = run3(model)
            fast_dwconv.install()
            b, tb = run3(model)
            d = (a - b).abs()
            res.update({"unit": fast_dwconv.check(), "positions": int(a.shape[0]), "max_abs_delta": float(d.max()),
                        "max_abs_logit": float(a.abs().max()), "argmax_orig": a.argmax(-1).tolist(),
                        "argmax_fast": b.argmax(-1).tolist(), "bit_equal": bool(torch.equal(a, b)),
                        "seconds_orig": ta, "seconds_fast": tb})
        json.dump(res, open(args.out, "w"), indent=1)
        print(json.dumps(res, indent=1), flush=True)
        return

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["name"], r["mode"], bool(r.get("pos1d"))))

    def read_jobs_file():
        return [l.strip() for l in open(args.jobs_file) if l.strip() and not l.lstrip().startswith("#")]

    def job_stream():
        """--jobs, then --jobs-file; with --follow, keep reading lines appended to the file until a line STOP."""
        for j in (args.jobs.split(",") if args.jobs else []):
            yield j
        if not args.jobs_file:
            return
        seen = 0
        while True:
            lines = read_jobs_file()
            for j in lines[seen:]:
                if j == "STOP":
                    return
                yield j
            seen = len(lines)
            if not args.follow:
                return
            time.sleep(20)

    for job in job_stream():
        parts = job.split(":")
        name, mode = parts[0], parts[1]
        POS1D["on"] = bool(args.pos1d) or (len(parts) > 2 and parts[2] == "pos1d")
        if (name, mode, POS1D["on"]) in done:
            continue
        done.add((name, mode, POS1D["on"]))
        if name not in cases or (mode != "native" and mode[0] != "t" and mode not in cases[name]["files"]):
            cases = load_set()          # pictures added to set.json after this worker started
        try:
            case = cases[name]
            inputs, meta = prep(processor, case, mode, args.device, args.pix_tol)
        except Exception:               # a bad job line leaves a traceback in the log; the worker goes on
            import traceback
            traceback.print_exc()
            print(f"JOB_FAILED {job}", flush=True)
            continue
        ids = inputs["input_ids"]
        max_new = args.max_new or case["max_new_tokens"]
        probe.seen = None
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, eos_token_id=EOS,
                                 pad_token_id=248044)
        gen_s = time.time() - t1
        new = out[0, ids.shape[1]:].tolist()
        stop = "eos" if new and new[-1] in EOS else ("max_new_tokens" if len(new) >= max_new else "other")
        row = {"name": name, "mode": mode, "pos1d": POS1D["on"], "class": case["class"], **meta,
               "n_input": int(ids.shape[1]), "n_image_tokens": int((ids == IMAGE_PAD).sum()), "max_new_tokens": max_new,
               "gen_ids": new, "n_gen": len(new), "stop": stop,
               "text": processor.decode(new, skip_special_tokens=True),
               "text_raw": processor.decode(new, skip_special_tokens=False),
               "prompt_ids_head": ids[0, :8].tolist(), "prompt_ids_tail": ids[0, -40:].tolist(),
               "positions": probe.seen, "seconds": round(gen_s, 2), "tag": args.tag, **env}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name:18s} {mode:7s} pos1d={int(POS1D["on"])} in {row['n_input']:5d} img {row['n_image_tokens']:5d} "
              f"gen {len(new):4d} {stop:14s} {gen_s:7.1f}s | {row['text'][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
