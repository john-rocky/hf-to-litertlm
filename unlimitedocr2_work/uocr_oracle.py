#!/usr/bin/env python3
"""Reference runs of baidu/Unlimited-OCR-2 with the checkpoint's own code: fp32, greedy, one long-lived process.

  python uocr_oracle.py --model SNAPSHOT --set SET.json --jobs JOBS.json --out ROWS.jsonl [--device cpu|mps]
        [--threads N] [--resume] [--code DIR]

SNAPSHOT = the checkpoint at revision 04809669 (read with HF_HUB_OFFLINE=1); SET.json = build_set.py's set.
The checkpoint's Python files are copied to DIR/hf_mod (default: beside --out) with one change, device moves only:
the two `.cuda()` calls on the image sequence mask in modeling_unlimitedocr.py become `.to(inputs_embeds.device)`.
JOBS.json = [{"image": <set.json image name>, "arm": <arm>, "max_new_tokens": N}, ...]
  optional per job: "pages": [i, ...] (run only these pages of the picture, e.g. one page of the multi-page
  document = the one-page-at-a-time form), "ngram_window": W (ngram35 only), "tag": label (part of the resume key).
Arms:
  R        the author's base mode: _uocr_build_inputs(crop_mode=False, 1024) on the original picture(s), the author's
           DASH (configure_dash_inference topm 16, K 12), the author's positions (page2d), greedy, no n-gram,
           PageCountGuard on. The reference of score_uocr.py.
  s1024 / w1024 / a1024   R with the picture replaced by the 1024x1024 PNG variant of set.json.
  b        R with the hard top-K and the top-M window lifted (K = M = every block): all candidate blocks, biased.
  c        R with plain causal attention (no window, no block selection, no bias) in prefill and decode.
  d        R with the routing bias set to 0 (same selected blocks + window).
  p1       R with 1-D RoPE (main = config YaRN rotary_emb, router = indexer rotary_emb_1d).
  p2       R with the (tok, page) coordinates built from the 1-D position by the fixed-layout function f(p).
  gundam   crop_mode=True (the author's single-page default).  ngram35  no_repeat_ngram 35, window 128 (multi 1024).
Each row: image, arm, input sha256 per page, prompt id count + sha256, generated ids, text, n_gen, finish, seconds,
prefill seconds, raw top-2 logits per step (ids + values), steps where the chosen id != raw argmax (guard / n-gram),
DASH candidate stats (max candidate blocks per query and the rows over min(K, M)), and the layer-0 / layer-11
(tok, page) coordinates of every position with the call counts of the 2-D and 1-D rotary modules.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CODE_STAMP_LOG", "0")
HF = None    # the checkpoint snapshot (--model)
SET = None   # the directory of set.json (its files are relative to it)
CUDA_MOVE = ".unsqueeze(-1).cuda()"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

PAGE_ID, IMG_ID, EOS_ID = 128827, 128815, 1


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def prepare_code(snapshot, root):
    """root/hf_mod = the checkpoint's Python files, the two `.cuda()` moves of the image mask made device-neutral."""
    import glob
    import shutil
    dst = os.path.join(root, "hf_mod")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in glob.glob(os.path.join(snapshot, "*.py")):
        shutil.copyfile(f, os.path.join(dst, os.path.basename(f)))
    open(os.path.join(dst, "__init__.py"), "w").close()
    p = os.path.join(dst, "modeling_unlimitedocr.py")
    s = open(p).read()
    assert s.count(CUDA_MOVE) == 2, "expected the two .cuda() moves of revision 04809669"
    s = s.replace(CUDA_MOVE, ".unsqueeze(-1).to(inputs_embeds.device)")
    assert ".cuda()" not in s
    open(p, "w").write(s)
    sys.path.insert(0, root)


def load(device, threads):
    if threads:
        torch.set_num_threads(threads)
    from hf_mod import modeling_unlimitedocr as mu
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF, local_files_only=True)
    cfg = mu.UnlimitedOCRConfig.from_pretrained(HF, local_files_only=True)
    cfg._attn_implementation = "eager"
    t0 = time.time()
    model = mu.UnlimitedOCRForCausalLM.from_pretrained(HF, config=cfg, torch_dtype=torch.float32,
                                                       local_files_only=True, use_safetensors=True)
    model = model.eval().to(device)
    return tok, model, time.time() - t0


class Recorder:
    """Hooks for one job: coordinates, rotary call counts, DASH candidate stats, raw top-2 logits."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.reset()

    def reset(self):
        self.coords = {0: [], 11: []}
        self.idx_coords = {0: []}
        self.calls = {"main_2d": 0, "main_1d": 0, "idx_2d": 0, "idx_1d": 0}
        self.top2 = []
        self.cand_max = 0
        self.cand_over = 0
        self.cand_calls = 0

    def install(self):
        layers = self.model.model.layers
        for li, layer in enumerate(layers):
            a = layer.self_attn
            if a.rotary_emb_2d_main is not None:
                self.handles.append(a.rotary_emb_2d_main.register_forward_hook(self._mk_coord(li, "main_2d")))
            self.handles.append(a.rotary_emb.register_forward_hook(self._mk_count("main_1d")))
            if a.indexer is not None:
                if a.indexer.rotary_emb_2d is not None:
                    self.handles.append(a.indexer.rotary_emb_2d.register_forward_hook(self._mk_coord(li, "idx_2d")))
                self.handles.append(a.indexer.rotary_emb_1d.register_forward_hook(self._mk_count("idx_1d")))
        self.handles.append(self.model.lm_head.register_forward_hook(self._lm))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _mk_count(self, key):
        def hook(mod, inp, out):
            self.calls[key] += 1
        return hook

    def _mk_coord(self, li, key):
        def hook(mod, inp, out):
            self.calls[key] += 1
            if key == "main_2d" and li in self.coords:
                t, p = inp[1], inp[2]
                self.coords[li].append((t[0].tolist(), p[0].tolist()))
            if key == "idx_2d" and li in self.idx_coords:
                t, p = inp[1], inp[2]
                self.idx_coords[li].append((t[0].tolist(), p[0].tolist()))
        return hook

    def _lm(self, mod, inp, out):
        v, i = out.float()[0, -1].topk(2)
        self.top2.append((i.tolist(), [round(x, 5) for x in v.tolist()]))


@contextlib.contextmanager
def arm_patch(arm, model, rec):
    """Patch the author's inference functions for one arm; always restores."""
    from hf_mod import modeling_deepseekv2 as md
    from hf_mod import modeling_unlimitedocr as mu
    from hf_mod import dash_attention as da
    saved = {}
    orig_route = md._infer_route_select

    def route_stats(block_scores, block_cand, p):
        remote = block_scores > (da._SCORE_NEG * 0.5)
        cand = remote if block_cand is None else (remote & block_cand)
        cand = torch.where(~cand.any(dim=-1, keepdim=True), remote, cand)
        n = cand.sum(dim=-1)
        nblk = block_scores.shape[-1]
        M = max(1, min(int(p["topm"]), nblk)) if int(p["topm"]) > 0 else nblk
        Kc = max(1, min(int(p["K"]), M))
        rec.cand_max = max(rec.cand_max, int(n.max()))
        rec.cand_over += int((n > Kc).sum())
        rec.cand_calls += 1
        xi, bias, valid = orig_route(block_scores, block_cand, p)
        if arm == "d":
            bias = torch.zeros_like(bias)
        return xi, bias, valid

    md._infer_route_select = route_stats
    saved["route"] = orig_route
    orig_conf = mu.configure_dash_inference

    def conf(decoder, on, topm=16, page_scope_prefill=True, state=None):
        st = orig_conf(decoder, on, topm=topm, page_scope_prefill=page_scope_prefill, state=state)
        if on and arm == "b":
            for layer in decoder.layers:
                r = getattr(layer.self_attn, "dash_router", None)
                if r is not None:
                    r._infer_params["K"] = 10 ** 9
                    r._infer_params["topm"] = 10 ** 9
        return st

    mu.configure_dash_inference = conf
    saved["conf"] = orig_conf
    if arm == "c":
        saved["dense"] = da.DashAttention._forward_dense
        saved["dec"] = md._decode_attend

        def dense_causal(self, q, k, v, topi, blk_bias, valid, S, nblk, attention_mask):
            assert attention_mask is None and k.shape[2] == q.shape[2]
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self._scale)

        def dec_dense(q, k_all, v_all, topi, bias, valid, q_pos, block_size, local_window, scale):
            kk, vv = k_all.float(), v_all.float()
            s = torch.matmul(q.float(), kk.transpose(-1, -2)) * scale   # every cached key is <= q_pos
            return torch.matmul(torch.softmax(s, dim=-1), vv)

        da.DashAttention._forward_dense = dense_causal
        md._decode_attend = dec_dense
    modes = []
    if arm == "p1":
        for layer in model.model.layers:
            a = layer.self_attn
            modes.append((a, a.main_rope_mode, a.indexer.rope_mode))
            a.main_rope_mode = "1d"
            a.indexer.rope_mode = "1d"
    try:
        yield
    finally:
        md._infer_route_select = saved["route"]
        mu.configure_dash_inference = saved["conf"]
        if "dense" in saved:
            da.DashAttention._forward_dense = saved["dense"]
            md._decode_attend = saved["dec"]
        for a, m1, m2 in modes:
            a.main_rope_mode, a.indexer.rope_mode = m1, m2


def fixed_layout_coords(n_pages, per_page, inst_len, total_len, page_marks):
    """f(p) for the layout BOS + (<PAGE> + image*per_page)*n + instruction, then the reply.

    page_marks = positions (>= prompt length) of generated <PAGE> tokens assumed by the layout: the reply starts
    with <PAGE> and page k's text follows the k-th generated <PAGE>.  Returns tok[], page[] for p < total_len,
    the same running-count rule as _build_page_rope_pos / _main_rope_coords."""
    tok, page = [], []
    pf = [1] + [1 + per_page] * n_pages            # prefill count of page 0 (BOS) and of each page k
    S0 = 1 + n_pages * (1 + per_page) + inst_len
    cur_page, cur_tok = 0, inst_len
    for p in range(total_len):
        if p == 0:
            tok.append(0); page.append(0)
        elif p < 1 + n_pages * (1 + per_page):
            k = (p - 1) // (1 + per_page)
            tok.append((p - 1) % (1 + per_page)); page.append(k + 1)
        elif p < S0:
            tok.append(p - n_pages * (1 + per_page)); page.append(0)
        else:
            if p in page_marks:
                cur_page = min(cur_page + 1, n_pages)
                cur_tok = pf[cur_page]
            else:
                cur_tok += 1
            tok.append(cur_tok); page.append(cur_page)
    return tok, page


def p2_install(model, layout):
    """Arm p2: feed every layer the coordinates of f(p) instead of the ones read off the token ids."""
    from hf_mod import _sparse_base_impl as sb
    orig_build = sb.DeepseekV2Indexer._build_page_rope_pos
    orig_coords = sb.SlidingWindowLlamaAttention._main_rope_coords
    st = {"n": layout["prompt_len"]}

    def build(self, source_input_ids, key_length):
        t, p = fixed_layout_coords(layout["n_pages"], 273, layout["inst_len"], key_length, set())
        dev = self.q_proj.weight.device
        n_pg = layout["n_pages"] + 1
        self._page_rope_meta = {"pf_cnt": torch.tensor([[1] + [274] * layout["n_pages"]], device=dev),
                                "n_pg": n_pg}
        return (torch.tensor([t], device=dev), torch.tensor([p], device=dev))

    def coords(self, q_len, past_kv):
        if q_len > 1:
            return orig_coords(self, q_len, past_kv)
        # decode: position of the fed token = cache length (before update) -> f(p) with the layout's page marks
        pos = int(past_kv.get_seq_length(self.layer_idx))
        t, p = fixed_layout_coords(layout["n_pages"], 273, layout["inst_len"], pos + 1, layout["marks"])
        dev = self.q_proj.weight.device
        return torch.tensor([[t[pos]]], device=dev), torch.tensor([[p[pos]]], device=dev)

    sb.DeepseekV2Indexer._build_page_rope_pos = build
    sb.SlidingWindowLlamaAttention._main_rope_coords = coords

    def undo():
        sb.DeepseekV2Indexer._build_page_rope_pos = orig_build
        sb.SlidingWindowLlamaAttention._main_rope_coords = orig_coords
    return undo


def run_job(job, model, tok, setd, rec, device):
    from hf_mod import modeling_unlimitedocr as mu
    im = next(x for x in setd["images"] if x["name"] == job["image"])
    arm = job["arm"]
    var = arm if arm in ("s1024", "w1024", "a1024") else None
    files, shas = [], []
    sel = job.get("pages")                 # optional page subset (e.g. [1] = one page of a multi-page document)
    for pg in ([im["pages"][i] for i in sel] if sel is not None else im["pages"]):
        if var:
            p = os.path.join(SET, pg["variants"][var]["file"])
            assert sha_file(p) == pg["variants"][var]["sha256"]
        else:
            p = os.path.join(SET, pg["file"])
            assert sha_file(p) == pg["sha256"]
        files.append(p)
        shas.append(sha_file(p))
    images = [Image.open(f).convert("RGB") for f in files]
    n = len(images)
    prompt = "<PAGE><image>" * n + setd["prompt"]
    crop = True if arm == "gundam" else False
    input_ids, imgs, seq_mask, spatial = mu._uocr_build_inputs(tok, prompt, images, crop_mode=crop, base_size=1024,
                                                               image_size=640, max_crops=16)
    imgs = (imgs[0].float(), imgs[1].float())          # the author's bf16 pixel values, upcast (lossless)
    ng = 35 if arm == "ngram35" else 0
    ngw = job.get("ngram_window", (1024 if n > 1 else 128)) if ng else 100
    rec.reset()
    undo = None
    if arm == "p2":
        inst_len = len(tok.encode(setd["prompt"], add_special_tokens=False))
        ref = job["p2_marks"]          # positions of R's generated <PAGE> tokens (from R's row)
        undo = p2_install(model, {"n_pages": n, "inst_len": inst_len, "prompt_len": int(input_ids.numel()),
                                  "marks": set(ref)})
    t0 = time.time()
    try:
        with arm_patch(arm if arm in ("b", "c", "d", "p1") else "R", model, rec):
            res = model._uocr_generate(tok, input_ids, imgs, seq_mask, spatial, max_new_tokens=job["max_new_tokens"],
                                       max_length=131072, temperature=0.0, page_count_guard=True,
                                       max_page_tokens=20000, ngram_size=ng, ngram_window=ngw, dash_topm=16,
                                       stream=False)
    finally:
        if undo:
            undo()
    dt = time.time() - t0
    gen = res["token_ids"]
    raw_arg = [t[0][0] for t in rec.top2]
    chosen = gen + ([EOS_ID] if res["finish_reason"] == "stop" else [])
    moved = [i for i, (a, b) in enumerate(zip(raw_arg, chosen)) if a != b]
    S = int(input_ids.numel())
    # flatten the coordinate records (prefill = one call with S positions, decode = one per step)
    def flat(lst):
        t, p = [], []
        for tt, pp in lst:
            t += tt; p += pp
        return t, p
    t0c, p0c = flat(rec.coords[0])
    t11, p11 = flat(rec.coords[11])
    ti, pi = flat(rec.idx_coords[0])
    row = {"image": job["image"], "arm": arm, "tag": job.get("tag"), "pages": sel, "ngram": [ng, ngw] if ng else None,
           "device": device, "threads": torch.get_num_threads(),
           "n_pages": n, "input_sha256": shas, "prompt_len": S,
           "prompt_ids_sha256": hashlib.sha256(json.dumps(input_ids.tolist()).encode()).hexdigest(),
           "prompt_head": input_ids.tolist()[:4], "prompt_tail": input_ids.tolist()[-8:],
           "n_image_tokens": int(seq_mask.sum()), "max_new_tokens": job["max_new_tokens"],
           "gen_ids": gen, "text": res["text"], "n_gen": len(gen), "finish": res["finish_reason"],
           "num_pages_out": res["num_pages_out"], "seconds": round(dt, 2),
           "prefill_seconds": round(res["prefill_time"], 2) if res["prefill_time"] else None,
           "top2": rec.top2, "moved_steps": moved,
           "cand_max": rec.cand_max, "cand_over_rows": rec.cand_over, "route_calls": rec.cand_calls,
           "rope_calls": rec.calls, "coords_l0": [t0c, p0c], "coords_l11": [t11, p11], "coords_idx_l0": [ti, pi]}
    return row


def main():
    global HF, SET
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the checkpoint snapshot (revision 04809669)")
    ap.add_argument("--set", required=True, help="set.json of build_set.py")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--code", default=None, help="where hf_mod is written (default: the directory of --out)")
    a = ap.parse_args()
    HF = os.path.abspath(a.model)
    SET = os.path.dirname(os.path.abspath(a.set))
    setd = json.load(open(a.set))
    jobs = json.load(open(a.jobs))
    done = set()
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["image"], r["arm"], r.get("tag")))
    todo = [j for j in jobs if (j["image"], j["arm"], j.get("tag")) not in done]
    print(f"[oracle] {len(todo)} jobs ({len(done)} done) device={a.device}", flush=True)
    if not todo:
        return
    prepare_code(HF, os.path.abspath(a.code or os.path.dirname(os.path.abspath(a.out))))
    tok, model, t_load = load(a.device, a.threads)
    print(f"[oracle] loaded in {t_load:.1f} s, threads={torch.get_num_threads()}", flush=True)
    rec = Recorder(model)
    rec.install()
    for j in todo:
        row = run_job(j, model, tok, setd, rec, a.device)
        with open(a.out, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[oracle] {row['image']}:{row['arm']}{(':' + row['tag']) if row['tag'] else ''} "
              f"n_gen={row['n_gen']} finish={row['finish']} "
              f"{row['seconds']} s prefill={row['prefill_seconds']} cand_max={row['cand_max']} "
              f"moved={len(row['moved_steps'])} pages_out={row['num_pages_out']}", flush=True)
        if a.device == "mps":
            torch.mps.synchronize(); torch.mps.empty_cache()
    print("[oracle] ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
