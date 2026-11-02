#!/usr/bin/env python3
"""HF fp32's own margin at the token where a bundle row's text leaves HF's text.

For each case: the reference = hf_xocr.py's row of the same picture with the runtime's 1-D positions (mode p1344x768,
pos1d true, its gen_ids); the bundle row = the first row of --bundle-rows with that name. c = the first character where
the two texts differ; d = the first generation step whose token makes HF's decoded prefix leave the bundle's text.
One greedy HF generation (fp32, 1-D positions, hf_xocr.load / prep; eos [248046, 248044]) to step d + 1 with the
raw logits kept; its ids must equal the row's first d + 1 ids (ids_equal_row). At step d: the top-10, HF's token, and
the candidate tokens whose text the bundle wrote next (the top-10 entries whose decoded text is a prefix of the bundle's
continuation, or that the continuation is a prefix of), with prob / logit / rank; margin = HF's token minus the best
such candidate (prob and logit). No bundle token ids are needed (the runtime's rows hold text only).

  python3 -B margin_bundle.py --set SET.json --snap SNAP --device mps --fast-dwconv --vision-qchunk 2048 \
      --cases qianziwen,receipt_jpg_kie,receipt_jpg --bundle-rows rows_cpu.jsonl --hf-rows 'hf/runs_*.jsonl' \
      --out margin_cpu.json
"""
import argparse
import glob
import importlib.util
import json
import os

import torch

_spec = importlib.util.spec_from_file_location(
    "xiaomiocr_hf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_xocr.py"))
hf_xocr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf_xocr)


def entry(lg, tok, i):
    pr = torch.softmax(lg, -1)
    return {"id": int(i), "token": tok.decode([int(i)]), "logit": round(float(lg[i]), 4), "prob": round(float(pr[i]), 6),
            "rank": int((lg > lg[i]).sum()) + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="set.json of make_set.py")
    ap.add_argument("--snap", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fast-dwconv", action="store_true")
    ap.add_argument("--vision-qchunk", type=int, default=0)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--mode", default="p1344x768")
    ap.add_argument("--bundle-rows", required=True)
    ap.add_argument("--hf-rows", required=True, help="glob of hf_xocr.py rows (quote it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    hf_xocr.load_set(a.set)
    if a.fast_dwconv:
        hf_xocr.fast_dwconv_module().install()
    names = a.cases.split(",")
    hf = {}
    for p in glob.glob(a.hf_rows):
        for line in open(p):
            r = json.loads(line)
            if r["name"] in names and r["mode"] == a.mode and r.get("pos1d"):
                hf[r["name"]] = r
    brow = {}
    for line in open(a.bundle_rows):
        r = json.loads(line)
        brow.setdefault(r["name"], r)
    cases = hf_xocr.load_set()
    processor, model = hf_xocr.load(a.snap, a.device, "sdpa", a.vision_qchunk)
    hf_xocr.install_pos1d(model)
    hf_xocr.POS1D["on"] = True
    tok = processor.tokenizer
    res = {"mode": a.mode, "pos1d": True, "device": a.device, "bundle_rows": a.bundle_rows, "cases": {}}
    for n in names:
        ref, b = hf[n], brow[n]
        ids, htext, btext = ref["gen_ids"], ref["text"], b["text"]
        c = next((i for i, (x, y) in enumerate(zip(htext, btext)) if x != y), min(len(htext), len(btext)))
        d = next((k for k in range(len(ids)) if not btext.startswith(tok.decode(ids[:k + 1]))), None)
        e = {"bundle_label": b.get("label"), "first_char_diff": c, "hf_prefix_tail": htext[max(0, c - 30):c],
             "hf_next": htext[c:c + 30], "bundle_next": btext[c:c + 30], "step": d}
        if d is None:
            e["note"] = "HF's whole decoded text is a prefix of the bundle's"
            res["cases"][n] = e
            continue
        inputs, meta = hf_xocr.prep(processor, cases[n], a.mode, a.device, 1e-6)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=d + 1, do_sample=False, eos_token_id=hf_xocr.EOS,
                                 pad_token_id=248044, output_logits=True, return_dict_in_generate=True)
        gen = out.sequences[0, n_in:].tolist()
        lg = out.logits[d][0].float().cpu()
        e["ids_equal_row"] = gen == ids[:d + 1]
        pre = tok.decode(ids[:d])
        cont = btext[len(pre):len(pre) + 40] if btext.startswith(pre) else None
        e["bundle_continuation"] = cont
        top = torch.topk(lg, 10)
        e["top10"] = [entry(lg, tok, int(i)) for i in top.indices]
        e["hf_token"] = entry(lg, tok, ids[d])
        cands = []
        if cont is not None:
            for x in e["top10"]:
                t = x["token"]
                if x["id"] != ids[d] and t and (cont.startswith(t) or t.startswith(cont)):
                    cands.append(x)
        e["bundle_candidates"] = cands
        if cands:
            best = max(cands, key=lambda x: x["prob"])
            e["margin_prob"] = round(e["hf_token"]["prob"] - best["prob"], 6)
            e["margin_logit"] = round(e["hf_token"]["logit"] - best["logit"], 4)
        res["cases"][n] = e
        print(n, "step", d, "hf", repr(e["hf_token"]["token"]), e["hf_token"]["prob"], "bundle",
              [(x["token"], x["prob"]) for x in cands], "ids_equal_row", e["ids_equal_row"], flush=True)
        del out
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("MARGIN_DONE", a.out)


if __name__ == "__main__":
    main()
