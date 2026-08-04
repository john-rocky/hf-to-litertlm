#!/usr/bin/env python3
"""Task-level verification + device-fixture generation for the LFM2.5-Encoder
finetune conversions (router / linter / pii / gec).

    python verify_lfm25_finetunes.py router|linter|pii|gec

Per model: builds REAL task inputs (offset-pooled prompts for router/linter,
task sentences for pii/gec), compares torch reference vs fp32 vs wi8fc tflite
at task level, and appends a manifest-v2 case (expected = Mac wi8fc outputs)
to lfm25_encoder_work/device_fixtures_ft/ for the generic iPhone harness.
"""
import json
import os
import sys

import types


class _D:
    def __getattr__(self, n):
        return lambda *a, **k: None

    def __call__(self, *a, **k):
        return None


_pp = types.ModuleType("scipy.sparse.linalg._propack")
_pp.__file__ = "<stub>"
_pp.__spec__ = None
_pp.__getattr__ = lambda n: _D()  # noqa: E731
sys.modules.setdefault("scipy.sparse.linalg._propack", _pp)

import numpy as np
import torch

KIND = sys.argv[1] if len(sys.argv) > 1 else "router"
R = 8
FIX = "device_fixtures_ft"
os.makedirs(FIX, exist_ok=True)

SPEC = {
    "router": ("LiquidAI/LFM2.5-Encoder-350M-Prompt-Router", "out/lfm25_ft_router", 512, "route_512"),
    "linter": ("LiquidAI/LFM2.5-Encoder-350M-Policy-Linter", "out/lfm25_ft_linter", 512, "lint_512"),
    "pii": ("LiquidAI/LFM2.5-Encoder-350M-PII-Detector", "out/lfm25_ft_pii", 128, "pii_128"),
    "gec": ("LiquidAI/LFM2.5-Encoder-350M-Spellchecker", "out/lfm25_ft_gec", 128, "gec_128"),
}
MODEL, OUT_DIR, S, SIG = SPEC[KIND]

ROUTES = ["coding question", "cooking recipe", "travel planning", "medical advice"]
ROUTER_TEXT = "Should I use Python or pandas dataframes to aggregate my sales CSV?"
RULES = ["Do not share customer email addresses",
         "Never promise a specific delivery date"]
LINTER_TEXT = ("Hi team, please reach out to the buyer at maria.lopez@example.com "
               "and tell her the package will definitely arrive on Tuesday.")
PII_TEXT = ("My name is John Smith, my email is john.smith@example.com and my "
            "phone number is 555-0192.")
GEC_TEXT = "She go to school every day ."


def offsets_pools(tok, prefix_header, items, text):
    """Router/linter prompt build (mirrors the model cards' route() helpers):
    header + '- item' lines + text; returns ids, pools over token offsets."""
    body = "\n".join(f"- {it}" for it in items)
    prefix = f"{prefix_header}:\n{body}\n\nText:\n"
    full = prefix + text
    enc = tok(full, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    text_start = len(prefix)
    text_idxs = [i for i, (a, b) in enumerate(offsets) if b > text_start and a != b]
    text_pool = np.zeros((1, 1, S), np.float32)
    if text_idxs:
        text_pool[0, 0, text_idxs] = 1.0 / len(text_idxs)
    ranges = []
    pos = len(f"{prefix_header}:\n")
    for it in items:
        a = pos + 2
        b = a + len(it)
        ranges.append((a, b))
        pos = b + 1
    item_pool = np.zeros((1, R, S), np.float32)
    for r, (a, b) in enumerate(ranges):
        idxs = [i for i, (ta, tb) in enumerate(offsets)
                if ta < b and tb > a and ta != tb]
        if idxs:
            item_pool[0, r, idxs] = 1.0 / len(idxs)
    return ids, text_pool, item_pool


def pad(ids):
    x = np.zeros((1, S), np.int32)
    m = np.zeros((1, S), np.int32)
    x[0, :len(ids)] = ids
    m[0, :len(ids)] = 1
    return x, m


def load_ref():
    from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cls = AutoModelForTokenClassification if KIND == "pii" else AutoModel
    m = cls.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).eval()
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    g = Lfm2ShortConv.slow_forward.__globals__

    def _always(h, am):
        if am is not None:
            h = h * am[:, :, None].to(h.dtype)
        return h

    g["apply_mask_to_padding_states"] = _always
    return tok, m


def build_case(tok):
    """Returns (inputs dict name->np array, n_tokens, extra host info)."""
    if KIND == "router":
        ids, tp, cp = offsets_pools(tok, "Categories", ROUTES, ROUTER_TEXT)
        x, mm = pad(ids)
        return ({"input_ids": x, "attention_mask": mm,
                 "text_pool": tp, "category_pool": cp}, len(ids), ROUTES)
    if KIND == "linter":
        ids, _, rp = offsets_pools(tok, "Policy", RULES, LINTER_TEXT)
        x, mm = pad(ids)
        return ({"input_ids": x, "attention_mask": mm, "rule_pool": rp}, len(ids), RULES)
    if KIND == "pii":
        ids = tok(PII_TEXT)["input_ids"]
        x, mm = pad(ids)
        return ({"input_ids": x, "attention_mask": mm}, len(ids), None)
    bos = tok.bos_token_id if tok.bos_token_id is not None else 0
    ids = [bos] + tok.encode(GEC_TEXT, add_special_tokens=False)
    x, mm = pad(ids)
    return ({"input_ids": x, "attention_mask": mm}, len(ids), None)


def torch_forward(m, kw):
    t = {k: torch.tensor(v) for k, v in kw.items()}
    am = t["attention_mask"].to(torch.float32)
    with torch.no_grad():
        if KIND == "router":
            return (m(input_ids=t["input_ids"], attention_mask=am,
                      text_pool=t["text_pool"], category_pool=t["category_pool"])
                    ["logits"].numpy(),)
        if KIND == "linter":
            s = m(input_ids=t["input_ids"], attention_mask=am,
                  rule_pool=t["rule_pool"])["logits"]
            return ((s * am[:, :, None]).numpy(),)
        if KIND == "pii":
            lg = m(input_ids=t["input_ids"], attention_mask=am).logits
            return ((lg * am[:, :, None]).numpy(),)
        out = m(input_ids=t["input_ids"], attention_mask=am)
        return ((out["label_logits"] * am[:, :, None]).numpy(),
                (out["detect_logits"] * am[:, :, None]).numpy())


def tflite_forward(path, kw):
    from ai_edge_litert.interpreter import Interpreter

    rn = Interpreter(model_path=path, num_threads=os.cpu_count()).get_signature_runner(SIG)
    got = rn(**kw)
    return tuple(got[k] for k in sorted(got)), sorted(got)


def task_summary(tok, outs, n, extra):
    if KIND == "router":
        lg = outs[0][0, :len(extra)]
        p = np.exp(lg - lg.max()); p /= p.sum()
        return {r: round(float(v), 4) for r, v in zip(extra, p)}
    if KIND == "linter":
        s = outs[0][0, :n, :len(extra)]
        hits = 1 / (1 + np.exp(-s)) > 0.5
        return {rule: int(hits[:, r].sum()) for r, rule in enumerate(extra)}
    if KIND == "pii":
        lab = outs[0][0, :n].argmax(-1)
        return {"nonzero_label_positions": [int(i) for i in np.nonzero(lab)[0]],
                "labels": [int(v) for v in lab[lab != 0]]}
    best = outs[0][0, :n].argmax(-1)
    err = outs[1][0, :n]
    err_p = np.exp(err) / np.exp(err).sum(-1, keepdims=True)
    edits = [(int(i), int(t)) for i, t in enumerate(best)
             if t != 0 and err_p[i, 1] >= 0.5]
    return {"edit_positions_tags": edits}


def main():
    tok, m = load_ref()
    kw, n, extra = build_case(tok)
    ref = torch_forward(m, kw)

    report = {"model": MODEL, "sig": SIG, "task_ref": task_summary(tok, ref, n, extra)}
    print(f"[{KIND}] torch task result:", report["task_ref"])

    kinds = {}
    for tag in ("fp32", "fp16", "wi8fc"):
        path = os.path.join(OUT_DIR, f"{KIND}_{tag}.tflite")
        if not os.path.exists(path):
            continue
        outs, keys = tflite_forward(path, kw)
        stats = []
        for r0, g0 in zip(ref, outs):
            stats.append({
                "maxdiff": float(np.abs(r0 - g0).max()),
                "corr": float(np.corrcoef(r0.ravel(), g0.ravel())[0, 1]),
            })
        ts = task_summary(tok, outs, n, extra)
        kinds[tag] = {"stats": stats, "task": ts,
                      "task_match": ts == report["task_ref"]}
        print(f"[{KIND}] {tag}: {stats} task_match={ts == report['task_ref']}")
        print(f"[{KIND}] {tag} task result:", ts)
        if tag == "wi8fc":
            wi8_outs, wi8_keys = outs, keys
    report["tflite"] = kinds

    # ---- device fixture (expected = Mac wi8fc outputs) ----
    mpath = os.path.join(FIX, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"models": {}}
    ins = []
    for name, arr in kw.items():
        f = f"{KIND}_c0_{name}.bin"
        arr.tofile(os.path.join(FIX, f))
        ins.append({"name": name, "file": f})
    outs_spec = []
    for key, arr in zip(wi8_keys, wi8_outs):
        f = f"{KIND}_c0_{key}.bin"
        arr.astype(np.float32).tofile(os.path.join(FIX, f))
        outs_spec.append({"key": key, "file": f, "count": int(arr.size)})
    manifest["models"][f"350m_{KIND}_wi8fc"] = {
        "file": f"LFM2.5-{KIND}-350M_wi8fc.tflite",
        "cases": [{"name": f"{KIND} demo", "sig": SIG,
                   "inputs": ins, "outputs": outs_spec}],
    }
    json.dump(manifest, open(mpath, "w"), indent=2)
    with open(os.path.join(OUT_DIR, "verify_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[{KIND}] fixture + manifest updated; report {OUT_DIR}/verify_report.json")


if __name__ == "__main__":
    main()
