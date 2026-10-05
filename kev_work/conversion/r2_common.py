"""Shared helpers: model selection, the reference, the host readout and the parity statistics.

`--model 0.8b` (default) or `--model 4b` on the command line (or KEV_MODEL) selects the checkpoint family before
anything binds the constants: the merged fp32 checkpoint (merged/kev-<model>-v1.0/checkpoint), the reference
(oracle/oracle_<model>.json + oracle/hidden_<model>.npz, written by oracle_kev.py), the hidden size (1024 / 2560) and
the file names (exports/kev08b_... / exports/kev4b_..., results/<name>_L{L}_4b... for 4B).
Readout = the checkpoint's PointerHead in numpy float32:
    z = ((h_opts @ Wk^T + bk) @ (h_dec @ Wq^T + bq)) / sqrt(head_dim) / T,  probs = softmax(z)
(the reference's z_post; oracle_kev.py asserts it equals DecisionModel.forward). `summarize` gives the statistics every
gate reports: argmax per question, max / mean / p95 |dp| over all options, h_sel max |diff|, near-tie questions
(reference top-2 gap <= 0.02) apart, flips, and the red arm (graph(red_arm_000) vs reference(tv4_000) must exceed 0.02)."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
# `--model 4b` on the command line (or KEV_MODEL=4b) selects the constants below before anything binds them;
# without it every constant, path and file name is the 0.8B one.
MODELS = {
    "0.8b": {"ckpt": "merged/kev-0.8b-v1.0/checkpoint", "oracle": "oracle/oracle_0.8b.json",
             "npz": "oracle/hidden_0.8b.npz", "hidden": 1024, "prefix": "kev08b", "suffix": "", "cache": None},
    "4b": {"ckpt": "merged/kev-4b-v1.0/checkpoint", "oracle": "oracle/oracle_4b.json",
           "npz": "oracle/hidden_4b.npz", "hidden": 2560, "prefix": "kev4b", "suffix": "_4b", "cache": "cache/r4b"},
}


def _model_from_argv(argv=None):
    argv = sys.argv if argv is None else argv
    for i, x in enumerate(argv):
        if x == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--model="):
            return x.split("=", 1)[1]
    return os.environ.get("KEV_MODEL", "0.8b")


MODEL = _model_from_argv()
assert MODEL in MODELS, f"--model {MODEL!r}: one of {sorted(MODELS)}"
CKPT = K / MODELS[MODEL]["ckpt"]
ORACLE_JSON = K / MODELS[MODEL]["oracle"]
ORACLE_NPZ = K / MODELS[MODEL]["npz"]
HIDDEN = MODELS[MODEL]["hidden"]            # d of the hidden output
FILE_PREFIX = MODELS[MODEL]["prefix"]       # exports/<prefix>_rowprefill_L{L}_...
RESULT_SUFFIX = MODELS[MODEL]["suffix"]     # results/<name>_L{L}<suffix>...; "" for 0.8B


def cache_dir(default):
    """Intermediate-file dir: the caller's default for 0.8B (cache/r2, cache/r3), cache/r4b for 4B."""
    return K / (MODELS[MODEL]["cache"] or default)


def add_model_arg(ap):
    """Every script that imports this module declares --model (the value was already read from argv above)."""
    ap.add_argument("--model", choices=sorted(MODELS), default=MODEL,
                    help="checkpoint family (0.8b = the default; 4b = Kev-4B)")


PAD_ID = 248044
NEAR_TIE = 0.02
BAR = {"argmax": "100%", "max_abs_dp": 0.02, "mean_abs_dp": 0.002}
RED_ARM = ("red_arm_000", "tv4_000")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def dump_json(path, doc, overwrite=False):
    path = Path(path)
    if not overwrite:
        assert not path.exists(), f"refusing to overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


def load_oracle():
    doc = json.loads(ORACLE_JSON.read_text())
    assert doc["pad_token_id"] == PAD_ID
    return doc


def qkey(q):
    return f"{q['id']}/{q['qid']}"


class Head:
    """head.pt -> numpy float32 readout (weights_only=True loads it: plain dict of tensors and python values)."""

    def __init__(self, path=CKPT / "head.pt"):
        import torch
        d = torch.load(path, map_location="cpu", weights_only=True)
        sd = d["head"]
        self.Wq = sd["q.weight"].numpy().astype(np.float32)
        self.bq = sd["q.bias"].numpy().astype(np.float32)
        self.Wk = sd["k.weight"].numpy().astype(np.float32)
        self.bk = sd["k.bias"].numpy().astype(np.float32)
        self.T = float(d["temperature"])
        self.head_dim = int(d["head_dim"])
        assert self.Wq.shape == (self.head_dim, HIDDEN) and self.Wk.shape == (self.head_dim, HIDDEN)
        self.scale = np.float32(1.0 / math.sqrt(self.head_dim))
        self.info = {"path": str(Path(path).relative_to(K)), "sha256": sha256_file(path), "weights_only": True,
                     "temperature": self.T, "head_dim": self.head_dim, "weights": d.get("weights"),
                     "shapes": {k: list(v.shape) for k, v in sd.items()}}

    def __call__(self, h_sel):
        """h_sel [1+K, d] (row order [decide, *opts]) -> (z_pre, z_post, probs), all float32."""
        h = np.asarray(h_sel, dtype=np.float32)
        q = h[0] @ self.Wq.T + self.bq
        k = h[1:] @ self.Wk.T + self.bk
        z_pre = (k @ q) * self.scale
        z_post = (z_pre / np.float32(self.T)).astype(np.float32)
        e = np.exp(z_post - z_post.max())
        return z_pre.astype(np.float32), z_post, (e / e.sum()).astype(np.float32)


def select(hidden, q):
    """hidden [L, d] of one row -> h_sel [1+K, d] in the oracle's order [decide, *opts]."""
    return np.concatenate([hidden[q["decide_idx"]][None], hidden[np.asarray(q["opt_idx"])]], 0)


def compare_question(q, probs, z_post, h_sel, ref_h_sel):
    p_ref = np.asarray(q["probs"], dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    dp = np.abs(p - p_ref)
    top = np.sort(p)[::-1]
    am = int(np.argmax(p))
    return {
        "key": qkey(q), "id": q["id"], "qid": q["qid"], "source": q["source"], "type": q["type"], "row_len": q["row_len"],
        "n_options": len(p), "argmax_key": q["keys"][am], "argmax_key_oracle": q["argmax_key"],
        "argmax_equal": q["keys"][am] == q["argmax_key"],
        "max_abs_dp": float(dp.max()), "sum_abs_dp": float(dp.sum()), "dp": dp.tolist(),
        "max_abs_dz_post": float(np.abs(np.asarray(z_post, np.float64) - np.asarray(q["z_post"], np.float64)).max()),
        "h_sel_max_abs": float(np.abs(np.asarray(h_sel, np.float64) - np.asarray(ref_h_sel, np.float64)).max()),
        "top2_gap": float(top[0] - top[1]) if len(top) > 1 else 1.0, "top2_gap_oracle": q["top2_gap"],
        "near_tie_oracle": bool(q["near_tie"]), "probs": [float(x) for x in probs],
    }


def aggregate(rows):
    if not rows:
        return {"questions": 0}
    dps = np.concatenate([np.asarray(r["dp"]) for r in rows])
    qmax = np.asarray([r["max_abs_dp"] for r in rows])
    return {
        "questions": len(rows), "argmax_equal": int(sum(r["argmax_equal"] for r in rows)),
        "max_abs_dp": float(qmax.max()), "mean_abs_dp_all_options": float(dps.mean()),
        "p95_abs_dp_all_options": float(np.percentile(dps, 95)), "p95_question_max_abs_dp": float(np.percentile(qmax, 95)),
        "options": int(dps.size), "max_abs_dz_post": float(max(r["max_abs_dz_post"] for r in rows)),
        "h_sel_max_abs": float(max(r["h_sel_max_abs"] for r in rows)),
    }


def summarize(rows, red_probs=None, oracle=None):
    """rows = compare_question outputs. Returns overall / by source / near-tie / flips / red arm / bar verdict."""
    sources = sorted({r["source"] for r in rows})
    overall = aggregate(rows)
    near = [r for r in rows if r["near_tie_oracle"]]
    flips = [{"key": r["key"], "argmax": r["argmax_key"], "argmax_oracle": r["argmax_key_oracle"],
              "top2_gap_oracle": r["top2_gap_oracle"], "top2_gap": r["top2_gap"], "max_abs_dp": r["max_abs_dp"]}
             for r in rows if not r["argmax_equal"]]
    doc = {
        "overall": overall,
        "by_source": {s: aggregate([r for r in rows if r["source"] == s]) for s in sources},
        "near_tie": {"definition": f"oracle top-2 gap <= {NEAR_TIE}", **aggregate(near),
                     "rows": [{"key": r["key"], "top2_gap_oracle": r["top2_gap_oracle"], "top2_gap": r["top2_gap"],
                               "argmax_equal": r["argmax_equal"], "max_abs_dp": r["max_abs_dp"]} for r in near]},
        "flips": flips,
        "bar": dict(BAR),
        "bar_pass": bool(overall["argmax_equal"] == overall["questions"] and overall["max_abs_dp"] <= BAR["max_abs_dp"]
                         and overall["mean_abs_dp_all_options"] <= BAR["mean_abs_dp"]),
    }
    if red_probs is not None and oracle is not None:
        base = next(q for q in oracle["questions"] if q["id"] == RED_ARM[1])
        dp = float(np.abs(np.asarray(red_probs, np.float64) - np.asarray(base["probs"], np.float64)).max())
        doc["red_arm"] = {"graph": RED_ARM[0], "vs_oracle": RED_ARM[1], "max_abs_dp": dp,
                          "exceeds_0.02": dp > BAR["max_abs_dp"],
                          "note": "red_arm_000 differs from tv4_000 by one word; the bar must catch it (0.8B reference: 0.0476)"}
    return doc


class Clock:
    def __init__(self):
        self.t0 = time.time()

    def stamp(self):
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def seconds(self):
        return round(time.time() - self.t0, 1)
