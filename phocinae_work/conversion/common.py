"""Shared paths, the sequence protocol and the decode rules of the Phocinae-Largha-150M-v1 lane.

Sequence protocol = laya's build_sequence (the publisher's phocinae-server `Engine._build_sequence` is the same
layout with its own option texts; the SDK's `DecisionSequenceBuilder` renders the dataset's criteria the way the
author's eval harness (laya.Agent.predict on typed_test rows) did):

    CLS <type> question: <instructions> SEP [MASK] opt0 [MASK] opt1 ... SEP <state> SEP

Temperatures: rl_agent_config.json `temperature` = [choice, score, noul] (QTYPES choice=0, score=1, noul=2 in
phocinae/engine.py). The brief of 2026-10-10 listed them as noul/choice/score — that order is wrong; engine.py is
the source.
"""
import json
import math
import os
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]          # phocinae_work
HF = W / "hf"                                     # the Hub snapshot (model.safetensors, encoder/, tokenizer/, rl_agent_config.json)
SRC_SERVER = W / "src" / "phocinae-server"        # github.com/Phocinae/phocinae-server @ 705508c3 (0.1.5)
SRC_DATASETS = W / "src" / "gh_datasets"          # github.com/Phocinae/Phocinae-Largha-150M-v1/datasets @ f78d42e6
MODEL_ID = "Phocinae/Phocinae-Largha-150M-v1"
HUB_SHA = "3807157407d14497a7ebd7114a3d71847b5b10a5"
WEIGHTS_SHA256 = "b6472511eea30729985374f43968cf7f4b16bbe0827de6c6ca07cf92afbb778a"

QTYPES = {"choice": 0, "score": 1, "noul": 2}
TYPE_NAMES = ["choice", "score", "noul"]
NOUL_DEFAULT = ("no, the statement does not hold", "yes, the statement holds")
HIDDEN = 384
NEAR_TIE = 0.02
BAR = {"argmax": "100% on non-near-tie rows (reference top-2 gap > 0.02)", "max_abs_dp": 0.02, "mean_abs_dp": 0.002,
       "control_row_moves": 0.02}

if str(SRC_SERVER) not in sys.path:
    sys.path.insert(0, str(SRC_SERVER))


def load_cfg():
    return json.loads((HF / "rl_agent_config.json").read_text())


def load_encoder_cfg():
    return json.loads((HF / "encoder" / "config.json").read_text())


def temperatures():
    t = load_cfg()["temperature"]
    assert len(t) == 3
    return [float(x) for x in t]


def author_tokenizer():
    """The publisher's pure-python tokenizer replica (phocinae/tokenizer.py)."""
    from phocinae.tokenizer import GemmaTokenizer
    return GemmaTokenizer(str(HF / "tokenizer"))


def hf_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(HF / "tokenizer" / "tokenizer.json"))


def serialize_state(state):
    """laya: strings as they are, structured state as json.dumps(state, ensure_ascii=False) (HOST_CONTRACT B.7)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options_laya(q):
    """The SDK / laya rendering of a typed_test question: key: description, level i: text, false: … / true: …."""
    t = q["type"]
    crit = q.get("criteria")
    if t == "choice":
        assert isinstance(crit, dict)
        return [k if (v is None or v == "") else f"{k}: {v}" for k, v in crit.items()]
    if t == "score":
        assert isinstance(crit, list)
        return [f"level {i}: {c}" for i, c in enumerate(crit)]
    if t == "noul":
        f = (crit or {}).get("false") if isinstance(crit, dict) else None
        tr = (crit or {}).get("true") if isinstance(crit, dict) else None
        return ["false: " + (NOUL_DEFAULT[0] if not f else f), "true: " + (NOUL_DEFAULT[1] if not tr else tr)]
    raise ValueError(t)


class Builder:
    """`Engine._build_sequence` of phocinae-server 0.1.5 with the instruction text and options supplied by the caller."""

    def __init__(self, tok, max_len=512, head_max_len=192):
        self.tok = tok
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.mask = "<mask>"

    def encode(self, s):
        return list(self.tok.encode(s))

    def state_ids(self, state):
        return self.encode(serialize_state(state).replace(self.mask, " "))

    def build(self, qtype_name, instructions, options, state_ids, head_max_len=None, max_len=None):
        head_max_len = head_max_len or self.head_max_len
        max_len = max_len or self.max_len
        head = self.encode("%s question: %s" % (qtype_name, instructions.replace(self.mask, " ")))
        opt_ids = []
        for o in options:
            ot = self.encode(" " + o.replace(self.mask, " "))[:48]
            opt_ids.append([self.tok.mask_id] + ot)
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        squeezed = False
        if opt_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max_len - sum(len(o) for o in opt_ids)
            squeezed = True
        head = head[:max(8, opt_budget)]
        ids = [self.tok.cls_token_id] + head + [self.tok.sep_token_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.tok.sep_token_id)
        room = max(0, max_len - len(ids) - 1)
        truncated = len(state_ids) > room
        ids = ids + state_ids[:room] + [self.tok.sep_token_id]
        ids = ids[:max_len]
        markers = [m for m in markers if m < max_len]
        return dict(ids=ids, markers=markers, qtype=QTYPES[qtype_name], n_opts=len(options), head_len=len(head),
                    state_tokens=min(len(state_ids), room), state_truncated=truncated, squeezed=squeezed)


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def probs_from_logits(logits, qtype, temps):
    """phocinae-server `_probs`: logits / T[qtype], softmax in float64."""
    t = temps[qtype]
    return softmax([float(v) / t for v in logits])


def act_features(raw_logits):
    """model.py `_score`: top1, top1-top2, normalized entropy, k/255 from softmax of the RAW marker logits (T=1)."""
    p = softmax([float(v) for v in raw_logits])
    k = max(len(p), 2)
    s = sorted(p, reverse=True)
    top1 = s[0]
    top2 = s[1] if len(s) > 1 else 0.0
    ent = -sum(x * math.log(max(x, 1e-9)) for x in p) / math.log(k)
    return [top1, top1 - top2, ent, k / 255.0]


def act_probability(act_logits):
    return softmax([float(v) for v in act_logits])[0]


def decode_answer(qtype, probs, threshold=0.5):
    """phocinae-server answers: choice = argmax index, score = 2 + argmax (server levels) / argmax (dataset levels), noul = p[1] >= thr."""
    if qtype == QTYPES["noul"]:
        return bool(probs[1] >= threshold)
    return int(max(range(len(probs)), key=lambda i: probs[i]))


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
