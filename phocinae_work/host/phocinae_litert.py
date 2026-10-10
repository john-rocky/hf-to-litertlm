"""Python host for the Phocinae-Largha-150M-v1 LiteRT graphs: a request (state + typed questions) -> answers.

    python phocinae_litert.py --dir <repo dir> --request request.json [--window 384] [--accel cpu|gpu] [--form int8emb|embeds]

Needs `tokenizers`, `numpy` and `ai-edge-litert` (2.2.0); no PyTorch, no phocinae-server. The sequence is the publisher's
(laya's build_sequence: `CLS <type> question: <instructions> SEP [MASK] option … SEP state SEP`, head budget 192 tokens,
window 256 / 384 / 512), the options are rendered as the hfmodels-android laya family does (`key: description`,
`level i: text`, `false: … / true: …`), the probabilities follow phocinae-server's `_probs` (logits / temperature per
question type from rl_agent_config.json, float64 softmax) and the answers its decoding: choice = the option with the
highest probability, score = the level with the highest probability (the server's own protocol uses nine fixed levels
2–10; here the levels are the question's criteria), noul = p(true) >= threshold (0.5). `act_probability` is the act
head's softmax[0], the publisher's escalation signal.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
NOUL_DEFAULT = ("no, the statement does not hold", "yes, the statement holds")
HIDDEN = 384
HEAD_MAX_LEN = 192
PAD_ID = 0


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def render_options(q):
    t = q["type"]
    crit = q.get("criteria")
    if t == "choice":
        return [k if (v is None or v == "") else f"{k}: {v}" for k, v in crit.items()], list(crit.keys())
    if t == "score":
        return [f"level {i}: {c}" for i, c in enumerate(crit)], [str(i) for i in range(len(crit))]
    f = crit.get("false") if isinstance(crit, dict) else None
    tr = crit.get("true") if isinstance(crit, dict) else None
    return ["false: " + (f or NOUL_DEFAULT[0]), "true: " + (tr or NOUL_DEFAULT[1])], ["false", "true"]


class PhocinaeLiteRT:
    def __init__(self, repo_dir, window=384, accel="cpu", form="int8emb", threads=4, gpu_precision="fp32"):
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, HardwareAccelerator, Options
        from tokenizers import Tokenizer
        d = Path(repo_dir)
        self.window = window
        self.form = form
        self.tok = Tokenizer.from_file(str(d / "tokenizer" / "tokenizer.json"))
        self.mask_id, self.cls_id, self.sep_id = self.tok.token_to_id("<mask>"), self.tok.token_to_id("<bos>"), self.tok.token_to_id("<eos>")
        cfg = json.loads((d / "rl_agent_config.json").read_text())
        self.temperature = [float(x) for x in cfg["temperature"]]       # [choice, score, noul]
        self.head_max_len = int(cfg.get("head_max_len", HEAD_MAX_LEN))
        main = d / f"phocinae_largha_150m_L{window}_fp16fc_{form}.tflite"
        if accel == "gpu":
            opts = Options(HardwareAccelerator.GPU, gpu_options=GpuOptions(enforce_f32=(gpu_precision == "fp32")))
        else:
            opts = Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
        self.model = CompiledModel.from_file(str(main), options=opts)
        self.key = next(iter(self.model.get_signature_list()))
        self.ins = {n: self.model.create_input_buffer_by_name(self.key, n) for n in self.model.get_input_tensor_details(self.key)}
        self.outs = {n: self.model.create_output_buffer_by_name(self.key, n) for n in self.model.get_output_tensor_details(self.key)}
        self.act = CompiledModel.from_file(str(d / "phocinae_act_head_fp32.tflite"), options=Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=1)))
        self.akey = next(iter(self.act.get_signature_list()))
        self.ains = {n: self.act.create_input_buffer_by_name(self.akey, n) for n in self.act.get_input_tensor_details(self.akey)}
        self.aouts = {n: self.act.create_output_buffer_by_name(self.akey, n) for n in self.act.get_output_tensor_details(self.akey)}
        self.table = None
        if form == "embeds":
            raw = np.fromfile(d / "tables" / "tok_embeddings_fp16.bin", dtype="<f2")
            self.table = raw.reshape(-1, HIDDEN).astype(np.float32)

    def encode(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def build(self, q, state_ids):
        opts, keys = render_options(q)
        ins = q["instructions"].replace("<mask>", " ")
        head = self.encode(f"{q['type']} question: {ins}")
        opt_ids = [[self.mask_id] + self.encode(" " + o.replace("<mask>", " "))[:48] for o in opts]
        budget = self.head_max_len - sum(len(o) for o in opt_ids)
        if budget < 16:
            per = max(4, (self.head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            budget = self.head_max_len - sum(len(o) for o in opt_ids)
        head = head[: max(8, budget)]
        ids = [self.cls_id] + head + [self.sep_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.sep_id)
        room = max(0, self.window - len(ids) - 1)
        truncated = len(state_ids) > room
        ids = (ids + state_ids[:room] + [self.sep_id])[: self.window]
        markers = [m for m in markers if m < self.window]
        if len(markers) != len(opts):
            raise ValueError(f"question does not fit the {self.window}-token window: {len(markers)} of {len(opts)} options kept")
        return ids, markers, keys, truncated

    def forward(self, ids, markers, qtype):
        L = self.window
        arr = np.full((1, L), PAD_ID, dtype=np.int32)
        arr[0, : len(ids)] = np.asarray(ids, dtype=np.int32)
        mask = np.zeros((1, L), dtype=np.float32)
        mask[0, : len(ids)] = 1.0
        onehot = np.zeros((1, 3), dtype=np.float32)
        onehot[0, qtype] = 1.0
        if self.table is not None:
            self.ins["inputs_embeds"].write(np.ascontiguousarray(self.table[arr[0]][None]))
        else:
            self.ins["input_ids"].write(arr)
        self.ins["attention_mask"].write(mask)
        self.ins["qtype_onehot"].write(onehot)
        self.model.run_by_name(self.key, self.ins, self.outs)
        tl = np.asarray(self.outs["token_logits"].read(L, np.float32), dtype=np.float32).reshape(L)
        pc = np.asarray(self.outs["pooled_cls"].read(HIDDEN, np.float32), dtype=np.float32).reshape(HIDDEN)
        raw = tl[markers].astype(np.float64)
        p0 = softmax(list(raw))
        k = max(len(p0), 2)
        s = sorted(p0, reverse=True)
        ent = -sum(x * math.log(max(x, 1e-9)) for x in p0) / math.log(k)
        feats = np.asarray([[s[0], s[0] - (s[1] if len(s) > 1 else 0.0), ent, k / 255.0]], dtype=np.float32)
        self.ains["pooled_cls"].write(np.ascontiguousarray(pc[None]))
        self.ains["feats"].write(feats)
        self.act.run_by_name(self.akey, self.ains, self.aouts)
        al = np.asarray(self.aouts["act_logits"].read(2, np.float32), dtype=np.float64).reshape(2)
        return raw, al

    def decide(self, request):
        t0 = time.perf_counter()
        state = request["state"]
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        state_ids = self.encode(state.replace("<mask>", " "))
        answers, truncated, tokens = {}, False, 0
        for qid, q in request["questions"].items():
            ids, markers, keys, cut = self.build(q, state_ids)
            truncated |= cut
            tokens += len(ids)
            raw, al = self.forward(ids, markers, QTYPES[q["type"]])
            p = softmax([v / self.temperature[QTYPES[q["type"]]] for v in raw])
            actp = round(softmax(list(al))[0], 4)
            probs = {k: round(v, 4) for k, v in zip(keys, p)}
            if q["type"] == "choice":
                best = int(np.argmax(p))
                answers[qid] = dict(type="choice", choice=keys[best], confidence=round(max(p), 4), probabilities=probs, action=dict(act_probability=actp))
            elif q["type"] == "score":
                best = int(np.argmax(p))
                answers[qid] = dict(type="score", level=best, legend={str(i): c for i, c in enumerate(q["criteria"])}, confidence=round(max(p), 4),
                                    probabilities=probs, action=dict(act_probability=actp))
            else:
                thr = float(q.get("threshold", 0.5))
                answers[qid] = dict(type="noul", noul=bool(p[1] >= thr), probability_true=round(p[1], 4), confidence=round(max(p), 4), action=dict(act_probability=actp))
        return dict(model="Phocinae-Largha-150M-v1", answers=answers, usage=dict(input_tokens=tokens, output_tokens=0, state_truncated=truncated),
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--request", required=True)
    ap.add_argument("--window", type=int, default=384)
    ap.add_argument("--accel", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--form", choices=["int8emb", "embeds"], default="int8emb")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    host = PhocinaeLiteRT(a.dir, a.window, a.accel, a.form, a.threads)
    req = json.loads(Path(a.request).read_text())
    print(json.dumps(host.decide(req), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
