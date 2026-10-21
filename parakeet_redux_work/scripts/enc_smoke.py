#!/usr/bin/env python3
"""Encoder smoke, part 1 (venv .venv-lt: litert-torch 0.9.4, torch 2.13, transformers 5.20.2): reference B's
encoder + encoder_projector wrapped as one module and written as an fp32 .tflite by litert_torch.convert, without
NeMo and without quantization.

Signature: mel [1, 128, 3000] fp32 (n_mels first, the 30 s window of the existing export) + length [1] int32 (valid
mel frames; the module builds the attention mask in the graph) -> enc [1, 375, 640] fp32 (encoder_projector
output, the joint's encoder input). -> out/enc/enc30_fp32.tflite, results/enc_census.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redux_ref as R  # noqa: E402

import litert_torch  # noqa: E402

OUT = "out/enc/enc30_fp32.tflite"


class EncWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.proj = model.encoder_projector

    def forward(self, mel, length):
        x = mel.transpose(1, 2)
        t = x.shape[1]
        mask = torch.arange(t, dtype=torch.int32)[None, :] < length[:, None]
        h = self.encoder(input_features=x, attention_mask=mask).last_hidden_state
        return self.proj(h)


def census(path):
    from ai_edge_litert import schema_py_generated as S
    buf = open(path, "rb").read()
    m = S.Model.GetRootAsModel(buf, 0)
    tt = {v: k for k, v in S.TensorType.__dict__.items() if not k.startswith("_")}
    bo = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}
    ops, flex, custom, rank_gt4, int64, fc_w = {}, 0, {}, [], [], {}
    for s in range(m.SubgraphsLength()):
        sg = m.Subgraphs(s)
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            if t.ShapeLength() > 4:
                rank_gt4.append((t.Name().decode(), t.ShapeAsNumpy().tolist()))
            if tt.get(t.Type()) == "INT64":
                int64.append(t.Name().decode())
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            oc = m.OperatorCodes(op.OpcodeIndex())
            code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
            name = bo.get(code, str(code))
            if name == "CUSTOM":
                cc = oc.CustomCode().decode() if oc.CustomCode() else "?"
                if cc.startswith("Flex"):
                    flex += 1
                else:
                    custom[cc] = custom.get(cc, 0) + 1
            ops[name] = ops.get(name, 0) + 1
            if name == "FULLY_CONNECTED":
                w = sg.Tensors(op.Inputs(1))
                k = tt.get(w.Type())
                fc_w[k] = fc_w.get(k, 0) + 1
    sigs = []
    for k in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(k)
        sg = m.Subgraphs(sd.SubgraphIndex())

        def io(n, get):
            res = []
            for j in range(n):
                tm = get(j)
                t = sg.Tensors(tm.TensorIndex())
                res.append({"name": tm.Name().decode(), "dtype": tt.get(t.Type()), "shape": t.ShapeAsNumpy().tolist()})
            return res
        sigs.append({"key": sd.SignatureKey().decode(), "inputs": io(sd.InputsLength(), sd.Inputs),
                     "outputs": io(sd.OutputsLength(), sd.Outputs)})
    return {"bytes": len(buf), "n_ops": sum(ops.values()), "ops": dict(sorted(ops.items(), key=lambda x: -x[1])),
            "flex": flex, "custom": custom, "rank_gt4_tensors": len(rank_gt4), "rank_gt4_examples": rank_gt4[:5],
            "int64_tensors": len(int64), "int64_examples": int64[:5], "fc_weight_dtypes": fc_w, "signatures": sigs}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    R.set_threads(8)
    t0 = time.time()
    model, rep = R.build("B")
    wrap = EncWrap(model).eval()
    mel = torch.zeros(1, 128, 3000)
    length = torch.tensor([3000], dtype=torch.int32)
    with torch.no_grad():
        y = wrap(mel, length)
    rec = {"load_report": rep, "env": R.env_record(), "litert_torch": litert_torch.__version__,
           "torch_out_shape": list(y.shape), "attn_implementation": model.config._attn_implementation}
    t1 = time.time()
    em = litert_torch.convert(wrap, (mel, length), enable_x64=False)
    em.export(OUT)
    rec["convert_s"] = round(time.time() - t1, 1)
    rec["total_s"] = round(time.time() - t0, 1)
    rec["census"] = census(OUT)
    json.dump(rec, open("results/enc_census.json", "w"), indent=1)
    c = rec["census"]
    print("bytes", c["bytes"], "ops", c["n_ops"], "flex", c["flex"], "custom", c["custom"], "rank>4",
          c["rank_gt4_tensors"], "int64", c["int64_tensors"], "fc", c["fc_weight_dtypes"], flush=True)
    print(json.dumps(c["signatures"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
