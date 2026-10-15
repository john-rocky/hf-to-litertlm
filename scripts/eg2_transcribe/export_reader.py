#!/usr/bin/env python3
"""The EmbeddingGemma 2 Transcribe reader + its word table + the argmax -> one LiteRT graph per window length.

Reader = mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader (reader.pt, reader_config.json, vocab.txt, word_table.npy,
RELEASE.md); its model code = reader/train/{model,decode}.py of github.com/john-rocky/eg2-transcribe.

ReaderGraph(T): soft [1, T, 512] float32 + mask [1, T] float32 (1 = a real token) ->
  ids [1, T] int32    argmax over the 200,000 words of scale x cos(h, table[w])
  best [1, T] float32 that logit
  blank [1, T] float32 the blank logit
The host's label per token: blank when blank > best, else ids (model.greedy_labels' rule); then merge repeats and drop
blanks. No 200,001-wide output leaves the graph.

The rewrites of model.Reader's forward (checked here in torch on the 20 fixtures, ids and labels equal):
  - qkv as three Linears (the q / k / v rows of the fused weight); q's rows and bias times 1/8 (SDPA's 1/sqrt(64), a power
    of two: exact through the rotary), so no scale op on the [8, T, T] scores
  - attention written out: softmax(q k^T + bias) v with bias = (mask - 1) * 30000 on the keys (the masked exp is 0 in
    float32 as with -inf, and the constant is finite in fp16)
  - rotary cos / sin as constant [1, 1, T, 64] tables (model.rope for positions 0..T-1)
  - normalize: v / max(|v|, 1e-12)
  - the table: FC 768 -> 200,000 with the release's float16 rows, times a per-word constant scale / |row| (the float32
    re-normalization of decode.load_table and the scale in one MUL)
  - ARG_MAX (int32 out) + REDUCE_MAX over the words
Files: reader_fp32_w<S>.tflite (the converter's output, every weight float32), then reader_w<S>.tflite = the table's FC
weight stored as float16 with an explicit DEQUANTIZE (the stored values are the release's float16 rows, so nothing is
lost; the body stays float32). Signatures reader_30s (T = 750) and reader_10s (T = 250).
Names: tensor names carry the class path "__main__.build_graph_class.<locals>.ReaderGraph"; run this file as a script and
keep the class where it is to get the published bytes.

Usage:
  export_reader.py export --reader-dir <Hub snapshot> --reader-code <eg2-transcribe>/reader/train
                          --fixtures <eg2-transcribe>/reader/fixtures --out <dir> [--windows 30,10]
  export_reader.py check  --reader-dir ... --reader-code ... --fixtures ... --out <dir> [--threads 4]
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common

MASK_VALUE = 30000.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reader_modules(code_dir):
    sys.path.insert(0, code_dir)
    import model as M  # noqa: E402
    import decode as D  # noqa: E402
    return M, D


def build_graph_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ReaderGraph(nn.Module):
        def __init__(self, reader, table16, T, chunks=1, rope=None):
            super().__init__()
            body = reader.body
            self.T, self.heads, self.chunks = T, body.heads, chunks
            d = body.inp.out_features
            self.dh = d // self.heads
            self.ln_in, self.inp, self.ln_out, self.out = body.ln_in, body.inp, body.ln_out, body.out
            self.blank = reader.blank
            self.ln1, self.ln2, self.ff1, self.ff2, self.proj = (nn.ModuleList() for _ in range(5))
            self.q, self.k, self.v = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            qscale = 1.0 / math.sqrt(self.dh)
            assert qscale == 0.125
            for blk in body.blocks:
                W, b = blk.qkv.weight.detach(), blk.qkv.bias.detach()
                for i, (dst, s) in enumerate(((self.q, qscale), (self.k, 1.0), (self.v, 1.0))):
                    lin = nn.Linear(d, d)
                    lin.weight.data = (W[i * d:(i + 1) * d] * s).clone()
                    lin.bias.data = (b[i * d:(i + 1) * d] * s).clone()
                    dst.append(lin)
                self.ln1.append(blk.ln1); self.ln2.append(blk.ln2); self.ff1.append(blk.ff1); self.ff2.append(blk.ff2)
                self.proj.append(blk.proj)
            cos, sin = rope(T, self.dh, body.rope_base, torch.device("cpu"))
            self.register_buffer("cos", cos.reshape(1, 1, T, self.dh).clone())
            self.register_buffer("sin", sin.reshape(1, 1, T, self.dh).clone())
            N = table16.shape[0]
            assert N % chunks == 0
            n = N // chunks
            t32 = table16.astype(np.float32)
            norms = np.linalg.norm(table16.astype(np.float64), axis=1)
            rs = (float(reader.scale.detach()) / norms).astype(np.float32)
            self.tables = nn.ModuleList()
            self.rs = []
            for c in range(chunks):
                lin = nn.Linear(t32.shape[1], n, bias=False)
                lin.weight.data = torch.from_numpy(t32[c * n:(c + 1) * n].copy())
                self.tables.append(lin)
                self.register_buffer(f"rs{c}", torch.from_numpy(rs[c * n:(c + 1) * n].copy()))
                self.rs.append(f"rs{c}")

        def rot(self, x):
            h = self.dh // 2
            return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

        def forward(self, soft, mask):
            T, H, dh = self.T, self.heads, self.dh
            bias = ((mask - 1.0) * MASK_VALUE).reshape(1, 1, 1, T)
            x = self.inp(self.ln_in(soft))
            for i in range(len(self.q)):
                y = self.ln1[i](x)
                q = self.q[i](y).reshape(1, T, H, dh).transpose(1, 2)
                k = self.k[i](y).reshape(1, T, H, dh).transpose(1, 2)
                v = self.v[i](y).reshape(1, T, H, dh).transpose(1, 2)
                q = q * self.cos + self.rot(q) * self.sin
                k = k * self.cos + self.rot(k) * self.sin
                a = torch.softmax(torch.matmul(q, k.transpose(2, 3)) + bias, dim=-1)
                a = torch.matmul(a, v).transpose(1, 2).reshape(1, T, H * dh)
                x = x + self.proj[i](a)
                x = x + self.ff2[i](F.gelu(self.ff1[i](self.ln2[i](x))))
            v = self.out(self.ln_out(x))
            blank = self.blank(v).reshape(1, T)
            n = torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True))
            hn = v / torch.clamp(n, min=1e-12)
            if self.chunks == 1:
                logits = self.tables[0](hn) * getattr(self, self.rs[0])
                best = torch.amax(logits, dim=-1)
                ids = torch.argmax(logits, dim=-1).to(torch.int32)
            else:
                bs, ix = [], []
                for c in range(self.chunks):
                    lg = self.tables[c](hn) * getattr(self, self.rs[c])
                    bs.append(torch.amax(lg, dim=-1, keepdim=True))
                    ix.append(torch.argmax(lg, dim=-1, keepdim=True).to(torch.int32))
                best, ids = torch.cat(bs, dim=-1), torch.cat(ix, dim=-1)
            return {"ids": ids, "best": best, "blank": blank}

    return ReaderGraph


def torch_labels(M, reader, table32n, x, n):
    """model.Reader on tokens 0..n-1 -> (word ids [n], best [n], blank [n], labels [n])."""
    import torch
    with torch.no_grad():
        h, bl = reader(torch.from_numpy(x[:n])[None], torch.tensor([n]))
        h, bl = h[0], bl[0]
        v, i = (h @ table32n.T).max(dim=1)
        lab = M.greedy_labels(h, bl, table32n, reader.scale).numpy()
        return i.numpy(), (v * reader.scale).numpy(), bl.numpy(), lab


def table_cast(src, dst):
    """FLOAT_CASTING (float16 weights, explicit DEQUANTIZE) of the table FCs only (the FCs with >= 20,000 outputs)."""
    import mmap
    from ai_edge_quantizer import quantizer
    from ai_edge_litert import schema_py_generated as schema
    cfg = {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}, "compute_precision": "FLOAT",
           "explicit_dequantize": True}
    with open(src, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    m = schema.Model.GetRootAs(mm, 0)
    sg = m.Subgraphs(0)
    names = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        code = m.OperatorCodes(op.OpcodeIndex())
        if max(code.BuiltinCode(), code.DeprecatedBuiltinCode()) != schema.BuiltinOperator.FULLY_CONNECTED:
            continue
        w = sg.Tensors(op.Inputs(1))
        if w.ShapeLength() == 2 and w.Shape(0) >= 20000:
            names.append(sg.Tensors(op.Outputs(0)).Name().decode())
    del m, sg
    mm.close()
    assert names, "no table FC found"
    # the quantizer's op scope is the op's output tensor names joined by ";" plus a trailing ";": without the "$" anchor
    # on that trailing ";" the regex matches nothing and the file comes out unchanged
    recipe = [{"regex": "^" + re.escape(nm) + ";$", "operation": "FULLY_CONNECTED", "algorithm_key": "float_casting",
               "op_config": dict(cfg)} for nm in names]
    qt = quantizer.Quantizer(src)
    qt.load_quantization_recipe(recipe)
    qt.quantize().export_model(dst + ".tmp.tflite", overwrite=True)
    os.replace(dst + ".tmp.tflite", dst)
    return len(recipe)


def cmd_export(a):
    import litert_torch  # noqa: F401  (before torch models are built: it registers its torch ops)
    import torch
    from litert_torch._convert import interface as converter_utils
    M, D = reader_modules(a.reader_code)
    torch.set_num_threads(a.threads)
    model, table32n, vocab, cfg = D.load_reader_dir(a.reader_dir, "cpu")
    N = len(vocab)
    table16 = np.load(os.path.join(a.reader_dir, cfg["table"]["file"]))
    rep = {"litert_torch": getattr(litert_torch, "__version__", "?"), "torch": torch.__version__,
           "reader_pt_sha256": common.sha256_file(os.path.join(a.reader_dir, "reader.pt")),
           "scale": float(model.scale.detach()), "n_vocab": N, "mask_value": -MASK_VALUE, "torch_check": {}, "files": {}}
    RG = build_graph_class()
    wss = [int(s) for s in a.windows.split(",")]
    graphs = {ws: RG(model, table16, common.WINDOW_TOKENS[ws], 1, rope=M.rope).eval() for ws in wss}
    ids = common.fixture_ids(a.fixtures)
    for ws, g in graphs.items():
        T = common.WINDOW_TOKENS[ws]
        same_ids = same_lab = total = 0
        for uid in ids:
            x = common.load_fixture(a.fixtures, uid, "soft_tokens")
            n = min(x.shape[0], T)
            ref_ids, _, _, ref_lab = torch_labels(M, model, table32n, x, n)
            xp = np.zeros((1, T, 512), np.float32); xp[0, :n] = x[:n]
            mk = np.zeros((1, T), np.float32); mk[0, :n] = 1.0
            with torch.no_grad():
                o = g(torch.from_numpy(xp), torch.from_numpy(mk))
            gi, gb, gbl = o["ids"][0, :n].numpy(), o["best"][0, :n].numpy(), o["blank"][0, :n].numpy()
            lab = np.where(gbl > gb, N, gi)
            same_ids += int((gi == ref_ids).sum()); same_lab += int((lab == ref_lab).sum()); total += n
        rep["torch_check"][f"w{ws}"] = {"tokens": total, "ids_equal": same_ids, "labels_equal": same_lab}
        log(f"torch check w{ws}: ids {same_ids}/{total}, labels {same_lab}/{total}")
        if same_lab != total:
            raise SystemExit("the graph module does not give model.Reader's labels: not converted")
    os.makedirs(a.out, exist_ok=True)
    for ws in wss:
        T = common.WINDOW_TOKENS[ws]
        f32 = os.path.join(a.out, f"reader_fp32_w{ws}.tflite")
        dst = os.path.join(a.out, f"reader_w{ws}.tflite")
        for p in (f32, dst):
            if os.path.exists(p):
                raise SystemExit(f"refusing to overwrite {p}")
        t1 = time.time()
        conv = converter_utils.Converter()
        conv.add_signature(f"reader_{ws}s", graphs[ws], sample_kwargs={"soft": torch.zeros(1, T, 512), "mask": torch.ones(1, T)})
        conv.convert(strict_export=False).export(f32 + ".tmp")
        os.replace(f32 + ".tmp", f32)
        rep["files"][os.path.basename(f32)] = {"signature": f"reader_{ws}s", "bytes": os.path.getsize(f32),
                                              "sha256": common.sha256_file(f32), "seconds": round(time.time() - t1, 1)}
        log(f"w{ws}: {f32} {os.path.getsize(f32):,} B in {time.time() - t1:.1f} s")
        t1 = time.time()
        n_fc = table_cast(f32, dst)
        rep["files"][os.path.basename(dst)] = {"from": os.path.basename(f32), "float16": f"table FC weight ({n_fc} op)",
                                              "bytes": os.path.getsize(dst), "sha256": common.sha256_file(dst),
                                              "seconds": round(time.time() - t1, 1)}
        log(f"w{ws}: {dst} {os.path.getsize(dst):,} B in {time.time() - t1:.1f} s")
        json.dump(rep, open(os.path.join(a.out, "export_reader_report.json"), "w"), indent=1)
    for name, f in rep["files"].items():
        print(f"{f['sha256']}  {name}", flush=True)
    log("READER_EXPORT_DONE")


def cmd_check(a):
    """reader_w<S>.tflite on the CPU (LiteRT Interpreter) on the 20 fixtures' soft tokens: labels against model.Reader on
    the same tokens, transcripts against the RELEASE.md table (30 s: all 20; 10 s: the utterances of <= 250 tokens)."""
    import torch
    from ai_edge_litert.interpreter import Interpreter
    M, D = reader_modules(a.reader_code)
    model, table32n, vocab, cfg = D.load_reader_dir(a.reader_dir, "cpu")
    N = len(vocab)
    rel = common.release_table(os.path.join(a.reader_dir, "RELEASE.md"))
    ids = common.fixture_ids(a.fixtures)
    rep = {"runtime": "ai_edge_litert Interpreter (XNNPACK)", "threads": a.threads, "release_rows": len(rel), "files": {}}
    for ws in [int(s) for s in a.windows.split(",")]:
        T = common.WINDOW_TOKENS[ws]
        path = os.path.join(a.out, f"reader_w{ws}.tflite")
        run = Interpreter(model_path=path, num_threads=a.threads).get_signature_runner(f"reader_{ws}s")
        same = total = whole = equal_release = 0
        rows = []
        for uid in ids:
            x = common.load_fixture(a.fixtures, uid, "soft_tokens")
            n = min(x.shape[0], T)
            _, _, _, ref_lab = torch_labels(M, model, table32n, x, n)
            xp = np.zeros((1, T, 512), np.float32); xp[0, :n] = x[:n]
            mk = np.zeros((1, T), np.float32); mk[0, :n] = 1.0
            o = run(soft=xp, mask=mk)
            gi = np.asarray(o["ids"])[0, :n]
            lab = np.where(np.asarray(o["blank"])[0, :n] > np.asarray(o["best"])[0, :n], N, gi)
            same += int((lab == ref_lab).sum()); total += n
            text = " ".join(vocab[i] for i in M.collapse(lab.tolist(), N))
            row = {"id": uid, "tokens": int(x.shape[0]), "labels_equal": int((lab == ref_lab).sum())}
            if n == x.shape[0]:
                whole += 1
                row["transcript_equal_release"] = text == rel.get(uid)
                equal_release += int(row["transcript_equal_release"])
            rows.append(row)
        r = {"tokens": total, "labels_equal_torch": same, "whole_utterances": whole, "transcripts_equal_release": equal_release,
             "pass": bool(same == total and equal_release == whole), "rows": rows}
        rep["files"][os.path.basename(path)] = r
        log(f"{os.path.basename(path)}: labels {same}/{total} = model.Reader, transcripts {equal_release}/{whole} = RELEASE.md "
            f"({'PASS' if r['pass'] else 'FAIL'})")
        del run
    json.dump(rep, open(os.path.join(a.out, "reader_check.json"), "w"), indent=1)
    log("READER_CHECK_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "check"])
    ap.add_argument("--reader-dir", required=True, help="snapshot of mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader")
    ap.add_argument("--reader-code", required=True, help="eg2-transcribe/reader/train (model.py, decode.py)")
    ap.add_argument("--fixtures", required=True, help="eg2-transcribe reader/fixtures (MANIFEST.json + rebuilt arrays)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="30,10")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    n = common.check_fixtures(a.fixtures, names=("soft_tokens.npy",))
    print(f"fixtures: {n} files equal to MANIFEST.json", flush=True)
    {"export": cmd_export, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
