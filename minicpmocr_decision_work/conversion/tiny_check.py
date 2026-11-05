#!/usr/bin/env python3
"""Tiny random graphs of the contract's I/O through the whole path, in about a minute, before any heavy
export: litert-torch -> the static scan and both rank tools -> CompiledModel CPU (XNNPACK) and Metal fp32 (and Metal
default) against the torch module. No checkpoint read.

    PYTHONHASHSEED=0 $EXPORT conversion/tiny_check.py [--which decide,tower] [--only TAG]
        -> results/tiny_<tag>.json, out/tiny/*.tflite, logs/tiny_<tag>_<acc>.runtime.log

Cases:
  decide_L64 / L128 (one chunk / two), L256, L512, L512_qb128 (the query-block form), each with T
  smaller than L (padded) and T = L (no pads);
  tower_N64 / N256 / N576, N1024_qb256, each with n < N (padded) and n = N.
Truths: decide = the stock classes of transformers 5.14.1 (sdpa, the stock kernel, no patch) on exactly T tokens; tower = the
same TowerRow at N = n (no pads) in torch, and against the patched / padded module too.
Recorded per file x accelerator: rel_rms and max|d| against the truth, finite, a second call bit-equal, the delegation lines
(Replacing N out of M), a GPU-only CompiledModel; COS / SIN / INT64 / GATHER / BROADCAST_TO / custom / rank > 4 = 0; the lines
of a rank-3 lint and tfl_rank4.py (dry run through `rewrite --report`), when the environment variables RANK3_PRELINT and TFL_RANK4 name them
(tfl_rank4.py: tools/gpu_graph/ of github.com/john-rocky/hf-to-litertlm)."""
import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
WORK = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORK / "conversion"))
TINY_TEXT = {"hidden_size": 64, "intermediate_size": 256, "num_hidden_layers": 8,
             "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
             "linear_key_head_dim": 16, "linear_value_head_dim": 16}
ROT, SECTION = 8, (2, 1, 1)
FORBID = ("COS", "SIN", "GATHER", "GATHER_ND", "BROADCAST_TO", "SELECT_V2", "EMBEDDING_LOOKUP")
PRELINT = os.environ.get("RANK3_PRELINT", "")
RANK4 = os.environ.get("TFL_RANK4", "")


def inv_freq(dim):
    expo = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    return (np.float32(1.0) / np.power(np.float32(1e7), expo, dtype=np.float32)).astype(np.float32)


def mrope(pos3):
    ang = (pos3.astype(np.float32)[:, :, None] * inv_freq(ROT)[None, None, :]).astype(np.float32)
    a = ang[0].copy()
    for plane, offset in ((1, 1), (2, 2)):
        length = SECTION[plane] * 3
        a[:, offset:length:3] = ang[plane][:, offset:length:3]
    emb = np.concatenate([a, a], axis=-1).astype(np.float64)
    return np.cos(emb).astype(np.float32)[None], np.sin(emb).astype(np.float32)[None]


def text_planes(T):
    """Mixed planes: a text run, an image run (grid 4 x 8 merged), a text run, so h / w differ from t."""
    H, W = 4, 8
    n = H * W
    a = min(T // 4, 16)
    if T < a + n + 2:
        return np.broadcast_to(np.arange(T, dtype=np.int64), (3, T)).copy()
    pos = np.zeros((3, T), np.int64)
    pos[:, :a] = np.arange(a)
    k = np.arange(n)
    pos[0, a:a + n] = a
    pos[1, a:a + n] = a + k // W
    pos[2, a:a + n] = a + k % W
    cur = a + max(H, W)
    rest = T - a - n
    pos[:, a + n:] = cur + np.arange(rest)
    return pos


def padded(pos3, L):
    n = pos3.shape[1]
    out = np.empty((3, L), np.int64)
    out[:, :n] = pos3
    out[:, n:] = int(pos3.max()) + 1 + np.arange(L - n)[None, :]
    return out


def build_decide():
    import copy

    import torch

    import decide_graph as G
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    G.register_attention()
    cfg = copy.deepcopy(G.text_config())
    for k, v in TINY_TEXT.items():
        setattr(cfg, k, v)
    cfg.rope_parameters = dict(cfg.rope_parameters, mrope_section=list(SECTION))
    G._set_classes(G.PATCHED)
    torch.manual_seed(0)
    model = M.Qwen3_5TextModel(cfg)
    model.embed_tokens = None
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.normal_(1.0, 0.1)
        g = torch.Generator().manual_seed(7)
        for name, mod in model.named_modules():
            if hasattr(mod, "A_log") and hasattr(mod, "dt_bias"):
                mod.A_log.data = torch.log(torch.rand(mod.A_log.shape, generator=g) * 0.009 + 0.001)
    model = model.eval().requires_grad_(False)
    stock = G.stock_twin(model)
    return model, stock, cfg


def scan(path) -> dict:
    from ai_edge_litert import schema_py_generated as schema

    data = Path(path).read_bytes()
    m = schema.Model.GetRootAs(data, 0)
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    hist = collections.Counter()
    rank_gt4, int64, custom = 0, 0, 0
    fc_out_rank = collections.Counter()
    for gi in range(m.SubgraphsLength()):
        g = m.Subgraphs(gi)
        for ti in range(g.TensorsLength()):
            t = g.Tensors(ti)
            rank_gt4 += int(t.ShapeLength() > 4)
            int64 += int(type_names.get(t.Type()) == "INT64")
        for oi in range(g.OperatorsLength()):
            op = g.Operators(oi)
            c = m.OperatorCodes(op.OpcodeIndex())
            name = op_names.get(max(c.BuiltinCode(), c.DeprecatedBuiltinCode()), "?")
            hist[name] += 1
            custom += int(name == "CUSTOM")
            if name == "FULLY_CONNECTED":
                fc_out_rank[g.Tensors(op.Outputs(0)).ShapeLength()] += 1
    return {"ops": sum(hist.values()), "hist": dict(hist.most_common()), "rank_gt4": rank_gt4, "int64": int64, "custom": custom,
            "forbidden": {k: hist.get(k, 0) for k in FORBID if hist.get(k, 0)}, "fc_output_rank": dict(fc_out_rank)}


def prelint(path) -> dict:
    if not PRELINT:
        return {"not_run": "RANK3_PRELINT is not set"}
    out = Path(str(path) + ".prelint.json")
    r = subprocess.run([sys.executable, str(PRELINT), str(path), "--json", str(out), "--quiet"], capture_output=True, text=True)
    if not out.exists():
        return {"rc": r.returncode, "stderr": r.stderr[-400:]}
    d = json.loads(out.read_text())
    out.unlink()
    rows = d[0]["findings"] if isinstance(d, list) else d.get("findings", d.get("rows", []))
    c = collections.Counter()
    ex = {}
    for x in rows:
        key = f'{x.get("pattern")}|{x.get("confidence")}'
        c[key] += 1
        ex.setdefault(key, {k: x.get(k) for k in ("op", "op_index", "operands", "detail") if k in x})
    return {"rows": len(rows), "by_pattern": dict(c), "examples": ex}


def rank4_dry(path) -> dict:
    if not RANK4:
        return {"not_run": "TFL_RANK4 is not set"}
    out = Path(str(path) + ".rank4.tflite")
    rep = Path(str(path) + ".rank4.json")
    r = subprocess.run([sys.executable, str(RANK4), "rewrite", str(path), str(out), "--report", str(rep), "--unique-names"],
                       capture_output=True, text=True)
    doc = {"rc": r.returncode}
    if rep.exists():
        d = json.loads(rep.read_text())
        doc["rewrites"] = {k: (len(v) if isinstance(v, list) else v) for k, v in d.items() if k in ("rules", "sites", "counts", "by_rule", "changed")} or list(d)[:8]
        rep.unlink()
    else:
        doc["stderr"] = r.stderr[-400:]
    if out.exists():
        out.unlink()
    return doc


def run_all(path, sig, feeds, truths, tag, res_key, res) -> None:
    import lrt
    from rt_log import PARTITIONED, REPLACING, runtime_log_verbose

    for acc in ("cpu", "gpu_f32", "gpu_default"):
        item = {}
        log = WORK / f"logs/tiny_{tag}_{acc}.runtime.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        lf = open(log, "w")
        saved = os.dup(2)
        os.dup2(lf.fileno(), 2)
        try:
            item["logger"] = runtime_log_verbose().get("applied")
            g = lrt.Graph(path, sig, acc)
            outs = []
            for f in feeds:
                outs.append(g(**f))
            again = g(**feeds[0])
            item["load_s"] = round(g.load_s, 2)
            item["fully_accelerated"] = g.fully_accelerated
            item["vs_truth"] = []
            for o, (name, n_rows, mod, want) in zip(outs, truths):
                whole = o[list(o)[0]][0]
                got = whole[:n_rows]
                item["vs_truth"].append({"case": name, **lrt.stats(got, want), "vs_module": lrt.stats(got, mod),
                                         "nonfinite_real": int((~np.isfinite(got)).sum()), "nonfinite_all": int((~np.isfinite(whole)).sum())})
            a0, a1 = again[list(again)[0]][0], outs[0][list(outs[0])[0]][0]
            n0 = truths[0][1]
            item["repeat_bit_equal"] = bool(np.array_equal(a0, a1))
            item["repeat_bit_equal_real"] = bool(np.array_equal(a0[:n0], a1[:n0]))
            item["repeat_max_abs_real"] = float(np.abs(a0[:n0].astype(np.float64) - a1[:n0]).max())
            g.close()
            item["status"] = "OK"
        except BaseException as e:  # a refused compile is a result
            item.update(status="FAIL", error=f"{type(e).__name__}: {e}"[:400])
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            lf.close()
        txt = log.read_text(errors="replace")
        item["replacing"] = [{"delegated": int(m[0]), "total": int(m[1]), "partitions": int(m[3])} for m in REPLACING.findall(txt)]
        item["partitioned"] = [{"selected": int(m[1]), "total": int(m[2]), "partitions": int(m[3])} for m in PARTITIONED.findall(txt)]
        item["errors"] = [ln for ln in txt.splitlines() if ("ERROR" in ln or "No support" in ln or "not supported" in ln.lower())][:6]
        res[f"{res_key}_{acc}"] = item
        print(tag, acc, item.get("status"), item.get("error", ""),
              [(v["case"], float(f'{v["rel_rms"]:.3g}'), float(f'{v["vs_module"]["rel_rms"]:.3g}'), v["nonfinite_all"]) for v in item.get("vs_truth", [])],
              item.get("repeat_bit_equal_real"), item["replacing"][:1], flush=True)


def fp16fc(src, dst):
    from ai_edge_quantizer import qtyping, quantizer, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=16,
                              granularity=qtyping.QuantGranularity.CHANNELWISE, algorithm_key=AlgorithmName.FLOAT_CASTING)
    qt = quantizer.Quantizer(str(src), rm.get_quantization_recipe())
    qt.quantize().export_model(str(dst))


def decide_T(L):
    return (L - 37 if L > 64 else 41, L)


def decide_feed(cfg, L, T):
    d = cfg.hidden_size
    rng = np.random.default_rng(T)
    e = np.zeros((1, L, d), np.float32)
    e[0, :T] = rng.standard_normal((T, d)).astype(np.float32)
    v = np.zeros((1, L), np.float32)
    v[0, :T] = 1.0
    pos = text_planes(T)
    cos, sin = mrope(padded(pos, L))
    return {"embeds": e, "valid": v, "cos": cos, "sin": sin}, pos


def stock_truths(stock, cfg, cases):
    """Before the form is applied: the form patches Qwen3_5RMSNormGated at class level, and the stock twin shares the class."""
    import torch

    out = {}
    for L, _ in cases:
        for T in decide_T(L):
            feed, pos = decide_feed(cfg, L, T)
            with torch.inference_mode():
                out[(L, T)] = stock(inputs_embeds=torch.from_numpy(feed["embeds"][:, :T]), position_ids=torch.from_numpy(pos[:, None, :]).long(),
                                    use_cache=False).last_hidden_state[0].numpy()
    return out


def decide_case(model, truth, cfg, L, qblock, res, reuse=False):
    import torch
    import litert_torch

    import decide_graph as G

    tag = "decide_" + G.tag(L, qblock)
    gr = G.graph(model, L, qblock)
    cases = []
    for T in decide_T(L):
        feed, _ = decide_feed(cfg, L, T)
        tf = {k: torch.from_numpy(x) for k, x in feed.items()}
        with torch.inference_mode():
            mod = gr(**tf)["hidden"][0, :T].numpy()
        cases.append({"T": T, "feed": feed, "module": mod, "stock": truth[(L, T)]})
    import lrt
    doc = {"tag": tag, "L": L, "qblock": qblock}
    doc["module_vs_stock"] = {f"T{c['T']}": lrt.stats(c["module"], c["stock"]) for c in cases}
    print(tag, "module vs stock", {k: float(f'{v["rel_rms"]:.3g}') for k, v in doc["module_vs_stock"].items()}, flush=True)
    f32 = WORK / f"out/tiny/{tag}_fp32.tflite"
    f16 = WORK / f"out/tiny/{tag}_fp16fc.tflite"
    if not (reuse and f32.exists() and f16.exists()):
        for p in (f32, f16):
            if p.exists():
                p.unlink()
        t0 = time.time()
        sample = {k: torch.from_numpy(x) for k, x in cases[0]["feed"].items()}
        litert_torch.signature("decide", gr, sample_kwargs=sample).convert().export(str(f32))
        doc["export_s"] = round(time.time() - t0, 1)
        fp16fc(f32, f16)
    doc["scan"] = scan(f32)
    doc["prelint"] = prelint(f32)
    doc["rank4"] = rank4_dry(f32)
    feeds = [c["feed"] for c in cases]
    truths = [(f"T{c['T']}", c["T"], c["module"], c["stock"]) for c in cases]
    res_run = {}
    run_all(f32, "decide", feeds, truths, tag + "_fp32", "fp32", res_run)
    run_all(f16, "decide", feeds, truths, tag + "_fp16fc", "fp16fc", res_run)
    doc["runtime"] = res_run
    res[tag] = doc
    return doc


def tower_case(n_bucket, qblock, res):
    import torch
    import litert_torch

    import lrt
    import tower_graph as T

    tag = "tower_" + T.tag(n_bucket, qblock)
    hidden, heads = 64, 4
    d = hidden // heads
    sd = T.tiny_state(hidden=hidden, heads=heads)
    gr = T.TowerRow(sd, n_bucket, heads=heads, qblock=qblock).eval()
    g = torch.Generator().manual_seed(2)
    cases = []
    for n in sorted({n_bucket, max(4, n_bucket // 2 - (n_bucket // 2) % 4)}):
        pv = torch.randn(n, T.PATCH_DIM, generator=g).numpy()
        pos = (torch.randn(n, hidden, generator=g) * 0.1).numpy()
        ang = (torch.rand(n, d, generator=g) * 6.0).numpy()
        feed = T.pad_inputs(pv, pos, np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32), n_bucket)
        exact = T.TowerRow(sd, n, heads=heads).eval()
        with torch.inference_mode():
            mod = gr(**{k: torch.from_numpy(x) for k, x in feed.items()})["rows"][0, :n // 4].numpy()
            want = exact(torch.from_numpy(pv)[None], torch.ones(1, n), torch.from_numpy(pos)[None],
                         torch.from_numpy(np.cos(ang).astype(np.float32))[None],
                         torch.from_numpy(np.sin(ang).astype(np.float32))[None])["rows"][0].numpy()
        cases.append({"n": n, "feed": feed, "module": mod, "exact": want})
    doc = {"tag": tag, "N": n_bucket, "qblock": qblock}
    doc["module_vs_exact"] = {f"n{c['n']}": lrt.stats(c["module"], c["exact"]) for c in cases}
    f32 = WORK / f"out/tiny/{tag}_fp32.tflite"
    f16 = WORK / f"out/tiny/{tag}_fp16fc.tflite"
    for p in (f32, f16):
        if p.exists():
            p.unlink()
    t0 = time.time()
    sample = {k: torch.from_numpy(x) for k, x in cases[0]["feed"].items()}
    litert_torch.signature("tower", gr, sample_kwargs=sample).convert().export(str(f32))
    doc["export_s"] = round(time.time() - t0, 1)
    fp16fc(f32, f16)
    doc["scan"] = scan(f32)
    doc["prelint"] = prelint(f32)
    doc["rank4"] = rank4_dry(f32)
    feeds = [c["feed"] for c in cases]
    truths = [(f"n{c['n']}", c["n"] // 4, c["module"], c["exact"]) for c in cases]
    res_run = {}
    run_all(f32, "tower", feeds, truths, tag + "_fp32", "fp32", res_run)
    run_all(f16, "tower", feeds, truths, tag + "_fp16fc", "fp16fc", res_run)
    doc["runtime"] = res_run
    res[tag] = doc
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="decide,tower")
    ap.add_argument("--only", default="")
    ap.add_argument("--reuse", action="store_true", help="keep existing tiny files, only re-read them")
    a = ap.parse_args()
    import torch

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    (WORK / "out/tiny").mkdir(parents=True, exist_ok=True)
    (WORK / "results").mkdir(parents=True, exist_ok=True)
    res = {}
    t0 = time.time()
    which = a.which.split(",")
    if "decide" in which:
        import kernel_q35

        import decide_graph as G

        model, stock, cfg = build_decide()
        todo = [(L, qb) for L, qb in ((64, 0), (128, 0), (256, 0), (512, 0), (512, 128)) if not a.only or a.only == G.tag(L, qb)]
        truth = stock_truths(stock, cfg, todo)
        res_form = G.apply_form(model)
        for L, qb in todo:
            doc = decide_case(model, truth, cfg, L, qb, res, a.reuse)
            (WORK / f"results/tiny_decide_{G.tag(L, qb)}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        _ = kernel_q35, res_form
    if "tower" in which:
        import tower_graph as T

        for n, qb in ((64, 0), (256, 0), (576, 0), (1024, 256)):
            if a.only and a.only != T.tag(n, qb):
                continue
            doc = tower_case(n, qb, res)
            (WORK / f"results/tiny_tower_{T.tag(n, qb)}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"seconds": round(time.time() - t0, 1), "cases": list(res)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
