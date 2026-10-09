"""Round 2 step 5: the fp16 headroom of every norm site of D1Decision (checkpoint weights) over the whole fixture
-> results/norm_range.json

    cd K
    ~/code/standup/tools/quiet/quiet_wait.py -- venv-ref/bin/python scripts/norm_probe.py

Why (memory fp16-storage-norm-overflow): the converter lowers a norm's mean(x^2) to SUM + MUL; under fp16 storage
(GPU default precision, FP16_WITH_FP32_ACCUM) a sum of squares above 65,504 is inf, rsqrt 0, and the row comes back
exactly 0 with no non-finite value. So, per norm site, over the real positions (pad = 1: the prefix and the text) of
every fixture row at its smallest bucket (inputs = host build_inputs, media prefix from ref/npz), torch fp32 CPU:
  - RMSNorm (trunk operator_norm / ffn_norm x 16 layers, q_layernorm / k_layernorm per head (64) x 6 attention layers,
    embedding_norm): the max and min of sum(x^2) over the normalised dim, max |x|;
  - LayerNorm (head layers 0 / 1 norm1 / norm2, scorer.0): sum(x^2), sum((x - mean)^2), |sum(x)|, max |x|;
  - the same maxima over all L positions (pads included: an inf there can reach real rows as 0 * inf);
  - the residual stream absmax per layer (the operator_norm / ffn_norm / embedding_norm inputs), each operator's and
    MLP's output absmax, and the attention scores q.k^T (before the scale) on allowed pairs (real query, key the mask
    lets through) and over every entry, trunk (6 layers, [1, 8, 2L, L]) and head (2 layers, [1, 16, L, L]).
Per site: the margin 65,504 / max sum, the top 5 rows (id, position) and k = the smallest k >= 0 with
max sum * 4^-k <= 65,504 / 4 (input x 2^-k, eps x 4^-k leave fp32 bit-identical; computed only, applied later), with
the smallest real-position sum after the scale (fp16 normal floor 6.1e-5) and eps x 4^-k beside it.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

import d1_graph as G  # noqa: E402
import d1_src as S  # noqa: E402

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05
BUCKETS = (128, 256, 512, 1024, 2048, 4096)


class Site:
    def __init__(self, name, kind, dim, eps):
        self.name, self.kind, self.dim, self.eps = name, kind, dim, eps
        self.max_ss, self.min_ss, self.max_abs = 0.0, math.inf, 0.0
        self.max_ss_all, self.max_abs_all = 0.0, 0.0
        self.max_var_ss, self.max_abs_sum = 0.0, 0.0
        self.per_row = []          # (max_ss_real, row id, position)

    def note(self, x, n, rid):
        x = x.detach().float()
        L = x.shape[1]
        flat = x.reshape(L, -1, x.shape[-1])                      # [L, H, d] (H = 1 for [1, L, d])
        ss = (flat * flat).sum(-1)                                 # [L, H]
        real = ss[:n]
        mx, pos = real.max(dim=1).values.max(dim=0)
        self.max_ss = max(self.max_ss, float(mx))
        self.min_ss = min(self.min_ss, float(real.min()))
        self.max_abs = max(self.max_abs, float(flat[:n].abs().max()))
        self.max_ss_all = max(self.max_ss_all, float(ss.max()))
        self.max_abs_all = max(self.max_abs_all, float(flat.abs().max()))
        if self.kind == "layernorm":
            mu = flat[:n].mean(-1, keepdim=True)
            self.max_var_ss = max(self.max_var_ss, float(((flat[:n] - mu) ** 2).sum(-1).max()))
            self.max_abs_sum = max(self.max_abs_sum, float(flat[:n].sum(-1).abs().max()))
        self.per_row.append((float(mx), rid, int(pos)))

    def report(self):
        k = 0
        while self.max_ss * 4.0 ** -k > FP16_MAX / 4:
            k += 1
        top, seen = [], set()
        for v, rid, pos in sorted(self.per_row, key=lambda t: -t[0]):
            if rid in seen:
                continue
            seen.add(rid)
            top.append({"row": rid, "position": pos, "sum_sq": v})
            if len(top) == 5:
                break
        out = {"kind": self.kind, "dim": self.dim, "eps": self.eps, "max_sum_sq": self.max_ss,
               "min_sum_sq": self.min_ss, "max_abs_x": self.max_abs, "max_sum_sq_all_positions": self.max_ss_all,
               "max_abs_x_all_positions": self.max_abs_all, "margin_to_fp16_max": FP16_MAX / self.max_ss,
               "margin_all_positions": FP16_MAX / self.max_ss_all, "k": k,
               "max_sum_sq_after_k": self.max_ss * 4.0 ** -k, "min_sum_sq_after_k": self.min_ss * 4.0 ** -k,
               "min_after_k_below_fp16_min_normal": self.min_ss * 4.0 ** -k < FP16_MIN_NORMAL,
               "eps_after_k": self.eps * 4.0 ** -k, "top5": top}
        if self.kind == "layernorm":
            out["max_sum_sq_centered"] = self.max_var_ss
            out["max_abs_sum_x"] = self.max_abs_sum
            out["margin_centered"] = FP16_MAX / self.max_var_ss
        return out


class AbsMax:
    def __init__(self):
        self.real, self.all, self.where = 0.0, 0.0, None

    def note(self, x, n, rid, valid=None):
        x = x.detach().float()
        a = float(x.abs().max())
        r = float((x[valid] if valid is not None else x[:, :n]).abs().max())
        self.all = max(self.all, a)
        if r > self.real:
            self.real, self.where = r, rid

    def report(self):
        return {"max_abs_real": self.real, "max_abs_all": self.all, "row": self.where,
                "margin_real": FP16_MAX / self.real if self.real else None}


def main():
    t0 = time.time()
    torch.set_num_threads(12)
    sys.path.insert(0, str(S.K / "host"))
    import d1_host as H

    ref = json.loads((S.K / "ref/records_ref.json").read_text())
    fixtures = {r["id"]: r for r in json.loads((S.K / "fixtures/requests.json").read_text())["records"]}
    cfg_all = S.config()
    cfg = cfg_all["text_config"]
    sd = G.checkpoint_state(S.WEIGHTS)
    rows = []
    for e in ref["records"]:
        prefix = None
        if e["mode"] != "text":
            with np.load(S.K / "ref/npz" / f"{e['id']}.npz") as z:
                prefix = z["prefix"].copy()
        for q in e["questions"]:
            hq = H.Pm.as_question(fixtures[e["id"]]["request"]["questions"][q["qid"]])
            rows.append({"rid": f"{e['id']}/{q['qid']}", "ids": q["ids"], "prefix": prefix, "P": e["prefix"],
                         "n": q["positions"], "L": q["bucket"], "hq": hq, "mode": e["mode"]})
    cur = {"n": 0, "rid": None, "pad": None, "media": None}
    sites, res_in, res_mid, op_out, mlp_out = {}, {}, {}, {}, {}
    att_trunk, att_head = {}, {}
    final_in, scores_abs = AbsMax(), AbsMax()

    def pre(site):
        def hook(m, args):
            site.note(args[0], cur["n"], cur["rid"])
        return hook

    def build_hooks(m):
        hs = []
        for i, layer in enumerate(m.encoder.layers):
            for nm in ("operator_norm", "ffn_norm"):
                key = f"trunk.L{i:02d}.{nm}"
                sites.setdefault(key, Site(key, "rmsnorm", cfg["hidden_size"], cfg["norm_eps"]))
                hs.append(getattr(layer, nm).register_forward_pre_hook(pre(sites[key])))
            res_in.setdefault(i, AbsMax())
            res_mid.setdefault(i, AbsMax())
            op_out.setdefault(i, AbsMax())
            mlp_out.setdefault(i, AbsMax())
            hs.append(layer.operator_norm.register_forward_pre_hook(
                lambda mm, a, i=i: res_in[i].note(a[0], cur["n"], cur["rid"])))
            hs.append(layer.ffn_norm.register_forward_pre_hook(
                lambda mm, a, i=i: res_mid[i].note(a[0], cur["n"], cur["rid"])))
            hs.append(layer.feed_forward.register_forward_hook(
                lambda mm, a, o, i=i: mlp_out[i].note(o, cur["n"], cur["rid"])))
            opmod = layer.self_attn if layer.is_attention_layer else layer.conv
            hs.append(opmod.register_forward_hook(lambda mm, a, o, i=i: op_out[i].note(o, cur["n"], cur["rid"])))
            if layer.is_attention_layer:
                at = layer.self_attn
                for nm in ("q_layernorm", "k_layernorm"):
                    key = f"trunk.L{i:02d}.{nm}"
                    sites.setdefault(key, Site(key, "rmsnorm", at.head_dim, cfg["norm_eps"]))
                    hs.append(getattr(at, nm).register_forward_pre_hook(pre(sites[key])))
                att_trunk.setdefault(i, AbsMax())

                def att_hook(mm, args, out, i=i):
                    x, cos, sin, mask2 = args
                    L = x.shape[1]
                    q = mm.q_layernorm(mm.q_proj(x).reshape(1, L, mm.heads, mm.head_dim)).transpose(1, 2)
                    k = mm.k_layernorm(mm.k_proj(x).reshape(1, L, mm.kv_heads, mm.head_dim)).transpose(1, 2)
                    q = q * cos + G.rotate_half(q) * sin
                    k = k * cos + G.rotate_half(k) * sin
                    qg = q.reshape(1, mm.kv_heads, mm.groups * L, mm.head_dim)
                    rowpos = torch.arange(2 * L) % L
                    valid = (mask2[0, 0] == 0) & (rowpos < cur["n"])[:, None]
                    for g in range(mm.kv_heads):        # one kv group at a time ([2L, L]: 134 MB at L4096)
                        qk = torch.matmul(qg[0, g], k[0, g].transpose(0, 1))
                        att_trunk[i].note(qk, cur["n"], cur["rid"], valid=valid)
                hs.append(at.register_forward_hook(att_hook))
        key = "trunk.embedding_norm"
        sites.setdefault(key, Site(key, "rmsnorm", cfg["hidden_size"], cfg["norm_eps"]))
        hs.append(m.encoder.embedding_norm.register_forward_pre_hook(pre(sites[key])))
        hs.append(m.encoder.embedding_norm.register_forward_pre_hook(
            lambda mm, a: final_in.note(a[0], cur["n"], cur["rid"])))
        for j, hl in enumerate(m.head.layers):
            for nm in ("norm1", "norm2"):
                key = f"head.L{j}.{nm}"
                sites.setdefault(key, Site(key, "layernorm", cfg["hidden_size"], getattr(hl, nm).eps))
                hs.append(getattr(hl, nm).register_forward_pre_hook(pre(sites[key])))
            att_head.setdefault(j, AbsMax())

            def head_hook(mm, args, out, j=j):
                x, kmask = args
                L = x.shape[1]
                y = mm.norm1(x)
                q = mm.q_proj(y).reshape(1, L, mm.heads, mm.head_dim).transpose(1, 2)
                k = mm.k_proj(y).reshape(1, L, mm.heads, mm.head_dim).transpose(1, 2)
                valid = (kmask[0, 0] == 0) & (torch.arange(L) < cur["n"])[:, None]
                for hh in range(mm.heads):              # one head at a time ([L, L])
                    qk = torch.matmul(q[0, hh], k[0, hh].transpose(0, 1))
                    att_head[j].note(qk, cur["n"], cur["rid"], valid=valid)
            hs.append(hl.register_forward_hook(head_hook))
        key = "head.scorer.0"
        sites.setdefault(key, Site(key, "layernorm", cfg["hidden_size"], m.head.scorer[0].eps))
        hs.append(m.head.scorer[0].register_forward_pre_hook(pre(sites[key])))
        return hs

    by_L = {}
    for r in rows:
        by_L.setdefault(r["L"], []).append(r)
    with torch.no_grad():
        for L in sorted(by_L):
            m = G.D1Decision(cfg, cfg_all["head_layers"], L).eval()
            G.load_state_dict_from_provider(m, sd)
            hs = build_hooks(m)
            for r in by_L[L]:
                x = H.build_inputs(r["ids"], r["prefix"], L)
                x["qtype_onehot"] = H.qtype_onehot(r["hq"])
                t = {k: torch.from_numpy(v) for k, v in x.items()}
                cur.update(n=r["n"], rid=r["rid"])
                s = m(**t)["scores"]
                scores_abs.note(s[:, :, None], r["n"], r["rid"])
            for h in hs:
                h.remove()
            del m
    site_rep = {k: v.report() for k, v in sites.items()}
    kinds = {}
    for k, v in site_rep.items():
        typ = k.split(".")[-1] if k.startswith("trunk.L") else k
        if k.startswith("head.L"):
            typ = "head." + k.split(".")[-1]
        t = kinds.setdefault(typ, {"sites": 0, "max_sum_sq": 0.0, "min_margin": math.inf, "k_max": 0, "k_min": 99,
                                   "worst_site": None})
        t["sites"] += 1
        if v["max_sum_sq"] > t["max_sum_sq"]:
            t["max_sum_sq"], t["worst_site"] = v["max_sum_sq"], k
        t["min_margin"] = min(t["min_margin"], v["margin_to_fp16_max"])
        t["k_max"], t["k_min"] = max(t["k_max"], v["k"]), min(t["k_min"], v["k"])
    out = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "rows": len(rows), "by_L": {str(L): len(v) for L, v in sorted(by_L.items())},
        "oracle_sha256": S.sha256_file(S.K / "ref/records_ref.json"), "d1_graph_sha256": S.sha256_file(G.__file__),
        "fp16_max": FP16_MAX, "target_after_k": FP16_MAX / 4, "fp16_min_normal": FP16_MIN_NORMAL,
        "sites_needing_k_gt_0": sorted(k for k, v in site_rep.items() if v["k"] > 0),
        "sites_margin_below_4x": sorted(k for k, v in site_rep.items() if v["margin_to_fp16_max"] < 4),
        "by_kind": kinds,
        "sites": site_rep,
        "residual_stream": {f"L{i:02d}": {"in": res_in[i].report(), "after_operator": res_mid[i].report()}
                            for i in sorted(res_in)} | {"final_norm_input": final_in.report()},
        "operator_output": {f"L{i:02d}": op_out[i].report() for i in sorted(op_out)},
        "mlp_output": {f"L{i:02d}": mlp_out[i].report() for i in sorted(mlp_out)},
        "attention_qk_unscaled": {"trunk": {f"L{i:02d}": att_trunk[i].report() for i in sorted(att_trunk)},
                                  "head": {f"L{j}": att_head[j].report() for j in sorted(att_head)}},
        "scores": scores_abs.report(),
        "seconds": round(time.time() - t0, 1),
    }
    out["residual_absmax_max"] = max(max(v["in"]["max_abs_real"], v["after_operator"]["max_abs_real"])
                                     for k, v in out["residual_stream"].items() if k.startswith("L"))
    (S.K / "results/norm_range.json").write_text(json.dumps(out, indent=1) + "\n")
    show = {k: out[k] for k in ("rows", "by_L", "sites_needing_k_gt_0", "sites_margin_below_4x", "by_kind",
                                "residual_absmax_max", "scores", "seconds")}
    show["attention"] = {g: {k: v["max_abs_real"] for k, v in d.items()} for g, d in out["attention_qk_unscaled"].items()}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
