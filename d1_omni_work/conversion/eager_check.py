"""Step 5: D1Decision (scripts/d1_graph.py) against the provider's encoder.Trunk + encoder.DecisionHead, both holding
the same random weights, eval, no_grad, CPU fp32 -> results/eager_random_check.json

    venv-ref/bin/python scripts/eager_check.py

Weights: torch.manual_seed(0), the provider's own init, then every norm weight drawn as 1 + 0.1 N(0, 1) and every bias
as 0.02 N(0, 1) (generator seed 1), so a mis-mapped norm or bias shows up; loaded into D1Decision through
load_state_dict_from_provider(). L = 64, three cases: (a) text 40 + pad 24, (b) prefix 16 + text 36 + pad 12,
(c) text 64, each with qtype 0 / 1 / 2 and three markers.
Reference = the provider's path for one row (no padding: _forward on a single row pads nothing); a second reference
pads the row to L (what _run's batching does to a shorter row). The provider's head is run with the
TransformerEncoderLayer fast path enabled (the default) and disabled.
Bars: logits max|d| <= 1e-5 with argmax equal; trunk output (after the final norm, all text positions) max|d| <= 1e-5;
pad-content / keep_right / mask invariances bit-equal (torch.equal); no repeat_interleave / expand / repeat /
broadcast_to call and no tensor above rank 4 in D1Decision (AST scan of the source + a TorchFunctionMode op audit).

Round 2: `--ckpt` -> results/eager_ckpt_check.json (see ckpt_main()): D1Decision with the checkpoint's weights
against the provider's oracle (ref/records_ref.json, round 2 step 2) on every row.
"""
import ast
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

import d1_graph as G
import d1_src as S

L = 64
BAR = 1e-5
HEADER = Path.home() / "code/standup/handoffs/assets/2026-10-08-d1/source/d1-omni-600m_safetensors_header.json"
CASES = {  # name: (prefix length, text length, state / instruction / option random-token counts)
    "a_text40_pad24": (0, 40, (10, 8, 3)),
    "b_prefix16_text36_pad12": (16, 36, (8, 6, 3)),
    "c_text64_nopad": (0, 64, (25, 11, 5)),
}


def text_row(n, parts, g):
    """[bos, state_delim, s.., q_delim, i.., (opt, marker, o.., opt_end) x 3, decide] of length n + marker positions."""
    ns, ni, no = parts
    r = lambda k: torch.randint(22, 64900, (k,), generator=g).tolist()  # noqa: E731
    ids = [1, 17] + r(ns) + [18] + r(ni)
    markers = []
    for _ in range(3):
        markers.append(len(ids) + 1)
        ids += [19, 16] + r(no) + [20]
    ids.append(21)
    assert len(ids) == n, (len(ids), n)
    return ids, markers


def our_inputs(text_ids, prefix_embeds, qtype):
    """The host's build_inputs() rule (host/d1_host.py), inlined: right padding with id 0, prefix rows first."""
    P, n = (0 if prefix_embeds is None else prefix_embeds.shape[0]), len(text_ids)
    t = torch.arange(L)
    ids = torch.zeros(1, L, dtype=torch.int32)
    ids[0, P:P + n] = torch.tensor(text_ids, dtype=torch.int32)
    prefix = torch.zeros(1, L, G.D)
    if P:
        prefix[0, :P] = prefix_embeds
    return {"ids": ids, "prefix": prefix, "media": (t < P).float()[None], "pad": (t < P + n).float()[None],
            "keep_right": (t != P - 1).float()[None], "qtype_onehot": F.one_hot(torch.tensor([qtype]), 3).float()}


def provider_logits(PT, PH, text_ids, prefix_embeds, markers, qtype, padded):
    P, n = (0 if prefix_embeds is None else prefix_embeds.shape[0]), len(text_ids)
    emb = PT.embed_tokens(torch.tensor(text_ids))
    h = emb if P == 0 else torch.cat([prefix_embeds, emb])
    total = P + n
    if padded:
        h = torch.cat([h, torch.zeros(L - total, G.D)])
    h = h[None]
    pad = (torch.arange(h.shape[1]) < total)[None]
    hp = PT(h, pad, torch.tensor([P]))
    text = hp[:, P:P + n]
    if padded:   # the head input padded the way pad_sequence pads a shorter row of a batch
        text = torch.cat([text, torch.zeros(1, L - P - n, G.D)], dim=1)
    text_pad = (torch.arange(text.shape[1]) < n)[None]
    logits = PH(text, text_pad, torch.tensor([markers]), torch.ones(1, len(markers), dtype=torch.bool),
                torch.tensor([qtype]))
    return logits[0], hp[0, P:P + n], hp[0, :P]


def maxabs(a, b):
    return float((a - b).abs().max())


def randomize(trunk, head):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in sorted(list(trunk.named_parameters(prefix="encoder")) + list(head.named_parameters(prefix="head"))):
            if p.dim() == 1 and name.endswith("bias"):
                p.copy_(0.02 * torch.randn(p.shape, generator=g))
            elif p.dim() == 1 and ("norm" in name or "scorer.0" in name):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))


def source_scan():
    src = Path(G.__file__).read_text()
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            calls.append(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "?"))
    banned = ("repeat_interleave", "expand", "expand_as", "repeat", "broadcast_to", "tile")
    raw = {w: len(re.findall(rf"\b{w}\(", src)) for w in banned}
    return {"ast_call_hits": {w: calls.count(w) for w in banned}, "raw_text_call_hits": raw,
            "raw_text_word_hits_incl_docstrings": {w: len(re.findall(rf"\b{w}\b", src)) for w in banned},
            "pass": all(calls.count(w) == 0 for w in banned) and all(v == 0 for v in raw.values())}


class OpAudit(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.names, self.max_rank, self.rank_by_op = set(), 0, {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "__name__", str(func))
        self.names.add(name)
        outs = out if isinstance(out, (tuple, list)) else [out]
        for o in outs:
            if isinstance(o, torch.Tensor):
                self.max_rank = max(self.max_rank, o.dim())
                self.rank_by_op[name] = max(self.rank_by_op.get(name, 0), o.dim())
        return out


def header_keymap(model):
    """Map the real checkpoint's header (names + shapes, no weights) through KEY_RULES onto the module."""
    hdr = json.loads(HEADER.read_text())
    ours = {k: list(v.shape) for k, v in model.state_dict().items()}
    mapped, unmatched, ignored, mismatch = {}, [], 0, []
    for key, meta in hdr.items():
        if key == "__metadata__":
            continue
        t = G.map_key(key, meta["shape"])
        if t is None:
            unmatched.append(key)
        elif not t:
            ignored += 1
        else:
            for ok, how, shape in t:
                mapped[ok] = shape
                if ours.get(ok) != shape:
                    mismatch.append({"key": ok, "header_derived": shape, "module": ours.get(ok)})
    missing = sorted(set(ours) - set(mapped))
    return {"header": str(HEADER), "header_tensors": len(hdr) - ("__metadata__" in hdr),
            "encoder_head_tensors": sum(1 for k in hdr if k.startswith(("encoder.", "head."))),
            "module_parameters": len(ours), "mapped_module_parameters": len(mapped), "unmatched": unmatched,
            "ignored_vision_audio": ignored, "shape_mismatch": mismatch, "module_params_not_covered": missing,
            "pass": not unmatched and not mismatch and not missing}


def main():
    t0 = time.time()
    torch.manual_seed(0)
    PE = S.provider_encoder()
    cfg_all = S.config()
    cfg = cfg_all["text_config"]
    PT, PH = PE.Trunk(cfg), PE.DecisionHead(cfg["hidden_size"], cfg_all["head_layers"])
    randomize(PT, PH)
    ours = G.D1Decision(cfg, cfg_all["head_layers"], L)
    load_report = G.load_state_dict_from_provider(ours, (PT, PH))
    PT.eval(), PH.eval(), ours.eval()
    out = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "L": L, "bar": BAR, "torch": torch.__version__,
           "threads": torch.get_num_threads(), "mha_fastpath_default": torch.backends.mha.get_fastpath_enabled(),
           "provider_encoder_sha256": S.sha256_file(S.SNAP / "encoder.py"),
           "d1_graph_sha256": S.sha256_file(G.__file__), "load_report": load_report,
           "weights": "manual_seed(0) provider init + norm weights 1 + 0.1 N(0,1), biases 0.02 N(0,1) (seed 1)"}
    g = torch.Generator().manual_seed(2)
    cases, inv = {}, {}
    with torch.no_grad():
        for cname, (P, n, parts) in CASES.items():
            text_ids, markers = text_row(n, parts, g)
            prefix_embeds = torch.randn(P, G.D, generator=g) if P else None
            rec = {"P": P, "n_text": n, "n_pad": L - P - n, "markers_text": markers,
                   "markers_graph": [P + m for m in markers], "qtypes": {}}
            trunk_ours = None
            for qt in (0, 1, 2):
                x = our_inputs(text_ids, prefix_embeds, qt)
                s = ours(**x)["scores"][0]
                z_ours = s[[P + m for m in markers]]
                if trunk_ours is None:
                    trunk_ours = ours.trunk(ours.embed(x["ids"], x["prefix"], x["media"]), x["media"], x["pad"],
                                            x["keep_right"])[0]
                row = {}
                for ref_name, padded in (("unpadded", False), ("padded_to_L", True)):
                    for fp in (True, False):
                        torch.backends.mha.set_fastpath_enabled(fp)
                        z_ref, h_text, h_pre = provider_logits(PT, PH, text_ids, prefix_embeds, markers, qt, padded)
                        torch.backends.mha.set_fastpath_enabled(True)
                        key = f"{ref_name}_fastpath_{'on' if fp else 'off'}"
                        row[key] = {"max_abs_dlogit": maxabs(z_ours, z_ref),
                                    "argmax_equal": int(z_ours.argmax()) == int(z_ref.argmax()),
                                    "logits_ours": z_ours.tolist(), "logits_ref": z_ref.tolist()}
                        if qt == 0 and fp:
                            rec[f"trunk_text_max_abs_d_{ref_name}"] = maxabs(trunk_ours[P:P + n], h_text)
                            if P:
                                rec[f"trunk_prefix_max_abs_d_{ref_name}"] = maxabs(trunk_ours[:P], h_pre)
                # the provider against itself: one row vs the row padded to L (batching), fast path on vs off
                zu, hu, _ = provider_logits(PT, PH, text_ids, prefix_embeds, markers, qt, False)
                zp, hpd, _ = provider_logits(PT, PH, text_ids, prefix_embeds, markers, qt, True)
                torch.backends.mha.set_fastpath_enabled(False)
                zo, _, _ = provider_logits(PT, PH, text_ids, prefix_embeds, markers, qt, False)
                torch.backends.mha.set_fastpath_enabled(True)
                row["provider_self"] = {"unpadded_vs_padded_max_abs_dlogit": maxabs(zu, zp),
                                        "unpadded_vs_padded_trunk_text_max_abs_d": maxabs(hu, hpd),
                                        "fastpath_on_vs_off_max_abs_dlogit": maxabs(zu, zo)}
                rec["qtypes"][str(qt)] = row
            rec["trunk_text_abs_max"] = float(trunk_ours[P:P + n].abs().max())
            cases[cname] = rec

            # ---- invariances on D1Decision (bit equality)
            x = our_inputs(text_ids, prefix_embeds, 0)
            valid = x["pad"][0] > 0
            base = ours(**x)["scores"][0]
            if L - P - n > 0:
                x2 = {k: v.clone() for k, v in x.items()}
                x2["ids"][0, P + n:] = torch.randint(22, 64900, (L - P - n,), generator=g, dtype=torch.int32)
                x2["prefix"][0, P + n:] = 5.0 * torch.randn(L - P - n, G.D, generator=g)
                s2 = ours(**x2)["scores"][0]
                inv.setdefault("pad_content", {})[cname] = {
                    "valid_scores_bit_equal": bool(torch.equal(base[valid], s2[valid])),
                    "max_abs_d_valid": maxabs(base[valid], s2[valid]),
                    "pad_scores_changed": bool(not torch.equal(base[~valid], s2[~valid]))}
                h0 = ours.embed(x["ids"], x["prefix"], x["media"])
                h0m = h0.clone()
                h0m[0, P + n:] *= 1e3
                a = ours.decide(h0, x["media"], x["pad"], x["keep_right"], x["qtype_onehot"])[0]
                b = ours.decide(h0m, x["media"], x["pad"], x["keep_right"], x["qtype_onehot"])[0]
                ctl_pad = torch.ones_like(x["pad"])           # control: the scaled rows counted as real tokens
                c = ours.decide(h0m, x["media"], ctl_pad, x["keep_right"], x["qtype_onehot"])[0]
                inv.setdefault("mask_h0_pad_x1e3", {})[cname] = {
                    "valid_scores_bit_equal": bool(torch.equal(a[valid], b[valid])),
                    "max_abs_d_valid": maxabs(a[valid], b[valid]),
                    "control_unmasked_valid_max_abs_d": maxabs(a[valid], c[valid])}
            if P:
                x3 = {k: v.clone() for k, v in x.items()}
                x3["ids"][0, P] = 5000                       # the first text token (bos) changed
                h_a = ours.trunk(ours.embed(x["ids"], x["prefix"], x["media"]), x["media"], x["pad"], x["keep_right"])[0]
                h_b = ours.trunk(ours.embed(x3["ids"], x3["prefix"], x3["media"]), x3["media"], x3["pad"],
                                 x3["keep_right"])[0]
                ones = torch.ones_like(x["keep_right"])        # control: the right tap not gated
                h_c = ours.trunk(ours.embed(x["ids"], x["prefix"], x["media"]), x["media"], x["pad"], ones)[0]
                h_d = ours.trunk(ours.embed(x3["ids"], x3["prefix"], x3["media"]), x3["media"], x3["pad"], ones)[0]
                inv["keep_right"] = {cname: {
                    "prefix_trunk_bit_equal": bool(torch.equal(h_a[:P], h_b[:P])),
                    "text_trunk_max_abs_d": maxabs(h_a[P:P + n], h_b[P:P + n]),
                    "text_trunk_moved": bool(maxabs(h_a[P:P + n], h_b[P:P + n]) > 0),
                    "control_ungated_prefix_max_abs_d": maxabs(h_c[:P], h_d[:P]),
                    "control_ungated_moves_prefix": bool(maxabs(h_c[:P], h_d[:P]) > 0)}}

        audit = OpAudit()
        x = our_inputs(*text_row(36, (8, 6, 3), g)[:1], torch.randn(16, G.D, generator=g), 1)
        with audit:
            ours(**x)
    banned = {"repeat_interleave", "expand", "expand_as", "repeat", "broadcast_to", "tile"}
    out["cases"] = cases
    out["invariances"] = inv
    out["source_scan"] = source_scan()
    out["op_audit"] = {"ops": sorted(audit.names), "max_rank": audit.max_rank,
                       "rank_by_op": dict(sorted(audit.rank_by_op.items())),
                       "banned_ops_seen": sorted(audit.names & banned),
                       "pass": audit.max_rank <= 4 and not (audit.names & banned)}
    out["checkpoint_header_keymap"] = header_keymap(ours)

    # ---- verdicts
    def worst(ref):
        return max(r[ref]["max_abs_dlogit"] for c in cases.values() for r in c["qtypes"].values())

    out["summary"] = {
        "max_abs_dlogit_unpadded_fastpath_on": worst("unpadded_fastpath_on"),
        "max_abs_dlogit_unpadded_fastpath_off": worst("unpadded_fastpath_off"),
        "max_abs_dlogit_padded_fastpath_on": worst("padded_to_L_fastpath_on"),
        "max_abs_dlogit_padded_fastpath_off": worst("padded_to_L_fastpath_off"),
        "argmax_all_equal": all(r[k]["argmax_equal"] for c in cases.values() for r in c["qtypes"].values() for k in r
                                if k != "provider_self"),
        "provider_self_unpadded_vs_padded_max_abs_dlogit": max(r["provider_self"]["unpadded_vs_padded_max_abs_dlogit"]
                                                               for c in cases.values() for r in c["qtypes"].values()),
        "provider_self_unpadded_vs_padded_trunk_max_abs_d": max(
            r["provider_self"]["unpadded_vs_padded_trunk_text_max_abs_d"] for c in cases.values()
            for r in c["qtypes"].values()),
        "provider_self_fastpath_on_vs_off_max_abs_dlogit": max(r["provider_self"]["fastpath_on_vs_off_max_abs_dlogit"]
                                                               for c in cases.values() for r in c["qtypes"].values()),
        "trunk_text_max_abs_d_unpadded": max(c["trunk_text_max_abs_d_unpadded"] for c in cases.values()),
        "trunk_text_max_abs_d_padded": max(c["trunk_text_max_abs_d_padded_to_L"] for c in cases.values()),
        "pad_content_bit_equal": all(v["valid_scores_bit_equal"] for v in inv["pad_content"].values()),
        "keep_right_prefix_bit_equal": all(v["prefix_trunk_bit_equal"] and v["text_trunk_moved"]
                                           and v["control_ungated_moves_prefix"] for v in inv["keep_right"].values()),
        "mask_bit_equal": all(v["valid_scores_bit_equal"] and v["control_unmasked_valid_max_abs_d"] > 0
                              for v in inv["mask_h0_pad_x1e3"].values()),
        "source_scan_pass": out["source_scan"]["pass"], "op_audit_pass": out["op_audit"]["pass"],
        "checkpoint_header_keymap_pass": out["checkpoint_header_keymap"]["pass"],
    }
    sm = out["summary"]
    sm["PASS"] = bool(max(sm["max_abs_dlogit_unpadded_fastpath_on"], sm["max_abs_dlogit_padded_fastpath_on"]) <= BAR
                      and sm["argmax_all_equal"]
                      and max(sm["trunk_text_max_abs_d_unpadded"], sm["trunk_text_max_abs_d_padded"]) <= BAR
                      and sm["pad_content_bit_equal"] and sm["keep_right_prefix_bit_equal"] and sm["mask_bit_equal"]
                      and sm["source_scan_pass"] and sm["op_audit_pass"] and sm["checkpoint_header_keymap_pass"])
    out["seconds"] = round(time.time() - t0, 1)
    (S.K / "results/eager_random_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"summary": sm, "load_report": load_report,
                      "per_case": {k: {kk: v[kk] for kk in v if kk.startswith("trunk")} for k, v in cases.items()},
                      "invariances": inv, "source_scan": out["source_scan"]["ast_call_hits"],
                      "op_audit": {k: out["op_audit"][k] for k in ("max_rank", "banned_ops_seen", "pass")},
                      "header_keymap": {k: out["checkpoint_header_keymap"][k] for k in (
                          "header_tensors", "encoder_head_tensors", "module_parameters", "mapped_module_parameters",
                          "unmatched", "ignored_vision_audio", "shape_mismatch", "pass")},
                      "seconds": out["seconds"]}, indent=1))


# ---------------------------------------------------------------- round 2: --ckpt

CKPT_BAR_DP, CKPT_BAR_DLOGIT, CKPT_BAR_F64 = 2e-5, 1e-4, 1e-12
NEAR_TIE = 0.02
BUCKETS = (128, 256, 512, 1024, 2048, 4096)


def _host():
    import sys

    sys.path.insert(0, str(S.K / "host"))
    import d1_host

    return d1_host


def _rows(ref):
    """Every oracle question as one row (ids, markers, prefix from the npz for media records)."""
    import numpy as np

    rows = []
    for e in ref["records"]:
        prefix = None
        if e["mode"] != "text":
            with np.load(S.K / "ref/npz" / f"{e['id']}.npz") as z:
                prefix = z["prefix"].copy()
            assert prefix.shape == (e["prefix"], G.D), (e["id"], prefix.shape)
        for q in e["questions"]:
            rows.append({"id": e["id"], "source": e["source"], "mode": e["mode"], "q": q, "prefix": prefix,
                         "P": e["prefix"], "n": len(q["ids"]), "positions": q["positions"], "L": q["bucket"],
                         "qd": None})
    return rows


def _question(fixtures, row, H):
    rec = fixtures[row["id"]]
    return H.Pm.as_question(rec["request"]["questions"][row["q"]["qid"]])


def _inputs(H, row, hq, L, dtype=torch.float32):
    x = H.build_inputs(row["q"]["ids"], row["prefix"], L)
    x["qtype_onehot"] = H.qtype_onehot(hq)
    t = {k: torch.from_numpy(v) for k, v in x.items()}
    if dtype != torch.float32:
        t = {k: (v if k == "ids" else v.to(dtype)) for k, v in t.items()}
    return t


def _model(cfg_all, L, sd):
    m = G.D1Decision(cfg_all["text_config"], cfg_all["head_layers"], L).eval()
    rep = G.load_state_dict_from_provider(m, sd)
    return m, rep


def ckpt_main():
    """D1Decision + the checkpoint (KEY_RULES) against the provider's fp32 CPU oracle, every row, at the row's
    smallest bucket L (right padding): scores -> host readout -> probs. Bars: argmax 100 % (near-tie = oracle top-2
    margin <= 0.02 listed apart), max |dp| <= 2e-5, max |dlogit| <= 1e-4 (against the oracle's natural call; the
    single-question call is reported too). Then: float64 identity on 50 rows (both modules in float64 at L = the row's
    own length, the provider's fp32 islands replaced: RMSNorm by forward hooks returning the float64 formula, the
    logits read from the scorer before DecisionHead's .float()), pad-content (20 rows), keep_right (5 media rows),
    bucket dependence (smallest vs next bucket), cutoff crossings at 0.5 / 0.9."""
    import numpy as np

    t0 = time.time()
    torch.set_num_threads(12)
    H = _host()
    ref_path = S.K / "ref/records_ref.json"
    ref = json.loads(ref_path.read_text())
    fixtures = {r["id"]: r for r in json.loads((S.K / "fixtures/requests.json").read_text())["records"]}
    assert ref["fixtures"]["sha256"] == S.sha256_file(S.K / "fixtures/requests.json"), "oracle / fixtures differ"
    cfg_all = S.config()
    temps = cfg_all["temperatures"]
    t_load = time.time()
    sd = G.checkpoint_state(S.WEIGHTS)
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    assert dtypes == ["torch.float32"], dtypes
    load_s = time.time() - t_load
    rows = _rows(ref)
    for r in rows:
        r["hq"] = _question(fixtures, r, H)
        assert r["hq"].options == r["q"]["K"]
    out = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__,
           "threads": torch.get_num_threads(), "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
           "d1_graph_sha256": S.sha256_file(G.__file__), "oracle": {"path": str(ref_path),
                                                                     "sha256": S.sha256_file(ref_path)},
           "weights": {"path": str(S.WEIGHTS), "sha256": ref["model"]["weights_sha256"],
                       "encoder_head_tensors": len(sd), "dtypes": dtypes, "read_seconds": round(load_s, 1)},
           "bars": {"max_abs_dp": CKPT_BAR_DP, "max_abs_dlogit": CKPT_BAR_DLOGIT, "argmax": "100 % (near-tie apart)",
                    "float64_max_abs_dlogit": CKPT_BAR_F64, "near_tie_top2": NEAR_TIE,
                    "basis": "launch r2 fact 5a: the provider's fp32 vs float64 = dp 9.3e-6 (d1d); two fp32 runs "
                             "-> dp <= 2e-5, logit <= 1e-4; same function = float64 <= 1e-12"}}

    # ---- 1. every row at its smallest bucket
    per_row, load_reports = [None] * len(rows), {}   # per_row[i] belongs to rows[i] (oracle order)
    by_L = {}
    for i, r in enumerate(rows):
        by_L.setdefault(r["L"], []).append(i)
    t1 = time.time()
    with torch.no_grad():
        for L in sorted(by_L):
            if not load_reports:   # the first bucket through load_checkpoint(): the file read, fp32 asserted
                m = G.D1Decision(cfg_all["text_config"], cfg_all["head_layers"], L).eval()
                rep = G.load_checkpoint(m, S.WEIGHTS)
            else:
                m, rep = _model(cfg_all, L, sd)
            load_reports[str(L)] = rep
            assert rep["loaded_parameters"] == 207 and not rep["unmatched"], rep
            for i in by_L[L]:
                r = rows[i]
                x = _inputs(H, r, r["hq"], L)
                s = m(**x)["scores"].numpy()
                P, mk = r["P"], r["q"]["markers"]
                z = s[0, [P + k for k in mk]].astype(np.float64)
                p = H.readout(s, P, mk, r["hq"], r["q"]["calibrate"], temps)
                q = r["q"]
                dp = max(abs(a - b) for a, b in zip(p, q["probs"]))
                dps = max(abs(a - b) for a, b in zip(p, q["probs_single"]))
                dl = float(np.abs(z - np.asarray(q["logits_raw"])).max())
                dls = float(np.abs(z - np.asarray(q["logits_raw_single"])).max())
                best = int(np.argmax(p))
                cross = {c: sum((u >= c) != (v >= c) for u, v in zip(p, q["probs"])) for c in (0.5, 0.9)}
                cross_s = {c: sum((u >= c) != (v >= c) for u, v in zip(p, q["probs_single"])) for c in (0.5, 0.9)}
                per_row[i] = ({"id": r["id"], "qid": q["qid"], "source": r["source"], "mode": r["mode"],
                                "type": q["type"], "K": q["K"], "P": P, "positions": r["positions"], "L": L,
                                "max_abs_dp": dp, "max_abs_dp_vs_single": dps, "max_abs_dlogit": dl,
                                "max_abs_dlogit_vs_single": dls, "argmax_equal": best == q["argmax_index"],
                                "near_tie": q["near_tie"], "top2_margin": q["top2_margin"],
                                "cross_0.5": cross[0.5], "cross_0.9": cross[0.9],
                                "cross_0.5_vs_single": cross_s[0.5], "cross_0.9_vs_single": cross_s[0.9],
                                "probs": [float(v) for v in p], "probs_ref": q["probs"],
                                "logits": [float(v) for v in z], "logits_ref": q["logits_raw"]})
                assert per_row[i]["id"] == rows[i]["id"] and per_row[i]["qid"] == rows[i]["q"]["qid"]
            del m
    pass1_s = time.time() - t1
    assert all(x is not None for x in per_row)
    dps = [x["max_abs_dp"] for x in per_row]
    dls = [x["max_abs_dlogit"] for x in per_row]
    miss = [x for x in per_row if not x["argmax_equal"]]
    out["load_reports"] = load_reports
    out["rows"] = {
        "n": len(per_row), "by_L": {str(L): len(v) for L, v in sorted(by_L.items())},
        "argmax_equal": sum(x["argmax_equal"] for x in per_row),
        "argmax_differ_near_tie": [f"{x['id']}/{x['qid']}" for x in miss if x["near_tie"]],
        "argmax_differ_not_near_tie": [f"{x['id']}/{x['qid']}" for x in miss if not x["near_tie"]],
        "near_tie_rows": sum(x["near_tie"] for x in per_row),
        "max_abs_dp": max(dps), "p99_abs_dp": float(np.percentile(dps, 99)), "p95_abs_dp": float(np.percentile(dps, 95)),
        "mean_abs_dp": float(np.mean(dps)), "max_abs_dlogit": max(dls), "p99_abs_dlogit": float(np.percentile(dls, 99)),
        "max_abs_dp_vs_single": max(x["max_abs_dp_vs_single"] for x in per_row),
        "max_abs_dlogit_vs_single": max(x["max_abs_dlogit_vs_single"] for x in per_row),
        "rows_over_dp_bar": [f"{x['id']}/{x['qid']} {x['max_abs_dp']:.3g}" for x in per_row if x["max_abs_dp"] > CKPT_BAR_DP],
        "rows_over_dlogit_bar": [f"{x['id']}/{x['qid']} {x['max_abs_dlogit']:.3g}" for x in per_row
                                 if x["max_abs_dlogit"] > CKPT_BAR_DLOGIT],
        "cutoff_crossings": {"0.5": sum(x["cross_0.5"] for x in per_row), "0.9": sum(x["cross_0.9"] for x in per_row),
                             "0.5_vs_single": sum(x["cross_0.5_vs_single"] for x in per_row),
                             "0.9_vs_single": sum(x["cross_0.9_vs_single"] for x in per_row),
                             "options": sum(x["K"] for x in per_row),
                             "rows": [f"{x['id']}/{x['qid']}" for x in per_row if x["cross_0.5"] or x["cross_0.9"]]},
        "by_mode": {md: {"rows": sum(1 for x in per_row if x["mode"] == md),
                         "max_abs_dp": max(x["max_abs_dp"] for x in per_row if x["mode"] == md),
                         "max_abs_dlogit": max(x["max_abs_dlogit"] for x in per_row if x["mode"] == md),
                         "argmax_equal": sum(x["argmax_equal"] for x in per_row if x["mode"] == md)}
                    for md in ("text", "image", "audio")},
        "by_L": {str(L): {"rows": len(v), "max_abs_dp": max(per_row[i]["max_abs_dp"] for i in v),
                          "max_abs_dlogit": max(per_row[i]["max_abs_dlogit"] for i in v)} for L, v in sorted(by_L.items())},
        "top_dp": sorted(({k: x[k] for k in ("id", "qid", "mode", "positions", "L", "max_abs_dp", "max_abs_dlogit")}
                          for x in per_row), key=lambda x: -x["max_abs_dp"])[:12],
        "seconds": round(pass1_s, 1),
    }
    out["batch_dependence"] = {"source": "ref/records_ref.json checks.batch_vs_single (step 2)",
                               **{k: v for k, v in ref["checks"]["batch_vs_single"].items() if k != "top"}}
    print(json.dumps({k: v for k, v in out["rows"].items() if k not in ("top_dp", "cutoff_crossings")}, indent=1),
          flush=True)

    # ---- 2. float64 identity (50 rows: text 40 + media 10)
    t2 = time.time()
    text_idx = [i for i, x in enumerate(per_row) if x["mode"] == "text"]
    by_dl = sorted(text_idx, key=lambda i: -per_row[i]["max_abs_dlogit"])
    pick = by_dl[:28]
    rest = sorted((i for i in text_idx if i not in pick), key=lambda i: per_row[i]["positions"])
    longest = [i for i in rest[-2:]]
    spread = rest[:-2][::max(1, len(rest[:-2]) // 10)][:10]
    pick += longest + spread
    pick = pick[:40]
    media_idx = [i for i, x in enumerate(per_row) if x["mode"] != "text"]
    seen, mpick = set(), []
    for i in media_idx:   # one row per media record first, then the rest
        if per_row[i]["id"] not in seen:
            seen.add(per_row[i]["id"])
            mpick.append(i)
    mpick = (mpick + [i for i in media_idx if i not in mpick])[:10]
    assert len(pick) == 40 and len(mpick) == 10 and all(rows[i]["mode"] == "text" for i in pick) \
        and all(rows[i]["mode"] != "text" for i in mpick), (len(pick), len(mpick))
    f64 = f64_identity(rows, per_row, pick + mpick, sd, cfg_all, H)
    out["float64_identity"] = {**f64, "seconds": round(time.time() - t2, 1)}
    print(json.dumps({k: v for k, v in out["float64_identity"].items() if k != "per_row"}, indent=1), flush=True)

    # ---- 3. invariances and bucket dependence
    t3 = time.time()
    out.update(invariances(rows, per_row, sd, cfg_all, H, temps))
    out["invariance_seconds"] = round(time.time() - t3, 1)
    print(json.dumps({k: out[k] for k in ("pad_content", "keep_right")}, indent=1)[:3000], flush=True)
    print(json.dumps({k: v for k, v in out["bucket_dependence"].items() if k != "per_row"}, indent=1), flush=True)

    rw = out["rows"]
    out["summary"] = {
        "parameters_loaded": 207, "argmax": f"{rw['argmax_equal']}/{rw['n']}",
        "argmax_differ_not_near_tie": rw["argmax_differ_not_near_tie"],
        "max_abs_dp": rw["max_abs_dp"], "max_abs_dlogit": rw["max_abs_dlogit"],
        "float64_max_abs_dlogit": out["float64_identity"]["max_abs_dlogit"],
        "pad_content_bit_equal": out["pad_content"]["all_bit_equal"],
        "keep_right_bit_equal": out["keep_right"]["all_bit_equal"],
        "bucket_dependence_max_abs_dp": out["bucket_dependence"]["max_abs_dp"],
        "bucket_dependence_max_abs_dlogit": out["bucket_dependence"]["max_abs_dlogit"],
        "batch_dependence_max_abs_dp": out["batch_dependence"]["max_abs_dp"],
        "cutoff_crossings_0.5": rw["cutoff_crossings"]["0.5"], "cutoff_crossings_0.9": rw["cutoff_crossings"]["0.9"],
    }
    sm = out["summary"]
    sm["PASS"] = bool(not rw["argmax_differ_not_near_tie"] and rw["max_abs_dp"] <= CKPT_BAR_DP
                      and rw["max_abs_dlogit"] <= CKPT_BAR_DLOGIT and sm["float64_max_abs_dlogit"] <= CKPT_BAR_F64
                      and sm["pad_content_bit_equal"] and sm["keep_right_bit_equal"])
    sm["PASS_strict_argmax_incl_near_tie"] = sm["PASS"] and rw["argmax_equal"] == rw["n"]
    out["per_row"] = per_row
    out["seconds"] = round(time.time() - t0, 1)
    (S.K / "results/eager_ckpt_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(sm, indent=1))


def f64_identity(rows, per_row, idx, sd, cfg_all, H):
    """Both modules in float64, L = the row's own length (no padding: the provider's RoPE table and ours are built
    for the same length); the provider's fp32 steps replaced (RMSNorm -> forward hook with the float64 formula, the
    logits read at the scorer's output, before DecisionHead's .float())."""
    import numpy as np

    F64 = torch.float64
    cfg = cfg_all["text_config"]
    PE = S.provider_encoder()
    PT, PH = PE.Trunk(cfg), PE.DecisionHead(cfg["hidden_size"], cfg_all["head_layers"])
    PT.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
    PH.load_state_dict({k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}, strict=True)
    PT.double().eval()
    PH.double().eval()
    rms_cls = type(PT.embedding_norm)
    hooks = 0

    def rms64(module, args, output):
        x = args[0]
        return module.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + module.eps))

    for mod in PT.modules():
        if isinstance(mod, rms_cls):
            mod.register_forward_hook(rms64)
            hooks += 1
    cap = {}
    PH.scorer.register_forward_hook(lambda m, a, o: cap.__setitem__("z", o.detach().clone()))
    ours, _ = _model(cfg_all, 64, sd)
    ours.double()
    theta, hd = cfg["rope_theta"], cfg["hidden_size"] // cfg["num_attention_heads"]
    res = []
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)   # the head's python path in float64 (same function as the fused one)
    with torch.no_grad():
        for i in idx:
            r, pr = rows[i], per_row[i]
            n = r["positions"]
            cos, sin = G.rope_tables(n, hd, theta)
            ours.encoder.cos, ours.encoder.sin = cos.to(F64), sin.to(F64)
            x = _inputs(H, r, r["hq"], n, F64)
            s = ours(**x)["scores"][0]
            P, mk = r["P"], r["q"]["markers"]
            z_ours = s[[P + k for k in mk]]
            ids = torch.tensor(r["q"]["ids"])
            h = PT.embed_tokens(ids)
            if P:
                h = torch.cat([torch.from_numpy(r["prefix"]).to(F64), h])
            h = h[None]
            hp = PT(h, torch.ones(1, n, dtype=torch.bool), torch.tensor([P]))
            Kq = len(mk)
            PH(hp[:, P:P + r["n"]], torch.ones(1, r["n"], dtype=torch.bool), torch.tensor([mk]),
               torch.ones(1, Kq, dtype=torch.bool), torch.tensor([H.Pm.QTYPES[r["hq"].type]]))
            z_pub = cap["z"][0, :Kq, 0]
            assert z_pub.dtype == F64 and z_ours.dtype == F64
            res.append({"id": r["id"], "qid": r["q"]["qid"], "mode": r["mode"], "positions": n, "P": P,
                        "max_abs_dlogit": float((z_ours - z_pub).abs().max()),
                        "fp32_max_abs_dlogit_vs_oracle": pr["max_abs_dlogit"],
                        "logits_f64_ours": [float(v) for v in z_ours], "logits_f64_provider": [float(v) for v in z_pub]})
    torch.backends.mha.set_fastpath_enabled(fast)
    worst = max(x["max_abs_dlogit"] for x in res)
    return {"rows": len(res), "text_rows": sum(x["mode"] == "text" for x in res),
            "media_rows": sum(x["mode"] != "text" for x in res), "rms_hooks": hooks,
            "provider_head_fastpath": False,
            "max_abs_dlogit": worst, "pass": worst <= CKPT_BAR_F64,
            "selection": "text: the 28 rows with the largest fp32 |dlogit| vs the oracle, the 2 longest, 10 spread by "
                         "length; media: one row per media record (10)", "per_row": res}


def invariances(rows, per_row, sd, cfg_all, H, temps):
    """pad-content (20 rows with padding: 15 text across buckets + 5 media), keep_right (5 media rows), bucket
    dependence (each chosen row at its smallest and the next bucket)."""
    import numpy as np

    g = torch.Generator().manual_seed(7)
    padded = [i for i, r in enumerate(rows) if r["L"] > r["positions"]]
    text_p = [i for i in padded if rows[i]["mode"] == "text"]
    media_p = [i for i in padded if rows[i]["mode"] != "text"]
    pick_pad = []
    for L in BUCKETS:   # three text rows per bucket (as available), then media rows to 20
        pick_pad += [i for i in text_p if rows[i]["L"] == L][:3]
    pick_pad = pick_pad[:15]
    seen = set()
    for i in media_p:
        if rows[i]["id"] not in seen and len(pick_pad) < 20:
            seen.add(rows[i]["id"])
            pick_pad.append(i)
    kr = []
    seen = set()
    for i in media_p:
        if rows[i]["id"] not in seen and len(kr) < 5:
            seen.add(rows[i]["id"])
            kr.append(i)
    bucket_rows = []
    for L in BUCKETS[:-1]:
        bucket_rows += [i for i, r in enumerate(rows) if r["L"] == L and r["mode"] == "text"][:4]
        bucket_rows += [i for i, r in enumerate(rows) if r["L"] == L and r["mode"] != "text"][:2]
    need = sorted({rows[i]["L"] for i in pick_pad + kr} | {rows[i]["L"] for i in bucket_rows}
                  | {BUCKETS[BUCKETS.index(rows[i]["L"]) + 1] for i in bucket_rows})
    pad_res, kr_res, bk = [], [], {}
    with torch.no_grad():
        for L in need:
            m, _ = _model(cfg_all, L, sd)
            for i in pick_pad:
                r = rows[i]
                if r["L"] != L:
                    continue
                x = _inputs(H, r, r["hq"], L)
                base = m(**x)["scores"][0]
                n0 = r["positions"]
                x2 = {k: v.clone() for k, v in x.items()}
                x2["ids"][0, n0:] = torch.randint(22, 64000, (L - n0,), generator=g, dtype=torch.int32)
                x2["prefix"][0, n0:] = 5.0 * torch.randn(L - n0, G.D, generator=g)
                s2 = m(**x2)["scores"][0]
                pad_res.append({"id": r["id"], "qid": r["q"]["qid"], "mode": r["mode"], "L": L, "positions": n0,
                                "pad_positions": L - n0, "valid_bit_equal": bool(torch.equal(base[:n0], s2[:n0])),
                                "pad_scores_changed": bool(not torch.equal(base[n0:], s2[n0:]))})
            for i in kr:
                r = rows[i]
                if r["L"] != L:
                    continue
                x = _inputs(H, r, r["hq"], L)
                P = r["P"]
                x3 = {k: v.clone() for k, v in x.items()}
                x3["ids"][0, P] = 5000                      # the first text token (bos) changed
                ha = m.trunk(m.embed(x["ids"], x["prefix"], x["media"]), x["media"], x["pad"], x["keep_right"])[0]
                hb = m.trunk(m.embed(x3["ids"], x3["prefix"], x3["media"]), x3["media"], x3["pad"], x3["keep_right"])[0]
                ones = torch.ones_like(x["keep_right"])
                hc = m.trunk(m.embed(x["ids"], x["prefix"], x["media"]), x["media"], x["pad"], ones)[0]
                hd_ = m.trunk(m.embed(x3["ids"], x3["prefix"], x3["media"]), x3["media"], x3["pad"], ones)[0]
                kr_res.append({"id": r["id"], "qid": r["q"]["qid"], "mode": r["mode"], "L": L, "P": P,
                               "prefix_trunk_bit_equal": bool(torch.equal(ha[:P], hb[:P])),
                               "text_trunk_max_abs_d": float((ha[P:r["positions"]] - hb[P:r["positions"]]).abs().max()),
                               "control_ungated_prefix_max_abs_d": float((hc[:P] - hd_[:P]).abs().max())})
            for i in bucket_rows:
                r = rows[i]
                if L not in (r["L"], BUCKETS[BUCKETS.index(r["L"]) + 1]):
                    continue
                x = _inputs(H, r, r["hq"], L)
                s = m(**x)["scores"].numpy()
                P, mk = r["P"], r["q"]["markers"]
                bk.setdefault(i, {})[L] = (s[0, [P + k for k in mk]].astype(np.float64),
                                           H.readout(s, P, mk, r["hq"], r["q"]["calibrate"], temps))
            del m
    bres = []
    for i, d in bk.items():
        r = rows[i]
        L0, L1 = r["L"], BUCKETS[BUCKETS.index(r["L"]) + 1]
        (z0, p0), (z1, p1) = d[L0], d[L1]
        bres.append({"id": r["id"], "qid": r["q"]["qid"], "mode": r["mode"], "positions": r["positions"],
                     "L": [L0, L1], "max_abs_dp": max(abs(a - b) for a, b in zip(p0, p1)),
                     "max_abs_dlogit": float(np.abs(z0 - z1).max()),
                     "bit_equal": bool(np.array_equal(z0, z1)),
                     "next_bucket_max_abs_dp_vs_oracle": max(abs(a - b) for a, b in zip(p1, r["q"]["probs"]))})
    return {
        "pad_content": {"rows": len(pad_res), "all_bit_equal": all(x["valid_bit_equal"] for x in pad_res),
                        "pad_scores_changed_all": all(x["pad_scores_changed"] for x in pad_res), "per_row": pad_res},
        "keep_right": {"rows": len(kr_res), "all_bit_equal": all(x["prefix_trunk_bit_equal"] for x in kr_res),
                       "text_moved_all": all(x["text_trunk_max_abs_d"] > 0 for x in kr_res),
                       "control_moves_prefix_all": all(x["control_ungated_prefix_max_abs_d"] > 0 for x in kr_res),
                       "per_row": kr_res},
        "bucket_dependence": {"rows": len(bres), "max_abs_dp": max(x["max_abs_dp"] for x in bres),
                              "max_abs_dlogit": max(x["max_abs_dlogit"] for x in bres),
                              "bit_equal_rows": sum(x["bit_equal"] for x in bres), "per_row": bres},
    }


if __name__ == "__main__":
    import sys

    if "--ckpt" in sys.argv[1:]:
        ckpt_main()
    else:
        main()
