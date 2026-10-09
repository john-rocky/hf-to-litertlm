"""Round 2 acceptance 2 (and the torch-side probs of acceptance 7): D1Prefill in PyTorch against the provider's code.

    $EXPORT scripts/d1_graph_check.py                          # tiny, L 64 and 256
    $EXPORT scripts/d1_graph_check.py --source <snapshot> --tag real --L 1024   (round 3)

Float32 on the CPU. One model object: the provider's references are computed first (its own Attention, SDPA), then
`to_row_form` swaps RowAttention onto the same object and the D1Prefill side runs (same weights, no second copy).
Rows are cut from fixtures/rows.json: `base` = every row's ids back to back (folded into the tiny vocabulary with
i % 255 for the tiny model; 255 = the tiny pad id), a row of n tokens = base[:n]; the pad id is 124893 for the real
model and 255 for the tiny one.
  1 rope    the constant cos/sin tables + `rotate` on [1, H, L, D] equal the provider's `hybrid.rope` on [1, L, H, D]
            (random q, bit for bit).
  2 parity  per L, rows of n = L, L-1, L/2+1, L/4+3, 1 tokens (--rows 2: n = L, L/2+1, round 6c's long buckets):
            D1Prefill(right-padded ids, valid) vs the provider's
            `LanguageModel.run(embed(ids[:n]))` (no pad) at the real positions; bar max |diff| <= --bar (1e-5, the
            tiny model's; round 6b runs the real weights at 1e-4, the expected bar: their hidden is larger,
            and the largest |hidden| of each row is recorded beside the difference).
  3 pad     the same rows with (a) the pad ids replaced by random in-vocab ids, (b) `valid` set to 1 on the pads,
            (c) D1PrefillEmbeds with the pad embeddings replaced by random vectors x 100: real positions bit-equal.
  4 embeds  D1PrefillEmbeds(embed(ids)) vs D1Prefill(ids): bit-equal at every position.
  5 tree    fixture requests card_text_001 and own_sensor_08 (a trunk + 3 branches each, from rows.json): the provider's
            `LanguageModel.answer(trunk, questions, lengths)` vs the 3 rows through `run` (no pad) and through
            D1Prefill (L = the smallest of --L that holds the rows): max |diff| of the final-position hidden (no bar).
  6 host    (tiny only) the end-to-end requests of d1_tiny_rows.E2E_RECORDS through the host's render + TinyTokenizer:
            hidden at the answer slot from the provider's run and from D1Prefill (bucket = the smallest of --L), read
            out with the random table cache/tiny/readout_tiny.safetensors (written here if missing) -> probs; the host
            test (host/test_d1_host_tiny.py) compares its LiteRT probs with probs_provider.
Outputs (never overwritten): results/{tag}_graph_check.json (default windows; with --L: results/{tag}_graph_check_L{L}
[_{L2}..].json); cache/{tag}/check_rows_L{L}.npz (the rows of check 2:
ids, valid, n, D1Prefill hidden at every position, provider hidden at the real positions) for d1_check.py.
--norm-scale <json or file> (round 5): the D1Prefill side runs with d1_prefill_graph.apply_norm_scale (applied after
the provider's references, on the same object); the record goes to the json (`norm_scale`), and d1_export.py takes a
graph check of the tag only when its k equal the export's.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_prefill_graph as G  # noqa: E402
from d1_common import K, ROWS, provider, sha256_file  # noqa: E402

BAR = 1e-5
REAL_PAD_ID = 124893
TREE_RECORDS = ("card_text_001", "own_sensor_08")


def lengths(L: int, rows: int = 5) -> list[int]:
    """Check-2 rows: 5 = n L, L-1, L/2+1, L/4+3, 1; 2 (round 6c, the long buckets) = n L and L/2+1 (no pad / half
    padded; the pair d1_clock_mac.py times)."""
    return [L, L - 1, L // 2 + 1, L // 4 + 3, 1] if rows == 5 else [L, L // 2 + 1]


def maxabs(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tiny", help="tiny, or a snapshot directory with model.safetensors")
    ap.add_argument("--tag", default="tiny")
    ap.add_argument("--L", type=int, action="append", help="window(s); default 64 and 256")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--norm-scale", default="", help="per-site k (JSON or file); default none")
    ap.add_argument("--bar", type=float, default=BAR, help=f"check 2 bar on max |diff| (default {BAR:g})")
    ap.add_argument("--rows", type=int, choices=[5, 2], default=5, help="check-2 rows per L (lengths())")
    a = ap.parse_args()
    Ls = sorted(a.L or [64, 256])
    # results/{tag}_graph_check.json for the default windows, results/{tag}_graph_check_L{L}[_{L2}..].json otherwise
    out_json = K / f"results/{a.tag}_graph_check{'' if not a.L else '_L' + '_'.join(map(str, Ls))}.json"
    cache = K / f"cache/{a.tag}"
    npz_paths = {L: cache / f"check_rows_L{L}.npz" for L in Ls}
    for p in (out_json, *npz_paths.values()):
        assert not p.exists(), f"refusing to overwrite {p}"
    cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tiny = a.source == "tiny"
    lm, info = G.load(a.source, row_form=False)
    vocab, d = info["vocab"], info["hidden"]
    pad_id = G.TINY_PAD_ID if tiny else REAL_PAD_ID
    fold = (lambda ids: [i % 255 for i in ids]) if tiny else (lambda ids: list(ids))
    rows_doc = json.loads(ROWS.read_text())
    base = [i for r in rows_doc["rows"] for i in fold(r["ids"])]
    assert len(base) >= max(Ls)
    hybrid = provider("hybrid")
    rng = np.random.default_rng(7)
    doc = {"what": "round 2 acceptance 2: D1Prefill (row form) vs the provider's LanguageModel, float32 CPU",
           "source": info, "Ls": Ls, "pad_id": pad_id, "bar_max_abs": a.bar, "threads": a.threads,
           "check2_rows_per_L": a.rows,
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "litert-torch")},
           "rows_json_sha256": sha256_file(ROWS)}

    # ---------------- references on the provider's own classes ----------------
    refs, tree_refs, host_refs = {}, {}, {}
    with torch.no_grad():
        for L in Ls:
            for n in lengths(L, a.rows):
                ids = torch.tensor([base[:n]])
                refs[(L, n)] = lm.run(lm.embed_tokens(ids))[0].numpy()
        rows_by = {}
        for r in rows_doc["rows"]:
            rows_by.setdefault(r["id"], []).append(r)
        for rid in TREE_RECORDS:
            rs = rows_by[rid]
            assert all(r["path"] == "tree" and r["split_equal"] for r in rs) and len(rs) == 3, rid
            ids = [fold(r["ids"]) for r in rs]
            p = rs[0]["state_len"]
            trunk = ids[0][:p]
            assert all(x[:p] == trunk for x in ids)
            branches = [x[p:] for x in ids]
            lens = torch.tensor([len(b) for b in branches])
            tree_h = lm.answer(lm.embed_tokens(torch.tensor([trunk])), torch.tensor([i for b in branches for i in b]),
                               lens).numpy()
            row_h = np.stack([lm.run(lm.embed_tokens(torch.tensor([x])))[0, -1].numpy() for x in ids])
            tree_refs[rid] = {"rows": ids, "trunk_len": p, "branch_lens": lens.tolist(), "tree": tree_h, "row": row_h}
        if tiny:
            import d1_tiny_rows as T

            if not T.TABLE.exists():
                doc["readout_table_written"] = T.make_table(d)
            table, tok = T.table(), T.tokenizer()
            host = T.H.D1Host(tok, pad_id=pad_id)
            for rid, req in T.requests():
                for row in host.rows(req):
                    if len(row.ids) <= max(Ls):   # rows no window holds are left out
                        h = lm.run(lm.embed_tokens(torch.tensor([row.ids])))[0, -1].numpy()
                        host_refs[(rid, row.name)] = (row, h)

    # ---------------- check 1: rope tables ----------------
    attn = next(layer.self_attn for layer in lm.layers if layer.operator_name == "self_attn")
    rope_rows = []
    for L in Ls:
        q = torch.randn(1, L, attn.heads, attn.head_dim, generator=torch.Generator().manual_seed(L))
        want = hybrid.rope(q, torch.arange(L), attn.theta, attn.head_dim // 2)
        cos, sin = G.rope_tables(L, attn.head_dim, attn.theta)
        got = hybrid.rotate(q.transpose(1, 2), cos, sin).transpose(1, 2)
        rope_rows.append({"L": L, "bit_equal": bool(torch.equal(want, got)), "max_abs": float((want - got).abs().max())})
    doc["check1_rope"] = rope_rows

    # ---------------- row form ----------------
    G.to_row_form(lm)
    doc["norm_scale"] = G.apply_norm_scale(lm, a.norm_scale or None)
    parity, pad_rows, emb_rows, tree_rows, host_rows = [], [], [], [], []
    with torch.no_grad():
        for L in Ls:
            g, ge = G.graph(lm, L), G.graph(lm, L, embeds=True)
            store = {}
            for k, n in enumerate(lengths(L, a.rows)):
                ids, valid = G.row_inputs(base[:n], L, pad_id)
                hid = g(ids, valid)["hidden"][0].numpy()
                d_real = maxabs(hid[:n], refs[(L, n)])
                parity.append({"L": L, "n": n, "pads": L - n, "max_abs_real": d_real, "pass": d_real <= a.bar,
                               "max_abs_hidden_real": float(np.abs(hid[:n]).max()),
                               "finite_all_positions": bool(np.isfinite(hid).all())})
                store.update({f"ids_{k}": ids.numpy(), f"valid_{k}": valid.numpy(), f"n_{k}": np.int64(n),
                              f"hidden_{k}": hid, f"provider_{k}": refs[(L, n)]})
                emb = lm.embed_tokens(ids)
                hid_e = ge(emb, valid)["hidden"][0].numpy()
                emb_rows.append({"L": L, "n": n, "bit_equal_all_positions": bool(np.array_equal(hid_e, hid))})
                if n < L:
                    ids_b = ids.clone()
                    ids_b[0, n:] = torch.from_numpy(rng.integers(0, vocab, L - n).astype(np.int32))
                    valid_b = valid.clone()
                    valid_b[0, n:] = 1.0
                    emb_c = emb.clone()
                    emb_c[0, n:] = torch.from_numpy((rng.standard_normal((L - n, d)) * 100).astype(np.float32))
                    ha = g(ids_b, valid)["hidden"][0, :n].numpy()
                    hb = g(ids, valid_b)["hidden"][0, :n].numpy()
                    hc = ge(emb_c, valid)["hidden"][0, :n].numpy()
                    pad_rows.append({"L": L, "n": n, "a_pad_ids_random": bool(np.array_equal(ha, hid[:n])),
                                     "b_valid_1_on_pads": bool(np.array_equal(hb, hid[:n])),
                                     "c_pad_embeds_random_x100": bool(np.array_equal(hc, hid[:n]))})
            store["L"] = np.int64(L)
            np.savez(npz_paths[L], **store)
        for rid, t in tree_refs.items():
            L = next((x for x in Ls if x >= max(len(r) for r in t["rows"])), None)   # None: no window holds the rows
            d1 = None
            if L is not None:
                g = G.graph(lm, L)
                d1 = np.stack([g(*G.row_inputs(r, L, pad_id))["hidden"][0, len(r) - 1].numpy() for r in t["rows"]])
            tree_rows.append({"record": rid, "trunk_len": t["trunk_len"], "branch_lens": t["branch_lens"], "L": L,
                              "tree_vs_provider_row": maxabs(t["tree"], t["row"]),
                              "tree_vs_d1prefill_row": None if d1 is None else maxabs(t["tree"], d1),
                              "provider_row_vs_d1prefill_row": None if d1 is None else maxabs(t["row"], d1)})
        if tiny:
            graphs = {L: G.graph(lm, L) for L in Ls}
            for (rid, qname), (row, h_ref) in host_refs.items():
                L = T.H.pick_L(len(row.ids), tuple(Ls))
                ids, valid = T.H.pad_row(row.ids, L, pad_id)
                h = graphs[L](torch.from_numpy(ids), torch.from_numpy(valid))["hidden"][0, row.answer_slot].numpy()
                p_ref, p_row = T.H.readout(h_ref, table, row.groups), T.H.readout(h, table, row.groups)
                host_rows.append({"id": rid, "qid": qname, "type": row.question.type, "row_len": len(row.ids), "L": L,
                                  "ids": row.ids, "groups": row.groups, "probs_provider": p_ref, "probs_d1prefill": p_row,
                                  "hidden_slot_max_abs": maxabs(h, h_ref),
                                  "max_abs_dp": max(abs(x - y) for x, y in zip(p_ref, p_row))})
    doc["check2_parity"] = parity
    doc["check3_pad_content"] = pad_rows
    doc["check4_embeds"] = emb_rows
    doc["check5_tree_vs_row"] = tree_rows
    if tiny:
        doc["check6_host_rows"] = {"table": str(T.TABLE.relative_to(K)), "table_sha256": sha256_file(T.TABLE),
                                   "requests": list(T.E2E_RECORDS), "questions": len(host_rows), "rows": host_rows,
                                   "max_abs_dp_d1prefill_vs_provider": max((r["max_abs_dp"] for r in host_rows),
                                                                           default=None)}
    summary = {
        "rope_bit_equal": all(r["bit_equal"] for r in rope_rows),
        "parity_max_abs_real": max(r["max_abs_real"] for r in parity),
        "parity_pass": all(r["pass"] for r in parity),
        "parity_by_L": {str(L): max(r["max_abs_real"] for r in parity if r["L"] == L) for L in Ls},
        "pad_content_bit_equal": all(r["a_pad_ids_random"] and r["b_valid_1_on_pads"] and r["c_pad_embeds_random_x100"]
                                     for r in pad_rows),
        "pad_content_rows": len(pad_rows),
        "embeds_bit_equal": all(r["bit_equal_all_positions"] for r in emb_rows),
        "tree_vs_row_max_abs": max(r["tree_vs_provider_row"] for r in tree_rows),
        "tree_vs_d1prefill_max_abs": max((r["tree_vs_d1prefill_row"] for r in tree_rows
                                          if r["tree_vs_d1prefill_row"] is not None), default=None),
        "all_finite": all(r["finite_all_positions"] for r in parity),
    }
    summary["pass"] = (summary["parity_pass"] and summary["pad_content_bit_equal"] and summary["embeds_bit_equal"]
                       and summary["all_finite"])
    doc["summary"] = summary
    doc["check_rows_npz"] = {str(L): str(p.relative_to(K)) for L, p in npz_paths.items()}
    doc["seconds"] = round(time.time() - t0, 1)
    out_json.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(json.dumps({"parity": [(r["L"], r["n"], f"{r['max_abs_real']:.3e}") for r in parity],
                      "tree": [(r["record"], r["tree_vs_provider_row"], r["tree_vs_d1prefill_row"]) for r in tree_rows]}))
    if tiny:
        print(json.dumps({"host_rows": len(host_rows), "max_abs_dp_d1prefill_vs_provider":
                          doc["check6_host_rows"]["max_abs_dp_d1prefill_vs_provider"]}))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
