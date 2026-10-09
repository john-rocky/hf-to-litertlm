"""Round 8 acceptance 3-4: the host's picture path end to end on the real weights, against the provider's plain pass.

    $EXPORT host/test_d1_vision_real.py --files ship|fp32|mixed --accel cpu
        [--requests one_tile,split_thumbnail,two_pictures] [--threads 8]
    $EXPORT \
        host/test_d1_vision_real.py --files ship --accel gpu [--precision default]
    $EXPORT host/test_d1_vision_real.py --accel torch

The round-3 tiny test (host/test_d1_vision_tiny.py) on the real weights and files. Reference:
results/real_vision_e2e_ref.json + cache/real/image/real_e2e_ref.npz (scripts/d1_vision_ref.py --tag real: the
provider's D1Model / SystemOne `_request`, float32 CPU; three requests of one question: one_tile = card_cats_001/cats +
the card's COCO photo in one tile, split_thumbnail = tv4x_qnli_07 + a synthetic picture in 2 x 3 tiles + thumbnail,
two_pictures = tv4s_00 + two synthetic pictures of one tile each).
The host = host/d1_litert.py `D1Host` with its picture path (item 9), no PyTorch on the LiteRT path: the request as a
client sends it (`images` = the picture files) -> `rows` (render, cap_pixels, preprocess, `row_ids`) -> `picture_tokens`
(d1_vision.VisionPath: tower graph -> unshuffle -> projector graph per tile) -> `embeddings` (the float32 rows of
cache/real/tables/embed_table.safetensors, the picture tokens at the `<image>` positions) -> `hidden_at_slot` (right
padding with the pad id's row to the smallest embeds graph present that holds the row, `pick_L`) -> `readout`
(cache/real/tables/readout_table.safetensors); then `decide` on the same request, whose probabilities must equal the
stepwise ones bit for bit. The text graphs are loaded one at a time (a 4.9 GB graph per bucket).
--files: fp32 = exports/real_vision_tower_fp32 + real_projector_fp32 + real_rowprefill_embeds_L{L}_fp32 (the path
check: max |dp| <= 1e-4 is the stop line, 1e-5 class expected); ship = real_vision_tower_v2_fp16fc +
real_projector_v2_fp16fc + real_rowprefill_embeds_L{L}_v2e_fp16fc (the shipping forms: the FACTS section 7 bar on the CPU
and Metal float32); mixed = the float32 picture graphs + v2e (the ship error's text part alone; recorded).
--accel torch: the same host steps with the torch export forms in place of the files (d1_vision_graph VisionTower /
Projector, d1_prefill_graph D1PrefillEmbeds from the snapshot, float32 CPU) = path error vs conversion error. gpu =
Metal (inside the measurement window); --precision default is recorded without a bar.
Checks per request: text, ids, pixel_values (float32 bits of the provider's cut), pixel_attention_mask,
spatial_shapes, the bucket, the insertion order (the provider's own picture tokens through `insert_embeddings` with the
host's table rows = the provider's input embeddings, bit for bit), the text rows (bits), the tower's real patches and
the picture tokens vs the provider's (max |d|, relative = max |d| / max |provider| per tile), the embeddings, the
answer-slot hidden (max |d|, relative), every real position's hidden, the probabilities (|dp|, argmax, the provider's
near tie = top-2 gap <= 0.02), the host read-out on the provider's hidden, decide() vs the stepwise probabilities.
Output (never overwritten): results/real_vision_e2e_{files}_{accel}[_default][_{request} when one request].json and
cache/real/image/e2e_<same stem>.npz (picture tokens, every real position's hidden).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(K / "scripts"))
import d1_litert as H  # noqa: E402
import d1_vision as V  # noqa: E402

SNAP = Path.home() / ".cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/da1fe36a861f24690f27f622dca1d8688503d113"
REF = K / "results/real_vision_e2e_ref.json"
TABLES = K / "cache/real/tables"
PATH_BAR = 1e-4                          # the stop line for the float32 files and the torch forms
BAR_MAX_DP, BAR_MEAN_DP, NEAR_TIE = 0.02, 0.002, 0.02
TOWER = {"fp32": "exports/real_vision_tower_fp32.tflite", "v2": "exports/real_vision_tower_v2_fp16fc.tflite"}
PROJECTOR = {"fp32": "exports/real_projector_fp32.tflite", "v2": "exports/real_projector_v2_fp16fc.tflite"}
TEXT = {"fp32": "exports/real_rowprefill_embeds_L{L}_fp32.tflite", "v2e": "exports/real_rowprefill_embeds_L{L}_v2e_fp16fc.tflite"}
FILES = {"fp32": ("fp32", "fp32"), "ship": ("v2", "v2e"), "mixed": ("fp32", "v2e")}   # (picture graphs, text graph)


class OneAtATime:
    """{L: key} -> the graph for L made on first use; one graph alive at a time (D1Host reads the keys for pick_L)."""

    def __init__(self, keys: dict, make):
        self.keys_, self.make, self.L, self.g, self.info = dict(keys), make, None, None, {}

    def __iter__(self):
        return iter(self.keys_)

    def __len__(self):
        return len(self.keys_)

    def __getitem__(self, L):
        if L != self.L:
            self.close()
            t = time.perf_counter()
            self.g, self.L = self.make(self.keys_[L]), L
            self.info[L] = {"graph": rel(self.keys_[L]) if isinstance(self.keys_[L], Path) else str(self.keys_[L]), "compile_s": round(time.perf_counter() - t, 3),
                            "fully_accelerated": getattr(self.g.inner, "fully_accelerated", None)}
        return self.g

    def close(self):
        if self.g is not None and hasattr(self.g.inner, "close"):
            self.g.inner.close()
        self.g, self.L = None, None


class Last:
    """A graph callable that keeps its last output (the host never sees the difference)."""

    def __init__(self, inner):
        self.inner, self.out = inner, []

    def __call__(self, *args, **kw):
        y = self.inner(*args, **kw)
        self.out.append(np.array(y[0]))
        return y


def torch_forms(threads: int):
    import torch

    import d1_prefill_graph as G
    import d1_vision_graph as VG

    torch.set_num_threads(threads)
    tower, proj, vinfo = VG.graphs(str(SNAP))
    lm, info = G.load(str(SNAP))

    def vis(m, out):
        def call(**kw):
            with torch.no_grad():
                return m(**{k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in kw.items()})[out].numpy()
        return call

    def text(L):
        g = G.graph(lm, L, embeds=True)

        def call(embeds, valid):
            with torch.no_grad():
                return g(embeds=torch.from_numpy(np.ascontiguousarray(embeds)), valid=torch.from_numpy(valid))["hidden"].numpy()
        return call

    return vis(tower, "features"), vis(proj, "mm"), text, {"vision": vinfo, "text": {k: v for k, v in info.items()
                                                                                       if k != "layer_types"}}


def rel(p) -> str:
    p = Path(p)
    return str(p.relative_to(K)) if p.is_absolute() and p.is_relative_to(K) else str(p)


def probs_of(q: H.Question, ans: dict) -> list:
    if q.type == "noul":
        return [ans["noul"]]
    if q.type == "choice":
        return [ans["probabilities"][k] for k in q.criteria]
    return [ans["probabilities"][str(i)] for i in range(len(q.criteria))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", choices=sorted(FILES), default="ship")
    ap.add_argument("--accel", choices=["cpu", "gpu", "torch"], default="cpu")
    ap.add_argument("--precision", choices=["fp32", "default"], default="fp32")
    ap.add_argument("--requests", default="", help="comma list of request names (default every reference request)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    strict = a.precision == "fp32"
    assert strict or a.accel == "gpu", "--precision default is a GPU option"
    ref = json.loads(REF.read_text())
    npz = np.load(K / ref["npz"])
    reqs = [r for r in ref["requests"] if not a.requests or r["name"] in a.requests.split(",")]
    assert reqs and (not a.requests or len(reqs) == len(a.requests.split(","))), a.requests
    files = "torch" if a.accel == "torch" else a.files
    stem = f"real_vision_e2e_{files}{'' if a.accel == 'torch' else '_' + a.accel}{'' if strict else '_' + a.precision}"
    if a.requests and len(reqs) < len(ref["requests"]):
        stem += "_" + "_".join(r["name"] for r in reqs)
    out, out_npz = K / f"results/{stem}.json", K / f"cache/real/image/e2e_{stem}.npz"
    for p in (out, out_npz):
        assert not p.exists(), f"refusing to overwrite {p}"
    contract = json.loads((K / "host/contract.json").read_text())
    ids_map = V.token_ids(contract)
    tok = H.D1Tokenizer(K / "hf_small" / H.TOKENIZER_FILE)
    table = H.EmbedTable(TABLES / "embed_table.safetensors")
    rt = H.ReadoutTable.from_file(TABLES / "readout_table.safetensors")
    pos_table = V.load_position_table(TABLES / "vision_position_table.safetensors")
    t0 = time.time()
    if a.accel == "torch":
        tower, projector, text_of, tinfo = torch_forms(a.threads)
        texts = OneAtATime({L: L for L in H.BUCKETS}, lambda L: Last(text_of(L)))
        graphs_info = {"tower": "VisionTower (torch, float32)", "projector": "Projector (torch, float32)",
                       "text": "D1PrefillEmbeds (torch, float32)", "torch": tinfo, "load_seconds": round(time.time() - t0, 1)}
    else:
        pv, tv = FILES[a.files]
        tower = V.LiteRTGraph(K / TOWER[pv], a.accel, a.precision, a.threads)
        projector = V.LiteRTGraph(K / PROJECTOR[pv], a.accel, a.precision, a.threads)
        present = {L: K / TEXT[tv].format(L=L) for L in H.BUCKETS if (K / TEXT[tv].format(L=L)).is_file()}
        texts = OneAtATime(present, lambda f: Last(H.LiteRTEmbedsGraph(f, a.accel, a.precision, a.threads)))
        graphs_info = {"tower": {"file": TOWER[pv], "fully_accelerated": tower.fully_accelerated},
                       "projector": {"file": PROJECTOR[pv], "fully_accelerated": projector.fully_accelerated},
                       "text_present": {L: rel(f) for L, f in present.items()}}
    tower_rec = Last(tower)
    vision = V.VisionPath(tower_rec, projector, pos_table, ids_map)
    host = H.D1Host(tok, {}, rt, int(contract["token_ids"]["pad"]), embed_table=table, vision=vision)
    host.embeds_graphs = texts          # after construction: D1Host copies a mapping into a dict, which would load all
    rows, store = [], {}
    for r in reqs:
        name = r["name"]
        qn, qd, state = r["question"], r["request"]["question"], r["request"]["state"]
        request = {"state": state, "questions": {qn: qd}, "images": [K / p for p in r["pictures"]]}
        t_req = time.perf_counter()
        row = host.rows(request)[0]
        pics = row.pictures
        ref_ids = npz[f"ids__{name}"].tolist()
        ref_pv, ref_mask = npz[f"pixel_values__{name}"], npz[f"pixel_attention_mask__{name}"]
        cut = ref_pv.shape[1]
        tiles = [t for p in pics for t in p.tiles]
        host_pv = np.stack([t.pixels for t in tiles])[:, :cut]
        host_mask = np.stack([t.mask for t in tiles])[:, :cut]
        L_std = H.pick_L(len(row.ids))
        L_host = H.pick_L(len(row.ids), tuple(host.embeds_graphs))
        assert L_host == L_std, f"{name}: the {L_std}-token graph is not present (the host would take L{L_host})"
        ref_mm, ref_embeds, ref_hidden = npz[f"mm__{name}"], npz[f"embeds__{name}"], npz[f"hidden__{name}"]
        ref_tower = npz[f"tower_last_hidden__{name}"]
        order_bits = np.array_equal(V.insert_embeddings(ref_ids, table.rows, ref_mm, ids_map["image"]).view(np.uint32),
                                    ref_embeds.view(np.uint32))
        tower_rec.out.clear()
        t = time.perf_counter()
        tokens = host.picture_tokens(row)
        picture_s = time.perf_counter() - t
        tower_rel, tower_abs = [], []
        for k, (tile, feat) in enumerate(zip(tiles, tower_rec.out)):
            n_real = tile.grid[0] * tile.grid[1]
            d = np.abs(feat[:n_real].astype(np.float64) - ref_tower[k, :n_real])
            tower_abs.append(float(d.max()))
            tower_rel.append(float(d.max() / np.abs(ref_tower[k, :n_real]).max()))
        mm_d = np.abs(tokens.astype(np.float64) - ref_mm)
        per_tile, start = [], 0
        for tile in tiles:
            n_tok = (tile.grid[0] // 2) * (tile.grid[1] // 2)
            seg = slice(start, start + n_tok)
            per_tile.append(float(mm_d[seg].max() / np.abs(ref_mm[seg]).max()))
            start += n_tok
        emb = host.embeddings(row, tokens)
        slots = np.asarray(row.ids) == ids_map["image"]
        ref_slots = np.asarray(ref_ids) == ids_map["image"]
        texts_g_before = len(texts.info)
        t = time.perf_counter()
        h = host.hidden_at_slot(row, tokens)
        text_s = time.perf_counter() - t
        hidden = texts.g.out[-1]
        n = len(row.ids)
        probs = H.readout(h, rt, row.groups)
        probs_on_ref_hidden = H.readout(npz[f"hidden_slot__{name}"], rt, row.groups)
        decided = host.decide(request)
        decide_probs = probs_of(row.question, decided["answers"][qn])
        ref_probs = r["probs"]
        keys = H.option_keys(row.question)
        row_out = {
            "name": name, "record": r["record"], "question": qn, "pictures": r["pictures"], "tokens": n, "L": L_host,
            "tiles": len(tiles), "picture_tokens": int(tokens.shape[0]),
            "text_equal": row.text == r["text"], "ids_equal": row.ids == ref_ids,
            "pixel_values_bits_equal": bool(host_pv.shape == ref_pv.shape and np.array_equal(host_pv.view(np.uint32), ref_pv.view(np.uint32))),
            "pixel_attention_mask_equal": bool(host_mask.shape == ref_mask.shape and np.array_equal(host_mask, ref_mask)),
            "spatial_shapes_equal": [list(t.grid) for t in tiles] == npz[f"spatial_shapes__{name}"].tolist(),
            "bucket_equal_contract": L_host == L_std,
            "insertion_order_bits_equal": bool(order_bits),
            "text_rows_bits_equal": bool(np.array_equal(emb[~slots].view(np.uint32), ref_embeds[~ref_slots].view(np.uint32))),
            "tower_vs_provider_max_abs": max(tower_abs), "tower_vs_provider_rel_max": max(tower_rel),
            "tower_vs_provider_rel_per_tile": tower_rel,
            "picture_tokens_vs_provider_max_abs": float(mm_d.max()),
            "picture_tokens_vs_provider_rel": float(mm_d.max() / np.abs(ref_mm).max()),
            "picture_tokens_vs_provider_rel_per_tile": per_tile,
            "embeds_vs_provider": float(np.abs(emb.astype(np.float64) - ref_embeds).max()),
            "hidden_slot_vs_provider": float(np.abs(h.astype(np.float64) - npz[f"hidden_slot__{name}"]).max()),
            "hidden_slot_vs_provider_rel": float(np.abs(h.astype(np.float64) - npz[f"hidden_slot__{name}"]).max()
                                                 / np.abs(npz[f"hidden_slot__{name}"]).max()),
            "hidden_all_real_vs_provider": float(np.abs(hidden[:n].astype(np.float64) - ref_hidden).max()),
            "max_abs_hidden_slot": float(np.abs(h).max()),
            "keys": keys, "probs": probs, "provider_probs": ref_probs,
            "abs_dp": [abs(x - y) for x, y in zip(probs, ref_probs)],
            "max_abs_dp": max(abs(x - y) for x, y in zip(probs, ref_probs)),
            "argmax_key": keys[int(np.argmax(probs))], "provider_argmax_key": keys[int(np.argmax(ref_probs))],
            "argmax_equal": int(np.argmax(probs)) == int(np.argmax(ref_probs)),
            "provider_top2_gap": r["top2_gap"], "near_tie": r["top2_gap"] <= NEAR_TIE,
            "gold": r.get("gold"), "argmax_equals_gold": None if r.get("gold") is None else keys[int(np.argmax(probs))] == r["gold"],
            "host_readout_on_provider_hidden_dp": max(abs(x - y) for x, y in zip(probs_on_ref_hidden, ref_probs)),
            "decide": decided, "decide_bits_equal_stepwise": [float(x) for x in decide_probs] == [float(x) for x in probs[:len(decide_probs)]],
            "decide_vs_stepwise_max_abs": max(abs(float(x) - float(y)) for x, y in zip(decide_probs, probs)),
            "input_tokens_equal_provider": decided["usage"]["input_tokens"] == r["tokens_read"],
            "nonfinite": int((~np.isfinite(hidden)).sum()) + int((~np.isfinite(tokens)).sum()),
            "text_graph_loaded_here": len(texts.info) > texts_g_before,
            "seconds": {"picture_tokens": round(picture_s, 3), "text_graph": round(text_s, 3),
                        "request_total_incl_decide": round(time.perf_counter() - t_req, 3)},
            "load1": round(os.getloadavg()[0], 2)}
        rows.append(row_out)
        store[f"tokens__{name}"] = tokens
        store[f"hidden__{name}"] = hidden[:n]
        store[f"tower__{name}"] = np.stack([f for f in tower_rec.out])
        print(f"{name}: tokens {n} (L{L_host}), tiles {len(tiles)}, ids {row_out['ids_equal']}, pixel bits "
              f"{row_out['pixel_values_bits_equal']}, order bits {row_out['insertion_order_bits_equal']}, tower rel "
              f"{row_out['tower_vs_provider_rel_max']:.2e}, mm rel {row_out['picture_tokens_vs_provider_rel']:.2e}, hidden slot "
              f"{row_out['hidden_slot_vs_provider']:.2e}, |dp| {row_out['max_abs_dp']:.2e}, argmax {row_out['argmax_key']} "
              f"(provider {row_out['provider_argmax_key']}), decide bits {row_out['decide_bits_equal_stepwise']}", flush=True)
    texts.close()
    for g in (tower, projector):
        if hasattr(g, "close"):
            g.close()
    graphs_info["text"] = {L: v for L, v in texts.info.items()} if a.accel != "torch" else graphs_info["text"]
    np.savez(out_npz, **store)
    exact_keys = ("text_equal", "ids_equal", "pixel_values_bits_equal", "pixel_attention_mask_equal", "spatial_shapes_equal",
                  "bucket_equal_contract", "insertion_order_bits_equal", "text_rows_bits_equal", "input_tokens_equal_provider")
    n_opts = sum(len(r["abs_dp"]) for r in rows)
    summary = {"requests": len(rows), **{k: sum(bool(r[k]) for r in rows) for k in exact_keys},
               "tower_vs_provider_rel_max": max(r["tower_vs_provider_rel_max"] for r in rows),
               "picture_tokens_vs_provider_rel_max": max(r["picture_tokens_vs_provider_rel"] for r in rows),
               "hidden_slot_vs_provider_max": max(r["hidden_slot_vs_provider"] for r in rows),
               "hidden_all_real_vs_provider_max": max(r["hidden_all_real_vs_provider"] for r in rows),
               "max_abs_dp": max(r["max_abs_dp"] for r in rows),
               "mean_abs_dp_all_options": sum(sum(r["abs_dp"]) for r in rows) / n_opts,
               "argmax_equal": sum(r["argmax_equal"] for r in rows),
               "argmax_equal_non_near_tie": f"{sum(r['argmax_equal'] for r in rows if not r['near_tie'])}/"
                                            f"{sum(not r['near_tie'] for r in rows)}",
               "near_tie": [r["name"] for r in rows if r["near_tie"]],
               "host_readout_on_provider_hidden_dp": max(r["host_readout_on_provider_hidden_dp"] for r in rows),
               "decide_bits_equal_stepwise": sum(bool(r["decide_bits_equal_stepwise"]) for r in rows),
               "nonfinite": sum(r["nonfinite"] for r in rows)}
    exact = all(summary[k] == len(rows) for k in exact_keys) and summary["nonfinite"] == 0
    summary["exact_checks_pass"] = exact
    summary["path_bar_1e-4"] = exact and summary["max_abs_dp"] <= PATH_BAR
    summary["bar_near_tie_apart"] = exact and all(r["argmax_equal"] for r in rows if not r["near_tie"]) and \
        summary["max_abs_dp"] <= BAR_MAX_DP and summary["mean_abs_dp_all_options"] <= BAR_MEAN_DP
    kind = ("path" if files in ("fp32", "torch") else "ship" if files == "ship" else "recorded")
    if not strict or files == "mixed":
        kind = "recorded"
    summary["bar_applied"] = {"path": "exact checks + max |dp| <= 1e-4 (stop line)",
                              "ship": "exact checks + argmax (near ties apart) + max |dp| <= 0.02 + mean <= 0.002",
                              "recorded": "none (recorded)"}[kind]
    summary["pass"] = {"path": summary["path_bar_1e-4"], "ship": summary["bar_near_tie_apart"], "recorded": None}[kind]
    doc = {"what": "round 8: the host's picture path (D1Host, item 9) end to end on the real weights vs the provider's "
                   "float32 CPU plain pass", "files": files, "accel": a.accel, "precision": a.precision,
           "threads": a.threads if a.accel != "gpu" else None, "graphs": graphs_info,
           "reference": rel(REF), "reference_npz_sha256": ref["npz_sha256"],
           "tables": {"embed_table": rel(TABLES / "embed_table.safetensors"), "readout_table": rel(TABLES / "readout_table.safetensors"),
                      "position_table": rel(TABLES / "vision_position_table.safetensors")},
           "versions": {p: importlib.metadata.version(p) for p in ("ai-edge-litert", "numpy", "pillow", "tokenizers")},
           "seconds_wall": round(time.time() - t0, 1), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "npz": rel(out_npz), "summary": summary, "rows": rows}
    out.write_text(json.dumps(doc, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    print(json.dumps(summary, indent=1))
    if summary["pass"] is False:
        print(f"STOP: {kind} bar failed ({summary['bar_applied']})")
        return 1
    print("PASS" if summary["pass"] else "RECORDED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
