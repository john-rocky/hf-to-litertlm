#!/usr/bin/env python3
"""Acceptance test of the clef-flash host (host/clef_litert) against the reference, in the torch-free venv.

    VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 venv-host/bin/python host/test_host.py [--only a,b] [--out PATH]
        (from the work directory K) -> results/r5a_host_test.json; exit 1 when a check fails

Nothing here runs the author's code or a tensor framework: the expected values are files. Runs = the reference's 214
(ref/records_ref.json + ref/npz, the author's systemone() in float32 on the CPU), the 3 red arms
(ref/records_ref_red.json) and the 40 held-out runs (ref/heldout: records + npz, made by an earlier port's run of the same code on the
same weights; r2 re-ran two of them bit-equal). Requests = the fixtures' raw requests (fixtures/records.json,
fixtures/red_arms.json, fixtures/heldout/records.json) and their pictures. Torch-side values made apart in the
reference venv (scripts/r5a_torch_side.py): cache/r5a/torch_rope.npz (the author's rotary on the same planes) and
results/r5a_author_refusals.json (the author's systemone() messages on the refusal cases).

Sections (--only picks some; `torch_free` always runs last):
  erf        the head's erf (FreeBSD s_erf.c in NumPy) vs math.erf, and its float32 GELU
  inputs     ids, question / option spans, option ids, types, token_offset, the picture block, usage.input_tokens and
             the bucket of every run; the request path (ClefHost.encode with the picture) and its patches vs the
             processor's pixel_values on the fixed-tile runs
  positions  M-RoPE planes vs the rotary's input (npz rope_pos) on every run; cos / sin vs the author's rotary in ulps
  embeds     row_inputs vs the author's layer-0 input (npz hidden_states[0]: own_t01/text, img_01/g256); on every run
             the picture rows in place, valid, zero pads, shapes
  head       the reference's last_hidden -> JointHead -> logits and probabilities vs the reference's (max |dp| <= 1e-5
             and every argmax, near-ties included = the round's stop line)
  response   the response body vs the reference's systemone_response, value by value; control: the response function
             on the reference's own probabilities
  controls   every option / question span moved by one token, +1 and -1 (3 runs each), and the lexical rows zeroed
             (1 run): how far p moves from the reference (recorded; a control that stays under 0.02 is a record)
  encode_cuts  encode_record at small limits (the state cut, max_state_tokens, the schema-too-long and no-question
             errors) vs the author's encode_record on the same requests (results/r5a_author_refusals.json)
  refusals   the named refusals; the author-side messages when results/r5a_author_refusals.json exists
  decide     ClefHost.decide end to end with stand-in graphs that return the reference's hidden / picture rows
  torch_free torch / transformers never imported (sys.modules), not installed in this venv (find_spec)"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K = HERE.parent
sys.path.insert(0, str(HERE))
import clef_litert as C  # noqa: E402
from clef_litert.request import IMAGE_PAD, QUESTION_TYPES, VISION_END, VISION_START  # noqa: E402

SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
TABLES = K / "out/r5a"
HO = K / "ref/heldout"
HO_RECORDS = "records_ref.json"                      # clef_ref.py --fixtures fixtures/heldout --out-dir ref/heldout
TILE_OF = {"g256": 256, "g448": 448}
MAX_DP, NEAR_TIE, CONTROL_MIN = 1e-5, 0.02, 0.02
HEAVY = ("torch", "transformers", "torchvision")
SECTIONS = ("erf", "inputs", "positions", "embeds", "head", "response", "controls", "encode_cuts", "refusals",
            "decide")


def heavy_loaded() -> list[str]:
    return sorted(m for m in sys.modules if m.split(".")[0] in HEAVY)


assert not heavy_loaded(), heavy_loaded()           # importing the host loaded no framework


# --------------------------------------------------------------------------- runs
def load_runs() -> list[dict]:
    """Every reference run with its raw request and picture: set, id, arm, row, npz, request, image."""
    fx = {r["id"]: r for r in json.loads((K / "fixtures/records.json").read_text())["records"]}
    red = {r["id"]: r for r in json.loads((K / "fixtures/red_arms.json").read_text())["records"]}
    hof = {r["id"]: r for r in json.loads((K / "fixtures/heldout/records.json").read_text())["records"]}
    sets = (("ref", json.loads((K / "ref/records_ref.json").read_text())["rows"], fx, K / "ref/npz", K / "fixtures/images"),
            ("red", json.loads((K / "ref/records_ref_red.json").read_text())["rows"], red, K / "ref/npz", None),
            ("heldout", json.loads((HO / HO_RECORDS).read_text())["rows"], hof, HO / "npz",
             K / "fixtures/heldout/images"))
    runs = []
    for name, rows, recs, npz_dir, img_dir in sets:
        for row in rows:
            rec = recs[row["id"]]
            image = img_dir / rec["image_files"][0] if row["arm"] != "text" else None
            runs.append({"set": name, "id": row["id"], "arm": row["arm"], "row": row,
                         "npz": npz_dir / f"{row['id']}__{row['arm']}.npz", "request": rec["request"], "image": image})
    return runs


def ref_questions(row) -> tuple:
    return tuple(C.EncodedQuestion(q["question_id"], q["type"], QUESTION_TYPES[q["type"]], tuple(q["question_span"]),
                                   tuple(tuple(s) for s in q["option_spans"]), tuple(q["option_ids"]))
                 for q in row["questions"])


def grid_of(row):
    return tuple(row["merged_hw"]) if row["arm"] != "text" else None


def ulps(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    ia = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    ib = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    d = np.abs(ia - ib)
    return int(d.max()), int((d > 0).sum())


class Ctx:
    """Loaded once: tokenizer, tables, head, host, runs; the host's encodings and head outputs shared by sections."""

    def __init__(self):
        from tokenizers import Tokenizer

        t0 = time.perf_counter()
        self.tok = Tokenizer.from_file(str(SNAP / "tokenizer.json"))
        self.host = C.ClefHost.from_files(SNAP / "tokenizer.json", TABLES / "embed_tokens.safetensors",
                                          TABLES / "lm_head.safetensors", SNAP / "joint_head.safetensors",
                                          SNAP / "joint_head_config.json")
        self.load_s = time.perf_counter() - t0
        self.runs = load_runs()
        self.enc: dict = {}          # key -> EncodedRecord (host)
        self.probs: dict = {}        # key -> [float32 probs per question] (host head)

    @staticmethod
    def key(run) -> str:
        return f"{run['set']}:{run['id']}/{run['arm']}"

    def encoded(self, run) -> C.EncodedRecord:
        k = self.key(run)
        if k not in self.enc:
            self.enc[k] = C.encode_record(self.tok, run["request"], image_grid=grid_of(run["row"]))
        return self.enc[k]


# --------------------------------------------------------------------------- sections
def s_erf(cx: Ctx) -> dict:
    """The head's erf against Python's math.erf (the platform libm), and its GELU in float32."""
    from clef_litert._erf import erf, gelu

    rng = np.random.default_rng(0)
    xs = np.concatenate([np.linspace(-7.0, 7.0, 700_001), rng.standard_normal(300_000) * 3.0,
                         [0.0, -0.0, 1e-300, 2.0 ** -28, 0.84375, -0.84375, 1.25, -1.25, 1 / 0.35, 6.0, -6.0, 30.0,
                          np.inf, -np.inf]])
    mine, ref = erf(xs), np.array([math.erf(v) for v in xs])
    u = np.abs(mine.view(np.int64) - ref.view(np.int64))
    x32 = (rng.standard_normal(300_000) * 4.0).astype(np.float32)
    g = gelu(x32)
    gref = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x32.astype(np.float64)]).astype(np.float32)
    big = np.abs(gref) > 1e-6
    gu = np.abs(g.view(np.int32).astype(np.int64) - gref.view(np.int32).astype(np.int64))
    out = {"points": int(xs.size), "max_abs_diff": float(np.abs(mine - ref).max()), "max_ulps": int(u.max()),
           "exact": int((u == 0).sum()), "nan_in_nan_out": bool(np.isnan(erf(np.array([np.nan])))[0]),
           "gelu_points": int(x32.size), "gelu_max_abs_diff": float(np.abs(g.astype(np.float64) - gref).max()),
           "gelu_max_ulps_where_abs_gt_1e-6": int(gu[big].max())}
    return {"pass": out["max_abs_diff"] <= 1e-15 and out["gelu_max_ulps_where_abs_gt_1e-6"] <= 1 and out["nan_in_nan_out"],
            "summary": out}


def s_inputs(cx: Ctx) -> dict:
    rows, tile_rows = [], []
    for run in cx.runs:
        row = run["row"]
        rec = cx.encoded(run)
        rq = ref_questions(row)
        item = {"run": cx.key(run), "T": rec.tokens,
                "ids_equal": list(rec.ids) == row["ids"],
                "questions_equal": [(q.question_id, q.type, q.question_span, q.option_spans, q.option_ids)
                                    for q in rec.questions] == [(q.question_id, q.type, q.question_span, q.option_spans,
                                                                 q.option_ids) for q in rq],
                "input_tokens_equal": rec.tokens == row["tokens"] == row["systemone_response"]["usage"]["input_tokens"],
                "length": C.pick_length(rec.tokens)}
        if row["arm"] != "text":
            ids = np.asarray(rec.ids)
            p, (h, w) = rec.token_offset, rec.image_grid
            n = h * w
            item["image_block_equal"] = bool(
                p == row["token_offset"] and n == row["n_image_tokens"] == int((ids == IMAGE_PAD).sum())
                and ids[p] == VISION_START and (ids[p + 1:p + 1 + n] == IMAGE_PAD).all() and ids[p + 1 + n] == VISION_END)
        # the request path: ClefHost.encode (text runs and fixed-tile runs; the native arm has no tower graph)
        if row["arm"] in ("text", *TILE_OF):
            req = copy.deepcopy(run["request"])
            if row["arm"] in TILE_OF:
                req["images"] = [str(run["image"])]
            enc = cx.host.encode(req, tile=TILE_OF.get(row["arm"], 448))
            item["host_encode_equal"] = enc.record == rec and enc.length == item["length"]
            if row["arm"] in TILE_OF:
                pv = np.load(run["npz"])["pixel_values"]
                bit = enc.patches.shape == pv.shape and bool((enc.patches.view(np.uint32) == pv.view(np.uint32)).all())
                tile_rows.append({"run": cx.key(run), "tile": enc.tile, "patches_bit_equal": bit})
                item["patches_bit_equal"] = bit
        rows.append(item)
    keys = ("ids_equal", "questions_equal", "input_tokens_equal", "image_block_equal", "host_encode_equal",
            "patches_bit_equal")
    summary = {"runs": len(rows), "by_set": {s: sum(r["run"].startswith(s + ":") for r in rows)
                                             for s in ("ref", "red", "heldout")}}
    for k in keys:
        have = [r[k] for r in rows if k in r]
        summary[k] = f"{sum(have)}/{len(have)}"
    summary["lengths"] = {str(L): sum(r["length"] == L for r in rows) for L in C.BUCKETS}
    ok = all(all(r.get(k, True) for k in keys) for r in rows)
    return {"pass": ok, "summary": summary, "failures": [r for r in rows if not all(r.get(k, True) for k in keys)]}


def s_positions(cx: Ctx) -> dict:
    path = K / "cache/r5a/torch_rope.npz"
    tr = np.load(path) if path.exists() else None
    rows, max_u, n_el, n_cos, n_sin = [], 0, 0, 0, 0
    for run in cx.runs:
        rec = cx.encoded(run)
        z = np.load(run["npz"])
        pos = C.record_positions(rec.tokens, rec.token_offset, rec.image_grid)
        item = {"run": cx.key(run), "planes_equal": bool(np.array_equal(pos, z["rope_pos"].astype(np.int64)))}
        if tr is not None:
            L = C.pick_length(rec.tokens)
            cos, sin = C.mrope_tables(C.padded_positions(pos, L))
            k = f"{run['set']}__{run['id']}__{run['arm']}"
            uc, nc = ulps(cos, tr[k + "__cos"])
            us, ns = ulps(sin, tr[k + "__sin"])
            item.update(L=L, cos_max_ulps=uc, sin_max_ulps=us)
            max_u = max(max_u, uc, us)
            n_el += cos.size
            n_cos += nc
            n_sin += ns
        rows.append(item)
    summary = {"runs": len(rows), "planes_equal": f"{sum(r['planes_equal'] for r in rows)}/{len(rows)}"}
    if tr is not None:
        summary["cos_sin_vs_torch"] = {"elements_each": n_el, "max_ulps": max_u, "cos_differing": n_cos,
                                       "sin_differing": n_sin, "file": str(path.relative_to(K))}
    ok = all(r["planes_equal"] for r in rows) and (tr is None or max_u <= 1)
    return {"pass": ok, "torch_values_present": tr is not None, "summary": summary,
            "failures": [r for r in rows if not r["planes_equal"] or r.get("cos_max_ulps", 0) > 1
                         or r.get("sin_max_ulps", 0) > 1]}


def s_embeds(cx: Ctx) -> dict:
    layer0, rows = [], []
    for run in cx.runs:
        rec = cx.encoded(run)
        z = np.load(run["npz"])
        L = C.pick_length(rec.tokens)
        img = z["image_embeds"] if rec.image_grid is not None else None
        x = cx.host.row_inputs(C.Encoded(rec, L), img)
        T = rec.tokens
        ok = (x["embeds"].shape == (1, L, C.HIDDEN) and x["valid"].shape == (1, L) and x["cos"].shape == (1, L, 64)
              and x["sin"].shape == (1, L, 64) and all(v.dtype == np.float32 for v in x.values())
              and bool((x["valid"][0, :T] == 1).all()) and bool((x["valid"][0, T:] == 0).all())
              and bool((x["embeds"][0, T:] == 0).all()))
        if img is not None:
            p, (h, w) = rec.token_offset, rec.image_grid
            ok &= bool(np.array_equal(x["embeds"][0, p + 1:p + 1 + h * w], img))
            ok &= bool(np.array_equal(x["embeds"][0, p], cx.host.embed_tokens.rows([VISION_START])[0]))
        rows.append({"run": cx.key(run), "L": L, "ok": bool(ok)})
        if "hidden_states" in z.files:
            l0 = z["hidden_states"][0]
            layer0.append({"run": cx.key(run), "T": T,
                           "bit_equal": bool(np.array_equal(x["embeds"][0, :T].view(np.uint32), l0.view(np.uint32))),
                           "max_abs": float(np.abs(x["embeds"][0, :T].astype(np.float64) - l0).max())})
    ok = all(r["ok"] for r in rows) and len(layer0) == 2 and all(r["bit_equal"] for r in layer0)
    return {"pass": ok, "layer0_vs_author": layer0,
            "summary": {"runs_ok": f"{sum(r['ok'] for r in rows)}/{len(rows)}", "layer0_runs": len(layer0)},
            "failures": [r for r in rows if not r["ok"]]}


def s_head(cx: Ctx) -> dict:
    rows = []
    for run in cx.runs:
        row = run["row"]
        rec = cx.encoded(run)
        z = np.load(run["npz"])
        logits = cx.host.head.scores(z["last_hidden"], rec.ids, rec.questions, cx.host.lm_head)
        probs = [C.question_probs(lg) for lg in logits]
        cx.probs[cx.key(run)] = probs
        qs = []
        for q, lg, p in zip(row["questions"], logits, probs):
            dl = float(np.abs(lg.astype(np.float64) - np.asarray(q["logits"])).max())
            dp = float(np.abs(p.astype(np.float64) - np.asarray(q["probs"])).max())
            qs.append({"question_id": q["question_id"], "max_abs_dlogit": dl, "max_abs_dp": dp,
                       "argmax_equal": int(np.argmax(p)) == q["argmax_index"], "near_tie": bool(q["near_tie"]),
                       "logits_bit_equal": [float(v) for v in lg] == q["logits"]})
        rows.append({"run": cx.key(run), "T": rec.tokens, "questions": len(qs),
                     "max_abs_dlogit": max(q["max_abs_dlogit"] for q in qs), "max_abs_dp": max(q["max_abs_dp"] for q in qs),
                     "mean_abs_dp": float(np.mean(np.concatenate([np.abs(p.astype(np.float64) - np.asarray(q["probs"]))
                                                                  for q, p in zip(row["questions"], probs)]))),
                     "argmax_equal": all(q["argmax_equal"] for q in qs),
                     "near_tie_questions": [q["question_id"] for q in qs if q["near_tie"]],
                     "questions_logits_bit_equal": sum(q["logits_bit_equal"] for q in qs), "per_question": qs})
    worst = max(rows, key=lambda r: r["max_abs_dp"])
    worst_l = max(rows, key=lambda r: r["max_abs_dlogit"])
    nq = sum(r["questions"] for r in rows)
    summary = {"runs": len(rows), "questions": nq,
               "argmax_equal": f"{sum(q['argmax_equal'] for r in rows for q in r['per_question'])}/{nq}",
               "near_tie_argmax_equal": f"{sum(q['argmax_equal'] for r in rows for q in r['per_question'] if q['near_tie'])}/"
                                        f"{sum(q['near_tie'] for r in rows for q in r['per_question'])}",
               "max_abs_dp": worst["max_abs_dp"], "max_abs_dp_run": worst["run"],
               "max_abs_dlogit": worst_l["max_abs_dlogit"], "max_abs_dlogit_run": worst_l["run"],
               "mean_of_run_means_abs_dp": float(np.mean([r["mean_abs_dp"] for r in rows])),
               "questions_logits_bit_equal": f"{sum(r['questions_logits_bit_equal'] for r in rows)}/{nq}",
               "runs_over_line": [r["run"] for r in rows if r["max_abs_dp"] > MAX_DP or not r["argmax_equal"]],
               "by_set": {s: {"runs": sum(r["run"].startswith(s + ":") for r in rows),
                              "max_abs_dp": max(r["max_abs_dp"] for r in rows if r["run"].startswith(s + ":")),
                              "max_abs_dlogit": max(r["max_abs_dlogit"] for r in rows if r["run"].startswith(s + ":"))}
                          for s in ("ref", "red", "heldout")},
               "line": {"max_abs_dp": MAX_DP, "argmax": "all, near-ties included"}}
    return {"pass": not summary["runs_over_line"], "summary": summary, "rows": rows}


def numeric_fields(ans: dict) -> list[tuple[str, float]]:
    out = [(k, ans[k]) for k in ("noul", "score", "confidence") if k in ans]
    out += [(f"p[{k}]", v) for k, v in (ans.get("probabilities") or {}).items()]
    return out


def unrounded(question: dict, option_ids, probs, field: str) -> float:
    p = dict(zip(option_ids, np.asarray(probs, np.float32).tolist()))
    if field == "noul":
        return p["true"]
    if field.startswith("p["):
        return p[field[2:-1]]
    if field == "score":
        return sum(i * p[str(i)] for i in range(len(question["criteria"])))
    if question["type"] == "choice":
        return max(p[str(o)] for o in question["criteria"])
    return max(p[str(i)] for i in range(len(question["criteria"])))


def s_response(cx: Ctx) -> dict:
    if not cx.probs:
        s_head(cx)
    rows, diffs = [], []
    fields = equal = 0
    ctl_equal = ctl_text_equal = 0
    for run in cx.runs:
        row, req = run["row"], run["request"]
        rec = cx.encoded(run)
        ref = row["systemone_response"]
        mine = C.response(req, rec.questions, cx.probs[cx.key(run)], rec.tokens)
        ctl = C.response(req, rec.questions, [np.asarray(q["probs"], np.float32) for q in row["questions"]], rec.tokens)
        ctl_equal += ctl == ref
        ctl_text_equal += json.dumps(ctl, ensure_ascii=False) == json.dumps(ref, ensure_ascii=False)
        f = e = 0
        choices = []
        for (qid, a), q in zip(ref["answers"].items(), rec.questions):
            b = mine["answers"][qid]
            if a["type"] == "choice":
                choices.append(a["choice"] == b["choice"])
            if a.get("legend") != b.get("legend") or [n for n, _ in numeric_fields(a)] != [n for n, _ in numeric_fields(b)]:
                choices.append(False)
            fb = dict(numeric_fields(b))
            for name, x in numeric_fields(a):
                y = fb.get(name)
                f += 1
                if x == y:
                    e += 1
                    continue
                if y is None:
                    diffs.append({"run": cx.key(run), "question_id": qid, "field": name, "reference": x, "host": None,
                                  "abs_diff": math.inf, "boundary_between": False})
                    continue
                qd = req["questions"][qid]
                ref_p = [qq["probs"] for qq in row["questions"] if qq["question_id"] == qid][0]
                u_ref = unrounded(qd, q.option_ids, ref_p, name)
                u_mine = unrounded(qd, q.option_ids, cx.probs[cx.key(run)][list(ref["answers"]).index(qid)], name)
                boundary = (math.floor(u_ref * 1e4) + 0.5) / 1e4
                diffs.append({"run": cx.key(run), "question_id": qid, "field": name, "reference": x, "host": y,
                              "abs_diff": abs(x - y), "unrounded_reference": u_ref, "unrounded_host": u_mine,
                              "unrounded_abs_diff": abs(u_ref - u_mine), "nearest_rounding_boundary": boundary,
                              "reference_to_boundary": abs(u_ref - boundary),
                              "boundary_between": (u_ref - boundary) * (u_mine - boundary) <= 0})
        fields += f
        equal += e
        rows.append({"run": cx.key(run), "fields": f, "fields_equal": e, "choices_and_legends_equal": all(choices),
                     "body_equal": mine == ref, "model_usage_equal": mine["model"] == ref["model"]
                     and mine["usage"] == ref["usage"],
                     "json_text_equal": json.dumps(mine, ensure_ascii=False) == json.dumps(ref, ensure_ascii=False)})
    summary = {"runs": len(rows), "bodies_equal": f"{sum(r['body_equal'] for r in rows)}/{len(rows)}",
               "json_text_equal": f"{sum(r['json_text_equal'] for r in rows)}/{len(rows)}",
               "values_equal": f"{equal}/{fields}", "values_differing": fields - equal,
               "choices_and_legends_equal": f"{sum(r['choices_and_legends_equal'] for r in rows)}/{len(rows)}",
               "model_usage_equal": f"{sum(r['model_usage_equal'] for r in rows)}/{len(rows)}",
               "max_abs_diff": max((d["abs_diff"] for d in diffs), default=0.0),
               "every_difference_straddles_a_rounding_boundary": all(d["boundary_between"] for d in diffs),
               "control_reference_probs": {"bodies_equal": f"{ctl_equal}/{len(rows)}",
                                           "json_text_equal": f"{ctl_text_equal}/{len(rows)}"}}
    ok = (ctl_equal == len(rows) and all(r["choices_and_legends_equal"] and r["model_usage_equal"] for r in rows)
          and all(d["boundary_between"] and d["abs_diff"] <= 1.0001e-4 for d in diffs))
    return {"pass": ok, "summary": summary, "differences": diffs,
            "runs_not_equal": [r for r in rows if not r["body_equal"]]}


CONTROL_SHIFT = ("ref:own_t01/text", "ref:semif_a3f18f3a63d45345942b/text", "ref:img_01/g256")
CONTROL_LEXICAL = "ref:own_t01/text"


def s_controls(cx: Ctx) -> dict:
    by_key = {cx.key(r): r for r in cx.runs}
    out = {"span_shift_plus_1": [], "span_shift_minus_1": [], "lexical_zero": None}
    for shift, name in ((1, "span_shift_plus_1"), (-1, "span_shift_minus_1")):
        for k in CONTROL_SHIFT:
            run = by_key[k]
            row, rec = run["row"], cx.encoded(run)
            z = np.load(run["npz"])
            moved = tuple(C.EncodedQuestion(q.question_id, q.type, q.type_id,
                                            (q.question_span[0] + shift, q.question_span[1] + shift),
                                            tuple((a + shift, b + shift) for a, b in q.option_spans), q.option_ids)
                          for q in rec.questions)
            probs = [C.question_probs(lg)
                     for lg in cx.host.head.scores(z["last_hidden"], rec.ids, moved, cx.host.lm_head)]
            dps = {q["question_id"]: float(np.abs(p.astype(np.float64) - np.asarray(q["probs"])).max())
                   for q, p in zip(row["questions"], probs)}
            out[name].append({"run": k, "max_abs_dp": max(dps.values()), "per_question": dps,
                              "moves": max(dps.values()) > CONTROL_MIN,
                              "argmax_changed": [q["question_id"] for q, p in zip(row["questions"], probs)
                                                 if int(np.argmax(p)) != q["argmax_index"]]})
    run = by_key[CONTROL_LEXICAL]
    row, rec = run["row"], cx.encoded(run)
    z = np.load(run["npz"])
    zero = [np.zeros((len(q.option_spans), C.HIDDEN), np.float32) for q in rec.questions]
    probs = [C.question_probs(lg) for lg in cx.host.head.scores(z["last_hidden"], rec.ids, rec.questions, lexical=zero)]
    dps = {q["question_id"]: float(np.abs(p.astype(np.float64) - np.asarray(q["probs"])).max())
           for q, p in zip(row["questions"], probs)}
    out["lexical_zero"] = {"run": CONTROL_LEXICAL, "max_abs_dp": max(dps.values()), "per_question": dps,
                           "moves": max(dps.values()) > CONTROL_MIN,
                           "argmax_changed": [q["question_id"] for q, p in zip(row["questions"], probs)
                                              if int(np.argmax(p)) != q["argmax_index"]]}
    out["summary"] = {"span_shift_plus_1_moves": f"{sum(x['moves'] for x in out['span_shift_plus_1'])}/{len(CONTROL_SHIFT)}",
                      "span_shift_plus_1_max_abs_dp": [round(x["max_abs_dp"], 4) for x in out["span_shift_plus_1"]],
                      "span_shift_minus_1_moves": f"{sum(x['moves'] for x in out['span_shift_minus_1'])}/{len(CONTROL_SHIFT)}",
                      "span_shift_minus_1_max_abs_dp": [round(x["max_abs_dp"], 4) for x in out["span_shift_minus_1"]],
                      "lexical_zero_max_abs_dp": round(out["lexical_zero"]["max_abs_dp"], 4)}
    return {"pass": True, **out}                         # recorded, not a line: a control that does not move is a record


def refusal_cases() -> list[dict]:
    """(name, request, tile, expected class): the author's four ValueError checks and the encoder's two, then the
    host's own refusals. Shared with scripts/r5a_torch_side.py (the author's messages for the same requests)."""
    ok_q = {"q": {"type": "noul", "instructions": "Is it raining?"}}
    base = {"model": "clef-flash", "state": "It is raining.", "questions": ok_q}
    long_state = " ".join(["The parcel left the depot and the courier noted the time."] * 420)
    picture = str(K / "fixtures/images/shapes_224.png")
    return [
        {"name": "over_4096_tokens", "request": {**base, "state": long_state}, "expect": "RequestTooLong", "host_only": True},
        {"name": "no_questions", "request": {**base, "questions": {}}, "expect": "NoQuestions"},
        {"name": "questions_not_an_object", "request": {**base, "questions": [ok_q["q"]]}, "expect": "NoQuestions"},
        {"name": "bad_type", "request": {**base, "questions": {"q": {"type": "boolean", "instructions": "x"}}},
         "expect": "BadQuestionType"},
        {"name": "missing_type", "request": {**base, "questions": {"q": {"instructions": "x"}}}, "expect": "BadQuestionType"},
        {"name": "empty_choice_criteria", "request": {**base, "questions": {"q": {"type": "choice", "criteria": {}}}},
         "expect": "EmptyCriteria"},
        {"name": "missing_score_criteria", "request": {**base, "questions": {"q": {"type": "score"}}},
         "expect": "EmptyCriteria"},
        {"name": "missing_model", "request": {"state": "x", "questions": ok_q}, "expect": "MissingModelOrState"},
        {"name": "missing_state", "request": {"model": "clef-flash", "questions": ok_q}, "expect": "MissingModelOrState"},
        {"name": "two_pictures", "request": {**base, "images": [picture, picture]}, "expect": "UnsupportedMedia",
         "host_only": True},
        {"name": "videos", "request": {**base, "videos": ["clip.mp4"]}, "expect": "UnsupportedMedia", "host_only": True},
        {"name": "media_kwargs", "request": {**base, "images": [picture], "media_kwargs": {"max_pixels": 1 << 20}},
         "expect": "UnsupportedMedia", "host_only": True},
        {"name": "tile_300", "request": {**base, "images": [picture]}, "tile": 300, "expect": "UnsupportedMedia",
         "host_only": True},
        {"name": "url_picture", "request": {**base, "images": ["https://example.com/a.png"]}, "expect": "UnsupportedMedia",
         "host_only": True},
    ]


def encode_cases() -> list[dict]:
    """encode_record's own cuts and errors at small limits (paths the fixtures never reach): the state cut to
    max_length, max_state_tokens, both, a schema longer than max_length, no questions (encode_record called directly).
    Shared with scripts/r5a_torch_side.py (the author's encode_record on the same requests)."""
    q = {"a": {"type": "choice", "instructions": "Which colour?", "criteria": {"red": "Red", "blue": None}},
         "b": {"type": "score", "criteria": ["low", "mid", "high"]},
         "c": {"type": "noul", "instructions": {"claim": "the parcel arrived"}, "criteria": {"true": "It did."}}}
    state = {"note": " ".join(["The parcel left the depot and the courier noted the time."] * 40), "n": 3}
    base = {"model": "clef-flash", "state": state, "questions": q}
    return [
        {"name": "state_cut_max_length_400", "request": base, "kwargs": {"max_length": 400}},
        {"name": "max_state_tokens_7", "request": base, "kwargs": {"max_state_tokens": 7}},
        {"name": "max_state_tokens_and_max_length", "request": base, "kwargs": {"max_length": 350, "max_state_tokens": 300}},
        {"name": "no_cut_needed", "request": {**base, "state": "short"}, "kwargs": {"max_length": 400}},
        {"name": "schema_longer_than_max_length", "request": base, "kwargs": {"max_length": 60}},
        {"name": "no_questions_direct", "request": {**base, "questions": {}}, "kwargs": {}},
    ]


def s_encode_cuts(cx: Ctx) -> dict:
    path = K / "results/r5a_author_refusals.json"
    author = {c["name"]: c for c in json.loads(path.read_text()).get("encode_cases", [])} if path.exists() else {}
    rows = []
    for case in encode_cases():
        try:
            rec = C.encode_record(cx.tok, copy.deepcopy(case["request"]), **case["kwargs"])
            got = {"ids": list(rec.ids), "questions": [[q.question_id, list(q.question_span),
                                                         [list(s) for s in q.option_spans], list(q.option_ids)]
                                                        for q in rec.questions]}
            raised = msg = None
        except Exception as e:                                   # noqa: BLE001 - the class is the result
            got, raised, msg = None, type(e).__name__, str(e)
        a = author.get(case["name"])
        item = {"name": case["name"], "kwargs": case["kwargs"], "tokens": len(got["ids"]) if got else None,
                "raised": raised, "message": msg}
        if a is not None:
            if a.get("raised") is None:
                item["equal_to_author"] = got is not None and got["ids"] == a["ids"] and got["questions"] == a["questions"]
            else:
                item["equal_to_author"] = (a["raised"] == "ValueError" and msg == a["message"]
                                           and raised in ("SchemaTooLong", "EmptyRecord"))
        rows.append(item)
    ok = bool(author) and all(r.get("equal_to_author") for r in rows)
    return {"pass": ok, "author_present": bool(author),
            "summary": {"cases": len(rows), "equal_to_author": f"{sum(bool(r.get('equal_to_author')) for r in rows)}/"
                                                                f"{len(rows)}"}, "cases": rows}


def s_refusals(cx: Ctx) -> dict:
    author_path = K / "results/r5a_author_refusals.json"
    author = {c["name"]: c for c in json.loads(author_path.read_text())["cases"]} if author_path.exists() else {}
    rows = []
    for case in refusal_cases():
        got, msg = None, None
        try:
            cx.host.encode(copy.deepcopy(case["request"]), tile=case.get("tile", 448))
        except Exception as e:                                   # noqa: BLE001 - the class is the result
            got, msg = type(e).__name__, str(e)
        a = author.get(case["name"])
        item = {"name": case["name"], "expect": case["expect"], "raised": got, "message": msg,
                "is_value_error": got is not None and issubclass(getattr(C, got, Exception), ValueError),
                "named": got == case["expect"], "host_only": bool(case.get("host_only"))}
        if a is not None:
            item["author"] = {"raised": a["raised"], "message": a["message"]}
            if not case.get("host_only"):
                item["same_message_as_author"] = a["raised"] == "ValueError" and a["message"] == msg
            elif case["name"] == "over_4096_tokens":
                item["author_tokens"] = a.get("tokens")
        rows.append(item)
    the_four = ("over_4096_tokens", "no_questions", "bad_type", "empty_choice_criteria")
    ok = all(r["named"] and r["is_value_error"] for r in rows) and all(
        r.get("same_message_as_author", True) for r in rows)
    return {"pass": ok, "author_messages_present": bool(author),
            "summary": {"cases": len(rows), "named": f"{sum(r['named'] for r in rows)}/{len(rows)}",
                        "four_named": {n: next(r["raised"] for r in rows if r["name"] == n) for n in the_four},
                        "same_message_as_author": f"{sum(r.get('same_message_as_author', False) for r in rows)}/"
                                                  f"{sum('same_message_as_author' in r for r in rows)}"},
            "cases": rows}


DECIDE_RUNS = ("ref:own_t01/text", "ref:own_t16/text", "ref:img_04/g448", "ref:photo_02/g256", "heldout:ho_l02/text",
               "heldout:ho_i03/g448", "red:red_own_j01_large/text")


def s_decide(cx: Ctx) -> dict:
    by_key = {cx.key(r): r for r in cx.runs}
    rows = []
    for k in DECIDE_RUNS:
        run = by_key[k]
        row = run["row"]
        z = np.load(run["npz"])
        req = copy.deepcopy(run["request"])
        seen = {}
        if run["image"] is not None:
            req["images"] = [str(run["image"])]

        def tower(tile, patches, z=z, seen=seen):
            seen["tower_patches_equal"] = bool(np.array_equal(patches.view(np.uint32), z["pixel_values"].view(np.uint32)))
            return z["image_embeds"]

        def row_graph(length, x, z=z, seen=seen):
            T = z["last_hidden"].shape[0]
            seen["length"] = length
            seen["inputs"] = {n: [list(v.shape), str(v.dtype)] for n, v in x.items()}
            hidden = np.zeros((1, length, C.HIDDEN), np.float32)
            hidden[0, :T] = z["last_hidden"]
            hidden[0, T:] = 7.0                               # rows past the record must never be read
            return hidden

        tile = TILE_OF.get(run["arm"], 448)
        body = cx.host.decide(req, row_graph, tower, tile=tile)
        rows.append({"run": k, "length": seen["length"], "inputs": seen["inputs"],
                     "tower_patches_equal": seen.get("tower_patches_equal"),
                     "equals_response_section": body == C.response(run["request"], cx.encoded(run).questions,
                                                                  cx.probs[k], cx.encoded(run).tokens)
                     if cx.probs else None,
                     "equals_reference_body": body == row["systemone_response"]})
    ok = all(r["tower_patches_equal"] in (None, True) and r["equals_response_section"] in (None, True) for r in rows)
    return {"pass": ok, "rows": rows}


def torch_free() -> dict:
    return {"heavy_modules_loaded": heavy_loaded(),
            "find_spec": {m: importlib.util.find_spec(m) is not None for m in (*HEAVY, "huggingface_hub")},
            "pass": not heavy_loaded()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=",".join(SECTIONS))
    ap.add_argument("--out", default=str(K / "results/r5a_host_test.json"))
    a = ap.parse_args()
    only = [s for s in a.only.split(",") if s]
    bad = [s for s in only if s not in SECTIONS]
    assert not bad, bad
    t0 = time.perf_counter()
    cx = Ctx()
    import PIL
    import safetensors
    import tokenizers

    doc = {"what": "round 5a: the torch-free host vs the reference (author's float32 systemone() on the CPU)",
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "tokenizers": tokenizers.__version__,
                        "pillow": PIL.__version__, "safetensors": safetensors.__version__,
                        "clef_litert": C.__version__},
           "threads_env": {k: __import__("os").environ.get(k) for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
                                                                        "OPENBLAS_NUM_THREADS")},
           "host_files_sha256": {p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((HERE / "clef_litert").glob("*.py"))},
           "load_seconds": round(cx.load_s, 2), "runs": len(cx.runs), "sections": {}}
    for s in SECTIONS:
        if s not in only:
            continue
        t1 = time.perf_counter()
        r = globals()["s_" + s](cx)
        r["seconds"] = round(time.perf_counter() - t1, 2)
        doc["sections"][s] = r
        print(json.dumps({"section": s, "pass": r["pass"], "seconds": r["seconds"], **({"summary": r["summary"]}
                                                                                      if "summary" in r else {})}),
              flush=True)
    doc["sections"]["torch_free"] = torch_free()
    doc["pass"] = all(r["pass"] for r in doc["sections"].values())
    doc["seconds"] = round(time.perf_counter() - t0, 2)
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"torch_free": doc["sections"]["torch_free"], "pass": doc["pass"], "seconds": doc["seconds"]}))
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
