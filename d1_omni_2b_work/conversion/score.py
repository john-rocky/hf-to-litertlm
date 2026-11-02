"""Score a candidate run (host + graphs on one accelerator) against the provider's reference, by the lines set before
any number ((a)-(e) below), the d1-2B conversion's lines unchanged for d1-Omni-2B (the Omni snapshot's calibration and
box code; the audio set is scored as its own rows with --ids and its own --red-arms, the same bars). The d1-2B scorer
with the snapshot path changed, --ids-file, the over-30-s clip kind (audio_long_refused) checked like long_refused,
and the reference's own refusals checked under --ids too.

Both files are {"records": {id: {"answers": {name: answer}, "probs": [per question, the input of api.answer], "seqs":
[...], ...}}} (the reference from conversion/ref_run.py, the candidate from conversion/run_gate.py). A question's type is read
off the reference's answer (the provider's own answer form): noul, choice ("choice"), multiple ("choices"), score,
text / date / number with a span (reads), number with an interval (estimate), box, boxes. The request files give the
estimate questions' ranges (their grid step).

  $REF conversion/score.py --ref out/ref/ref_fp32.json --cand out/gate/<run>.json \
      --files set/requests.json ... [--red-arms conversion/red_arms_chosen_arms.json] [--ids a,b,...] \
      --out results/gate_<run>.json

Lines (bars as registered):
 (a) noul / choice / score / multiple: argmax equal on every question that is not a near tie (choice: the picked
     name; noul: the P(yes) >= 0.5 side; score: argmax, and the median level = the API's `score`; multiple: each
     option's side of 0.5). Near tie: the reference's top-2 gap <= 0.02; multiple per option |P_ref(yes) - 0.5| <=
     0.01; the score median is not asked when some cumulative C_k, k < K-1, is within 0.02 of 0.5. Over every option
     of every question (API values, after the temperature or the `multiple` table): max |dp| <= 0.02, mean <= 0.002.
 (b) text / number / date: |dP(found)| <= 0.02 (after `found:<type>`), the found side equal where |P_ref(found) -
     0.5| > 0.02; when both found, span and value (number: and unit) equal unless the reference's best and second-best
     valid spans (length <= the type's cap) are within 0.02 in P(start) * P(end) (the reference's pointer logits).
 (c) estimate: the grid distribution max |dp| <= 0.02, mean <= 0.002; median, p10, p90 equal, or within one grid step
     when the reference's cumulative is within 0.02 of the level at either grid point straddling it.
 (d) box / boxes: |dfound| <= 0.02 (box: found; boxes: each block's presence), the side equal where |found_ref -
     0.5| > 0.01, the count equal where every block is > 0.01 from 0.5; edges of blocks found in both (paired by
     block number, before the reading-order sort) |d| <= 0.002 unless the candidate's edge lies inside the
     reference's [CDF 0.48 bin, CDF 0.52 bin] interval +- 0.002; |dconfidence| <= 0.02 (after the `box` / `boxes`
     table, per block for boxes).
 (e) red arms: each arm moves its watched value by > 0.02 against its base on the candidate, same direction as on the
     reference.
Instruments (no bar): the raw slot values' max |dp| (before the typed tables), 0.5 / 0.9 side changes, near-tie
flips, the slot hidden states (cos, max |d|) when both runs saved them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
SNAP = os.path.join(WORK, "snapshot")
BAR_P, BAR_MEAN, BAR_TIE, BAR_TIE_HALF, BAR_EDGE = 0.02, 0.002, 0.02, 0.01, 0.002
SPAN_CAPS = {"text": 512, "number": 32, "date": 32}
TOKEN_BUDGET = 65536    # runner.SlotReader's default (the reference engine's)
LONG_REFUSED_KINDS = ("long_refused", "audio_long_refused")  # answered by the reference, refused by the host's contract


def provider_pkg():
    """The provider's torch-free modules (box, calibration) as a package of the snapshot."""
    import importlib

    name = "d1p_snapshot"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [SNAP]
        sys.modules[name] = pkg
    return importlib.import_module(name + ".box"), importlib.import_module(name + ".calibration")


def kind_of(ans: dict) -> str:
    t = ans["type"]
    if t == "choice":
        return "multiple" if "choices" in ans else "choice"
    if t == "number":
        return "estimate" if "interval" in ans else "number"
    return t


def option_vec(ans: dict, kind: str) -> list[float]:
    if kind == "noul":
        return [ans["noul"], 1.0 - ans["noul"]]
    if kind == "score":
        return [ans["probabilities"][str(i)] for i in range(len(ans["probabilities"]))]
    return list(ans["probabilities"].values())


def top2_gap(p):
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def log_softmax(x):
    import numpy as np

    x = np.asarray(x, np.float64)
    fin = np.isfinite(x)
    z = np.where(fin, x - np.max(x[fin]), -np.inf)
    return z - math.log(np.exp(z[fin]).sum())


def span_top2(start_logits, end_logits, cap):
    """The best and second-best valid spans' P(start) * P(end) (start >= 1, end >= start, end - start < cap), from
    the reference's pointer logits of one read."""
    import numpy as np

    ls, le = log_softmax(start_logits), log_softmax(end_logits)
    T = len(ls)
    best = []
    for s in range(1, T):
        if not np.isfinite(ls[s]):
            continue
        seg = le[s:min(T, s + cap)]
        for o in np.argsort(-seg)[:2]:
            if np.isfinite(seg[o]):
                best.append(ls[s] + seg[o])
    best.sort(reverse=True)
    p = [math.exp(v) for v in best[:2]] + [0.0, 0.0]
    return p[0], p[1]


def batches(seqs):
    """runner.SlotReader._batches for one request's sequences (lengths only): shortest first, split at the budget."""
    order = sorted(range(len(seqs)), key=lambda j: len(seqs[j]["ids"]))
    out = [[]]
    for j in order:
        if out[-1] and len(seqs[j]["ids"]) * (len(out[-1]) + 1) > TOKEN_BUDGET:
            out.append([])
        out[-1].append(j)
    return [b for b in out if b]


def read_pairs(seqs, kinds):
    """{question index: (batch index, pair index)}: where each read's start / end logits are in the reference's
    pointer arrays (one array per batch; a read's start then its end; rows in batch order, questions in order)."""
    out = {}
    for bi, batch in enumerate(batches(seqs)):
        n = 0
        for j in batch:
            s = seqs[j]
            for k in range(len(s["ends"])):
                qi = s["first_question"] + k
                if kinds[qi] in ("text", "number", "date"):
                    out[qi] = (bi, n)
                    n += 1
    return out


def slot_rows_ref(seqs):
    """{sequence j (original order): (batch index, first row, rows)} in the reference's per-batch slot arrays."""
    out = {}
    for bi, batch in enumerate(batches(seqs)):
        row = 0
        for j in batch:
            n = sum(len(e) for e in seqs[j]["ends"])
            out[j] = (bi, row, n)
            row += n
    return out


def stat(xs):
    return {"n": len(xs), "max": max(xs) if xs else 0.0, "mean": (sum(xs) / len(xs)) if xs else 0.0}


def watched(ans, field):
    """A red arm's watched value: noul / found / confidence / count / number, `probabilities.<name>`,
    `presence.<k>`, `box.<edge>`, `interval.<0|1>`."""
    if ans is None:
        return None
    head, _, tail = field.partition(".")
    if not tail:
        v = ans.get(field)
        return None if v is None else float(v)
    v = ans.get(head)
    if v is None:
        return None
    if isinstance(v, list):
        return float(v[int(tail)])
    return None if v.get(tail) is None else float(v[tail])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--files", nargs="*", default=[], help="request files (estimate ranges)")
    ap.add_argument("--red-arms", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None, help="comma-separated record ids (default: every reference record)")
    ap.add_argument("--ids-file", default=None, help="a JSON list of record ids (one row of the set)")
    a = ap.parse_args()
    if a.ids_file:
        a.ids = ",".join(json.load(open(os.path.join(WORK, a.ids_file))))
    import numpy as np

    box_mod, cal_mod = provider_pkg()
    cal = cal_mod.Calibration.load(os.path.join(SNAP, "calibration.json"))
    ref_path, cand_path = os.path.join(WORK, a.ref), os.path.join(WORK, a.cand)
    ref = json.load(open(ref_path))["records"]
    cand = json.load(open(cand_path))["records"]
    ref_npz, cand_npz = ref_path.replace(".json", "_slots.npz"), cand_path.replace(".json", "_slots.npz")
    rarr = np.load(ref_npz) if os.path.exists(ref_npz) else None
    carr = np.load(cand_npz) if os.path.exists(cand_npz) else None
    arms = json.load(open(os.path.join(WORK, a.red_arms))) if a.red_arms else []
    qdefs = {}
    for f in a.files + ([a.red_arms] if a.red_arms else []):
        doc = json.load(open(os.path.join(WORK, f)))
        for rec in (doc if isinstance(doc, list) else doc["records"]):
            qdefs[rec["id"]] = rec["request"]["questions"]
    ids = a.ids.split(",") if a.ids else sorted(rid for rid in ref if not ref[rid].get("error"))
    # refusals: records the reference refuses (the provider's own checks) must be refused by the candidate too; records
    # of kind long_refused (longer than the largest graph) are answered by the reference and refused by the host by
    # contract: they leave the lines and are checked for the refusal alone
    kinds_by_id = {}
    for f in a.files:
        doc = json.load(open(os.path.join(WORK, f)))
        for rec in (doc if isinstance(doc, list) else doc["records"]):
            kinds_by_id[rec["id"]] = rec.get("kind")
    long_refused = sorted(rid for rid in ids if kinds_by_id.get(rid) in LONG_REFUSED_KINDS)
    ids = [rid for rid in ids if rid not in set(long_refused)]
    refusals = {"reference_refused": [], "long_refused": []}
    # with --ids the reference's own refusals among those ids are checked too (the d1-2B scorer checked them only
    # without --ids)
    for rid in (sorted(r_ for r_ in ref if ref[r_].get("error")) if not a.ids else
                sorted(r_ for r_ in ids if r_ in ref and ref[r_].get("error"))):
        c = cand.get(rid)
        refusals["reference_refused"].append({"id": rid, "candidate_refused": bool(c and c.get("error")),
                                              "same_message": bool(c and c.get("error") == ref[rid]["error"])})
    for rid in long_refused:
        c = cand.get(rid)
        refusals["long_refused"].append({"id": rid, "candidate_refused": bool(c and c.get("error")),
                                         "error": (c or {}).get("error")})
    lv = {"p10": cal.level("estimate", 0.1), "median": cal.level("estimate", 0.5), "p90": cal.level("estimate", 0.9)}

    opt = {"dp": [], "q": 0, "near_tie": 0, "flips_near_tie": 0, "argmax_fail": [], "median_fail": [],
           "median_excused": 0, "side05": 0, "side09": 0, "by_kind": {}, "raw_dp": []}
    rd = {"q": 0, "dfound": [], "side_fail": [], "side_excluded": 0, "span_checked": 0, "span_excused": 0,
          "span_fail": [], "raw_dfound": []}
    est = {"q": 0, "dp": [], "q_fail": [], "q_excused": 0}
    bx = {"q": 0, "dfound": [], "side_fail": [], "count_checked": 0, "count_fail": [], "edge_d": [], "edge_fail": [],
          "edge_excused": 0, "dconf": [], "raw_dp": [], "edge_dist_dp": []}
    hid = {"cos": [], "max_abs": [], "sequences": 0}
    errors, missing = [], []

    for rid in ids:
        r, c = ref.get(rid), cand.get(rid)
        if r is None or r.get("error"):
            continue
        if c is None:
            missing.append(rid)
            continue
        if c.get("error"):
            errors.append({"id": rid, "error": c["error"]})
            continue
        names = list(r["answers"])
        kinds = [kind_of(r["answers"][n]) for n in names]
        pairs = read_pairs(r["seqs"], kinds) if any(k in ("text", "number", "date") for k in kinds) else {}
        for i, (name, kind) in enumerate(zip(names, kinds)):
            ra, ca = r["answers"][name], c["answers"][name]
            rp, cp = r["probs"][i], c["probs"][i]
            key = f"{rid}/{name}"
            if kind in ("noul", "choice", "score", "multiple"):
                pr, pc = option_vec(ra, kind), option_vec(ca, kind)
                d = [abs(x - y) for x, y in zip(pr, pc)]
                opt["dp"].extend(d)
                bk = opt["by_kind"].setdefault(kind, {"q": 0, "dp": []})
                bk["q"] += 1
                bk["dp"].extend(d)
                opt["q"] += 1
                opt["side05"] += sum((x >= 0.5) != (y >= 0.5) for x, y in zip(pr, pc))
                opt["side09"] += sum((x >= 0.9) != (y >= 0.9) for x, y in zip(pr, pc))
                opt["raw_dp"].extend(abs(x - y) for x, y in zip(np.ravel(rp), np.ravel(cp)))
                if kind == "multiple":
                    for o, x, y in zip(ra["probabilities"], pr, pc):
                        if abs(x - 0.5) <= BAR_TIE_HALF:
                            opt["near_tie"] += 1
                            opt["flips_near_tie"] += int((x >= 0.5) != (y >= 0.5))
                        elif (x >= 0.5) != (y >= 0.5):
                            opt["argmax_fail"].append({"q": key, "option": o, "ref": x, "cand": y})
                    continue
                if top2_gap(pr) <= BAR_TIE:
                    opt["near_tie"] += 1
                    opt["flips_near_tie"] += int(np.argmax(pr) != np.argmax(pc))
                elif int(np.argmax(pr)) != int(np.argmax(pc)):
                    opt["argmax_fail"].append({"q": key, "ref": pr, "cand": pc})
                if kind == "score" and ra["score"] != ca["score"]:
                    if np.any(np.abs(np.cumsum(pr)[:-1] - 0.5) <= BAR_TIE):
                        opt["median_excused"] += 1
                    else:
                        opt["median_fail"].append({"q": key, "ref": [ra["score"], pr], "cand": [ca["score"], pc]})
            elif kind in ("text", "number", "date"):
                rd["q"] += 1
                fr, fc = ra["found"], ca["found"]
                rd["dfound"].append(abs(fr - fc))
                rd["raw_dfound"].append(abs(rp["found"] - cp["found"]))
                if abs(fr - 0.5) > BAR_TIE:
                    if (fr >= 0.5) != (fc >= 0.5):
                        rd["side_fail"].append({"q": key, "ref": fr, "cand": fc})
                else:
                    rd["side_excluded"] += 1
                if fr >= 0.5 and fc >= 0.5:
                    rd["span_checked"] += 1
                    same = (ra.get("span") == ca.get("span") and ra.get(kind) == ca.get(kind)
                            and ra.get("unit") == ca.get("unit"))
                    if not same:
                        gap = None
                        if rarr is not None and i in pairs:
                            bi, n = pairs[i]
                            pos = rarr[f"{rid}::positions{bi}"]
                            p1, p2 = span_top2(pos[2 * n], pos[2 * n + 1], SPAN_CAPS[kind])
                            gap = p1 - p2
                        if gap is not None and gap <= BAR_TIE:
                            rd["span_excused"] += 1
                        else:
                            rd["span_fail"].append({"q": key, "ref": [ra.get("span"), ra.get(kind)],
                                                    "cand": [ca.get("span"), ca.get(kind)], "ref_top2_gap": gap})
            elif kind == "estimate":
                est["q"] += 1
                pr, pc = np.asarray(rp, np.float64), np.asarray(cp, np.float64)
                est["dp"].extend(np.abs(pr - pc).tolist())
                qd = qdefs.get(rid, {}).get(name, {})
                step = (qd["range"][1] - qd["range"][0]) / (len(pr) - 1) if "range" in qd else None
                cum = np.cumsum(pr / pr.sum())
                vals = {"median": (ra["number"], ca["number"]), "p10": (ra["interval"][0], ca["interval"][0]),
                        "p90": (ra["interval"][1], ca["interval"][1])}
                for lk, (vr, vc) in vals.items():
                    if vr == vc:
                        continue
                    lvl = lv[lk]
                    j = int(np.argmax(cum >= lvl)) if np.any(cum >= lvl) else len(cum) - 1
                    near = abs(cum[j] - lvl) <= BAR_TIE or (j > 0 and abs(cum[j - 1] - lvl) <= BAR_TIE)
                    if near and step is not None and abs(vr - vc) <= step * (1 + 1e-9):
                        est["q_excused"] += 1
                    else:
                        est["q_fail"].append({"q": key, "level": lk, "ref": vr, "cand": vc, "step": step})
            elif kind in ("box", "boxes"):
                bx["q"] += 1
                blocks_r = [rp[k:k + 5] for k in range(0, len(rp), 5)] if kind == "boxes" else [rp]
                blocks_c = [cp[k:k + 5] for k in range(0, len(cp), 5)] if kind == "boxes" else [cp]
                founds = [(ra["found"], ca["found"])] if kind == "box" else list(zip(ra["presence"], ca["presence"]))
                for br, bc in zip(blocks_r, blocks_c):
                    bx["raw_dp"].append(abs(br[0][0] - bc[0][0]))
                    for er_, ec_ in zip(br[1:], bc[1:]):
                        bx["edge_dist_dp"].append(float(np.max(np.abs(np.asarray(er_) - np.asarray(ec_)))))
                for b, (fr, fc) in enumerate(founds):
                    bx["dfound"].append(abs(fr - fc))
                    if abs(fr - 0.5) > BAR_TIE_HALF and (fr >= 0.5) != (fc >= 0.5):
                        bx["side_fail"].append({"q": key, "block": b, "ref": fr, "cand": fc})
                if kind == "boxes" and all(abs(fr - 0.5) > BAR_TIE_HALF for fr, _ in founds):
                    bx["count_checked"] += 1
                    if ra["count"] != ca["count"]:
                        bx["count_fail"].append({"q": key, "ref": ra["count"], "cand": ca["count"]})
                for b, (br, bc) in enumerate(zip(blocks_r, blocks_c)):
                    fr, fc = founds[b]
                    if fr < 0.5 or fc < 0.5:
                        continue
                    er, ec = box_mod.medians(br[1:]), box_mod.medians(bc[1:])
                    gvals = np.asarray(box_mod.read_as(br[1]).values)
                    for e_i, edge in enumerate(box_mod.EDGES):
                        d = abs(er[edge] - ec[edge])
                        bx["edge_d"].append(d)
                        if d <= BAR_EDGE + 1e-9:
                            continue
                        dist = np.asarray(br[1 + e_i], np.float64)
                        cum = np.cumsum(dist / dist.sum())
                        lo_v = gvals[min(len(gvals) - 1, int(np.argmax(cum >= 0.48)))]
                        hi_v = gvals[min(len(gvals) - 1, int(np.argmax(cum >= 0.52)))]
                        if lo_v - BAR_EDGE - 1e-9 <= ec[edge] <= hi_v + BAR_EDGE + 1e-9:
                            bx["edge_excused"] += 1
                        else:
                            bx["edge_fail"].append({"q": key, "block": b, "edge": edge, "ref": er[edge],
                                                    "cand": ec[edge], "ref_bins_048_052": [lo_v, hi_v]})
                    table = "box" if kind == "box" else "boxes"
                    conf_r = cal.through(table, box_mod.confidence(br[1:], er))
                    conf_c = cal.through(table, box_mod.confidence(bc[1:], ec))
                    bx["dconf"].append(abs(conf_r - conf_c))
        # instruments: slot hidden states, when both runs saved them (candidate: per sequence, original order)
        if rarr is not None and carr is not None:
            for j, (bi, row, n) in slot_rows_ref(r["seqs"]).items():
                kr, kc = f"{rid}::slots{bi}", f"{rid}::seq{j}"
                if kr in rarr.files and kc in carr.files:
                    hr, hc = rarr[kr][row:row + n].astype(np.float64), carr[kc].astype(np.float64)
                    if hr.shape != hc.shape:
                        continue
                    cos = (hr * hc).sum(-1) / (np.linalg.norm(hr, axis=-1) * np.linalg.norm(hc, axis=-1))
                    hid["cos"].append(float(cos.min()))
                    hid["max_abs"].append(float(np.abs(hr - hc).max()))
                    hid["sequences"] += 1

    res = {"ref": a.ref, "cand": a.cand, "records": len(ids), "missing": missing, "errors": errors}
    o = {"questions": opt["q"], "near_tie": opt["near_tie"], "flips_near_tie": opt["flips_near_tie"],
         "dp": stat(opt["dp"]), "argmax_fail_n": len(opt["argmax_fail"]), "argmax_fail": opt["argmax_fail"][:30],
         "median_fail_n": len(opt["median_fail"]), "median_fail": opt["median_fail"][:20],
         "median_excused": opt["median_excused"], "side05_changes": opt["side05"], "side09_changes": opt["side09"],
         "raw_dp": stat(opt["raw_dp"]),
         "by_kind": {k: {"questions": v["q"], "dp": stat(v["dp"])} for k, v in opt["by_kind"].items()}}
    o["pass"] = (not opt["argmax_fail"] and not opt["median_fail"] and o["dp"]["max"] <= BAR_P
                 and o["dp"]["mean"] <= BAR_MEAN)
    b = {"questions": rd["q"], "dfound": stat(rd["dfound"]), "side_fail_n": len(rd["side_fail"]),
         "side_fail": rd["side_fail"][:20], "side_excluded": rd["side_excluded"], "span_checked": rd["span_checked"],
         "span_excused": rd["span_excused"], "span_fail_n": len(rd["span_fail"]), "span_fail": rd["span_fail"][:20],
         "raw_dfound": stat(rd["raw_dfound"])}
    b["pass"] = not rd["side_fail"] and not rd["span_fail"] and b["dfound"]["max"] <= BAR_P
    c_ = {"questions": est["q"], "dp": stat(est["dp"]), "q_fail_n": len(est["q_fail"]), "q_fail": est["q_fail"][:20],
          "q_excused": est["q_excused"], "levels": lv}
    c_["pass"] = not est["q_fail"] and c_["dp"]["max"] <= BAR_P and c_["dp"]["mean"] <= BAR_MEAN
    d_ = {"questions": bx["q"], "dfound": stat(bx["dfound"]), "side_fail_n": len(bx["side_fail"]),
          "side_fail": bx["side_fail"][:20], "count_checked": bx["count_checked"], "count_fail_n": len(bx["count_fail"]),
          "count_fail": bx["count_fail"][:20], "edge_d": stat(bx["edge_d"]), "edge_excused": bx["edge_excused"],
          "edge_fail_n": len(bx["edge_fail"]), "edge_fail": bx["edge_fail"][:20], "dconf": stat(bx["dconf"]),
          "raw_presence_dp": stat(bx["raw_dp"]), "edge_dist_max_dp": stat(bx["edge_dist_dp"])}
    d_["pass"] = (not bx["side_fail"] and not bx["count_fail"] and not bx["edge_fail"]
                  and d_["dfound"]["max"] <= BAR_P and d_["dconf"]["max"] <= BAR_P)
    e_rows = []
    for arm in arms:
        aid, base, w = arm["id"], arm["base_id"], arm["watch"]

        def val(doc, rid_):
            rec = doc.get(rid_)
            if not rec or rec.get("error"):
                return None
            return watched(rec["answers"].get(w["qid"]), w["field"])
        rr, cc = (val(ref, aid), val(ref, base)), (val(cand, aid), val(cand, base))
        row = {"id": aid, "base": base, "watch": w, "ref_move": None if None in rr else rr[0] - rr[1],
               "cand_move": None if None in cc else cc[0] - cc[1]}
        row["pass"] = (row["cand_move"] is not None and abs(row["cand_move"]) > BAR_P
                       and (row["ref_move"] is None or row["ref_move"] * row["cand_move"] > 0))
        e_rows.append(row)
    res.update(a_options=o, b_reads=b, c_estimate=c_, d_boxes=d_,
               e_red_arms={"arms": e_rows, "pass": all(x["pass"] for x in e_rows) if e_rows else None},
               hidden={"sequences": hid["sequences"], "cos_min": min(hid["cos"]) if hid["cos"] else None,
                       "max_abs": max(hid["max_abs"]) if hid["max_abs"] else None})
    res["refusals"] = refusals
    refusals_ok = all(x["candidate_refused"] for v in refusals.values() for x in v)
    res["pass"] = (not missing and not errors and o["pass"] and b["pass"] and c_["pass"] and d_["pass"]
                   and res["e_red_arms"]["pass"] is not False and refusals_ok)
    json.dump(res, open(os.path.join(WORK, a.out), "w"), indent=1)
    print(f"(a) {o['questions']}q max|dp| {o['dp']['max']:.3g} mean {o['dp']['mean']:.3g} argmax_fail "
          f"{o['argmax_fail_n']} median_fail {o['median_fail_n']} near_tie {o['near_tie']} -> {o['pass']} | "
          f"(b) {b['questions']}q max|dfound| {b['dfound']['max']:.3g} side_fail {b['side_fail_n']} span "
          f"{b['span_checked']} checked {b['span_excused']} excused {b['span_fail_n']} fail -> {b['pass']} | "
          f"(c) {c_['questions']}q max|dp| {c_['dp']['max']:.3g} mean {c_['dp']['mean']:.3g} q_fail "
          f"{c_['q_fail_n']} -> {c_['pass']} | (d) {d_['questions']}q max|dfound| {d_['dfound']['max']:.3g} edge "
          f"max {d_['edge_d']['max']:.3g} fail {d_['edge_fail_n']} exc {d_['edge_excused']} count_fail "
          f"{d_['count_fail_n']} max|dconf| {d_['dconf']['max']:.3g} -> {d_['pass']} | (e) "
          f"{sum(x['pass'] for x in e_rows)}/{len(e_rows)} | hidden cos_min {res['hidden']['cos_min']} | missing "
          f"{len(missing)} errors {len(errors)} | refusals ok {refusals_ok} "
          f"({sum(len(v) for v in refusals.values())}) | PASS={res['pass']}")


if __name__ == "__main__":
    main()
