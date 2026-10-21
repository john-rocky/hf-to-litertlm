"""Round 5b: the e2e run set (every reference run the host can make), the comparison of one run's answers with its
reference, and the line of this conversion (unchanged). NumPy and the standard library only (no LiteRT, no torch).

Runs (round 5a's 257 minus the 14 native runs, whose picture grid is the processor's dynamic one, not a tower graph):
  ref      ref/records_ref.json (the author's float32 systemone() on the CPU), arms text / g256 / g448      200 runs
  red      ref/records_ref_red.json, the three one-word red arms of fixtures/red_arms.json                    3 runs
  heldout  the held-out reference (ref/heldout, an earlier port's run of the same code and weights), arms text / g256 / g448  40 runs
Each run's request = the fixture's raw request (fixtures/records.json, fixtures/red_arms.json,
fixtures/heldout/records.json), its picture path in `images` for a picture arm, the tile = 256 / 448 by the arm.

Line (pre-registered): on every question that is not a near-tie of the reference (top-2 margin <= 0.02)
the most likely option is the reference's; max |dp| <= 0.02 over every option of every question; the mean over runs of
each run's mean |dp| <= 0.002 (unrounded probabilities). Red arms: each arm's probabilities against its BASE record's
reference must move some option by > 0.02 (a red arm that does not move is recorded, as the line says)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
HO = K / "ref/heldout"
HO_RECORDS = "records_ref.json"                      # clef_ref.py --fixtures fixtures/heldout --out-dir ref/heldout
TILE_OF = {"g256": 256, "g448": 448}
NEAR_TIE, MAX_DP, MEAN_DP, RED_MIN = 0.02, 0.02, 0.002, 0.02
BUCKETS = (256, 512, 1024, 2048, 4096)


def bucket(tokens: int) -> int | None:
    return next((L for L in BUCKETS if tokens <= L), None)


def load_runs() -> list[dict]:
    """Every run the host can make, in the order ref, red, heldout (each in its reference file's order)."""
    fx = {r["id"]: r for r in json.loads((K / "fixtures/records.json").read_text())["records"]}
    red = {r["id"]: r for r in json.loads((K / "fixtures/red_arms.json").read_text())["records"]}
    hof = {r["id"]: r for r in json.loads((K / "fixtures/heldout/records.json").read_text())["records"]}
    sets = (("ref", json.loads((K / "ref/records_ref.json").read_text())["rows"], fx, K / "fixtures/images"),
            ("red", json.loads((K / "ref/records_ref_red.json").read_text())["rows"], red, None),
            ("heldout", json.loads((HO / HO_RECORDS).read_text())["rows"], hof, K / "fixtures/heldout/images"))
    runs = []
    for name, rows, recs, img_dir in sets:
        for row in rows:
            if row["arm"] not in ("text", "g256", "g448"):
                continue
            rec = recs[row["id"]]
            image = img_dir / rec["image_files"][0] if row["arm"] != "text" else None
            runs.append({"set": name, "id": row["id"], "arm": row["arm"], "key": f"{name}:{row['id']}/{row['arm']}",
                         "row": row, "request": rec["request"], "image": image, "tile": TILE_OF.get(row["arm"], 448),
                         "tokens": int(row["tokens"]), "length": bucket(int(row["tokens"]))})
    return runs


def red_bases() -> dict:
    """red arm id -> (base id, the base's reference row)."""
    arms = {a["arm_id"]: a["base_id"] for a in json.loads((K / "fixtures/red_arms.json").read_text())["arms"]}
    ref = {r["id"]: r for r in json.loads((K / "ref/records_ref.json").read_text())["rows"] if r["arm"] == "text"}
    return {arm: (base, ref[base]) for arm, base in arms.items()}


def request_of(run: dict) -> dict:
    req = copy.deepcopy(run["request"])
    if run["image"] is not None:
        req["images"] = [str(run["image"])]
    return req


def select(runs: list[dict], length: int, subset: str = "all") -> list[dict]:
    """The runs of one row-graph length. subset "all" = every run; "cpu20" = the first 20 non-red runs of the length
    in load order, plus the red arms of the length."""
    mine = [r for r in runs if r["length"] == length]
    if subset == "all":
        return mine
    if subset == "cpu20":
        return [r for r in mine if r["set"] != "red"][:20] + [r for r in mine if r["set"] == "red"]
    raise ValueError(subset)


def decision(answer: dict) -> str:
    """The option a response answer picks: choice = `choice`; noul = true when p(true) > 0.5, false when < 0.5 (a 0.5
    after rounding is reported as "tie"); score = the most likely level (first on a tie, as max() of the criteria order)."""
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        p = answer["noul"]
        return "true" if p > 0.5 else "false" if p < 0.5 else "tie"
    probs = answer["probabilities"]
    return max(probs, key=probs.__getitem__)


def numeric_fields(ans: dict) -> list[tuple[str, float]]:
    out = [(k, ans[k]) for k in ("noul", "score", "confidence") if k in ans]
    out += [(f"p[{k}]", v) for k, v in (ans.get("probabilities") or {}).items()]
    return out


def compare(run: dict, probs: list, body: dict) -> dict:
    """One run's float32 probabilities (per question, the reference's option order) and response body against the
    reference row: per question argmax / near-tie / max |dp|; the run's mean |dp|; the body's structure, decisions and
    4-digit values."""
    row = run["row"]
    qs, max_dp, all_dp = [], 0.0, []
    for q, p in zip(row["questions"], probs):
        p = np.asarray(p, np.float64)
        ref = np.asarray(q["probs"], np.float64)
        d = np.abs(p - ref)
        top = int(np.argmax(p))
        qs.append({"question_id": q["question_id"], "near_tie": bool(q["near_tie"]),
                   "argmax_ref": q["argmax_id"], "argmax": q["option_ids"][top],
                   "argmax_equal": q["option_ids"][top] == q["argmax_id"], "max_abs_dp": float(d.max()),
                   "probs": [float(x) for x in p]})
        max_dp = max(max_dp, float(d.max()))
        all_dp.extend(d.tolist())
    ref_body = row["systemone_response"]
    structure = (body.get("model") == ref_body["model"] and body.get("usage") == ref_body["usage"]
                 and list(body["answers"]) == list(ref_body["answers"])
                 and all(a["type"] == body["answers"][k]["type"] and a.get("legend") == body["answers"][k].get("legend")
                         and [n for n, _ in numeric_fields(a)] == [n for n, _ in numeric_fields(body["answers"][k])]
                         and list((a.get("probabilities") or {})) == list((body["answers"][k].get("probabilities") or {}))
                         for k, a in ref_body["answers"].items()))
    near = {q["question_id"]: q["near_tie"] for q in row["questions"]}
    dec = {}
    fields = equal = 0
    for k, a in ref_body["answers"].items():
        b = body["answers"].get(k)
        if b is None:
            continue
        dec[k] = {"reference": decision(a), "host": decision(b), "near_tie": near[k]}
        fb = dict(numeric_fields(b))
        for name, x in numeric_fields(a):
            fields += 1
            equal += fb.get(name) == x
    return {"key": run["key"], "set": run["set"], "id": run["id"], "arm": run["arm"], "tokens": run["tokens"],
            "length": run["length"], "questions": qs, "max_abs_dp": max_dp,
            "mean_abs_dp": float(np.mean(all_dp)) if all_dp else 0.0,
            "argmax_equal_non_near_tie": all(q["argmax_equal"] for q in qs if not q["near_tie"]),
            "body_structure_equal": bool(structure),
            "decisions_equal_non_near_tie": all(v["reference"] == v["host"] for v in dec.values() if not v["near_tie"]),
            "decisions": dec, "digits_fields": fields, "digits_equal": equal,
            "body_equal": body == ref_body}


def red_move(run: dict, probs: list, bases: dict) -> dict | None:
    """A red arm's probabilities against its base record's reference (same questions, same option order)."""
    if run["set"] != "red":
        return None
    base_id, base = bases[run["id"]]
    moves = []
    for q, bq, p in zip(run["row"]["questions"], base["questions"], probs):
        assert q["question_id"] == bq["question_id"] and q["option_ids"] == bq["option_ids"], (run["key"], base_id)
        moves.append(float(np.abs(np.asarray(p, np.float64) - np.asarray(bq["probs"], np.float64)).max()))
    return {"base": base_id, "max_move": max(moves), "moves": moves, "moves_over_line": max(moves) > RED_MIN}


def line(rows: list[dict]) -> dict:
    """The line over the rows (compare() outputs)."""
    qs = [(r["key"], q) for r in rows for q in r["questions"]]
    non = [(k, q) for k, q in qs if not q["near_tie"]]
    flips = [f"{k}/{q['question_id']}" for k, q in non if not q["argmax_equal"]]
    worst = max(((q["max_abs_dp"], k, q["question_id"]) for k, q in qs), default=(0.0, None, None))
    mean = float(np.mean([r["mean_abs_dp"] for r in rows])) if rows else 0.0
    ok = bool(rows) and not flips and worst[0] <= MAX_DP and mean <= MEAN_DP
    return {"runs": len(rows), "questions": len(qs), "non_near_tie": len(non),
            "argmax_equal_non_near_tie": f"{len(non) - len(flips)}/{len(non)}", "flips": flips,
            "near_tie": f"{sum(q['argmax_equal'] for k, q in qs if q['near_tie'])}/{sum(q['near_tie'] for k, q in qs)}",
            "max_abs_dp": worst[0], "max_abs_dp_at": f"{worst[1]} / {worst[2]}", "mean_of_run_means": mean,
            "p95_question_max_abs_dp": float(np.percentile([q["max_abs_dp"] for k, q in qs], 95)) if qs else 0.0,
            "line": {"max_abs_dp": MAX_DP, "mean_of_run_means": MEAN_DP, "near_tie": NEAR_TIE}, "pass": ok}


def finite_or_none(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x
