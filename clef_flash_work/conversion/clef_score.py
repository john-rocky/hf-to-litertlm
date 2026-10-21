#!/usr/bin/env python3
"""Round 1: hidden states from a graph (torch module or LiteRT file) -> the author's JointSchemaHead (torch fp32,
venv-ref = the reference's versions) -> option probabilities -> the line against the reference.

    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/clef_score.py --hidden-dir cache/r1/torch --tag torch_L1024
        -> results/score_<tag>.json (refuses to overwrite)

Head = joint_schema_model.py's JointSchemaHead (sha256-pinned file of the snapshot), joint_head.safetensors loaded as
load_release_model loads it (strict, then float32), lm_head.weight of the shards widened to float32 (the reference
model's fp32 output embedding); per run head(hidden [1, T, 4096], ids, ones, [EncodedRecord from the reference row's
spans], lm_head) -> per question float softmax, exactly the reference's path (clef_ref.py records head(...) on the
hooked hidden == the hooked logits). Control (always): the reference's own last_hidden of every scored run through this
scorer must give the reference's logits within the reference's own re-run tolerance (max |dlogit| <= 1e-6; the count
of bit-equal runs is recorded; torch threads default 10 = the reference's), else the scorer is not the reference's
head and the run stops.

Every `<id>__<arm>.npy` (float32 [T, 4096], the real positions) in --hidden-dir is scored: fixture runs against
ref/records_ref.json, red arms (source red_arm) against ref/records_ref_red.json (the same input) and, for the control,
against the BASE record's reference (fixtures/red_arms.json `base_id`): the arm moves when some option of the base's
questions differs by > 0.02.
Line (pre-registered, unrounded probabilities): every question whose reference top-2 margin is > 0.02 keeps its argmax; max
|dp| over all options <= 0.02; the mean over runs of each run's mean |dp| (over its options) <= 0.002. Near-tie
questions (margin <= 0.02) are reported apart. Red arms are reported apart and are not in the line's statistics."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
REV = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots" / REV
JSM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
NEAR_TIE, MAX_DP, MEAN_DP, ARM_MIN = 0.02, 0.02, 0.002, 0.02


def load_jsm():
    p = SNAP / "joint_schema_model.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == JSM_SHA256
    spec = importlib.util.spec_from_file_location("joint_schema_model", p)
    jsm = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = jsm
    spec.loader.exec_module(jsm)
    return jsm


class Scorer:
    def __init__(self, threads: int):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self.torch = torch
        self.jsm = jsm = load_jsm()
        cfg = json.loads((SNAP / "joint_head_config.json").read_text())
        head = jsm.JointSchemaHead(**cfg)
        head.load_state_dict(load_file(str(SNAP / "joint_head.safetensors")), strict=True)
        self.head = head.to(device="cpu", dtype=torch.float32).eval()
        wm = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(str(SNAP / wm["lm_head.weight"]), framework="pt") as f:
            self.lm_head = f.get_tensor("lm_head.weight").to(torch.float32)
        self.info = {"head_config": cfg, "lm_head_shape": list(self.lm_head.shape), "torch": torch.__version__,
                     "threads": torch.get_num_threads()}

    def logits(self, hidden: np.ndarray, row: dict) -> list:
        torch, jsm = self.torch, self.jsm
        ids = torch.tensor([row["ids"]], dtype=torch.long)
        qs = tuple(jsm.EncodedQuestion(question_id=q["question_id"], question_type=["noul", "choice", "score"].index(q["type"]),
                                       question_span=tuple(q["question_span"]),
                                       option_spans=tuple(tuple(s) for s in q["option_spans"]),
                                       option_ids=tuple(q["option_ids"])) for q in row["questions"])
        enc = jsm.EncodedRecord(input_ids=tuple(row["ids"]), questions=qs, record_id=row["id"])
        h = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32))[None]
        assert h.shape[1] == len(row["ids"]), (h.shape, len(row["ids"]))
        out = self.head(h, ids, torch.ones_like(ids), [enc], self.lm_head)[0]
        return [t.float() for t in out]

    def probs(self, logits: list) -> list:
        return [self.torch.softmax(t, -1).tolist() for t in logits]


def compare(row_ref: dict, probs: list) -> dict:
    """One run against its reference row: per question argmax, |dp| per option, margin."""
    qrows, all_dp = [], []
    for q, p in zip(row_ref["questions"], probs):
        pr = np.asarray(q["probs"], np.float64)
        pg = np.asarray(p, np.float64)
        dp = np.abs(pg - pr)
        all_dp += dp.tolist()
        am = int(np.argmax(pg))
        qrows.append({"question_id": q["question_id"], "type": q["type"], "n_options": len(pr),
                      "argmax_ref": q["argmax_index"], "argmax": am, "argmax_equal": am == q["argmax_index"],
                      "top2_margin_ref": q["top2_margin"], "near_tie": bool(q["top2_margin"] <= NEAR_TIE),
                      "max_abs_dp": float(dp.max()), "probs": [float(x) for x in pg]})
    return {"questions": qrows, "max_abs_dp": float(max(all_dp)), "mean_abs_dp": float(np.mean(all_dp)),
            "options": len(all_dp)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    out = K / f"results/score_{a.tag}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    hdir = K / a.hidden_dir
    ref = json.loads((K / "ref/records_ref.json").read_text())
    red = json.loads((K / "ref/records_ref_red.json").read_text())
    arms = {x["arm_id"]: x for x in json.loads((K / "fixtures/red_arms.json").read_text())["arms"]}
    rows_ref = {(r["id"], r["arm"]): r for r in ref["rows"]}
    rows_red = {(r["id"], r["arm"]): r for r in red["rows"]}
    files = sorted(p for p in hdir.glob("*__*.npy") if not p.name.startswith("stock__") and ".tmp" not in p.name)
    assert files, f"no hidden files in {hdir}"
    sc = Scorer(a.threads)
    runs, red_runs, control = [], [], {"runs": 0, "logits_bit_equal": 0, "max_abs_dlogit": 0.0}
    for p in files:
        rid, arm = p.stem.split("__")
        is_red = (rid, arm) in rows_red
        row = rows_red[(rid, arm)] if is_red else rows_ref[(rid, arm)]
        zr = np.load(K / f"ref/npz/{rid}__{arm}.npz")
        lg_ctl = sc.logits(zr["last_hidden"], row)
        eq = all(lg.tolist() == q["logits"] for lg, q in zip(lg_ctl, row["questions"]))
        dl = max(float(np.abs(np.asarray(lg.tolist()) - np.asarray(q["logits"])).max())
                 for lg, q in zip(lg_ctl, row["questions"]))
        control["runs"] += 1
        control["logits_bit_equal"] += eq
        control["max_abs_dlogit"] = max(control["max_abs_dlogit"], dl)
        hid = np.load(p)
        pr = sc.probs(sc.logits(hid, row))
        c = compare(row, pr)
        c.update(id=rid, arm=arm, source=row["source"], tokens=row["tokens"], hidden_file=str(p.relative_to(K)),
                 control_logits_bit_equal=eq)
        if is_red:
            arm_info = arms[rid]
            base = rows_ref[(arm_info["base_id"], "text")]
            mv = compare(base, pr)
            ref_mv = compare(base, [q["probs"] for q in row["questions"]])
            c["vs_base_reference"] = {"base_id": arm_info["base_id"], "changed_question": arm_info["question_id"],
                                      "max_abs_dp": mv["max_abs_dp"], "moves": mv["max_abs_dp"] > ARM_MIN,
                                      "per_question": {q["question_id"]: q["max_abs_dp"] for q in mv["questions"]},
                                      "argmax_changed": [q["question_id"] for q in mv["questions"] if not q["argmax_equal"]],
                                      "reference_arm_vs_base_max_abs_dp": ref_mv["max_abs_dp"]}
            red_runs.append(c)
        else:
            runs.append(c)
    assert control["max_abs_dlogit"] <= 1e-6, f"scorer is not the reference head: {control}"
    qs = [q for r in runs for q in r["questions"]]
    nn = [q for q in qs if not q["near_tie"]]
    nt = [q for q in qs if q["near_tie"]]
    worst = max(runs, key=lambda r: r["max_abs_dp"]) if runs else None
    summary = {"runs": len(runs), "questions": len(qs), "near_tie_questions": len(nt),
               "argmax_non_near_tie": f"{sum(q['argmax_equal'] for q in nn)}/{len(nn)}",
               "argmax_near_tie": f"{sum(q['argmax_equal'] for q in nt)}/{len(nt)}",
               "max_abs_dp": max(r["max_abs_dp"] for r in runs) if runs else None,
               "max_abs_dp_run": f"{worst['id']}/{worst['arm']}" if worst else None,
               "max_abs_dp_question": (max(worst["questions"], key=lambda q: q["max_abs_dp"])["question_id"] if worst else None),
               "mean_of_run_means": float(np.mean([r["mean_abs_dp"] for r in runs])) if runs else None,
               "p95_option_abs_dp": None,
               "flips": [f"{r['id']}/{q['question_id']}" for r in runs for q in r["questions"] if not q["argmax_equal"]]}
    if runs:
        allo = []
        for r in runs:
            ref_row = rows_ref[(r["id"], r["arm"])]
            for q, qr in zip(r["questions"], ref_row["questions"]):
                allo += np.abs(np.asarray(q["probs"]) - np.asarray(qr["probs"])).tolist()
        summary["p95_option_abs_dp"] = float(np.percentile(allo, 95))
        summary["line_pass"] = bool(all(q["argmax_equal"] for q in nn) and summary["max_abs_dp"] <= MAX_DP
                                    and summary["mean_of_run_means"] <= MEAN_DP)
    red_summary = {"arms": len(red_runs),
                   "moved": [f"{r['id']}: {r['vs_base_reference']['max_abs_dp']:.4f}" for r in red_runs],
                   "all_move": all(r["vs_base_reference"]["moves"] for r in red_runs) if red_runs else None,
                   "vs_own_reference_max_abs_dp": max((r["max_abs_dp"] for r in red_runs), default=None)}
    doc = {"what": f"round 1: {a.hidden_dir} hidden -> author's JointSchemaHead (venv-ref) -> parity line vs the reference",
           "tag": a.tag, "hidden_dir": a.hidden_dir, "scorer": sc.info, "control": control,
           "line": {"near_tie": NEAR_TIE, "max_abs_dp": MAX_DP, "mean_of_run_means": MEAN_DP, "red_arm_min": ARM_MIN},
           "summary": summary, "red_arms": red_summary, "rows": runs, "red_rows": red_runs}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"control": control, "summary": summary, "red_arms": red_summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
